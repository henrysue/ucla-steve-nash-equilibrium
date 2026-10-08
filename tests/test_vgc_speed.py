"""The deterministic turn-order prior (agent/vgc/speed.py).

Turn order drives every doubles decision, so a wrong hint is worse than
none. These tests pin the base-speed ordering, Trick Room reversal, Tailwind
and paralysis multipliers, speed-tie flagging, and — just as important —
that every signal degrades safely to a base-speed prior when the data it
needs is absent. Base speeds come from a tiny fake lookup so the tests do
not depend on the live CSV's exact numbers.
"""

from agent.vgc import speed

# A fixed base-speed lookup for the species these tests use.
_SPEEDS = {
    "fluttermane": 135, "koraidon": 135, "incineroar": 60,
    "amoonguss": 30, "landorustherian": 91, "calyrexice": 50,
}


def _bs(species):
    import re
    return _SPEEDS.get(re.sub(r"[^a-z0-9]", "", (species or "").lower()))


def _order(obs):
    result = speed.turn_order(obs, _bs)
    return [x["species"] for x in result["order_fast_to_slow"]] if result else None


BASE_OBS = {
    "turn": 3,
    "active_pokemon": [{"species": "Incineroar"}, {"species": "Flutter Mane"}],
    "opponent_active_pokemon": [{"species": "Amoonguss"}, {"species": "Landorus-Therian"}],
}


def test_orders_fast_to_slow_by_base_speed():
    order = _order(BASE_OBS)
    assert order[0] == "Flutter Mane"       # 135
    assert order[-1] == "Amoonguss"         # 30
    assert order.index("Landorus-Therian") < order.index("Incineroar")  # 91 > 60


def test_speed_tie_is_flagged_as_fifty_fifty():
    obs = {
        "turn": 1,
        "active_pokemon": [{"species": "Flutter Mane"}],       # 135
        "opponent_active_pokemon": [{"species": "Koraidon"}],  # 135
    }
    result = speed.turn_order(obs, _bs)
    assert result["speed_ties_50_50"] == [["Flutter Mane", "Koraidon"]] or \
           result["speed_ties_50_50"] == [["Koraidon", "Flutter Mane"]]


def test_trick_room_reverses_the_speed_order():
    tr = dict(BASE_OBS, trick_room=True)
    result = speed.turn_order(tr, _bs)
    assert result["order_fast_to_slow"][0]["species"] == "Amoonguss"   # slowest moves first
    assert result.get("trick_room_active") is True
    assert "Trick Room" in result["mode"]


def test_trick_room_detected_in_a_nested_field_container():
    tr = dict(BASE_OBS)
    tr["field"] = {"trickroom": 5}
    assert _order(tr)[0] == "Amoonguss"


def test_trick_room_does_not_change_which_mons_are_present():
    assert set(_order(BASE_OBS)) == set(_order(dict(BASE_OBS, trick_room=True)))


def test_tailwind_doubles_our_speed():
    tw = dict(BASE_OBS, tailwind=True)
    result = speed.turn_order(tw, _bs)
    inc = next(x for x in result["order_fast_to_slow"] if x["species"] == "Incineroar")
    assert inc["effective_speed"] == 120.0   # 60 x 2
    assert result["tailwind"]["ours"] is True


def test_paralysis_halves_speed():
    par = {
        "turn": 2,
        "active_pokemon": [{"species": "Flutter Mane", "status": "par"}],
        "opponent_active_pokemon": [{"species": "Incineroar"}],
    }
    result = speed.turn_order(par, _bs)
    fm = next(x for x in result["order_fast_to_slow"] if x["species"] == "Flutter Mane")
    assert fm["effective_speed"] == 67.5    # 135 x 0.5


def test_basis_labels_the_result_a_prior():
    assert "prior" in speed.turn_order(BASE_OBS, _bs)["basis"].lower()


# -- safe degradation ---------------------------------------------------------


def test_none_observation_returns_none():
    assert speed.turn_order(None, _bs) is None


def test_no_known_speeds_returns_none():
    assert speed.turn_order({"active_pokemon": [{"species": "Unlisted"}]}, _bs) is None


def test_faster_than_is_none_on_a_tie():
    assert speed.faster_than(BASE_OBS, _bs, "Flutter Mane", "Koraidon") is None


def test_faster_than_true_when_clearly_faster():
    assert speed.faster_than(BASE_OBS, _bs, "Flutter Mane", "Amoonguss") is True


def test_faster_than_flips_under_trick_room():
    tr = dict(BASE_OBS, trick_room=True)
    assert speed.faster_than(tr, _bs, "Flutter Mane", "Amoonguss") is False


def test_faster_than_none_on_unknown_species():
    assert speed.faster_than(BASE_OBS, _bs, "Flutter Mane", "Unlisted") is None
