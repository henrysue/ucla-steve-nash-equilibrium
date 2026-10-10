"""lookahead.best_line: the deterministic fallback and the model's anchor row."""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, rules
from agent.vgc.predict import FoeExpectation
from agent.vgc.moverules import MoveTracker

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

TABLE = meta.table()
FOES = [
    {"target": 1, "side": "opponent", "species": "landorustherian"},
    {"target": 2, "side": "opponent", "species": "amoonguss"},
]
TYPES = {
    "incineroar": ["FIRE", "DARK"], "fluttermane": ["GHOST", "FAIRY"], "landorustherian": ["GROUND", "FLYING"],
    "amoonguss": ["GRASS", "POISON"], "rillaboom": ["GRASS"], "gyarados": ["WATER", "FLYING"], "tinglu": ["DARK", "GROUND"],
}


def _observation(ours, theirs, *, hp=None, turn=3):
    hp = hp or {}
    return {
        "turn": turn,
        "active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": 1.0} for s in ours],
        "opponent_active_pokemon": [
            {"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in theirs
        ],
    }


def _best(action, observation, odds=None):
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(offense=TABLE.offense), TABLE, lambda line: None)
    answer = lookahead.best_line(turn_rules, observation, lookahead.Calculator(observation, TABLE), odds)
    return turn_rules.check(answer)


def test_never_attacks_into_an_immunity():
    # Thunderbolt-like Electric move: Landorus-T (Ground) is immune.
    action = _doubles("incineroar", [_move("thunderpunch", [1, 2], FOES)], "fluttermane", [_move("moonblast", [1, 2], FOES)])
    payload = _best(action, _observation(["incineroar", "fluttermane"], ["landorustherian", "amoonguss"]))
    assert payload["slot_0"]["target"] == 2


def test_prefers_the_super_effective_target():
    action = _doubles("incineroar", [_move("flareblitz", [1, 2], FOES)], "fluttermane", [_move("moonblast", [1, 2], FOES)])
    payload = _best(action, _observation(["incineroar", "fluttermane"], ["landorustherian", "amoonguss"]))
    assert payload["slot_0"] == {"type": "move", "move_id": "flareblitz", "target": 2}  # 2x on Amoonguss


def test_finishes_a_weakened_foe():
    action = _doubles("fluttermane", [_move("moonblast", [1, 2], FOES)], "incineroar", [_move("protect", [])])
    observation = _observation(["fluttermane", "incineroar"], ["landorustherian", "amoonguss"], hp={"landorustherian": 0.15})
    assert _best(action, observation)["slot_0"]["target"] == 1


def test_prefers_damage_over_a_plain_status_move():
    action = _doubles("incineroar", [_move("willowisp", [1, 2], FOES), _move("flareblitz", [1, 2], FOES)],
                      "fluttermane", [_move("moonblast", [1, 2], FOES)])
    payload = _best(action, _observation(["incineroar", "fluttermane"], ["landorustherian", "amoonguss"]))
    assert payload["slot_0"]["move_id"] == "flareblitz"


def test_the_target_less_likely_to_protect_is_hit():
    # One attacker, so the choice is only the target (with two, splitting the
    # attacks can beat doubling into the open foe, which the race now sees).
    action = _doubles("incineroar", [_move("flareblitz", [1, 2], FOES)], "fluttermane", [_move("protect", [])])
    observation = _observation(["incineroar", "fluttermane"], ["landorustherian", "amoonguss"])
    payload = _best(action, observation, {"amoonguss": FoeExpectation(protect=0.8), "landorustherian": FoeExpectation(protect=0.05)})
    assert payload["slot_0"]["target"] == 1


def test_sets_up_when_every_foe_will_protect():
    action = _doubles("incineroar", [_move("flareblitz", [1, 2], FOES), _move("tailwind", [])],
                      "fluttermane", [_move("moonblast", [1, 2], FOES)])
    observation = _observation(["incineroar", "fluttermane"], ["landorustherian", "amoonguss"])
    expected = {"amoonguss": FoeExpectation(protect=0.9), "landorustherian": FoeExpectation(protect=0.9)}
    assert _best(action, observation, expected)["slot_0"]["move_id"] == "tailwind"


def test_never_earthquakes_beside_a_partner_that_is_not_immune():
    foes = [{"target": 1, "side": "opponent", "species": "amoonguss"}, {"target": 2, "side": "opponent", "species": "incineroar"}]
    action = _doubles("tinglu", [_move("earthquake", []), _move("ruination", [1, 2], foes)],
                      "rillaboom", [_move("woodhammer", [1, 2], foes)])
    payload = _best(action, _observation(["tinglu", "rillaboom"], ["amoonguss", "incineroar"]))
    assert payload["slot_0"]["move_id"] != "earthquake"


def test_a_forced_switch_sends_the_safest_replacement():
    action = _doubles("fluttermane", [{"type": "switch", "species": "incineroar"}, {"type": "switch", "species": "gyarados"}],
                      "rillaboom", [{"type": "pass"}], force=(True, False))
    observation = _observation(["fluttermane", "rillaboom"], ["landorustherian", "tinglu"])
    # Ground attackers: Incineroar takes 2x, Gyarados (Flying) is immune to Ground.
    assert _best(action, observation)["slot_0"] == {"type": "switch", "species": "gyarados"}


def test_passes_when_pass_is_the_only_option_and_never_double_switches():
    action = _doubles("fluttermane", [{"type": "switch", "species": "gyarados"}],
                      "rillaboom", [{"type": "switch", "species": "gyarados"}, {"type": "pass"}], force=(True, True))
    payload = _best(action, _observation(["fluttermane", "rillaboom"], ["landorustherian", "tinglu"]))
    assert payload["slot_0"] == {"type": "switch", "species": "gyarados"} and payload["slot_1"] == {"type": "pass"}


def test_opening_turn_uses_fake_out():
    action = _doubles("incineroar", [_move("fakeout", [1, 2], FOES), _move("flareblitz", [1, 2], FOES)],
                      "fluttermane", [_move("moonblast", [1, 2], FOES)])
    observation = _observation(["incineroar", "fluttermane"], ["landorustherian", "amoonguss"], turn=1)
    assert _best(action, observation)["slot_0"]["move_id"] == "fakeout"
