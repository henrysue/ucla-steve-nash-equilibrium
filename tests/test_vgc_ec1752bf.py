"""Regression tests from match ec1752bf (2026-10-09), built from its real boards."""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, predict, rules
from agent.vgc.moverules import MoveTracker

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

TABLE = meta.table()
TYPES = {
    "talonflame": ["FIRE", "FLYING"], "irontreads": ["GROUND", "STEEL"], "hatterene": ["PSYCHIC", "FAIRY"],
    "tinglu": ["DARK", "GROUND"], "landorustherian": ["GROUND", "FLYING"], "fluttermane": ["GHOST", "FAIRY"],
    "gyarados": ["WATER", "FLYING"],
}


def _obs(ours, theirs, hp=None, turn=5):
    hp = hp or {}
    return {
        "turn": turn,
        "active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in ours],
        "opponent_active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in theirs],
    }


def _best(action, observation, prediction, tracker=None):
    slots = act.parse_doubles(_as_legal(action))
    tracker = tracker or MoveTracker(known_moves=TABLE.common_moves)
    turn_rules = rules.build(slots, observation, tracker, TABLE, lambda line: None)
    ours = [e["species"] for e in observation["active_pokemon"]]
    expected = predict.expectations(prediction, ours)
    calc = lookahead.Calculator(observation, TABLE)
    return turn_rules, turn_rules.check(lookahead.best_line(turn_rules, observation, calc, expected))


FOES_TH = [
    {"target": 1, "side": "opponent", "species": "irontreads"},
    {"target": 2, "side": "opponent", "species": "hatterene"},
]


def test_a_lone_talonflame_finishes_the_nearly_dead_foe_instead_of_protecting():
    """Turns 12-14: Protect, Protect, Tailwind while Hatterene sat at 9%."""
    action = _doubles(
        "talonflame", [_move("bravebird", [1, 2], FOES_TH), _move("protect", []), _move("tailwind", []), _move("taunt", [1, 2], FOES_TH)],
        "fluttermane", [{"type": "pass"}], force=(False, True),
    )
    observation = _obs(["talonflame", "fluttermane"], ["irontreads", "hatterene"],
                       hp={"irontreads": 0.02, "hatterene": 0.09, "fluttermane": 0.0})
    _, payload = _best(action, observation, {
        "irontreads": {"action": "protect", "confidence": 0.55},
        "hatterene": {"action": "attack", "target": "talonflame", "confidence": 0.55},
    })
    assert payload["slot_0"] == {"type": "move", "move_id": "bravebird", "target": 2}


def test_a_second_protect_in_a_row_is_never_played():
    """Turns 3-4 (seat 1): Gyarados and Ting-Lu double-Protected twice in a row."""
    foes = [{"target": 1, "side": "opponent", "species": "landorustherian"},
            {"target": 2, "side": "opponent", "species": "fluttermane"}]
    action = _doubles("gyarados", [_move("waterfall", [1, 2], foes), _move("protect", [])],
                      "tinglu", [_move("ruination", [1, 2], foes), _move("protect", [])])
    observation = _obs(["gyarados", "tinglu"], ["landorustherian", "fluttermane"], hp={"gyarados": 0.44})
    prediction = {"landorustherian": {"action": "attack", "target": "gyarados", "confidence": 0.65},
                  "fluttermane": {"action": "attack", "target": "tinglu", "confidence": 0.5}}
    slots = act.parse_doubles(_as_legal(action))
    tracker = MoveTracker(known_moves=TABLE.common_moves)
    tracker.begin(slots, observation)
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "protect"},
                    "slot_1": {"type": "move", "move_id": "protect"}})
    turn_rules, payload = _best(action, observation, prediction, tracker)
    assert turn_rules.per_slot[0].protect_success == 1 / 3
    assert payload["slot_0"].get("move_id") != "protect" and payload["slot_1"].get("move_id") != "protect"


def test_a_choice_locked_move_every_foe_resists_is_switched_out_of():
    """Turns 6-10 (seat 0): Landorus-T locked into Rock Slide into Iron Treads (0.25x) and Ting-Lu (0.5x)."""
    action = _doubles("landorustherian", [_move("rockslide", []), {"type": "switch", "species": "talonflame"}],
                      "fluttermane", [_move("moonblast", [1, 2], [
                          {"target": 1, "side": "opponent", "species": "irontreads"},
                          {"target": 2, "side": "opponent", "species": "tinglu"}])])
    observation = _obs(["landorustherian", "fluttermane"], ["irontreads", "tinglu"])
    turn_rules, payload = _best(action, observation, {
        "irontreads": {"action": "attack", "target": "fluttermane", "confidence": 0.6},
        "tinglu": {"action": "attack", "target": "fluttermane", "confidence": 0.55},
    })
    assert "resisted by every foe" in turn_rules.per_slot[0].blocked[0]
    assert payload["slot_0"] == {"type": "switch", "species": "talonflame"}


def test_rising_hp_after_our_hit_is_not_read_as_a_protect():
    """Turns 5-9 (seat 0): Ting-Lu's HP rose (Leftovers) and was misread as a Protect."""
    foes = [{"target": 1, "side": "opponent", "species": "gyarados"}, {"target": 2, "side": "opponent", "species": "tinglu"}]
    action = _doubles("landorustherian", [_move("rockslide", [])], "fluttermane", [_move("moonblast", [1, 2], foes)])
    slots = act.parse_doubles(_as_legal(action))
    tracker = MoveTracker()
    tracker.begin(slots, _obs(["landorustherian", "fluttermane"], ["gyarados", "tinglu"], hp={"tinglu": 0.62}))
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "rockslide"},
                    "slot_1": {"type": "move", "move_id": "moonblast", "target": 2}})
    tracker.begin(slots, _obs(["landorustherian", "fluttermane"], ["gyarados", "tinglu"], hp={"tinglu": 0.68}))
    assert "tinglu" not in tracker.protected_last_turn
