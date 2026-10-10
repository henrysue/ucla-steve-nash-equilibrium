"""Regression tests from match e4546428 (2026-10-09, seat 0), built from its real boards."""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta
from agent.vgc.moverules import MoveTracker

from test_vgc_ec1752bf import _best
from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

TABLE = meta.table()
TYPES = {
    "incineroar": ["FIRE", "DARK"], "urshifurapidstrike": ["FIGHTING", "WATER"], "thundurus": ["ELECTRIC", "FLYING"],
    "tornadus": ["FLYING"], "chiyu": ["DARK", "FIRE"], "ironvaliant": ["FAIRY", "FIGHTING"],
    "archaludon": ["STEEL", "DRAGON"], "arcanine": ["FIRE"],
}


def _obs(ours, theirs, hp=None, turn=1):
    hp = hp or {}
    return {
        "turn": turn,
        "active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in ours],
        "opponent_active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in theirs],
    }


def test_chi_yus_real_attack_is_counted_against_urshifu():
    """Turn 1: Chi-Yu KO'd Urshifu before it moved; the old generic 90-power
    stand-in (no Beads of Ruin) had it barely scratching Urshifu."""
    calc = lookahead.Calculator(_obs(["incineroar", "urshifurapidstrike"], ["chiyu", "ironvaliant"]), TABLE)
    hit = calc.threats()["chiyu"]["urshifurapidstrike"]
    assert hit["expected_pct"] >= 45


FOES_1 = [{"target": 1, "side": "opponent", "species": "chiyu"}, {"target": 2, "side": "opponent", "species": "ironvaliant"}]


def _tracker_after_surging_strikes():
    action = _doubles("incineroar", [_move("knockoff", [1, 2], FOES_1)],
                      "urshifurapidstrike", [_move("surgingstrikes", [1, 2], FOES_1)])
    tracker = MoveTracker()
    tracker.begin(act.parse_doubles(_as_legal(action)), _obs(["incineroar", "urshifurapidstrike"], ["chiyu", "ironvaliant"]))
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "knockoff", "target": 2},
                    "slot_1": {"type": "move", "move_id": "surgingstrikes", "target": 1}})
    return tracker


def test_an_attacker_knocked_out_before_moving_is_no_evidence_of_a_protect():
    """Turn 2: Urshifu fainted first, so Chi-Yu's untouched HP meant nothing."""
    tracker = _tracker_after_surging_strikes()
    action = _doubles("incineroar", [_move("knockoff", [1, 2], FOES_1)],
                      "thundurus", [_move("thunderbolt", [1, 2], FOES_1)])
    tracker.begin(act.parse_doubles(_as_legal(action)),
                  _obs(["incineroar", "thundurus"], ["chiyu", "ironvaliant"], hp={"ironvaliant": 0.6}, turn=2))
    assert "chiyu" not in tracker.protected_last_turn


def test_an_attacker_still_on_the_field_is_still_evidence_of_a_protect():
    tracker = _tracker_after_surging_strikes()
    action = _doubles("incineroar", [_move("knockoff", [1, 2], FOES_1)],
                      "urshifurapidstrike", [_move("surgingstrikes", [1, 2], FOES_1)])
    tracker.begin(act.parse_doubles(_as_legal(action)),
                  _obs(["incineroar", "urshifurapidstrike"], ["chiyu", "ironvaliant"], hp={"ironvaliant": 0.6}, turn=2))
    assert "chiyu" in tracker.protected_last_turn


def test_a_lone_tornadus_attacks_instead_of_stalling():
    """Turns 5-9: a lone Tornadus alternated Protect and Bleakwind Storm into
    full-HP Arcanine and Archaludon; Protect only delayed the same hits."""
    action = _doubles("tornadus", [_move("bleakwindstorm", []), _move("protect", []), _move("tailwind", [])],
                      "thundurus", [{"type": "pass"}], force=(False, True))
    observation = _obs(["tornadus", "thundurus"], ["arcanine", "archaludon"], hp={"tornadus": 0.7, "thundurus": 0.0}, turn=5)
    _, payload = _best(action, observation, {
        "arcanine": {"action": "attack", "target": "tornadus", "confidence": 0.6},
        "archaludon": {"action": "attack", "target": "tornadus", "confidence": 0.6},
    })
    assert payload["slot_0"]["move_id"] != "protect"
