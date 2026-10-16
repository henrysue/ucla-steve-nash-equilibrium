"""Regression tests from match a938ac5c (2026-10-16), a win over Riffo.

Turn 1 the strategist vetoed the code's pick (Chien-Pao Protect, 53.7) for Icicle
Crash into Flutter Mane (50.4), calling it a "sure KO" while its own read had
Indeedee-F using Follow Me (55%), which would have pulled that very attack. A veto is
now locked in on a second, thinking look that sees the code's pick beside it, the
value gap, and the code's checks of the reason against the strategist's own read.
"""

import json

from agent.vgc import actions as act
from agent.vgc.strategist import DECISION_SECONDS, MIN_LOCK_IN_SECONDS, _name_payload, _veto_checks

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import (BATTLE_OBSERVATION, POKEMON, FakeClock, FakeProvider, _agent, _doubles_action,
                                 _doubles_answer, _logged, _move, _review, _state)


def _veto_state():
    return _state(_doubles_action(), observation=BATTLE_OBSERVATION)


# -- the lock-in --------------------------------------------------------------------------------


def test_a_veto_is_sent_only_once_locked_in_on_a_thinking_look():
    provider = FakeProvider(_doubles_answer(), _review("veto", "2"))
    agent = _agent(provider)
    state = _veto_state()
    decision = agent.choose_action(state, POKEMON)
    review = json.loads(provider.calls[1]["messages"][1]["content"])
    assert len(provider.lock_ins) == 1 and provider.lock_ins[0]["effort"] != "none"  # it thinks
    lock = json.loads(provider.lock_ins[0]["messages"][1]["content"])
    veto = lock["veto"]
    assert veto["code_pick"] == review["options"][0]["line"] and veto["your_line"] == review["options"][1]["line"]
    assert veto["value_gap"] >= 0 and veto["your_reason"] and veto["checks"]
    assert "your_read" in lock and "game_plan" in lock
    assert _name_payload(decision.action, act.parse_doubles(state.legal_actions[0])) == veto["your_line"]


def test_a_withdrawn_veto_plays_the_codes_pick(tmp_path):
    provider = FakeProvider(_doubles_answer(), _review("veto", "2"),
                            {"lock_in": False, "why": "the Follow Me read pulls that attack", "reasoning_summary": ""})
    text, records = _logged(tmp_path, provider, _veto_state())
    analysis = records[0]["analysis"]
    assert analysis["played"] == 0
    assert analysis["review"]["withdrawn"] == 2 and analysis["review"]["lock_in"]["locked"] is False
    assert "withdrawn at lock-in: played option 1" in text and "**Lock-in:** withdrawn" in text


def test_no_time_to_lock_in_plays_the_codes_pick():
    clock = FakeClock()
    provider = FakeProvider(_doubles_answer(), _review("veto", "2"))
    original = provider.complete_structured

    def slow(messages, schema_name, *args, **kwargs):
        if schema_name == "doubles_turn_review":
            clock.now += DECISION_SECONDS - MIN_LOCK_IN_SECONDS + 1  # the final call used up the budget
        return original(messages, schema_name, *args, **kwargs)

    provider.complete_structured = slow
    log = []
    agent = _agent(provider, log=log, clock=clock)
    state = _veto_state()
    decision = agent.choose_action(state, POKEMON)
    assert not provider.lock_ins
    assert any("withdrew its veto" in line and "no time to lock in" in line for line in log)
    review = json.loads(provider.calls[1]["messages"][1]["content"])
    assert _name_payload(decision.action, act.parse_doubles(state.legal_actions[0])) == review["options"][0]["line"]


# -- the checks of the reason against the strategist's own read -----------------------------------


F = [{"target": 1, "side": "opponent", "species": "fluttermane"}, {"target": 2, "side": "opponent", "species": "indeedeef"}]
SLOTS = act.parse_doubles(_as_legal(_doubles("ironhands", [_move("closecombat", [1, 2], F)],
                                             "chienpao", [_move("iciclecrash", [1, 2], F)])))
TURN_1 = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "closecombat", "target": 2},
          "slot_1": {"type": "move", "move_id": "iciclecrash", "target": 1}}


def test_a_follow_me_read_flags_the_attack_it_pulls():
    read = {"fluttermane": {"action": "attack", "target": "chienpao", "move": "Moonblast", "confidence": 0.55},
            "indeedeef": {"action": "support", "move": "Follow Me", "confidence": 0.55}}
    checks = _veto_checks(TURN_1, read, SLOTS)
    assert len(checks) == 1 and "Icicle Crash at fluttermane is pulled into indeedeef" in checks[0]


def test_a_protect_read_flags_the_wasted_attack():
    read = {"fluttermane": {"action": "protect", "move": "Protect", "confidence": 0.6},
            "indeedeef": {"action": "attack", "target": "ironhands", "move": "Psychic", "confidence": 0.5}}
    checks = _veto_checks(TURN_1, read, SLOTS)
    assert checks == ["Your read: fluttermane Protects (60%). chienpao's Icicle Crash into it is wasted."]


def test_no_checks_when_the_read_agrees_with_the_line():
    read = {"fluttermane": {"action": "attack", "target": "chienpao", "move": "Moonblast", "confidence": 0.55},
            "indeedeef": {"action": "attack", "target": "ironhands", "move": "Dazzling Gleam", "confidence": 0.5}}
    assert _veto_checks(TURN_1, read, SLOTS) == []
