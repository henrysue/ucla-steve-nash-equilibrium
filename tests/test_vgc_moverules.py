"""The move tracker: history-dependent moves (Fake Out, Protect chains) and the
simplified turn flow built around it.

The first tests replay the two wasted Fake Outs from the 2026-10-08 match
logs: Incineroar on turn 2 after using it on turn 1, and Rillaboom on turn 4
after being on the field since turn 1.
"""

import json

from agent.vgc import actions as act
from agent.vgc.moverules import MoveTracker
from agent.vgc.providers import ProviderError

from test_vgc_strategist import (
    BATTLE_OBSERVATION,
    POKEMON,
    FakeClock,
    FakeProvider,
    _agent,
    _doubles_action,
    _doubles_answer,
    _move,
    _state,
    _templated,
    FOES,
)


def _slots(action=None):
    return act.parse_doubles((action or _as_legal(_doubles_action())))


def _as_legal(raw):
    from altruagent.models import LegalAction

    return LegalAction.from_dict(raw)


def _doubles(slot_0_active, slot_0_options, slot_1_active, slot_1_options, *, force=(False, False)):
    return _templated("doubles_turn", {
        "type": "doubles_turn",
        "slots": [
            {"slot": 0, "active": slot_0_active, "force_switch": force[0], "options": slot_0_options},
            {"slot": 1, "active": slot_1_active, "force_switch": force[1], "options": slot_1_options},
        ],
    })


def _move_payload(slot_0, slot_1="moonblast"):
    return {
        "type": "doubles_turn",
        "slot_0": {"type": "move", "move_id": slot_0, "target": 1},
        "slot_1": {"type": "move", "move_id": slot_1, "target": 1},
    }


FAKEOUT_INDEX = 0  # in _doubles_action, slot 0's first option is Fake Out
PROTECT_INDEX = 2


# -- the tracker on its own ----------------------------------------------------------------


def test_fake_out_works_on_the_first_turn_only():
    """Replays match 2b63 seat0: Fake Out turn 1, then again on turn 2."""
    tracker = MoveTracker()
    slots = _slots()
    rules = tracker.begin(slots, BATTLE_OBSERVATION)
    assert rules[0].fresh and FAKEOUT_INDEX not in rules[0].blocked
    tracker.record(_move_payload("fakeout"))

    rules = tracker.begin(slots, BATTLE_OBSERVATION)
    assert not rules[0].fresh
    assert "already moved" in rules[0].blocked[FAKEOUT_INDEX]


def test_fake_out_stays_blocked_however_long_the_pokemon_stays_in():
    """Replays match 93a9: Rillaboom out since turn 1, Fake Out on turn 4."""
    tracker = MoveTracker()
    slots = _slots()
    for _ in range(3):
        tracker.begin(slots, BATTLE_OBSERVATION)
        tracker.record(_move_payload("flareblitz"))
    assert FAKEOUT_INDEX in tracker.begin(slots, BATTLE_OBSERVATION)[0].blocked


def test_a_pokemon_that_comes_back_in_can_fake_out_again():
    tracker = MoveTracker()
    first = _slots()
    tracker.begin(first, BATTLE_OBSERVATION)
    tracker.record(_move_payload("fakeout"))
    # Incineroar switches out for Rillaboom...
    away = act.parse_doubles(_as_legal(_doubles(
        "rillaboom", [_move("fakeout", [1, 2], FOES)], "fluttermane", [_move("moonblast", [1, 2], FOES)],
    )))
    assert away[0].options and tracker.begin(away, {})[0].fresh
    tracker.record(_move_payload("woodhammer"))
    # ...and back in: a new stint on the field.
    assert tracker.begin(first, BATTLE_OBSERVATION)[0].fresh


def test_a_replacement_after_a_faint_gets_its_fake_out():
    tracker = MoveTracker()
    tracker.begin(_slots(), BATTLE_OBSERVATION)
    tracker.record(_move_payload("flareblitz"))
    # Incineroar faints: a forced decision where slot 0 sends Rillaboom in.
    forced = act.parse_doubles(_as_legal(_doubles(
        "incineroar", [{"type": "switch", "species": "rillaboom"}], "fluttermane", [{"type": "pass"}],
        force=(True, False),
    )))
    tracker.begin(forced, {})
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "switch", "species": "rillaboom"}, "slot_1": {"type": "pass"}})
    nxt = act.parse_doubles(_as_legal(_doubles(
        "rillaboom", [_move("fakeout", [1, 2], FOES)], "fluttermane", [_move("moonblast", [1, 2], FOES)],
    )))
    rules = tracker.begin(nxt, {})
    assert rules[0].fresh and not rules[0].blocked
    # Flutter Mane passed the forced decision, but it had already moved.
    assert not rules[1].fresh


def test_a_second_protect_in_a_row_is_withheld():
    """Match fb4dcbc1 (seat 0): Tornadus protected on turns 5 and 6; the second failed."""
    tracker = MoveTracker()
    slots = _slots()
    rules = tracker.begin(slots, BATTLE_OBSERVATION)
    assert PROTECT_INDEX not in rules[0].blocked
    tracker.record(_move_payload("protect"))
    rules = tracker.begin(slots, BATTLE_OBSERVATION)
    assert rules[0].protect_success == 1 / 3 and PROTECT_INDEX in rules[0].blocked
    assert "protect #2 in a row" in rules[0].blocked[PROTECT_INDEX]
    tracker.record(_move_payload("flareblitz"))
    assert PROTECT_INDEX not in tracker.begin(slots, BATTLE_OBSERVATION)[0].blocked


def test_any_other_move_resets_the_protect_chain():
    tracker = MoveTracker()
    slots = _slots()
    for move in ("protect", "flareblitz", "protect"):
        tracker.begin(slots, BATTLE_OBSERVATION)
        tracker.record(_move_payload(move))
    assert tracker.begin(slots, BATTLE_OBSERVATION)[0].protect_chain == 1


def test_blood_moon_cannot_repeat():
    action = _as_legal(_doubles(
        "ursalunabloodmoon", [_move("bloodmoon", [1, 2], FOES), _move("earthpower", [1, 2], FOES)],
        "fluttermane", [_move("moonblast", [1, 2], FOES)],
    ))
    tracker = MoveTracker()
    slots = act.parse_doubles(action)
    tracker.begin(slots, {})
    tracker.record(_move_payload("bloodmoon"))
    assert 0 in tracker.begin(slots, {})[0].blocked
    tracker.record(_move_payload("earthpower"))
    assert 0 not in tracker.begin(slots, {})[0].blocked


def test_a_slot_is_never_left_with_nothing():
    action = _as_legal(_doubles("incineroar", [_move("fakeout", [1, 2], FOES)], "fluttermane", [{"type": "pass"}]))
    tracker = MoveTracker()
    slots = act.parse_doubles(action)
    tracker.begin(slots, {})
    tracker.record(_move_payload("fakeout"))
    assert tracker.begin(slots, {})[0].blocked == {}


def test_opposing_fake_out_threat_lasts_one_turn():
    tracker = MoveTracker(known_moves=lambda species: ["Fake Out", "Knock Off"] if species == "amoonguss" else [])
    slots = _slots()
    rules = tracker.begin(slots, BATTLE_OBSERVATION)
    theirs = tracker.snapshot(rules)["theirs"]
    assert theirs["amoonguss"] == {"first_turn_on_field": True, "fake_out_threat": True}
    assert theirs["landorustherian"]["fake_out_threat"] is False
    tracker.record(_move_payload("flareblitz"))
    rules = tracker.begin(slots, BATTLE_OBSERVATION)
    assert tracker.snapshot(rules)["theirs"]["amoonguss"]["fake_out_threat"] is False


def test_special_moves_used_are_stored_in_order():
    tracker = MoveTracker()
    slots = _slots()
    tracker.begin(slots, BATTLE_OBSERVATION)
    tracker.record(_move_payload("fakeout", "icywind"))
    tracker.begin(slots, BATTLE_OBSERVATION)
    tracker.record(_move_payload("protect"))
    assert [(u["species"], u["move"]) for u in tracker.used] == [("incineroar", "fakeout"), ("incineroar", "protect")]


# -- describing, validating and the heuristic ------------------------------------------------


def _stale_rules():
    tracker = MoveTracker()
    slots = _slots()
    tracker.begin(slots, BATTLE_OBSERVATION)
    tracker.record(_move_payload("fakeout"))
    return slots, tracker.begin(slots, BATTLE_OBSERVATION)


def test_the_model_never_sees_a_withheld_option_and_indices_keep_their_numbers():
    slots, rules = _stale_rules()
    shown = act.describe_slots(slots, _state(observation=BATTLE_OBSERVATION), rules)
    assert [o["option"] for o in shown[0]["options"]] == [1, 2, 3]


def test_a_withheld_option_is_rejected_if_named_anyway():
    slots, rules = _stale_rules()
    answer = {"slot_0": {"option": FAKEOUT_INDEX, "target": 1}, "slot_1": {"option": 0, "target": 1}}
    try:
        act.build_doubles(slots, answer, rules)
    except act.InvalidAnswer as exc:
        assert "already moved" in str(exc)
    else:
        raise AssertionError("a stale Fake Out was accepted")


# -- the strategist's simplified turn ----------------------------------------------------


def test_forced_switch_is_decided_without_a_model_call():
    provider = FakeProvider()
    action = _doubles(
        "fluttermane", [{"type": "switch", "species": "rillaboom"}], "incineroar", [{"type": "pass"}],
        force=(True, False),
    )
    decision = _agent(provider).choose_action(_state(action, observation=BATTLE_OBSERVATION), POKEMON)
    assert provider.calls == []
    assert decision.action["slot_0"] == {"type": "switch", "species": "rillaboom"}


def test_fake_out_is_not_played_on_the_second_turn(tmp_path):
    import json as _json
    import random as _random

    from agent.vgc.strategist import VGCDraftStrategist

    agent = VGCDraftStrategist(FakeProvider(_doubles_answer(), _doubles_answer()), log=[].append,
                               rng=_random.Random(1), clock=lambda: 0.0, log_dir=str(tmp_path))
    opening = {**BATTLE_OBSERVATION, "turn": 1}
    first = agent.choose_action(_state(_doubles_action(), observation=opening), POKEMON)
    second = agent.choose_action(_state(_doubles_action(), observation={**BATTLE_OBSERVATION, "turn": 2}), POKEMON)
    assert first.action["slot_0"].get("move_id") == "fakeout"  # Landorus-T threatens a KO: worth a Fake Out
    assert second.action["slot_0"].get("move_id") != "fakeout"  # and now Incineroar has moved
    records = [_json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert records[1]["move_state"]["ours"]["incineroar"]["first_turn_moves_work"] is False


def test_no_retry_when_less_than_half_the_budget_is_left():
    clock = FakeClock()

    class SlowThenBad(FakeProvider):
        def complete_structured(self, *args, **kwargs):
            clock.now += 25  # 25 s of a 40 s turn, then an unusable answer
            return super().complete_structured(*args, **kwargs)

    provider = SlowThenBad(ProviderError("garbled"), _doubles_answer())
    log = []
    _agent(provider, clock=clock, log=log).choose_action(
        _state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON
    )
    assert len(provider.calls) == 1
    assert any("FALLBACK" in line and "No usable model prediction" in line for line in log)


def test_nothing_strategic_reaches_the_public_summary():
    provider = FakeProvider(_doubles_answer(reasoning_summary="Double into Landorus; Amoonguss in back."))
    decision = _agent(provider).choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    assert decision.reasoning_summary == "Move submitted."


def test_battle_turns_ask_for_no_thinking_and_draft_keeps_it():
    from agent.vgc import decisions

    assert decisions.predict_decision([], [], {}).effort == "none"


def test_move_state_is_written_to_the_jsonl(tmp_path):
    from agent.vgc.strategist import VGCDraftStrategist
    import random

    agent = VGCDraftStrategist(FakeProvider(_doubles_answer()), log=[].append, rng=random.Random(1),
                               clock=lambda: 0.0, log_dir=str(tmp_path))
    agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    record = json.loads(next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()[-1])
    assert record["move_state"]["ours"]["incineroar"]["first_turn_moves_work"] is True


# -- the provider's no-thinking path -------------------------------------------------------


def _recording_provider(reject_disabled):
    import types

    from agent.vgc.providers import AnthropicProvider

    sdk = types.SimpleNamespace(
        NOT_GIVEN=object(),
        BadRequestError=type("BadRequestError", (Exception,), {}),
        APIStatusError=type("APIStatusError", (Exception,), {}),
        APIError=type("APIError", (Exception,), {}),
    )
    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            if reject_disabled and kwargs["thinking"]["type"] == "disabled":
                raise sdk.BadRequestError("thinking.type disabled is not supported")
            text = types.SimpleNamespace(type="text", text='{"ok": true}')
            return types.SimpleNamespace(stop_reason="end_turn", content=[text])

    class Client:
        messages = Messages()

        def with_options(self, **_kwargs):
            return self

    provider = object.__new__(AnthropicProvider)
    provider.model, provider.last_reasoning = "claude-sonnet-5", None
    provider._anthropic, provider._client, provider._thinking_off_ok = sdk, Client(), True
    return provider, calls


MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def test_no_thinking_sends_disabled_thinking_at_low_effort():
    provider, calls = _recording_provider(reject_disabled=False)
    assert provider.complete_structured(MESSAGES, "t", {}, timeout=30, effort="none") == {"ok": True}
    assert calls[0]["thinking"] == {"type": "disabled"}
    assert calls[0]["output_config"]["effort"] == "low"


def test_a_model_that_needs_thinking_falls_back_once_and_remembers():
    provider, calls = _recording_provider(reject_disabled=True)
    provider.complete_structured(MESSAGES, "t", {}, timeout=30, effort="none")
    provider.complete_structured(MESSAGES, "t", {}, timeout=30, effort="none")
    assert [c["thinking"]["type"] for c in calls] == ["disabled", "adaptive", "adaptive"]
