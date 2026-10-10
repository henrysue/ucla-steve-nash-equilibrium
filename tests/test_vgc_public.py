"""--show-reasoning / VGC_PUBLIC_REASONING: what the platform's public summary carries."""

from test_vgc_strategist import (
    BATTLE_OBSERVATION,
    DRAFT_OBSERVATION,
    POKEMON,
    FakeProvider,
    _agent,
    _doubles_action,
    _doubles_answer,
    _draft_actions,
    _state,
)


def test_by_default_the_platform_gets_a_neutral_line(monkeypatch):
    monkeypatch.delenv("VGC_PUBLIC_REASONING", raising=False)
    decision = _agent(FakeProvider(_doubles_answer())).choose_action(
        _state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON
    )
    assert decision.reasoning_summary == "Move submitted."


def test_with_the_flag_a_draft_pick_sends_the_models_note(monkeypatch):
    monkeypatch.setenv("VGC_PUBLIC_REASONING", "1")
    provider = FakeProvider({"action_id": "draft_pick:c2", "reasoning_summary": "Denies their wall."})
    decision = _agent(provider).choose_action(
        _state(*_draft_actions("c1", "c2"), phase="draft", observation=DRAFT_OBSERVATION), POKEMON
    )
    assert decision.reasoning_summary == "Denies their wall."


def test_with_the_flag_a_battle_turn_sends_the_read_and_the_play(monkeypatch):
    monkeypatch.setenv("VGC_PUBLIC_REASONING", "1")
    decision = _agent(FakeProvider(_doubles_answer())).choose_action(
        _state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON
    )
    text = decision.reasoning_summary
    assert text.startswith("Read: ") and "amoonguss protect" in text and "Play: slot 0:" in text
    assert len(text) <= 280


def test_the_models_private_thinking_is_never_sent_even_with_the_flag(monkeypatch):
    monkeypatch.setenv("VGC_PUBLIC_REASONING", "1")
    secret = "SECRET-DELIBERATION they will switch, so I bait it"
    decision = _agent(FakeProvider(_doubles_answer(), reasoning=secret)).choose_action(
        _state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON
    )
    assert secret not in decision.reasoning_summary


def test_the_cli_flag_turns_it_on_for_the_match_workers(monkeypatch):
    import agent.__main__ as cli

    monkeypatch.delenv("VGC_PUBLIC_REASONING", raising=False)
    monkeypatch.setattr(cli, "_check_tournament", lambda spec: 0)
    assert cli.main(["--check-tournament", "--show-reasoning"]) == 0
    import os

    assert os.environ["VGC_PUBLIC_REASONING"] == "1"
