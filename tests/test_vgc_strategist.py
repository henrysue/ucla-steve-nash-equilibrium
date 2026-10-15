"""The strategist end to end: draft, Team Preview, a doubles turn, fallbacks.

Nothing here touches the network. A scripted provider stands in for the
model so the deterministic half of the agent — validation, the equilibrium
solve, the opponent model, the clock, the fallbacks — is what gets tested.
"""

import json
import os
import random

import pytest

from altruagent import DecisionContext, WithReasoning
from altruagent.models import GameState
from agent.vgc import actions as act
from agent.vgc.providers import ProviderError
from agent.vgc.strategist import VGCDraftStrategist

POKEMON = DecisionContext(
    session_id="s", tournament_id=None, game_type="pokemon_vgc_doubles_draft",
    agent_id="me", seat_position=0,
)


# -- fixtures ------------------------------------------------------------------------------


class FakeProvider:
    """Returns scripted answers in order; an Exception value is raised."""

    model = "fake-model"

    def __init__(self, *answers, reasoning=None):
        self.answers = list(answers)
        self.calls = []
        self.reviews = []  # unscripted final-call reviews, answered "play"
        # Both real providers expose these: a repr that cannot leak a key,
        # and the model's private reasoning for the audit log.
        self.last_reasoning = reasoning

    def __repr__(self):
        return f"FakeProvider(model={self.model!r})"

    def complete_structured(self, messages, schema_name, schema, *, timeout, effort="medium"):
        if schema_name == "doubles_turn_review" and not (
            self.answers and isinstance(self.answers[0], dict) and "verdict" in self.answers[0]
        ):
            # The strategist's final call: unless a test scripts one, it plays the code's pick.
            self.reviews.append({"messages": messages, "schema": schema, "timeout": timeout})
            return {"verdict": "play", "play_option": "1", "why": "", "reasoning_summary": ""}
        self.calls.append({"messages": messages, "schema": schema, "timeout": timeout, "effort": effort})
        answer = self.answers.pop(0) if self.answers else ProviderError("no more scripted answers")
        if isinstance(answer, Exception):
            raise answer
        return answer

    def prompt(self, index=0):
        return json.loads(self.calls[index]["messages"][1]["content"])


class FakeClock:
    """A clock the test advances, so budgets can be exercised without sleeping."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _agent(provider=None, *, log=None, rng_seed=1, clock=None):
    return VGCDraftStrategist(
        provider,
        log=(log if log is not None else []).append,
        rng=random.Random(rng_seed),
        clock=clock or (lambda: 0.0),
    )


def _state(*legal, phase="moving", observation=None):
    return GameState.from_mcp_state({
        "session_id": "s",
        "status": "in_progress",
        "phase": phase,
        "is_current_actor": True,
        "is_terminal": False,
        "state_version": 3,
        "observation": observation or {},
        "legal_actions": {"session_id": "s", "state_version": 3, "actions": list(legal)},
    })


def _templated(action_id, template):
    return {"action_id": action_id, "label": action_id, "input": {"action_id": action_id, "action": template}}


def _as_action(template):
    """A bare LegalAction wrapping a template, for unit-testing the builders."""
    from altruagent.models import LegalAction

    return LegalAction.from_dict(_templated(template["type"], template))


def _draft_actions(*card_ids):
    return [
        {
            "action_id": f"draft_pick:{card}",
            "label": f"Draft {card}",
            "input": {"action_id": f"draft_pick:{card}", "action": {"type": "draft_pick", "card_id": card}},
        }
        for card in card_ids
    ]


ROSTER = ["incineroar", "amoonguss", "rillaboom", "fluttermane", "landorustherian", "ursaluna"]


def _lineup_action():
    return _templated("select_lineup", {"type": "select_lineup", "roster": ROSTER, "instructions": "Bring 4, lead 2."})


def _move(move_id, targets, target_options=None):
    """A template option: the server offers move_id and legal targets only.

    The move's element type, power and accuracy live in the observation's
    ``available_moves``, exactly as the platform's schema has them.
    """
    option = {"type": "move", "move_id": move_id, "targets": list(targets)}
    if target_options is not None:
        option["target_options"] = target_options
    return option


FOES = [
    {"target": 1, "side": "opponent", "species": "landorustherian"},
    {"target": 2, "side": "opponent", "species": "amoonguss"},
]


def _doubles_action():
    return _templated("doubles_turn", {
        "type": "doubles_turn",
        "target_legend": "1/2 = opponent A/B, -1/-2 = your slots, 0 = none",
        "instructions": "Choose for both slots.",
        "slots": [
            {
                "slot": 0,
                "active": "incineroar",
                "force_switch": False,
                "options": [
                    _move("fakeout", [1, 2], FOES),
                    _move("flareblitz", [1, 2], FOES),
                    _move("protect", []),
                    {"type": "switch", "species": "rillaboom"},
                ],
            },
            {
                "slot": 1,
                "active": "fluttermane",
                "force_switch": False,
                "options": [
                    _move("moonblast", [1, 2], FOES),
                    _move("icywind", []),
                    {"type": "switch", "species": "amoonguss"},
                ],
            },
        ],
    })


BATTLE_OBSERVATION = {
    "is_doubles": True,
    "turn": 3,
    "available_moves": [
        [
            {"id": "fakeout", "type": "NORMAL", "category": "PHYSICAL", "base_power": 40, "accuracy": 1.0, "priority": 3},
            {"id": "flareblitz", "type": "FIRE", "category": "PHYSICAL", "base_power": 120, "accuracy": 1.0},
            {"id": "protect", "type": "NORMAL", "category": "STATUS", "base_power": 0, "accuracy": 1.0},
        ],
        [
            {"id": "moonblast", "type": "FAIRY", "category": "SPECIAL", "base_power": 95, "accuracy": 1.0},
            {"id": "icywind", "type": "ICE", "category": "SPECIAL", "base_power": 55, "accuracy": 0.95},
        ],
    ],
    "active_pokemon": [
        {"species": "incineroar", "types": ["FIRE", "DARK"], "hp_fraction": 0.8},
        {"species": "fluttermane", "types": ["GHOST", "FAIRY"], "hp_fraction": 1.0},
    ],
    "opponent_active_pokemon": [
        {"species": "landorustherian", "types": ["GROUND", "FLYING"], "hp_fraction": 0.5},
        {"species": "amoonguss", "types": ["GRASS", "POISON"], "hp_fraction": 1.0},
    ],
    "clock": {"decision_seconds_left": 55, "bank_seconds_left": 300},
}


def _doubles_answer(foes=None, **overrides):
    """The model's battle answer: a prediction of each opposing active's action."""
    answer = {
        "state_summary": "Our Incineroar and Flutter Mane face a weakened Landorus and a healthy Amoonguss.",
        "foes": foes or [
            {"species": "landorustherian", "action": "attack", "target": "fluttermane", "move": "rockslide",
             "confidence": 0.6, "why": "Rock Slide is its best hit"},
            {"species": "amoonguss", "action": "protect", "target": "none", "move": "protect",
             "confidence": 0.7, "why": "it has Protect and did not use it last turn"},
        ],
        "reasoning_summary": "Flare Blitz should finish the weakened Landorus.",
    }
    answer.update(overrides)
    return answer


# -- draft ---------------------------------------------------------------------------------


DRAFT_OBSERVATION = {
    "phase": "draft",
    "pick_number": 3,
    "available_cards": [
        {"card_id": "c1", "species": "Rillaboom", "item": "Miracle Seed", "moves": ["grassyglide"]},
        {"card_id": "c2", "species": "Pyukumuku", "item": "Leftovers", "moves": ["recover"]},
    ],
    "rosters": {"me": [{"card_id": "c9", "species": "Incineroar"}], "them": []},
}


def test_draft_falls_back_to_the_highest_value_card_without_a_model():
    decision = _agent().choose_action(
        _state(*_draft_actions("c2", "c1"), phase="draft", observation=DRAFT_OBSERVATION), POKEMON
    )
    # Rillaboom (c1, 20 points) over Pyukumuku, despite c2 being listed first.
    assert decision.action.action_id == "draft_pick:c1"


def test_the_draft_picks_by_rule_without_a_model_call():
    """The pick comes from pick_priority's rules; the model is not asked, and its
    reason is kept for Team Preview and the battle."""
    provider = FakeProvider({"action_id": "draft_pick:c2", "reasoning_summary": "unused"})
    agent = _agent(provider)
    decision = agent.choose_action(_state(*_draft_actions("c1", "c2"), phase="draft", observation=DRAFT_OBSERVATION), POKEMON)
    assert isinstance(decision, WithReasoning)
    assert decision.action.action_id in ("draft_pick:c1", "draft_pick:c2")
    assert not provider.calls
    assert decision.reasoning_summary == "Move submitted."  # private by default
    assert agent.draft_strategy()["picks"][0].split(" - ")[1].startswith(("must pick", "partner", "strongest"))


# -- team preview --------------------------------------------------------------------------


GOOD_PREVIEW_BRING = ["incineroar", "rillaboom", "fluttermane", "landorustherian"]


def _lineup_answer(bring=None, leads=None, **overrides):
    answer = {
        "our_win_conditions": ["Fire/Fairy pressure"], "their_win_conditions": ["sun"],
        "their_playstyle": "sun offense", "their_likely_sets": [], "our_key_moves": [],
        "bring": GOOD_PREVIEW_BRING if bring is None else bring,
        "leads": ["incineroar", "fluttermane"] if leads is None else leads,
        "lead_plan": "Fake Out + Moonblast", "early_game": "e", "mid_game": "m", "end_game": "e",
        "preserve": ["fluttermane"], "remove_first": ["torkoal"], "avoid": [],
        "reasoning_summary": "Bringing the robust four.",
    }
    answer.update(overrides)
    return answer


def test_team_preview_builds_a_valid_lineup_payload():
    provider = FakeProvider(_lineup_answer())
    decision = _agent(provider).choose_action(
        _state(_lineup_action(), phase="team_preview", observation={"opponent_roster": [{"species": "Torkoal"}]}),
        POKEMON,
    )
    payload = decision.action
    assert payload["type"] == "select_lineup"
    assert len(payload["bring"]) == 4 and len(set(payload["bring"])) == 4
    assert set(payload["leads"]) <= set(payload["bring"])


def test_team_preview_rejects_a_lineup_that_is_not_four_species():
    bad = _lineup_answer(bring=ROSTER[:3], leads=ROSTER[:2])
    provider = FakeProvider(bad, _lineup_answer())
    log = []
    decision = _agent(provider, log=log).choose_action(
        _state(_lineup_action(), phase="team_preview"), POKEMON
    )
    assert len(provider.calls) == 2  # rejected, then retried
    assert any("bring must be 4 different species" in line for line in log)
    assert len(decision.action["bring"]) == 4


def test_team_preview_rejects_leads_outside_the_bring():
    bad = _lineup_answer(leads=[ROSTER[4], ROSTER[5]])
    provider = FakeProvider(bad, bad)
    log = []
    decision = _agent(provider, log=log).choose_action(_state(_lineup_action(), phase="team_preview"), POKEMON)
    assert any("leads must be 2 different species" in line for line in log)
    assert decision.action["type"] == "select_lineup"  # fell back to a legal lineup


def test_team_preview_without_a_model_brings_the_highest_value_four():
    decision = _agent().choose_action(_state(_lineup_action(), phase="team_preview"), POKEMON)
    assert decision.action["type"] == "select_lineup"
    assert len(decision.action["bring"]) == 4


# -- restricted legendaries (Reg I) --------------------------------------------------------


RESTRICTED_ROSTER = [
    "Calyrex-Shadow", "Miraidon", "Koraidon", "Incineroar", "Rillaboom", "Amoonguss",
]


def test_the_preview_fallback_brings_at_most_one_restricted():
    """Reg I allows one restricted among the four; an illegal bring loses."""
    action = _templated(
        "select_lineup", {"type": "select_lineup", "roster": RESTRICTED_ROSTER, "instructions": ""}
    )
    decision = _agent().choose_action(_state(action, phase="team_preview"), POKEMON)
    brought = decision.action["bring"]
    restricted = [name for name in brought if name in ("Calyrex-Shadow", "Miraidon", "Koraidon")]
    assert len(brought) == 4
    assert len(restricted) == 1


def test_the_preview_fallback_still_fields_four_from_an_all_restricted_roster():
    roster = ["Calyrex-Shadow", "Miraidon", "Koraidon", "Kyogre", "Groudon", "Lunala"]
    action = _templated("select_lineup", {"type": "select_lineup", "roster": roster, "instructions": ""})
    decision = _agent().choose_action(_state(action, phase="team_preview"), POKEMON)
    assert len(decision.action["bring"]) == 4


def test_the_draft_fallback_deprioritizes_a_second_restricted():
    """With a restricted already drafted, a cheaper non-restricted wins."""
    observation = {
        "phase": "draft",
        "available_cards": [
            {"card_id": "r", "species": "Miraidon"},      # 20 points, restricted
            {"card_id": "n", "species": "Amoonguss"},     # 16 points, not restricted
        ],
        "rosters": {"me": [{"card_id": "x", "species": "Calyrex-Shadow"}], "them": []},
    }
    decision = _agent().choose_action(
        _state(*_draft_actions("r", "n"), phase="draft", observation=observation), POKEMON
    )
    assert decision.action.action_id == "draft_pick:n"


def test_the_draft_fallback_takes_the_first_restricted_on_value():
    """With none drafted yet, the restricted is just the best card."""
    observation = {
        "phase": "draft",
        "available_cards": [
            {"card_id": "r", "species": "Miraidon"},
            {"card_id": "n", "species": "Amoonguss"},
        ],
        "rosters": {"me": [], "them": []},
    }
    decision = _agent().choose_action(
        _state(*_draft_actions("r", "n"), phase="draft", observation=observation), POKEMON
    )
    assert decision.action.action_id == "draft_pick:r"


# -- doubles turns -------------------------------------------------------------------------


def test_doubles_turn_builds_a_valid_payload():
    provider = FakeProvider(_doubles_answer())
    decision = _agent(provider).choose_action(
        _state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON
    )
    payload = decision.action
    assert payload["type"] == "doubles_turn"
    assert payload["slot_0"]["type"] in ("move", "switch", "pass")
    assert payload["slot_1"]["type"] in ("move", "switch", "pass")


def test_the_analysis_log_reports_the_prediction_and_our_answers():
    log = []
    provider = FakeProvider(_doubles_answer())
    _agent(provider, log=log).choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    rendered = "\n".join(log)
    assert "Turn 1 Analysis" in rendered
    assert "Predicted opponent (model)" in rendered and "amoonguss: protect" in rendered
    assert "Our best answers" in rendered and "<- played" in rendered


def test_a_bad_prediction_is_retried_then_the_code_guess_is_used():
    provider = FakeProvider({"state_summary": "s", "foes": [], "reasoning_summary": "r"}, ProviderError("down"))
    log = []
    decision = _agent(provider, log=log).choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    assert decision.action["type"] == "doubles_turn"
    assert any("FALLBACK" in line and "code's prediction" in line for line in log)


def test_both_slots_switching_the_same_pokemon_is_rejected():
    """GameAPI would reject it, so it must never leave the agent."""
    template = {
        "type": "doubles_turn",
        "slots": [
            {"slot": 0, "options": [{"type": "switch", "species": "ursaluna"}]},
            {"slot": 1, "options": [{"type": "switch", "species": "ursaluna"}]},
        ],
    }
    slots = act.parse_doubles(_as_action(template))
    with pytest.raises(act.InvalidAnswer, match="same Pok"):
        act.build_doubles(slots, {"slot_0": {"option": 0, "target": None}, "slot_1": {"option": 0, "target": None}})


# -- the opponent model across a match -----------------------------------------------------


# -- the clock -----------------------------------------------------------------------------


def test_a_slow_first_call_still_leaves_room_for_a_retry():
    clock = FakeClock()
    provider = FakeProvider(ProviderError("stalled"), _doubles_answer())

    original = provider.complete_structured

    def slow(*args, **kwargs):
        clock.now += 20.0  # burn 20 of the 32-second budget
        return original(*args, **kwargs)

    provider.complete_structured = slow
    decision = _agent(provider, clock=clock).choose_action(
        _state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON
    )
    assert len(provider.calls) == 2
    assert decision.action["type"] == "doubles_turn"


def test_the_fallback_plays_rather_than_overrun_the_clock():
    clock = FakeClock()
    provider = FakeProvider(ProviderError("stalled"), _doubles_answer())

    original = provider.complete_structured

    def very_slow(*args, **kwargs):
        clock.now += 38.0  # leaves under MIN_ATTEMPT_SECONDS of the 40s budget
        return original(*args, **kwargs)

    provider.complete_structured = very_slow
    log = []
    decision = _agent(provider, log=log, clock=clock).choose_action(
        _state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON
    )
    assert len(provider.calls) == 1
    assert any("out of time" in line for line in log)
    assert decision.action["type"] == "doubles_turn"  # still a legal move


def test_the_request_timeout_shrinks_with_the_remaining_budget():
    clock = FakeClock()
    provider = FakeProvider(ProviderError("x"), _doubles_answer())

    original = provider.complete_structured

    def slow(*args, **kwargs):
        clock.now += 10.0
        return original(*args, **kwargs)

    provider.complete_structured = slow
    _agent(provider, clock=clock).choose_action(
        _state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON
    )
    budget = 40.0  # DECISION_SECONDS
    assert provider.calls[0]["timeout"] == budget
    assert provider.calls[1]["timeout"] == budget - 10.0  # exactly what is left


# -- no model at all -----------------------------------------------------------------------


def test_the_strategist_plays_a_legal_turn_with_no_provider():
    decision = _agent().choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    payload = decision.action
    assert payload["type"] == "doubles_turn"
    assert payload["slot_0"]["type"] == "move"


def test_a_missing_key_is_diagnosed_once_at_startup():
    """The per-turn note is public and short, so the fix is said here."""
    log = []
    VGCDraftStrategist(None, log=log.append, rng=random.Random(1), clock=lambda: 0.0)
    assert len(log) == 1
    assert "No model configured" in log[0]
    assert "ANTHROPIC_API_KEY" in log[0] and "OPENAI_API_KEY" in log[0]


def test_a_configured_model_is_named_at_startup():
    log = []
    VGCDraftStrategist(FakeProvider(), log=log.append, rng=random.Random(1), clock=lambda: 0.0)
    assert log == ["[vgc] model: FakeProvider(model='fake-model')"]


def test_a_missing_reference_table_is_diagnosed_too(monkeypatch, tmp_path):
    from agent.vgc import meta

    meta.table.cache_clear()
    monkeypatch.setenv("VGC_META_CSV", str(tmp_path / "absent.csv"))
    try:
        log = []
        VGCDraftStrategist(FakeProvider(), log=log.append, rng=random.Random(1), clock=lambda: 0.0)
        assert any("No reference table loaded" in line for line in log)
    finally:
        meta.table.cache_clear()


def test_credentials_are_detected_beyond_the_api_key_variable(monkeypatch):
    """An `ant auth login` profile or auth token is a credential too."""
    from agent.vgc import providers

    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert providers.provider_from_env() is None

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-not-an-api-key")
    assert providers.anthropic_credentials() is True
    assert isinstance(providers.provider_from_env(), providers.AnthropicProvider)


def test_anthropic_is_preferred_over_openai(monkeypatch):
    from agent.vgc import providers

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-y")
    assert isinstance(providers.provider_from_env(), providers.AnthropicProvider)

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert isinstance(providers.provider_from_env(), providers.OpenAIProvider)


def test_the_default_claude_model_is_sonnet():
    from agent.vgc import providers

    assert providers.DEFAULT_ANTHROPIC_MODEL == "claude-sonnet-5"


def test_vgc_model_overrides_the_default_model(monkeypatch):
    from agent.vgc import providers

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.delenv("VGC_MODEL", raising=False)
    assert providers.provider_from_env().model == "claude-sonnet-5"

    monkeypatch.setenv("VGC_MODEL", "claude-opus-5")
    assert providers.provider_from_env().model == "claude-opus-5"


def _fake_anthropic(*, output_config: bool, version="9.9.9"):
    """A stand-in anthropic module whose create() does or does not take
    output_config, the way 1.x and 0.x differ."""
    import types

    if output_config:
        def create(self, *, model, max_tokens, system=None, messages=None,
                   thinking=None, output_config=None):
            raise AssertionError("not called in these tests")
    else:
        def create(self, *, model, max_tokens, system=None, messages=None):
            raise AssertionError("not called in these tests")

    messages_ns = type("Messages", (), {"create": create})

    class Client:
        def __init__(self, api_key=None, **_kwargs):
            self.messages = messages_ns()
            # The real SDK resolves ANTHROPIC_API_KEY when none is passed,
            # which is what anthropic_credentials() probes for.
            resolved = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.auth_headers = {"X-Api-Key": "probe"} if resolved else {}

    module = types.ModuleType("anthropic")
    module.__version__ = version
    module.Anthropic = Client
    module.APIStatusError = type("APIStatusError", (Exception,), {})
    module.APIError = type("APIError", (Exception,), {})
    module.NOT_GIVEN = object()
    return module


@pytest.mark.parametrize("has_output_config", [True, False])
def test_structured_output_support_is_detected_from_the_signature(has_output_config):
    from agent.vgc.providers import _supports_structured_outputs

    module = _fake_anthropic(output_config=has_output_config)
    assert _supports_structured_outputs(module) is has_output_config


def test_an_sdk_that_takes_kwargs_is_treated_as_supported():
    import types

    from agent.vgc.providers import _supports_structured_outputs

    class Client:
        class messages:
            @staticmethod
            def create(**kwargs):
                pass

        def __init__(self, **_kwargs):
            pass

    module = types.ModuleType("anthropic")
    module.Anthropic = Client
    assert _supports_structured_outputs(module) is True


def test_an_old_sdk_is_refused_at_construction_with_the_fix(monkeypatch):
    """anthropic 0.x has no output_config; failing here beats failing on turn 3."""
    from agent.vgc import providers

    monkeypatch.setitem(
        __import__("sys").modules, "anthropic", _fake_anthropic(output_config=False, version="0.75.0")
    )
    with pytest.raises(RuntimeError) as excinfo:
        providers.AnthropicProvider(api_key="sk-ant-x")

    message = str(excinfo.value)
    assert "0.75.0" in message and "output_config" in message
    assert "pip install -U" in message and "anthropic>=1.0" in message


def test_a_broken_claude_setup_falls_through_to_openai(monkeypatch):
    """A bad preference must not forfeit every match."""
    from agent.vgc import providers

    monkeypatch.setitem(
        __import__("sys").modules, "anthropic", _fake_anthropic(output_config=False, version="0.75.0")
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-x")

    log = []
    provider = providers.provider_from_env(log=log.append)
    assert isinstance(provider, providers.OpenAIProvider)
    assert any("Claude is configured but unusable" in line for line in log)


def test_a_broken_claude_setup_with_no_other_key_degrades_to_heuristics(monkeypatch):
    from agent.vgc import providers

    monkeypatch.setitem(
        __import__("sys").modules, "anthropic", _fake_anthropic(output_config=False, version="0.75.0")
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    log = []
    assert providers.provider_from_env(log=log.append) is None
    assert any("unusable" in line for line in log)


def test_any_provider_exception_falls_back_instead_of_failing_the_match():
    """Regression: a TypeError from the SDK used to escape and forfeit.

    The agent's one hard promise is that a model problem costs estimate
    quality, never a legal move.
    """
    class Exploding:
        model = "exploding-model"
        last_reasoning = None

        def complete_structured(self, *args, **kwargs):
            raise TypeError("Messages.create() got an unexpected keyword argument 'output_config'")

    log = []
    agent = VGCDraftStrategist(
        Exploding(), log=log.append, rng=random.Random(1), clock=lambda: 0.0
    )
    decision = agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)

    assert decision.action["type"] == "doubles_turn"  # a legal move was still played
    assert any("[TypeError]" in line for line in log)  # and the bug stayed visible


@pytest.mark.parametrize("blow_up", [
    TypeError("bad kwarg"),
    RuntimeError("something odd"),
    KeyError("missing"),
    AttributeError("no such attribute"),
])
def test_every_provider_failure_mode_still_yields_a_legal_move(blow_up):
    class Exploding:
        model = "exploding-model"
        last_reasoning = None

        def complete_structured(self, *args, **kwargs):
            raise blow_up

    agent = VGCDraftStrategist(
        Exploding(), log=lambda _l: None, rng=random.Random(1), clock=lambda: 0.0
    )
    decision = agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    assert decision.action["type"] == "doubles_turn"


def test_claude_reasoning_is_read_from_the_thinking_blocks():
    import types

    from agent.vgc.providers import _thinking_text

    response = types.SimpleNamespace(content=[
        types.SimpleNamespace(type="thinking", thinking="Weighing the Protect column."),
        types.SimpleNamespace(type="thinking", thinking="Their last three turns were Protect."),
        types.SimpleNamespace(type="text", text='{"x": 1}'),
    ])
    assert _thinking_text(response) == (
        "Weighing the Protect column.\n\nTheir last three turns were Protect."
    )


@pytest.mark.parametrize("content", [
    [],
    [{"type": "text"}],
    None,
])
def test_a_response_without_reasoning_does_not_break_the_turn(content):
    """Logging is a nicety; it must never cost a move."""
    import types

    from agent.vgc.providers import _thinking_text

    assert _thinking_text(types.SimpleNamespace(content=content)) == ""
    assert _thinking_text(types.SimpleNamespace()) == ""


def test_stale_reasoning_is_never_attributed_to_a_later_decision(tmp_path):
    """A failed call must clear it, or the log blames the wrong turn."""
    provider = FakeProvider(_doubles_answer(), _doubles_answer(), reasoning="turn one thinking")
    agent = VGCDraftStrategist(
        provider, log=lambda _l: None, rng=random.Random(1), clock=lambda: 0.0, log_dir=str(tmp_path)
    )
    state = _state(_doubles_action(), observation=BATTLE_OBSERVATION)
    agent.choose_action(state, POKEMON)
    provider.last_reasoning = None  # a provider that returned no thinking
    agent.choose_action(state, POKEMON)

    records = [
        json.loads(line)
        for line in sorted(tmp_path.glob("*.jsonl"))[0].read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["model_reasoning"] == "turn one thinking"
    assert records[1]["model_reasoning"] is None


def test_a_provider_repr_never_contains_the_key(monkeypatch):
    from agent.vgc import providers

    secret = "sk-ant-DO-NOT-LEAK-1234567890"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    assert secret not in repr(providers.provider_from_env())
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    assert secret not in repr(providers.provider_from_env())


def _logged(tmp_path, provider, *states, seed=1):
    """Play some decisions with a match log on disk; return (md, records)."""
    agent = VGCDraftStrategist(
        provider, log=lambda _l: None, rng=random.Random(seed), clock=lambda: 0.0,
        log_dir=str(tmp_path),
    )
    for state in states:
        agent.choose_action(state, POKEMON)
    markdown = sorted(tmp_path.glob("*.md"))
    jsonl = sorted(tmp_path.glob("*.jsonl"))
    assert len(markdown) == 1 and len(jsonl) == 1, "one pair of files per match"
    records = [json.loads(line) for line in jsonl[0].read_text(encoding="utf-8").splitlines()]
    return markdown[0].read_text(encoding="utf-8"), records


def test_the_match_log_records_every_decision_in_one_pair_of_files(tmp_path):
    provider = FakeProvider(_doubles_answer(), _doubles_answer())
    state = _state(_doubles_action(), observation=BATTLE_OBSERVATION)
    text, records = _logged(tmp_path, provider, state, state)

    assert len(records) == 2
    assert [r["turn"] for r in records] == ["Turn 1", "Turn 2"]
    assert "Turn 1 Analysis" in text and "Turn 2 Analysis" in text


def test_a_logged_record_carries_the_full_decision_basis(tmp_path):
    provider = FakeProvider(_doubles_answer(), reasoning="Landorus is in range; Protect loses tempo.")
    _, records = _logged(tmp_path, provider, _state(_doubles_action(), observation=BATTLE_OBSERVATION))
    record = records[0]

    assert record["kind"] == "doubles_turn"
    assert record["model"] == "fake-model"
    assert record["model_reasoning"] == "Landorus is in range; Protect loses tempo."
    assert record["attempts"] == 2  # the read, then the final call
    assert record["fallback_reason"] is None
    assert record["payload"]["type"] == "doubles_turn"
    assert record["public_summary"]
    assert "timestamp" in record

    analysis = record["analysis"]
    assert analysis["prediction_source"] == "model"
    assert analysis["prediction"]["amoonguss"]["action"] == "protect"
    assert analysis["code_guess"] and analysis["lines"][0]["line"]
    assert record["move_state"]["ours"]  # the move tracker's view is logged


def test_the_model_reasoning_is_logged_but_never_sent_to_the_platform(tmp_path):
    """It is private deliberation; reasoning_summary is public to spectators."""
    private = "SECRET-DELIBERATION they will switch, so I bait it"
    provider = FakeProvider(_doubles_answer(), reasoning=private)
    agent = VGCDraftStrategist(
        provider, log=lambda _l: None, rng=random.Random(1), clock=lambda: 0.0, log_dir=str(tmp_path)
    )
    decision = agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)

    assert private not in (decision.reasoning_summary or "")
    text = sorted(tmp_path.glob("*.md"))[0].read_text(encoding="utf-8")
    assert private in text
    assert "private - local log only" in text


def test_a_fallback_is_audited_with_its_reason(tmp_path):
    """The most important thing an audit can explain about a bad move."""
    provider = FakeProvider(ProviderError("boom"), ProviderError("boom"))
    text, records = _logged(tmp_path, provider, _state(_doubles_action(), observation=BATTLE_OBSERVATION))

    assert records[0]["fallback_reason"] is not None
    assert records[0]["analysis"]["prediction_source"] == "code"  # what we assumed instead
    assert records[0]["payload"]["type"] == "doubles_turn"  # still a legal move
    assert records[0]["attempts"] == 2
    assert "FALLBACK" in text


def test_a_no_model_decision_is_audited_too(tmp_path):
    text, records = _logged(tmp_path, None, _state(_doubles_action(), observation=BATTLE_OBSERVATION))
    assert records[0]["model"] is None
    assert records[0]["attempts"] == 0
    assert "No model configured" in records[0]["fallback_reason"]
    assert "no model call" in text


def test_the_log_records_how_long_the_model_took(tmp_path):
    clock = FakeClock()
    provider = FakeProvider(_doubles_answer())
    original = provider.complete_structured

    def slow(*args, **kwargs):
        clock.now += 12.0
        return original(*args, **kwargs)

    provider.complete_structured = slow
    agent = VGCDraftStrategist(
        provider, log=lambda _l: None, rng=random.Random(1), clock=clock, log_dir=str(tmp_path)
    )
    agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    records = [
        json.loads(line)
        for line in sorted(tmp_path.glob("*.jsonl"))[0].read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["seconds"] == 24.0  # the read and the final call, 12 s each
    assert records[0]["attempts"] == 2


def test_the_draft_and_preview_phases_are_audited(tmp_path):
    provider = FakeProvider(
        {"action_id": "draft_pick:c1", "reasoning_summary": "Best card."},
        _lineup_answer(),
    )
    text, records = _logged(
        tmp_path, provider,
        _state(*_draft_actions("c1", "c2"), phase="draft", observation=DRAFT_OBSERVATION),
        _state(_lineup_action(), phase="team_preview"),
    )
    assert [r["kind"] for r in records] == ["draft_pick", "game_plan", "select_lineup"]
    assert records[0]["payload"] == "draft_pick:c1"  # a LegalAction logs its id
    assert records[0]["phase"] == "draft"
    assert records[1]["payload"]["lead_plan"]  # the plan is logged before the lineup
    assert records[2]["payload"]["type"] == "select_lineup"
    assert "select_lineup" in text


def test_no_files_are_written_without_a_log_dir(tmp_path):
    agent = VGCDraftStrategist(
        FakeProvider(_doubles_answer()), log=lambda _l: None, rng=random.Random(1),
        clock=lambda: 0.0, log_dir=None,
    )
    agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_log_dir_is_reported_once_and_does_not_stop_play(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    log = []
    agent = VGCDraftStrategist(
        FakeProvider(_doubles_answer(), _doubles_answer()), log=log.append,
        rng=random.Random(1), clock=lambda: 0.0, log_dir=str(blocker / "sub"),
    )
    state = _state(_doubles_action(), observation=BATTLE_OBSERVATION)
    first = agent.choose_action(state, POKEMON)
    second = agent.choose_action(state, POKEMON)

    assert first.action["type"] == "doubles_turn"
    assert second.action["type"] == "doubles_turn"
    assert sum("match log disabled" in line for line in log) == 1


def test_each_match_gets_its_own_log_files(tmp_path):
    """Named per session and seat, so concurrent matches never share a file."""
    from altruagent import DecisionContext

    for index in (1, 2):
        agent = VGCDraftStrategist(
            FakeProvider(_doubles_answer()), log=lambda _l: None, rng=random.Random(1),
            clock=lambda: 0.0, log_dir=str(tmp_path),
        )
        context = DecisionContext(
            session_id=f"session-{index}", tournament_id=None,
            game_type="pokemon_vgc_doubles_draft", agent_id="me", seat_position=index,
        )
        agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), context)

    names = sorted(p.name for p in tmp_path.glob("*.md"))
    assert len(names) == 2
    assert any("session-1" in n and "seat1" in n for n in names)
    assert any("session-2" in n and "seat2" in n for n in names)


def test_no_legal_actions_is_an_error_not_a_guess():
    with pytest.raises(act.TemplateError, match="no legal actions"):
        _agent().choose_action(_state(), POKEMON)


def test_submit_team_is_refused_rather_than_guessed():
    action = _templated("submit_team", {"type": "submit_team", "instructions": "Build six."})
    with pytest.raises(act.TemplateError, match="submit_team"):
        _agent().choose_action(_state(action), POKEMON)


def test_a_timeout_retries_at_lower_effort():
    """A timeout means too slow, so the retry trades depth for speed."""
    clock = FakeClock()

    class TimesOutOnce:
        model = "m"
        last_reasoning = None

        def __init__(self):
            self.calls = []

        def complete_structured(self, messages, schema_name, schema, *, timeout, effort="medium"):
            self.calls.append({"timeout": timeout, "effort": effort, "n": len(messages)})
            if len(self.calls) == 1:
                clock.now += 20.0  # failed part-way through, leaving time to retry
                raise ProviderError("Claude request failed (APITimeoutError)")
            return _lineup_answer()

    provider = TimesOutOnce()
    agent = VGCDraftStrategist(provider, log=lambda _l: None, rng=random.Random(1), clock=clock)
    decision = agent.choose_action(_state(_lineup_action(), phase="team_preview"), POKEMON)

    first, second = provider.calls
    assert first["timeout"] == 78.0  # not capped: a capped call that would have finished is lost
    order = ["low", "medium", "high"]
    assert order.index(second["effort"]) <= order.index(first["effort"])  # never slower
    assert order.index(second["effort"]) == max(0, order.index(first["effort"]) - 1)
    assert second["n"] == first["n"]  # no "invalid answer" nudge for a timeout
    assert decision.action["type"] == "select_lineup"


# -- end-of-match summary (offline analysis) ----------------------------------


def test_summarize_match_records_opponent_archetype_and_counts(tmp_path):
    """The summary is the skimmable layer for no-time-pressure review."""
    agent = VGCDraftStrategist(
        FakeProvider(_doubles_answer()), log=lambda _l: None, rng=random.Random(1),
        clock=lambda: 0.0, log_dir=str(tmp_path),
    )
    agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    summary = agent.summarize_match(note="unit")

    assert summary["decisions"] == 1
    # BATTLE_OBSERVATION's opponents are Landorus-Therian + Amoonguss.
    assert any("landorus" in s.lower() for s in summary["opponent_species"])
    assert summary["outcome"] is None  # no final_result -> never guessed
    files = [p.name for p in tmp_path.iterdir()]
    assert any(f.endswith(".summary.json") for f in files)
    md = next(iter(tmp_path.glob("*.md"))).read_text(encoding="utf-8")
    assert "Match summary" in md


def test_summarize_match_reads_the_outcome_from_final_result_only(tmp_path):
    agent = VGCDraftStrategist(
        FakeProvider(_doubles_answer()), log=lambda _l: None, rng=random.Random(1),
        clock=lambda: 0.0, log_dir=str(tmp_path),
    )
    agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    terminal = _state(_doubles_action(), observation=BATTLE_OBSERVATION)
    terminal.final_result = {"winner_agent_id": "me"}
    terminal.is_terminal = True
    with pytest.raises(act.TemplateError):
        agent.choose_action(terminal, POKEMON)  # terminal auto-summarizes, then refuses
    summary_file = next(iter(tmp_path.glob("*.summary.json")))
    assert json.loads(summary_file.read_text(encoding="utf-8"))["outcome"] == "win"


def test_summarize_match_is_safe_without_a_log_dir():
    agent = VGCDraftStrategist(None, log=lambda _l: None, rng=random.Random(1), clock=lambda: 0.0)
    summary = agent.summarize_match()
    assert summary["decisions"] == 0 and summary["outcome"] is None


# -- opponent type profile ----------------------------------------------------




# -- the strategist's final call: play or veto -------------------------------------


def _review(verdict, option, why="Their read punishes the code's pick."):
    return {"verdict": verdict, "play_option": option, "why": why, "reasoning_summary": why}


def test_the_strategist_sees_the_code_pick_and_its_alternatives_by_name(tmp_path):
    provider = FakeProvider(_doubles_answer(), _review("play", "1"))
    text, records = _logged(tmp_path, provider, _state(_doubles_action(), observation=BATTLE_OBSERVATION))
    prompt = json.loads(provider.calls[1]["messages"][1]["content"])
    assert prompt["decision"] == "doubles_turn_review"
    assert prompt["your_read"]["prediction"]["landorustherian"]["action"] == "attack"
    lines = [o["line"] for o in prompt["options"]]
    assert len(lines) >= 2 and all(line.startswith("incineroar:") for line in lines)
    analysis = records[0]["analysis"]
    assert analysis["played"] == 0 and analysis["review"]["verdict"] == "play"
    assert "Strategist's call:** play option 1" in text


def test_a_veto_plays_the_strategists_chosen_line(tmp_path):
    provider = FakeProvider(_doubles_answer(), _review("veto", "2"))
    agent = _agent(provider)
    state = _state(_doubles_action(), observation=BATTLE_OBSERVATION)
    decision = agent.choose_action(state, POKEMON)
    prompt = json.loads(provider.calls[1]["messages"][1]["content"])
    code_pick, vetoed_for = prompt["options"][0]["line"], prompt["options"][1]["line"]
    assert code_pick != vetoed_for
    slots = act.parse_doubles(state.legal_actions[0])
    from agent.vgc.strategist import _name_payload
    assert _name_payload(decision.action, slots) == vetoed_for


def test_a_veto_that_names_the_code_pick_is_retried_then_the_code_pick_stands():
    provider = FakeProvider(_doubles_answer(), _review("veto", "1"), _review("veto", "1"))
    log = []
    agent = _agent(provider, log=log)
    decision = agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    assert decision.action["type"] == "doubles_turn"
    assert any("a veto must name another option" in line for line in log)
    assert any("played the code's pick" in line for line in log)


def test_the_final_call_shares_the_turn_budget_with_the_read():
    clock = FakeClock()
    provider = FakeProvider(_doubles_answer())
    original = provider.complete_structured

    def slow(*args, **kwargs):
        clock.now += 37.0
        return original(*args, **kwargs)

    provider.complete_structured = slow
    log = []
    agent = _agent(provider, log=log, clock=clock)
    decision = agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    assert decision.action["type"] == "doubles_turn"
    assert not provider.reviews  # 3 s of the 40 s budget left: no final call
    assert any("out of time" in line for line in log)
