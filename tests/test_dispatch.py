"""The game dispatcher (agent/dispatch.py).

Routes each match to the right specialist by ``context.game_type``, builds it
once, caches it, and forwards ``on_action_result`` only when the specialist
supports it. These tests inject fake specialists (via ``_build``) so they do
not drag in the VGC provider stack, and verify the dispatch logic itself:
routing, caching, the safe default, and forwarding.
"""

from __future__ import annotations

import pytest

from agent import dispatch


class _FakeRA:
    """Stands in for the Red Alert agent (has on_action_result)."""

    def __init__(self):
        self.action_results = 0

    def choose_action(self, state, context):
        return "RA"

    def on_action_result(self, result, context):
        self.action_results += 1


class _FakeVGC:
    """Stands in for the VGC strategist (no on_action_result, like the real one)."""

    def choose_action(self, state, context):
        return "VGC"


class _Ctx:
    def __init__(self, game_type):
        self.game_type = game_type


@pytest.fixture
def fakes(monkeypatch):
    ra, vgc = _FakeRA(), _FakeVGC()
    monkeypatch.setattr(dispatch, "_build", lambda gt: ra if gt in dispatch._RED_ALERT else vgc)
    return ra, vgc


def test_routes_red_alert_to_the_red_alert_agent(fakes):
    ra, _ = fakes
    agent = dispatch.create_agent()
    assert agent.choose_action(None, _Ctx("red_alert")) == "RA"
    assert agent._resolve(_Ctx("red_alert")) is ra


def test_routes_other_games_to_the_vgc_strategist(fakes):
    _, vgc = fakes
    agent = dispatch.create_agent()
    assert agent.choose_action(None, _Ctx("pokemon_vgc_doubles_draft")) == "VGC"
    assert agent._resolve(_Ctx("pokemon_vgc_doubles_draft")) is vgc


def test_specialist_is_cached_for_the_whole_match(fakes):
    ra, _ = fakes
    agent = dispatch.create_agent()
    agent.choose_action(None, _Ctx("red_alert"))
    # A later call with a different game_type must NOT switch specialists.
    assert agent._resolve(_Ctx("pokemon_vgc_doubles_draft")) is ra


def test_specialist_is_built_exactly_once(monkeypatch):
    builds = []
    monkeypatch.setattr(dispatch, "_build", lambda gt: builds.append(gt) or _FakeRA())
    agent = dispatch.create_agent()
    agent.choose_action(None, _Ctx("red_alert"))
    agent.choose_action(None, _Ctx("red_alert"))
    agent.on_action_result({}, _Ctx("red_alert"))
    assert builds == ["red_alert"]


def test_missing_or_empty_game_type_defaults_to_vgc(fakes):
    _, vgc = fakes
    assert dispatch.create_agent()._resolve(_Ctx(None)) is vgc
    assert dispatch.create_agent()._resolve(_Ctx("")) is vgc


def test_on_action_result_is_forwarded_when_supported(fakes):
    ra, _ = fakes
    agent = dispatch.create_agent()
    agent.choose_action(None, _Ctx("red_alert"))
    agent.on_action_result({"error": "INVALID_ACTION"}, _Ctx("red_alert"))
    assert ra.action_results == 1


def test_on_action_result_is_a_no_op_when_unsupported(fakes):
    agent = dispatch.create_agent()
    agent.choose_action(None, _Ctx("pokemon_vgc_doubles_draft"))
    # The VGC fake has no on_action_result; the dispatcher must not raise.
    agent.on_action_result({"error": "X"}, _Ctx("pokemon_vgc_doubles_draft"))
