"""The hyper-aggressive Red Alert agent (agent/redalert/).

Red Alert is real time with no live server here, so these drive the
deterministic layers against synthesized observations (the shape mirrors the
platform's, as in test_realtime.py). Every strategy hole flagged in the plan
has an explicit guard test, plus the invariant that the agent never stalls or
forfeits.
"""

from __future__ import annotations

from agent.redalert import brain, buildplan, executor, scout
from agent.redalert.agent import RedAlertAgent
from agent.redalert.modes import Mode
from agent.redalert.observe import read
from altruagent import WAIT, DecisionContext
from altruagent.models import GameState

CTX = DecisionContext(
    session_id="s", tournament_id=None, game_type="red_alert", agent_id="me", seat_position=0,
)


def _raw(**over):
    obs = {
        "tick": 100,
        "economy": {"cash": 1000, "power_provided": 100, "power_drained": 50},
        "military": {"army_value": 0, "visible_enemy_value": 0, "kills_cost": 0, "deaths_cost": 0, "assets_value": 2000},
        "units": [], "buildings": [], "visible_enemies": [], "visible_enemy_buildings": [],
        "available_production": ["powr", "tent", "proc", "weap", "e1", "e3", "1tnk", "harv"],
        "production": [], "events": [], "base_center": [4, 4],
    }
    for key in ("economy", "military"):
        if key in over:
            obs[key] = {**obs[key], **over.pop(key)}
    t = over.pop("_t", 100.0)
    r = over.pop("_r", 1000.0)
    obs.update(over)
    return {
        "phase": "playing", "is_terminal": False, "state_version": 1,
        "pacing": {"mode": "realtime"},
        "observation": obs,
        "time": {"game_seconds": t, "remaining_seconds": r},
        "map": {"spawns": [[4, 4], [40, 40]], "bounds": [0, 0, 48, 48]},
    }


def _view(**over):
    return read(_raw(**over))


def _state(**over):
    return GameState.from_mcp_state(_raw(**over))


BASE = [{"id": 10, "type": "fact"}, {"id": 11, "type": "tent"}]


# -- build order -------------------------------------------------------------


def test_deploys_mcv_before_anything():
    v = _view(units=[{"id": 1, "type": "mcv", "x": 4, "y": 4, "hp": 100}])
    o = buildplan.next_build_orders(v)
    assert o and o[0] == {"cmd": "deploy", "units": [1]}


def test_build_order_starts_with_power_plant():
    v = _view(buildings=[{"id": 10, "type": "fact"}])
    o = buildplan.next_build_orders(v)
    assert o and o[0] == {"cmd": "build", "item": "powr"}


# -- hole #1: low-power interrupt --------------------------------------------


def test_low_power_inserts_a_power_plant_before_other_buildings():
    v = _view(buildings=[{"id": 10, "type": "fact"}, {"id": 11, "type": "powr", "hp": 400}, {"id": 12, "type": "tent"}],
              economy={"power_provided": 100, "power_drained": 90})
    o = buildplan.next_build_orders(v)
    assert o and o[0]["item"] == "powr"


def test_does_not_double_queue_power():
    v = _view(buildings=[{"id": 10, "type": "fact"}, {"id": 11, "type": "powr", "hp": 400}],
              economy={"power_provided": 100, "power_drained": 95},
              production=[{"item": "powr", "queue": "building", "ready": False}])
    assert not any(x.get("item") == "powr" for x in buildplan.next_build_orders(v))


# -- hole #5: placement ------------------------------------------------------


def test_places_a_finished_building():
    v = _view(buildings=[{"id": 10, "type": "fact"}],
              production=[{"item": "powr", "queue": "building", "ready": True}])
    assert buildplan.placement_orders(v) == [{"cmd": "place", "item": "powr"}]


# -- hole #2: idle harvester -------------------------------------------------


def test_idle_harvester_is_reissued():
    v = _view(units=[{"id": 20, "type": "harv", "x": 6, "y": 6, "hp": 100, "idle": True}])
    o = executor._economy_orders(v)
    assert o == [{"cmd": "harvest", "units": [20]}]


def test_working_harvester_is_left_alone():
    v = _view(units=[{"id": 20, "type": "harv", "x": 6, "y": 6, "hp": 100, "idle": False}])
    assert executor._economy_orders(v) == []


# -- hole #3: cheese detection (conservative) + expiry -----------------------


def test_single_far_enemy_is_not_cheese():
    v = _view(visible_enemies=[{"id": 50, "type": "e1", "x": 45, "y": 45, "hp": 100}])
    assert scout.detect_cheese(v) is None


def test_two_enemies_near_base_is_cheese():
    v = _view(visible_enemies=[{"id": 50, "type": "e1", "x": 7, "y": 7, "hp": 100},
                               {"id": 51, "type": "e1", "x": 8, "y": 8, "hp": 100}])
    assert scout.detect_cheese(v) is not None


def test_near_enemy_harvester_is_not_cheese():
    v = _view(visible_enemies=[{"id": 52, "type": "harv", "x": 6, "y": 6, "hp": 100}])
    assert scout.detect_cheese(v) is None


def test_forward_enemy_mcv_is_cheese():
    v = _view(visible_enemy_buildings=[{"id": 60, "type": "mcv", "x": 7, "y": 7, "hp": 100}])
    assert scout.detect_cheese(v) is not None


def test_cheese_expires_when_the_threat_is_gone():
    gone = _view(_t=130.0, visible_enemies=[])
    assert brain.update_cheese_since(gone, 100.0) is None


def test_cheese_is_kept_within_the_window():
    recent = _view(_t=110.0, visible_enemies=[])
    assert brain.update_cheese_since(recent, 100.0) == 100.0


# -- hole #4 / #7: attack only with an edge, escalate late -------------------


def test_builds_when_outmatched():
    v = _view(buildings=BASE, military={"army_value": 1000, "visible_enemy_value": 2000})
    assert brain.choose_mode(v, cheese_since=None) is Mode.BUILD


def test_all_in_with_a_clear_edge():
    v = _view(buildings=BASE, military={"army_value": 3000, "visible_enemy_value": 1000})
    assert brain.choose_mode(v, cheese_since=None) is Mode.ALL_IN


def test_anti_passivity_escalates_late():
    early = _view(buildings=BASE, _t=120.0, military={"army_value": 1100, "visible_enemy_value": 1000})
    late = _view(buildings=BASE, _t=500.0, military={"army_value": 1100, "visible_enemy_value": 1000})
    assert brain.choose_mode(early, cheese_since=None) is Mode.BUILD
    assert brain.choose_mode(late, cheese_since=None) is Mode.ALL_IN


def test_protect_lead_near_the_limit_when_ahead():
    v = _view(buildings=BASE, _r=120.0,
              military={"army_value": 2000, "visible_enemy_value": 500, "kills_cost": 800, "deaths_cost": 200, "assets_value": 3000})
    assert brain.choose_mode(v, cheese_since=None) is Mode.PROTECT_LEAD


def test_cheese_overrides_a_huge_army():
    v = _view(buildings=BASE, military={"army_value": 5000, "visible_enemy_value": 100})
    assert brain.choose_mode(v, cheese_since=90.0) is Mode.DEFEND_RUSH


# -- executor army command ---------------------------------------------------

_ARMY = [{"id": 30, "type": "e1", "x": 5, "y": 5, "hp": 100}, {"id": 31, "type": "1tnk", "x": 5, "y": 6, "hp": 200}]


def test_defend_rush_rallies_home():
    v = _view(buildings=BASE, units=_ARMY)
    o = executor._army_orders(v, Mode.DEFEND_RUSH)
    assert o and o[0]["cmd"] == "attack_move" and o[0]["to"] == [4, 4]


def test_all_in_attacks_nearest_visible_enemy():
    v = _view(buildings=BASE, units=_ARMY,
              visible_enemy_buildings=[{"id": 99, "type": "proc", "x": 40, "y": 40, "hp": 500}])
    o = executor._army_orders(v, Mode.ALL_IN)
    assert o and o[0]["cmd"] == "attack" and o[0]["target"] == 99


def test_all_in_attack_moves_to_enemy_spawn_when_blind():
    v = _view(buildings=BASE, units=_ARMY)
    o = executor._army_orders(v, Mode.ALL_IN)
    assert o and o[0]["cmd"] == "attack_move" and o[0]["to"] == [40, 40]


def test_build_holds_the_army():
    assert executor._army_orders(_view(buildings=BASE, units=_ARMY), Mode.BUILD) == []


# -- production phases -------------------------------------------------------


def test_infantry_bridge_before_a_war_factory():
    v = _view(buildings=[{"id": 10, "type": "fact"}, {"id": 11, "type": "tent"}])
    assert any(x.get("item") == "e1" for x in executor._production_orders(v, Mode.BUILD))


def test_tank_core_once_the_war_factory_exists():
    v = _view(buildings=[{"id": 10, "type": "fact"}, {"id": 11, "type": "tent"}, {"id": 12, "type": "weap"}])
    assert any(x.get("item") == "1tnk" for x in executor._production_orders(v, Mode.BUILD))


def test_defend_rush_spams_cheap_infantry():
    v = _view(buildings=[{"id": 10, "type": "fact"}, {"id": 11, "type": "tent"}])
    o = executor._production_orders(v, Mode.DEFEND_RUSH)
    assert any(x.get("item") == "e1" and x.get("count", 1) >= 2 for x in o)


def test_protect_lead_trains_nothing():
    v = _view(buildings=[{"id": 10, "type": "fact"}])
    assert executor._production_orders(v, Mode.PROTECT_LEAD) == []


# -- hole #6 + robustness: the agent never stalls ----------------------------


def test_no_observation_waits():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    state = GameState.from_mcp_state({"phase": "playing", "pacing": {"mode": "realtime"}, "observation": None})
    assert agent.choose_action(state, CTX) is WAIT


def test_opening_returns_a_legal_batch_that_deploys_the_mcv():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    dec = agent.choose_action(_state(units=[{"id": 1, "type": "mcv", "x": 4, "y": 4, "hp": 100}]), CTX)
    assert hasattr(dec, "action") and dec.action["type"] == "orders"
    assert any(o["cmd"] == "deploy" for o in dec.action["orders"])


def test_a_crashing_model_hook_never_stalls():
    def boom(_view):
        raise RuntimeError("model down")

    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 1e9, model_pref_fn=boom)
    dec = agent.choose_action(
        _state(buildings=BASE, units=[{"id": 30, "type": "e1", "x": 5, "y": 5, "hp": 100}]), CTX
    )
    assert dec is WAIT or hasattr(dec, "action")  # a legal result either way, never an exception


def test_invalid_action_sets_the_no_resend_flag():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent.on_action_result({"error": "INVALID_ACTION"}, CTX)
    assert agent._refused_last is True
