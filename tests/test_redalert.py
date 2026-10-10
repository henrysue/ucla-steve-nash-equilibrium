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
    r = over.pop("_r", 1000.0)
    # A 20-minute match: when only the time left is given, the clock matches it.
    t = over.pop("_t", 100.0 if r >= 1000.0 else 1200.0 - r)
    obs.update(over)
    # Fixtures give our buildings' top-left cells; the live observation
    # reports each building's centre cell (w // 2, h // 2 in), so report that.
    obs["buildings"] = [_as_reported(b) for b in obs.get("buildings") or []]
    return {
        "phase": "playing", "is_terminal": False, "state_version": 1,
        "pacing": {"mode": "realtime"},
        "observation": obs,
        "time": {"game_seconds": t, "remaining_seconds": r},
        "map": {"spawns": [[4, 4], [40, 40]], "bounds": [0, 0, 48, 48]},
    }


def _as_reported(b):
    if not isinstance(b, dict) or "x" not in b or "y" not in b:
        return b
    from agent.redalert.config import DEFAULT_FOOTPRINT, FOOTPRINTS
    w, h = FOOTPRINTS.get(str(b.get("type")).lower(), DEFAULT_FOOTPRINT)
    return {**b, "x": b["x"] + w // 2, "y": b["y"] + h // 2}


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
    # the yard's position is unknown here, so no hint: the engine places it
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
    v = _view(_t=200.0, buildings=BASE, military={"army_value": 3000, "visible_enemy_value": 1000})
    assert brain.choose_mode(v, cheese_since=None) is Mode.ALL_IN


def test_anti_passivity_escalates_late():
    early = _view(buildings=BASE, _t=120.0, military={"army_value": 2200, "visible_enemy_value": 2000})
    late = _view(buildings=BASE, _t=500.0, military={"army_value": 2200, "visible_enemy_value": 2000})
    assert brain.choose_mode(early, cheese_since=None) is Mode.BUILD
    assert brain.choose_mode(late, cheese_since=None) is Mode.ALL_IN


def test_protect_lead_near_the_limit_when_ahead():
    v = _view(buildings=BASE, _r=50.0,
              military={"army_value": 2000, "visible_enemy_value": 1500, "kills_cost": 800, "deaths_cost": 200, "assets_value": 3000})
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


def test_all_in_attack_moves_to_nearest_visible_enemy():
    v = _view(buildings=BASE, units=_ARMY,
              visible_enemy_buildings=[{"id": 99, "type": "proc", "x": 30, "y": 32, "hp": 500}])
    o = executor._army_orders(v, Mode.ALL_IN)
    assert o and o[0]["cmd"] == "attack_move" and o[0]["to"] == [30, 32]


def test_all_in_attack_moves_to_enemy_spawn_when_blind():
    v = _view(buildings=BASE, units=_ARMY)
    o = executor._army_orders(v, Mode.ALL_IN)
    assert o and o[0]["cmd"] == "attack_move" and o[0]["to"] == [40, 40]


def test_build_leaves_busy_units_alone():
    assert executor._army_orders(_view(buildings=BASE, units=_ARMY), Mode.BUILD) == []


def test_build_stages_idle_units_at_the_choke():
    idle = [{**u, "idle": True} for u in _ARMY]
    o = executor._army_orders(_view(buildings=BASE, units=idle), Mode.BUILD)
    assert o == [{"cmd": "attack_move", "units": [30, 31], "to": [11, 11]}]


def test_units_already_staged_are_not_reordered():
    staged = [{"id": 30, "type": "e1", "x": 11, "y": 12, "hp": 100, "idle": True}]
    assert executor._army_orders(_view(buildings=BASE, units=staged), Mode.BUILD) == []


def test_blind_all_in_without_spawns_uses_the_servers_attack_now():
    raw = _raw(buildings=BASE, units=_ARMY)
    raw["map"] = {"bounds": [0, 0, 48, 48]}
    raw["legal_actions"] = {"attack_now": {"cmd": "attack_move", "units": [30, 31], "to": [41, 41]}}
    assert executor._army_orders(read(raw), Mode.ALL_IN) == [raw["legal_actions"]["attack_now"]]


# -- the live observation shape: no military values, spawns in game_config ---

LIVE_CONFIG = {
    "rules": {
        "units": {"e1": {"cost": 100}, "e3": {"cost": 300}, "1tnk": {"cost": 700}},
        "maps": {"singles": {"bounds": [0, 0, 48, 48], "spawns": [[4, 4], [40, 40]]}},
    }
}


def _live_raw(**over):
    raw = _raw(**over)
    del raw["observation"]["military"]
    raw["map"] = {"id": "singles", "bounds": [0, 0, 48, 48]}
    return raw


def test_live_shape_reads_spawns_from_game_config():
    v = read(_live_raw(), LIVE_CONFIG)
    assert v.spawns == [(4, 4), (40, 40)]


def test_live_shape_prices_the_army_from_unit_costs():
    units = [{"id": i, "type": "1tnk", "x": 5, "y": 5, "hp": 100} for i in range(30, 33)]
    v = read(_live_raw(units=units), LIVE_CONFIG)
    assert v.army_value == 2100


def test_live_shape_army_attacks_once_it_is_real():
    units = [{"id": i, "type": "1tnk", "x": 5, "y": 5, "hp": 100} for i in range(30, 33)]
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent.choose_action(GameState.from_mcp_state(_live_raw(_t=196.0, buildings=BASE, units=units)),
                        DecisionContext(session_id="s", tournament_id=None, game_type="red_alert", agent_id="me",
                                        seat_position=0, game_config=LIVE_CONFIG))     # the attack call...
    state = GameState.from_mcp_state(_live_raw(_t=200.0, buildings=BASE, units=units))   # ...confirmed 4 s on
    ctx = DecisionContext(session_id="s", tournament_id=None, game_type="red_alert", agent_id="me",
                          seat_position=0, game_config=LIVE_CONFIG)
    dec = agent.choose_action(state, ctx)
    assert agent._mode is Mode.ALL_IN
    assert {"cmd": "attack_move", "units": [30, 31, 32], "to": [40, 40]} in dec.action["orders"]


def test_a_token_force_does_not_all_in_into_the_fog():
    v = read(_live_raw(buildings=BASE, units=[{"id": 30, "type": "e1", "x": 5, "y": 5, "hp": 100}]), LIVE_CONFIG)
    assert brain.choose_mode(v, cheese_since=None) is Mode.BUILD


# -- production phases -------------------------------------------------------


def test_infantry_bridge_before_a_war_factory():
    v = _view(buildings=[{"id": 10, "type": "fact"}, {"id": 11, "type": "tent"}])
    assert any(x.get("item") == "e1" for x in executor._production_orders(v, Mode.BUILD))


_ECON_BASE = [{"id": 10, "type": "fact"}, {"id": 11, "type": "tent"}, {"id": 12, "type": "weap"},
              {"id": 13, "type": "proc"}]


def _harvs(n, start=20):
    return [{"id": start + i, "type": "harv", "x": 6, "y": 6, "hp": 100} for i in range(n)]


def test_tank_core_once_the_war_factory_exists():
    v = _view(buildings=_ECON_BASE, units=_harvs(3))
    assert any(x.get("item") == "1tnk" for x in executor._production_orders(v, Mode.BUILD))


# -- economy first -----------------------------------------------------------


def test_harvesters_come_before_tanks():
    o = executor._production_orders(_view(buildings=_ECON_BASE, units=_harvs(1)), Mode.BUILD)
    items = [x.get("item") for x in o]
    assert "harv" in items and "1tnk" not in items


def test_a_queued_harvester_counts_toward_the_target():
    v = _view(buildings=_ECON_BASE, units=_harvs(2), economy={"cash": 3000},
              production=[{"item": "harv", "queue": "Vehicle", "ready": False}])
    items = [x.get("item") for x in executor._production_orders(v, Mode.BUILD)]
    assert "harv" not in items and "1tnk" in items


def test_two_harvesters_per_refinery():
    v = _view(buildings=_ECON_BASE + [{"id": 14, "type": "proc"}], units=_harvs(3))
    assert executor.harvester_target(v) == 4  # refineries + 2
    assert any(x.get("item") == "harv" for x in executor._production_orders(v, Mode.BUILD))


def test_a_silo_when_ore_storage_is_nearly_full():
    v = _view(buildings=_ECON_BASE + [{"id": 15, "type": "powr"}],
              available_production=["powr", "tent", "proc", "weap", "silo", "e1"],
              economy={"cash": 0, "ore": 1800, "resource_capacity": 2000, "power_provided": 200, "power_drained": 80})
    assert buildplan.next_build_orders(v) == [{"cmd": "build", "item": "silo"}]  # full, but not rich enough to spend it


def test_no_silo_with_room_to_spare():
    v = _view(buildings=_ECON_BASE, available_production=["silo"],
              economy={"ore": 500, "resource_capacity": 2000})
    assert buildplan.next_build_orders(v) == []


# -- the roaming scout -------------------------------------------------------


def test_patrol_route_rings_the_map_from_our_corner():
    route = scout.patrol_route(_view())  # bounds [0, 0, 48, 48], base [4, 4]
    assert route[0] == (8, 8) and len(route) == 8
    assert all(8 <= x <= 39 and 8 <= y <= 39 for x, y in route)


def test_the_scout_corps_trains_a_scout_when_none_can_be_claimed():
    from agent.redalert.scouting import ScoutCorps
    corps = ScoutCorps()
    v = _view(buildings=_ECON_BASE, available_production=["e1", "e3", "1tnk", "harv", "dog"])
    corps.update(v)
    assert corps.production(v) == [{"cmd": "train", "item": "dog", "count": 1}]      # dogs first
    no_dogs = _view(buildings=_ECON_BASE, available_production=["e1", "e3", "1tnk", "harv"])
    assert corps.production(no_dogs) == [{"cmd": "train", "item": "e1", "count": 1}]


def test_the_scout_is_kept_out_of_the_army():
    units = _ARMY + [{"id": 40, "type": "jeep", "x": 5, "y": 5, "hp": 100}]
    o = executor._army_orders(_view(buildings=BASE, units=units), Mode.ALL_IN, scout_id=40)
    assert o and 40 not in o[0]["units"]


def test_defend_rush_spams_cheap_infantry():
    v = _view(buildings=[{"id": 10, "type": "fact"}, {"id": 11, "type": "tent"}])
    o = executor._production_orders(v, Mode.DEFEND_RUSH)
    assert {"cmd": "train", "item": "e1", "count": 1} in o                # one at a time, 2 deep at most
    full = dataclasses.replace(v, production=[{"item": "e1", "queue": "Infantry", "ready": False}] * 2)
    assert not any(x.get("item") == "e1" for x in executor._production_orders(full, Mode.DEFEND_RUSH))


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


def test_a_crashing_advisor_never_stalls():
    class Boom:
        def complete_structured(self, *a, **k):
            raise RuntimeError("model down")

    from agent.redalert.advisor import Advisor

    def factory(log, on_advice=None):
        return Advisor(Boom(), log=log, interval=0.0)

    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 1e9, advisor_factory=factory)
    state = _state(buildings=BASE, units=[{"id": 30, "type": "e1", "x": 5, "y": 5, "hp": 100}])
    for _ in range(3):
        dec = agent.choose_action(state, CTX)
        assert dec is WAIT or hasattr(dec, "action")  # a legal result either way, never an exception
        if agent._advisor._future is not None:
            agent._advisor._future.exception()  # let the failing call finish


def _mcv_state():
    return _state(units=[{"id": 1, "type": "mcv", "x": 4, "y": 4, "hp": 100}])


def test_a_refused_batch_is_held_back_then_retried():
    now = [0.0]
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: now[0])
    assert hasattr(agent.choose_action(_mcv_state(), CTX), "action")
    agent.on_action_result({"error": "INVALID_ACTION"}, CTX)
    assert agent.choose_action(_mcv_state(), CTX) is WAIT       # not resent right away
    now[0] = 10.0
    assert hasattr(agent.choose_action(_mcv_state(), CTX), "action")  # ...but not blocked for good


def test_a_single_rejected_row_is_held_back():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent.choose_action(_mcv_state(), CTX)
    agent.on_action_result({"accepted": True, "results": [{"index": 0, "cmd": "deploy", "status": "rejected"}]}, CTX)
    assert agent.choose_action(_mcv_state(), CTX) is WAIT


def test_an_unchanged_army_order_is_not_resent_every_view():
    now = [0.0]
    units = [{"id": i, "type": "1tnk", "x": 5, "y": 5, "hp": 100} for i in range(30, 33)]
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: now[0])
    state = _state(_t=200.0, buildings=BASE, units=units, military={"army_value": 3000})

    def army(dec):
        return [o for o in getattr(dec, "action", {}).get("orders", []) if o["cmd"] == "attack_move"]

    assert army(agent.choose_action(state, CTX))
    now[0] = 0.2
    assert not army(agent.choose_action(state, CTX))
    now[0] = 5.0
    assert army(agent.choose_action(state, CTX))


# -- the per-match review log ------------------------------------------------


def test_match_log_records_samples_refusals_and_the_result(tmp_path):
    import json

    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0, log_dir=str(tmp_path))
    raw = _raw(units=[{"id": 1, "type": "mcv", "x": 4, "y": 4, "hp": 100}])
    raw["recent_order_problems"] = [{"batch": 1, "index": 0, "cmd": "place", "status": "rejected", "reason": "invalid"}]
    agent.choose_action(GameState.from_mcp_state(raw), CTX)
    agent.on_action_result({"error": "INVALID_ACTION", "detail": "[0] deploy: invalid"}, CTX)
    final = GameState.from_mcp_state({**raw, "phase": "finished", "is_terminal": True,
                                      "result": {"outcome": "win", "reason": "time_limit"}})
    agent.on_match_end(final, CTX)

    [jsonl] = tmp_path.glob("*.jsonl")
    kinds = [json.loads(line)["kind"] for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert kinds[0] == "start" and "sample" in kinds and "mode" in kinds and kinds[-1] == "end"
    assert kinds.count("refused") >= 2  # the batch refusal (each order in it) and the engine's later verdict
    md = jsonl.with_suffix(".md").read_text(encoding="utf-8")
    assert "## Timeline" in md and "deploy (batch_refused): 1" in md


def test_match_log_is_off_without_a_directory():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    assert agent._matchlog.enabled is False


# -- enemy memory ------------------------------------------------------------

from agent.redalert.memory import EnemyMemory  # noqa: E402

_TANK_SEEN = {"id": 70, "type": "1tnk", "x": 30, "y": 30, "hp": 100}


def test_memory_keeps_an_enemy_army_that_left_our_sight():
    m = EnemyMemory()
    m.update(_view(_t=100.0, visible_enemies=[_TANK_SEEN]))
    m.update(_view(_t=130.0))
    v = m.apply(read(_raw(_t=130.0), LIVE_CONFIG))
    assert v.enemy_army_value == 700


def test_memory_forgets_old_unit_sightings():
    m = EnemyMemory()
    m.update(_view(_t=100.0, visible_enemies=[_TANK_SEEN]))
    m.update(_view(_t=300.0))
    assert m.units == []


def test_memory_drops_a_unit_when_we_look_and_it_is_gone():
    m = EnemyMemory()
    m.update(_view(_t=100.0, visible_enemies=[_TANK_SEEN]))
    m.update(_view(_t=110.0, units=[{"id": 30, "type": "e1", "x": 31, "y": 30, "hp": 100}]))
    assert m.units == []


def test_memory_drops_wrecks():
    m = EnemyMemory()
    m.update(_view(visible_enemies=[_TANK_SEEN]))
    m.update(_view(visible_enemies=[{**_TANK_SEEN, "wreck": True}]))
    assert m.units == []


def test_remembered_buildings_become_attack_targets():
    m = EnemyMemory()
    m.update(_view(_t=50.0, visible_enemy_buildings=[{"id": 80, "type": "proc", "x": 35, "y": 36, "hp": 100}]))
    v = m.apply(_view(_t=400.0, buildings=BASE, units=_ARMY))
    assert executor._army_orders(v, Mode.ALL_IN)[0]["to"] == [35, 36]


def test_a_building_we_find_destroyed_is_forgotten():
    m = EnemyMemory()
    m.update(_view(visible_enemy_buildings=[{"id": 80, "type": "proc", "x": 35, "y": 36, "hp": 100}]))
    m.update(_view(units=[{"id": 30, "type": "e1", "x": 35, "y": 35, "hp": 100}]))
    assert m.buildings == []


# -- retreat -----------------------------------------------------------------


def test_retreat_when_an_attack_turns_against_us():
    v = _view(buildings=BASE, military={"army_value": 1500, "visible_enemy_value": 2500})
    assert brain.choose_mode(v, cheese_since=None, prev_mode=Mode.ALL_IN) is Mode.RETREAT


def test_no_retreat_while_only_massing():
    v = _view(buildings=BASE, military={"army_value": 1500, "visible_enemy_value": 2500})
    assert brain.choose_mode(v, cheese_since=None, prev_mode=Mode.BUILD) is Mode.BUILD


def test_retreat_holds_until_regrouped_then_builds():
    far = [{"id": 30 + i, "type": "1tnk", "x": 40, "y": 40, "hp": 100} for i in range(3)]
    home = [{**u, "x": 15, "y": 15} for u in far]
    mil = {"army_value": 2100, "visible_enemy_value": 2000}  # edge 1.05 < 1.3
    assert brain.choose_mode(_view(_t=400.0, buildings=BASE, units=far, military=mil),
                             cheese_since=None, prev_mode=Mode.RETREAT) is Mode.RETREAT
    assert brain.choose_mode(_view(_t=400.0, buildings=BASE, units=home, military=mil),
                             cheese_since=None, prev_mode=Mode.RETREAT) is Mode.BUILD


def test_retreat_ends_in_an_attack_when_strong_again():
    v = _view(_t=200.0, buildings=BASE, military={"army_value": 4000, "visible_enemy_value": 1000})
    assert brain.choose_mode(v, cheese_since=None, prev_mode=Mode.RETREAT) is Mode.ALL_IN


def test_retreat_attack_moves_back_to_the_staging_point():
    far = [{"id": 30, "type": "1tnk", "x": 40, "y": 40, "hp": 100},
           {"id": 31, "type": "1tnk", "x": 11, "y": 12, "hp": 100}]
    o = executor._army_orders(_view(buildings=BASE, units=far), Mode.RETREAT)
    assert o == [{"cmd": "attack_move", "units": [30], "to": [11, 11]}]   # shot at, it shoots back


def test_the_agent_retreats_after_its_attack_goes_bad():
    units = [{"id": 30 + i, "type": "1tnk", "x": 38, "y": 38, "hp": 100} for i in range(3)]
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent.choose_action(_state(_t=396.0, buildings=BASE, units=units, military={"army_value": 2100}), CTX)
    agent.choose_action(_state(_t=400.0, buildings=BASE, units=units, military={"army_value": 2100}), CTX)
    assert agent._mode is Mode.ALL_IN                                       # confirmed 4 s later
    enemies = [{"id": 70 + i, "type": "1tnk", "x": 39, "y": 39, "hp": 100} for i in range(5)]
    dec = agent.choose_action(_state(_t=400.0, buildings=BASE, units=units, visible_enemies=enemies,
                                     military={"army_value": 2100, "visible_enemy_value": 3500}), CTX)
    assert agent._mode is Mode.RETREAT
    # they fight back on the way out, to the massing point in front of the base
    assert {"cmd": "attack_move", "units": [30, 31, 32], "to": [11, 11]} in dec.action["orders"]


# -- rules from https://api.altruagent-game.com/skill/redalert ---------------

from agent.redalert import orders as ra_orders  # noqa: E402


def test_batch_stays_under_the_engine_command_cap():
    big = [ra_orders.attack_move(list(range(i * 45, i * 45 + 45)), [10, 10]) for i in range(5)]  # 225 cmds
    move = ra_orders.batch(big + [ra_orders.build("powr")])
    assert sum(ra_orders.engine_commands(o) for o in move["orders"]) <= 200
    assert {"cmd": "build", "item": "powr"} in move["orders"]  # a small order still fits


def test_a_just_sent_build_is_not_queued_twice():
    now = [0.0]
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: now[0])
    state = _state(buildings=[{"id": 10, "type": "fact"}])  # the view lags: powr not shown queued
    assert {"cmd": "build", "item": "powr"} in agent.choose_action(state, CTX).action["orders"]
    now[0] = 0.3
    assert agent.choose_action(state, CTX) is WAIT
    now[0] = 2.0
    assert {"cmd": "build", "item": "powr"} in agent.choose_action(state, CTX).action["orders"]


def test_service_depot_closes_the_double_refinery_order():
    built = [{"id": 10, "type": "fact"}, {"id": 11, "type": "powr"}, {"id": 12, "type": "tent"},
             {"id": 13, "type": "proc"}, {"id": 14, "type": "weap"}, {"id": 15, "type": "proc"},
             {"id": 16, "type": "powr"}, {"id": 17, "type": "proc"}]
    v = _view(buildings=built, available_production=["powr", "tent", "proc", "weap", "fix"],
              economy={"power_provided": 300, "power_drained": 100})
    assert buildplan.next_build_orders(v) == [{"cmd": "build", "item": "fix"}]


def test_engineers_medics_and_husks_are_not_army():
    v = _view(units=[{"id": 1, "type": "e6"}, {"id": 2, "type": "medi"},
                     {"id": 3, "type": "1tnk.husk"}, {"id": 4, "type": "e1"}])
    assert [u.id for u in v.combat_units] == [4]


# -- lessons from match e4caa28c ---------------------------------------------


def test_underpowered_cancels_the_building_blocking_the_power_plant():
    v = _view(buildings=_ECON_BASE + [{"id": 15, "type": "powr"}],
              economy={"power_provided": 9, "power_drained": 130},
              production=[{"item": "weap", "queue": "Building", "ready": False},
                          {"item": "powr", "queue": "Building", "ready": False}])
    assert buildplan.next_build_orders(v) == [{"cmd": "cancel", "item": "weap"}]


def test_a_thin_margin_alone_cancels_nothing():
    v = _view(buildings=_ECON_BASE, economy={"power_provided": 100, "power_drained": 90},
              production=[{"item": "weap", "queue": "Building", "ready": False},
                          {"item": "powr", "queue": "Building", "ready": False}])
    assert not any(o["cmd"] == "cancel" for o in buildplan.next_build_orders(v))


def test_a_late_base_attack_defends_without_dropping_the_economy():
    v = _view(buildings=BASE, _t=400.0)
    assert brain.choose_mode(v, cheese_since=395.0) is Mode.DEFEND_BASE
    v = _view(buildings=_ECON_BASE, units=_harvs(1), _t=400.0)
    items = [o.get("item") for o in executor._production_orders(v, Mode.DEFEND_BASE)]
    assert "harv" in items


def test_an_opening_attack_is_still_a_rush():
    assert brain.choose_mode(_view(buildings=BASE, _t=90.0), cheese_since=85.0) is Mode.DEFEND_RUSH


def test_the_threat_window_runs_from_the_last_sighting():
    near = [{"id": 50 + i, "type": "3tnk", "x": 7, "y": 7, "hp": 100} for i in range(6)]   # a real threat
    assert brain.update_cheese_since(_view(_t=300.0, visible_enemies=near), 100.0) == 300.0
    assert brain.update_cheese_since(_view(_t=305.0), 300.0) == 300.0  # one clear view doesn't end it
    assert brain.update_cheese_since(_view(_t=311.0), 300.0) is None   # 10 s clear: back out (mid-game)


def test_a_small_raid_mid_game_does_not_recall_the_army():
    two = [{"id": 50, "type": "e1", "x": 7, "y": 7, "hp": 100}, {"id": 51, "type": "e1", "x": 8, "y": 8, "hp": 100}]
    assert scout.detect_cheese(_view(_t=300.0, visible_enemies=two)) is None            # home units handle it
    assert scout.detect_cheese(_view(_t=100.0, visible_enemies=two)) is not None        # early: a rush
    six = [{"id": 50 + i, "type": "3tnk", "x": 7, "y": 7, "hp": 100} for i in range(6)]
    assert scout.detect_cheese(_view(_t=300.0, visible_enemies=six)) is not None        # a real attack
    guards = [{"id": 30 + i, "type": "3tnk", "x": 6, "y": 6, "hp": 100} for i in range(8)]
    assert scout.detect_cheese(_view(_t=300.0, visible_enemies=six, units=guards)) is None   # home holds it


def test_an_old_under_attack_event_is_not_a_fresh_threat():
    old = _view(tick=5000, events=[{"seq": 1, "tick": 1000, "kind": "under_attack"}])
    fresh = _view(tick=5000, events=[{"seq": 1, "tick": 4900, "kind": "under_attack"}])
    assert scout.detect_cheese(old) is None and scout.detect_cheese(fresh) is not None


def test_units_only_spend_what_the_economy_does_not_need():
    army = [{"id": 30 + i, "type": "e1", "x": 5, "y": 5, "hp": 100} for i in range(6)]
    q = [{"item": "proc", "queue": "Building", "ready": False, "progress": 0.25}]
    tight = _view(buildings=BASE, units=army, production=q, economy={"cash": 1000})   # 1050 still owed + next step
    assert not any(o.get("item") == "e1" for o in executor._production_orders(tight, Mode.BUILD))
    flush = _view(buildings=BASE, units=army, production=q, economy={"cash": 3000})
    assert any(o.get("item") == "e1" for o in executor._production_orders(flush, Mode.BUILD))
    assert any(o.get("item") == "e1" for o in executor._production_orders(tight, Mode.BUILD, threat=True))


# -- lessons from match 9a9cd98e + harvester raids ---------------------------

from agent.redalert.raid import Raid  # noqa: E402


def test_defense_attacks_the_enemy_at_the_base_not_the_centre():
    near = [{"id": 50, "type": "1tnk", "x": 12, "y": 4, "hp": 100}, {"id": 51, "type": "e1", "x": 14, "y": 5, "hp": 100}]
    o = executor._army_orders(_view(buildings=BASE, units=_ARMY, visible_enemies=near), Mode.DEFEND_BASE)
    assert o == [{"cmd": "attack_move", "units": [30, 31], "to": [12, 4]}]


def test_a_ready_unplaced_building_is_not_queued_again():
    built = [{"id": 10, "type": "fact"}, {"id": 11, "type": "powr"}, {"id": 12, "type": "tent"}, {"id": 13, "type": "proc"}]
    v = _view(buildings=built, production=[{"item": "weap", "queue": "Building", "ready": True}],
              economy={"power_provided": 300, "power_drained": 100})
    assert {"cmd": "build", "item": "weap"} not in buildplan.next_build_orders(v)


def _raid_army(n_rockets=4, n_rifles=6):
    units = [{"id": 100 + i, "type": "e3", "x": 22, "y": 22, "hp": 100} for i in range(n_rockets)]
    units += [{"id": 200 + i, "type": "e1", "x": 22, "y": 22, "hp": 100} for i in range(n_rifles)]
    return units


_ENEMY_HARV = {"id": 90, "type": "harv", "x": 35, "y": 30, "hp": 100}


def test_raid_forms_from_rockets_and_hunts_a_visible_harvester():
    r = Raid()
    v = _view(_t=200.0, buildings=BASE, units=_raid_army(), visible_enemies=[_ENEMY_HARV])
    r.update(v, Mode.BUILD, scout_id=None)
    assert sorted(r.unit_ids) == [100, 101, 102, 103]
    assert r.orders(v) == [{"cmd": "attack", "units": [100, 101, 102, 103], "target": 90}]


def test_raid_waits_for_the_opening_to_pass_and_spares_defenders():
    r = Raid()
    r.update(_view(_t=100.0, buildings=BASE, units=_raid_army(), visible_enemies=[_ENEMY_HARV]), Mode.BUILD, None)
    assert r.unit_ids == []
    r.update(_view(_t=200.0, buildings=BASE, units=_raid_army(4, 2), visible_enemies=[_ENEMY_HARV]), Mode.BUILD, None)
    assert r.unit_ids == []  # 6 units can't spare 4 and keep 6 defenders


def test_raid_targets_a_known_refinery_when_no_harvester_is_known():
    r = Raid()
    v = _view(_t=200.0, buildings=BASE, units=_raid_army(),
              visible_enemy_buildings=[{"id": 80, "type": "proc", "x": 38, "y": 38, "hp": 100}])
    r.update(v, Mode.BUILD, None)
    assert r.orders(v) == [{"cmd": "attack_move", "units": [100, 101, 102, 103], "to": [38, 38]}]


def test_raid_without_a_target_never_forms():
    r = Raid()
    r.update(_view(_t=200.0, buildings=BASE, units=_raid_army()), Mode.BUILD, None)
    assert r.unit_ids == []


def test_raid_is_released_after_losing_half_then_cools_down():
    r = Raid()
    r.update(_view(_t=200.0, buildings=BASE, units=_raid_army(), visible_enemies=[_ENEMY_HARV]), Mode.BUILD, None)
    survivors = [u for u in _raid_army() if u["id"] not in (100, 101)]
    r.update(_view(_t=210.0, buildings=BASE, units=survivors, visible_enemies=[_ENEMY_HARV]), Mode.BUILD, None)
    assert r.unit_ids == []
    r.update(_view(_t=230.0, buildings=BASE, units=_raid_army(), visible_enemies=[_ENEMY_HARV]), Mode.BUILD, None)
    assert r.unit_ids == []  # still cooling down
    r.update(_view(_t=280.0, buildings=BASE, units=_raid_army(), visible_enemies=[_ENEMY_HARV]), Mode.BUILD, None)
    assert r.unit_ids


def test_raid_disbands_when_the_base_needs_defending():
    r = Raid()
    r.update(_view(_t=200.0, buildings=BASE, units=_raid_army(), visible_enemies=[_ENEMY_HARV]), Mode.BUILD, None)
    r.update(_view(_t=205.0, buildings=BASE, units=_raid_army(), visible_enemies=[_ENEMY_HARV]), Mode.DEFEND_BASE, None)
    assert r.unit_ids == []


def test_the_army_leaves_the_raid_squad_alone():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    state = _state(_t=400.0, buildings=BASE, units=_raid_army(4, 10), visible_enemies=[_ENEMY_HARV],  # 4 rifles scout
                   military={"army_value": 3000, "visible_enemy_value": 2500})   # massing, not attacking
    orders_ = agent.choose_action(state, CTX).action["orders"]
    raid = [o for o in orders_ if o.get("target") == 90]
    others = [o for o in orders_ if o["cmd"] in ("attack_move", "move") and o.get("target") is None]
    assert raid and not any(set(o["units"]) & set(raid[0]["units"]) for o in others)


# -- OpenRA wiki advice: placement, defense, repair, target priority ----------

_MID = dict(base_center=[20, 20])   # a base mid-map, so offsets aren't clamped


def test_refinery_goes_beside_the_ore_within_reach():
    v = dataclasses.replace(_view(base_center=[10, 10], buildings=[{"id": 10, "type": "fact", "x": 10, "y": 10}]), ore_cells=((20, 11), (21, 11)))
    x, y = buildplan.placement_cell(v, "proc")
    assert x == 14  # the yard spans x 10-12; one free cell (13), then the refinery, on the ore side


def test_ore_cells_come_from_the_map_layout():
    raw = _live_raw()
    cfg = {"rules": {"maps": {"singles": {**LIVE_CONFIG["rules"]["maps"]["singles"],
                                          "layout": {"origin": [2, 3], "rows": ["..$", "*.."]}}}}}
    assert read(raw, cfg).ore_cells == ((4, 3), (2, 4))


def test_barracks_toward_the_enemy_and_power_away():
    v = _view(base_center=[10, 10], buildings=[{"id": 10, "type": "fact", "x": 10, "y": 10}])  # enemy: south-east
    tx, ty = buildplan.placement_cell(v, "tent")
    px, py = buildplan.placement_cell(v, "powr")
    assert (tx - 10) + (ty - 10) > 0      # on the enemy-facing side (beside the kept-clear corridor)
    assert px < 10 and py < 10            # power at the back


def test_a_defense_goes_down_toward_the_attackers():
    near = [{"id": 50, "type": "e1", "x": 11, "y": 22, "hp": 100}, {"id": 51, "type": "e1", "x": 12, "y": 23, "hp": 100}]
    v = _view(visible_enemies=near, base_center=[10, 10], buildings=[{"id": 10, "type": "fact", "x": 10, "y": 10}])
    x, y = buildplan.placement_cell(v, "pbox")
    assert y > 12 and x >= 8    # below the yard, in the fan slot nearest the attackers


def test_one_standing_defense_and_more_only_under_attack():
    avail = ["pbox", "e1"]
    yard = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 11, "type": "tent", "x": 8, "y": 4}]
    weap = [{"id": 12, "type": "weap", "x": 4, "y": 8}]
    no_factory = _view(buildings=yard, available_production=avail)
    with_factory = _view(buildings=yard + weap, available_production=avail)
    one_built = _view(buildings=yard + weap + [{"id": 13, "type": "pbox", "x": 9, "y": 9}], available_production=avail)
    assert buildplan.defense_orders(no_factory, under_attack=False) == []
    assert buildplan.defense_orders(with_factory, under_attack=False) == [{"cmd": "build", "item": "pbox"}]
    assert buildplan.defense_orders(one_built, under_attack=False) == []
    assert buildplan.defense_orders(one_built, under_attack=True) == [{"cmd": "build", "item": "pbox"}]


def test_damaged_key_buildings_are_repaired():
    v = _view(buildings=[{"id": 10, "type": "fact", "hp": 40}, {"id": 12, "type": "weap", "hp": 30, "repairing": True},
                         {"id": 13, "type": "tent", "hp": 20}, {"id": 14, "type": "powr", "hp": 90}])
    assert executor._repair_orders(v) == [{"cmd": "repair", "building": 10}]


def test_attack_takes_out_power_first_within_the_cluster():
    v = _view(buildings=BASE, units=_ARMY, visible_enemy_buildings=[
        {"id": 80, "type": "weap", "x": 30, "y": 30, "hp": 100},
        {"id": 81, "type": "powr", "x": 34, "y": 33, "hp": 100},
        {"id": 82, "type": "proc", "x": 31, "y": 29, "hp": 100}])
    assert executor._army_orders(v, Mode.ALL_IN)[0]["to"] == [34, 33]


def test_priority_does_not_drag_the_army_across_the_map():
    v = _view(buildings=BASE, units=_ARMY, visible_enemy_buildings=[
        {"id": 80, "type": "proc", "x": 20, "y": 20, "hp": 100},
        {"id": 81, "type": "powr", "x": 44, "y": 44, "hp": 100}])
    assert executor._army_orders(v, Mode.ALL_IN)[0]["to"] == [20, 20]


# -- map knowledge: the enemy's ore field -------------------------------------

import dataclasses  # noqa: E402

from agent.redalert.observe import enemy_ore_field  # noqa: E402

_ORE = ((10, 10), (36, 34), (37, 34), (38, 35), (8, 9))  # ours near [4,4]; theirs near [40,40]


def test_enemy_ore_field_is_the_ore_near_their_spawn():
    v = dataclasses.replace(_view(), ore_cells=_ORE)
    assert enemy_ore_field(v) == (37, 34)


def test_scout_route_visits_the_enemy_ore_field():
    v = dataclasses.replace(_view(), ore_cells=_ORE)
    route = scout.patrol_route(v)
    assert (37, 34) in route and len(route) == 9


def test_raid_falls_back_to_the_enemy_ore_field():
    r = Raid()
    v = dataclasses.replace(_view(_t=200.0, buildings=BASE, units=_raid_army()), ore_cells=_ORE)
    r.update(v, Mode.BUILD, None)
    assert r.orders(v) == [{"cmd": "attack_move", "units": [100, 101, 102, 103], "to": [37, 34]}]


def test_no_layout_no_enemy_ore_field():
    assert enemy_ore_field(_view()) is None


# -- lessons from match 01930376: don't attack into defenses forever ---------


def test_a_losing_exchange_triggers_retreat_even_with_a_big_army_edge():
    v = _view(_t=200.0, buildings=BASE, military={"army_value": 7700, "visible_enemy_value": 1100})
    assert brain.choose_mode(v, cheese_since=None, prev_mode=Mode.ALL_IN) is Mode.ALL_IN
    assert brain.choose_mode(v, cheese_since=None, prev_mode=Mode.ALL_IN, losing_trades=True) is Mode.RETREAT


def test_no_new_attack_during_the_retreat_cooldown():
    v = _view(buildings=BASE, _t=300.0, military={"army_value": 7700, "visible_enemy_value": 1100})
    assert brain.choose_mode(v, cheese_since=None, prev_mode=Mode.BUILD, retreated_at=280.0) is Mode.BUILD
    assert brain.choose_mode(v, cheese_since=None, prev_mode=Mode.BUILD, retreated_at=200.0) is Mode.ALL_IN


def test_the_agent_measures_a_losing_exchange():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    assert not agent._commander.losing_trades(_view(_t=100.0, military={"kills_cost": 1000, "deaths_cost": 1000}))
    assert agent._commander.losing_trades(_view(_t=125.0, military={"kills_cost": 1400, "deaths_cost": 3000}))
    fair = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    fair._commander.losing_trades(_view(_t=100.0, military={"kills_cost": 1000, "deaths_cost": 1000}))
    assert not fair._commander.losing_trades(_view(_t=125.0, military={"kills_cost": 2500, "deaths_cost": 3000}))


def test_known_enemy_defenses_count_as_enemy_strength():
    m = EnemyMemory()
    m.update(_view(visible_enemy_buildings=[{"id": 85, "type": "pbox", "x": 30, "y": 30, "hp": 100}]))
    cfg = {"rules": {**LIVE_CONFIG["rules"], "buildings": {"pbox": {"cost": 600}}}}
    assert m.apply(read(_raw(), cfg)).enemy_army_value == 600


# -- model advisor, concave formation, perimeter defenses ----------------------

from agent.redalert import advisor as ra_advisor  # noqa: E402


class _FakeProvider:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def complete_structured(self, messages, name, schema, *, timeout, effort):
        self.calls += 1
        assert messages[0]["role"] == "system" and "professional" in messages[0]["content"]
        return self.answer


_ANSWER = {"posture": "hold", "attack_route": "auto", "expand": "auto", "army_target": [30, 30],
           "raid_target_id": None, "raid_target_cell": [37, 34], "defense_at": [9, 9],
           "plan": "Mass at home, raid their ore.", "reasoning": "Their army is bigger; harass instead."}


def _settled(adv, view):
    adv.poll(view, {})
    adv._future.result()
    return adv.poll(view, {})


def test_advisor_never_blocks_and_returns_validated_advice():
    adv = ra_advisor.Advisor(_FakeProvider(_ANSWER), log=lambda *_: None, clock=lambda: 0.0, interval=0.0)
    advice = _settled(adv, _view(_t=100.0))
    assert advice.posture == "hold" and advice.defense_at == (9, 9) and advice.raid_target_cell == (37, 34)


def test_advice_off_the_map_is_dropped_and_stale_advice_expires():
    bad = {**_ANSWER, "army_target": [500, 500]}
    adv = ra_advisor.Advisor(_FakeProvider(bad), log=lambda *_: None, clock=lambda: 0.0, interval=1e9)
    advice = _settled(adv, _view(_t=100.0))
    assert advice.army_target is None
    assert adv.poll(_view(_t=200.0), {}) is None  # 100 game seconds later: too old


def test_model_hold_keeps_the_army_home_but_cannot_override_defense():
    v = _view(buildings=BASE, military={"army_value": 2000, "visible_enemy_value": 1000})
    assert brain.choose_mode(v, cheese_since=None, model_pref=Mode.BUILD) is Mode.BUILD
    assert brain.choose_mode(_view(buildings=BASE, _t=400.0), cheese_since=395.0, model_pref=Mode.ALL_IN) is Mode.DEFEND_BASE


def test_model_cannot_force_an_attack_during_retreat_cooldown():
    v = _view(buildings=BASE, _t=300.0, military={"army_value": 4000, "visible_enemy_value": 1000})
    assert brain.choose_mode(v, cheese_since=None, model_pref=Mode.ALL_IN, retreated_at=290.0) is Mode.BUILD


def test_concave_slots_curve_toward_the_enemy():
    slots = executor.concave_slots(_view(), (20, 20), (20, 40), 5)
    assert slots[2] == (20, 20)                       # centre on the stage point
    assert slots[0][1] > 20 and slots[4][1] > 20      # the ends bend toward the enemy (+y)
    assert sorted([slots[0][0], slots[4][0]]) == [17, 23]  # spread evenly across the approach


def test_advice_drives_the_raid_and_turret_side_while_the_rules_form_the_concave():
    adv = ra_advisor.parse_advice(_ANSWER, _view(), 100.0)
    units = [{"id": 30 + i, "type": "e1", "x": 5, "y": 5, "hp": 100, "idle": True} for i in range(5)]
    o = executor._army_orders(_view(buildings=BASE, units=units), Mode.BUILD, advice=adv)
    assert len(o) == 5 and all(x["cmd"] == "attack_move" and len(x["units"]) == 1 for x in o)
    r = Raid()
    rv = _view(_t=200.0, buildings=BASE, units=_raid_army())
    r.update(rv, Mode.BUILD, None, adv)
    assert r.orders(rv)[0]["to"] == [37, 34]
    # defense_at picks the side (here: east, within 60 deg of the enemy to the south-east);
    # the rules pick a legal cell on it
    pv = _view(base_center=[10, 10], buildings=[{"id": 10, "type": "fact", "x": 10, "y": 10}], production=[{"item": "pbox", "queue": "Defense", "ready": True}])
    (order,) = buildplan.placement_orders(pv, (20, 11))
    assert order["item"] == "pbox" and order["at"][0] >= 13   # east of the yard, the advisor's side


def test_defenses_go_beyond_the_outermost_building_and_spread_out():
    buildings = [{"id": 10, "type": "fact", "x": 10, "y": 10}, {"id": 11, "type": "powr", "x": 14, "y": 14}]
    v = _view(buildings=buildings, base_center=[10, 10])   # enemy spawn guess: [40, 40]
    first = buildplan.placement_cell(v, "tsla")
    # on the edge facing the enemy: at least as far out as the power plant
    assert (first[0] - 10) + (first[1] - 10) >= (14 - 10) + (14 - 10)
    v2 = _view(buildings=buildings + [{"id": 12, "type": "tsla", "x": first[0], "y": first[1]}], base_center=[10, 10])
    second = buildplan.placement_cell(v2, "tsla")
    assert second != first and dist2_ok(first, second)


def dist2_ok(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 >= 9


# -- base layout like good players (web research) + home guard ---------------



def _layout_view(**over):
    # base at [10, 10]; enemy spawn guess [40, 40] (south-east); ore to the north-east
    over.setdefault("buildings", [{"id": 10, "type": "fact", "x": 10, "y": 10}])
    v = _view(base_center=[10, 10], **over)
    return dataclasses.replace(v, ore_cells=((20, 4), (21, 4)))


def test_power_goes_to_the_back_and_off_the_ore_side():
    x, y = buildplan.placement_cell(_layout_view(), "powr")
    # away from the enemy (south-east) and from the ore (north-east): west / south-west of the yard
    assert x < 10


def test_war_factory_and_depot_go_to_the_side_off_the_ore_not_into_the_front():
    wx, wy = buildplan.placement_cell(_layout_view(), "weap")
    assert (wy - 10) > (wx - 10)               # the side away from the ore (north-east)
    assert (wx - 10) + (wy - 10) >= 0          # not behind the base
    stage = scout.choke_cell(_layout_view())
    assert (wx + 1 - stage[0]) ** 2 + (wy + 1 - stage[1]) ** 2 > 3 ** 2   # off the army's massing point


def test_placement_keeps_a_gap_instead_of_cramming():
    v = _layout_view(buildings=[{"id": 10, "type": "fact", "x": 10, "y": 10},
                                {"id": 11, "type": "tent", "x": 14, "y": 14}])
    spot = buildplan.placement_cell(v, "barr")
    assert (spot[0] - 14) ** 2 + (spot[1] - 14) ** 2 >= 4


# -- the advisor's plan and reasoning in the logs -----------------------------


def test_advisor_plan_goes_to_console_only_when_it_changes(tmp_path):
    lines = []
    agent = RedAlertAgent(log=lines.append, clock=lambda: 0.0, log_dir=str(tmp_path))
    agent.choose_action(_state(), CTX)  # opens the match log
    a = ra_advisor.parse_advice(_ANSWER, _view(), 100.0)
    agent._on_advice(a, 2.5)
    agent._on_advice(ra_advisor.parse_advice(_ANSWER, _view(), 106.0), 2.4)       # same plan: quiet
    agent._on_advice(ra_advisor.parse_advice({**_ANSWER, "posture": "attack", "plan": "Push their power now."},
                                             _view(), 112.0), 3.0)
    console = [l for l in lines if "advisor t=" in l]
    assert len(console) == 2
    assert "hold: Mass at home, raid their ore. | why: Their army is bigger" in console[0]
    assert "attack: Push their power now." in console[1]

    final = GameState.from_mcp_state({**_raw(), "phase": "finished", "is_terminal": True,
                                      "result": {"outcome": "win", "reason": "time_limit"}})
    agent.on_match_end(final, CTX)
    md = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert "## Model plan" in md and "3 answers" in md
    assert "**100s, hold**: Mass at home" in md and "**112s, attack**: Push their power now." in md
    assert md.count("**106s") == 0  # a repeated plan is not listed again


def test_army_movement_always_attack_moves_even_in_retreat():
    """Every army order fights on the way; only RETREAT walks out with a plain move."""
    units = [{"id": 30 + i, "type": "1tnk", "x": 38, "y": 38, "hp": 100, "idle": True} for i in range(6)]
    adv = ra_advisor.parse_advice(_ANSWER, _view(), 100.0)
    enemy = [{"id": 50, "type": "e1", "x": 12, "y": 4, "hp": 100}, {"id": 51, "type": "e1", "x": 13, "y": 4, "hp": 100}]
    for mode, kw in ((Mode.BUILD, {}), (Mode.BUILD, {"advice": adv}), (Mode.ALL_IN, {}),
                     (Mode.ALL_IN, {"advice": adv}), (Mode.DEFEND_BASE, {}), (Mode.DEFEND_RUSH, {})):
        v = _view(buildings=BASE, units=units, visible_enemies=enemy if mode in (Mode.DEFEND_BASE, Mode.DEFEND_RUSH) else [])
        cmds = {o["cmd"] for o in executor._army_orders(v, mode, **kw)}
        assert cmds <= {"attack_move", "attack"}, (mode, cmds)
    assert {o["cmd"] for o in executor._army_orders(_view(buildings=BASE, units=units), Mode.RETREAT)} == {"attack_move"}


# -- lessons from match 1a2767cd: commit, don't turtle, raid with what we have --


def test_model_hold_cannot_stop_an_overwhelming_attack():
    v = _view(_t=200.0, buildings=BASE, military={"army_value": 7350, "visible_enemy_value": 2350})
    assert brain.choose_mode(v, cheese_since=None, model_pref=Mode.BUILD) is Mode.ALL_IN
    unseen = _view(_t=200.0, buildings=BASE, military={"army_value": 7350, "visible_enemy_value": 0})
    assert brain.choose_mode(unseen, cheese_since=None, model_pref=Mode.BUILD) is Mode.BUILD  # nothing known: hold is fine


def test_the_army_commits_instead_of_flip_flopping():
    tanks = [{"id": 30 + i, "type": "1tnk", "x": 22, "y": 22, "hp": 100} for i in range(4)]
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent.choose_action(_state(_t=366.0, buildings=BASE, units=tanks, military={"army_value": 2800}), CTX)
    agent.choose_action(_state(_t=370.0, buildings=BASE, units=tanks, military={"army_value": 2800}), CTX)
    assert agent._mode is Mode.ALL_IN
    behind = {"army_value": 2800, "visible_enemy_value": 2500}   # edge 1.12: the policy would say BUILD
    agent.choose_action(_state(_t=380.0, buildings=BASE, units=tanks, military=behind), CTX)
    assert agent._mode is Mode.ALL_IN                            # 10 s in: stays committed
    agent.choose_action(_state(_t=400.0, buildings=BASE, units=tanks, military=behind), CTX)
    assert agent._mode is Mode.BUILD                             # 30 s in: may switch


def test_raids_use_light_vehicles_when_infantry_is_short_and_spare_other_squads():
    tanks = [{"id": 300 + i, "type": "1tnk", "x": 22, "y": 22, "hp": 100} for i in range(10)]
    guard = [{"id": 400, "type": "e1", "x": 5, "y": 5, "hp": 100}, {"id": 401, "type": "e1", "x": 5, "y": 5, "hp": 100}]
    r = Raid()
    v = _view(_t=200.0, buildings=BASE, units=tanks + guard, visible_enemies=[_ENEMY_HARV])
    r.update(v, Mode.BUILD, None, taken={400, 401})
    assert len(r.unit_ids) == 4 and not {400, 401} & set(r.unit_ids)


# -- lessons from match 28e54539 ----------------------------------------------


def test_a_needed_harvester_stuck_behind_tanks_cancels_them():
    v = _view(buildings=_ECON_BASE, units=_harvs(1),
              production=[{"item": "1tnk", "queue": "Vehicle", "ready": False},
                          {"item": "1tnk", "queue": "Vehicle", "ready": False},
                          {"item": "harv", "queue": "Vehicle", "ready": False}])
    o = executor._production_orders(v, Mode.DEFEND_BASE)
    assert o[0] == {"cmd": "cancel", "item": "1tnk"}


def test_no_cancel_when_the_harvester_is_already_first_or_not_needed():
    first = _view(buildings=_ECON_BASE, units=_harvs(1),
                  production=[{"item": "harv", "queue": "Vehicle", "ready": False},
                              {"item": "1tnk", "queue": "Vehicle", "ready": False}])
    full = _view(buildings=_ECON_BASE, units=_harvs(3),
                 production=[{"item": "1tnk", "queue": "Vehicle", "ready": False},
                             {"item": "harv", "queue": "Vehicle", "ready": False}])
    for v in (first, full):
        assert not any(o["cmd"] == "cancel" for o in executor._production_orders(v, Mode.BUILD))


def test_losses_from_before_the_attack_do_not_trigger_a_retreat():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent._commander.losing_trades(_view(_t=470.0, military={"kills_cost": 1000, "deaths_cost": 1000}))
    agent._commander.losing_trades(_view(_t=490.0, military={"kills_cost": 1000, "deaths_cost": 2500}))  # hit while massing
    agent._mode_since = 493.0                                                                  # attack starts
    assert not agent._commander.losing_trades(_view(_t=496.0, military={"kills_cost": 1000, "deaths_cost": 2500}))
    assert agent._commander.losing_trades(_view(_t=515.0, military={"kills_cost": 1200, "deaths_cost": 4000}))  # losing *this* fight



# -- legal, growing placement (match 28e54539: hints too far were ignored) ----


def _legal(spot, item, buildings):
    """One free cell from some building, overlapping none (footprints top-left)."""
    fp = lambda t: __import__("agent.redalert.config", fromlist=["x"]).FOOTPRINTS.get(t, (2, 2))  # noqa: E731
    w, h = fp(item)
    touching = False
    for b in buildings:
        bw, bh = fp(b["type"])
        gx = max(b["x"] - (spot[0] + w - 1), spot[0] - (b["x"] + bw - 1)) - 1
        gy = max(b["y"] - (spot[1] + h - 1), spot[1] - (b["y"] + bh - 1)) - 1
        if gx < 1 and gy < 1:
            return False                   # overlaps or leaves no path
        if max(gx, gy) <= 2:
            touching = True
    return touching


def test_every_placement_is_within_reach_of_the_base():
    blds = [{"id": 10, "type": "fact", "x": 10, "y": 10}, {"id": 11, "type": "powr", "x": 6, "y": 7},
            {"id": 12, "type": "tent", "x": 14, "y": 13}, {"id": 13, "type": "proc", "x": 14, "y": 8}]
    v = dataclasses.replace(_view(base_center=[11, 11], buildings=blds), ore_cells=_ore_patch(20, 2, 6, 6))
    for item in ("proc", "tent", "weap", "fix", "powr", "silo", "pbox"):
        spot = buildplan.placement_cell(v, item)
        assert spot is not None and _legal(spot, item, blds), (item, spot)


def test_defenses_spread_along_the_edge_instead_of_piling_up():
    blds = [{"id": 10, "type": "fact", "x": 10, "y": 10}]
    first = buildplan.placement_cell(_view(base_center=[10, 10], buildings=blds), "pbox")
    blds2 = blds + [{"id": 20, "type": "pbox", "x": first[0], "y": first[1]}]
    second = buildplan.placement_cell(_view(base_center=[10, 10], buildings=blds2), "pbox")
    assert (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2 >= 9


# -- lessons from match 9ba271bb: don't flinch, don't hit the same wall ----------


def test_no_trade_retreat_in_the_first_seconds_of_a_fight():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent._mode_since, agent._commander.attack_army = 700.0, 14000.0
    agent._commander.losing_trades(_view(_t=700.0, military={"kills_cost": 15000, "deaths_cost": 22000}))
    assert not agent._commander.losing_trades(_view(_t=705.0, military={"kills_cost": 15200, "deaths_cost": 25000}))  # 5 s in


def test_trade_retreat_scales_with_the_army():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent._mode_since, agent._commander.attack_army = 700.0, 14000.0    # floor: 15% of 14000 = 2100
    agent._commander.losing_trades(_view(_t=700.0, military={"kills_cost": 15000, "deaths_cost": 22000}))
    assert not agent._commander.losing_trades(_view(_t=720.0, military={"kills_cost": 15300, "deaths_cost": 23500}))  # lost 1500
    assert agent._commander.losing_trades(_view(_t=725.0, military={"kills_cost": 15300, "deaths_cost": 25000}))      # lost 3000


def test_no_new_attack_until_the_army_has_regrown():
    v = _view(buildings=BASE, _t=700.0, military={"army_value": 11900, "visible_enemy_value": 3000})
    assert brain.choose_mode(v, cheese_since=None, retreated_at=620.0, retreat_army=10150.0) is Mode.BUILD   # 1.17x
    v2 = _view(buildings=BASE, _t=700.0, military={"army_value": 13500, "visible_enemy_value": 3000})
    assert brain.choose_mode(v2, cheese_since=None, retreated_at=620.0, retreat_army=10150.0) is Mode.ALL_IN  # 1.33x


def test_a_model_defense_side_behind_the_base_is_ignored():
    yard = [{"id": 10, "type": "fact", "x": 10, "y": 10}]
    v = _view(base_center=[10, 10], buildings=yard)            # enemy spawn guess: south-east
    behind = buildplan.placement_cell(v, "pbox", defense_at=(2, 2))
    default = buildplan.placement_cell(v, "pbox")
    assert behind == default and behind[0] >= 10 and behind[1] >= 10


# -- lessons from match 4d88e181: safe staging, economy, flanks ------------------


def test_double_refinery_order():
    """Power, barracks, refinery, refinery, power, war factory, refinery, depot."""
    avail = ["powr", "tent", "proc", "weap", "fix"]
    eco = {"power_provided": 300, "power_drained": 100}
    have = [{"id": 10, "type": "fact"}]
    expected = ["powr", "tent", "proc", "proc", "powr", "weap", "proc", "fix"]
    for i, item in enumerate(expected):
        order = buildplan.next_build_orders(_view(buildings=have, available_production=avail, economy=eco))
        assert order == [{"cmd": "build", "item": item}], (i, order)
        have = have + [{"id": 20 + i, "type": item}]


def test_four_harvesters_early_and_at_most_three_per_refinery():
    two_refs = _view(buildings=_ECON_BASE + [{"id": 14, "type": "proc"}])
    assert executor.harvester_target(two_refs) == 4   # refineries + the early pair
    one_ref = _view(buildings=_ECON_BASE)
    assert executor.harvester_target(one_ref) == 3   # 4 wanted early, but 1 refinery takes 3


def test_saturated_refineries_get_another_refinery_not_more_trucks():
    built = _ECON_BASE + [{"id": 15, "type": "powr"}, {"id": 16, "type": "powr"}, {"id": 17, "type": "fix"},
                      {"id": 18, "type": "tent"}]   # 2 barracks: the barracks schedule is met
    v = _view(buildings=built, units=_harvs(3), economy={"power_provided": 300, "power_drained": 100})
    assert buildplan.next_build_orders(v) == [{"cmd": "build", "item": "proc"}]


def test_when_broke_the_build_order_keeps_its_money():
    v = _view(buildings=_ECON_BASE, units=_harvs(3), economy={"cash": 100, "ore": 0})
    items = [o.get("item") for o in executor._production_orders(v, Mode.BUILD)]
    assert items == []                                    # the next power plant comes first
    rich = _view(buildings=_ECON_BASE, units=_harvs(3), economy={"cash": 3000})
    assert "e3" in [o.get("item") for o in executor._production_orders(rich, Mode.BUILD)]


def test_stored_ore_counts_as_money():
    assert executor.funds(_view(economy={"cash": 100, "ore": 900})) == 1000


def test_protect_lead_keeps_training_units_are_assets():
    v = _view(buildings=_ECON_BASE, units=_harvs(3), economy={"cash": 3000})
    assert executor._production_orders(v, Mode.PROTECT_LEAD)


def test_six_riflemen_are_not_an_army_to_attack_with():
    rifles = [{"id": 30 + i, "type": "e1", "x": 5, "y": 5, "hp": 100} for i in range(6)]
    v = _view(buildings=BASE, units=rifles, military={"army_value": 600})
    assert brain.choose_mode(v, cheese_since=None) is Mode.BUILD


def test_staging_is_on_our_side_and_falls_back_from_known_enemies():
    assert scout.choke_cell(_view()) == (11, 11)                # 10 cells in front of [4,4] toward [40,40]
    near = [{"id": 50, "type": "1tnk", "x": 12, "y": 11, "hp": 100}]
    assert scout.choke_cell(_view(visible_enemies=near)) == (6, 6)  # enemies there: right by the base


def test_the_advisor_does_not_pick_the_massing_point():
    adv = ra_advisor.parse_advice({**_ANSWER, "stage_at": [22, 22], "formation": "clump"}, _view(), 100.0)
    units = [{"id": 30, "type": "e1", "x": 5, "y": 5, "hp": 100, "idle": True}]
    o = executor._army_orders(_view(buildings=BASE, units=units), Mode.BUILD, advice=adv)
    assert o[0]["to"] == [11, 11]                               # the rules' massing point, not the advisor's


def test_flank_waypoint_is_off_to_the_side_and_passable():
    v = _view()                                                 # base [4,4], target [40,40]
    left, right = executor.flank_waypoint(v, (40, 40), -1), executor.flank_waypoint(v, (40, 40), 1)
    assert left != right and left != (22, 22)
    blocked = dataclasses.replace(v, blocked_cells=frozenset({right}))
    assert executor.flank_waypoint(blocked, (40, 40), 1) not in blocked.blocked_cells


def test_flank_picks_the_quieter_side():
    v = _view()
    center_threat = [{"id": 50 + i, "type": "1tnk", "x": 22, "y": 22, "hp": 100} for i in range(3)]
    side, via = executor.choose_flank(_view(visible_enemies=center_threat), (40, 40), attacks=0)
    assert side != 0 and via is not None
    side0, via0 = executor.choose_flank(v, (40, 40), attacks=0)
    assert side0 == 0 and via0 is None                          # quiet center: straight in


def test_a_flank_attack_goes_round_then_in():
    units = [{"id": 30 + i, "type": "1tnk", "x": 15, "y": 15, "hp": 100} for i in range(4)]
    o = executor._army_orders(_view(buildings=BASE, units=units), Mode.ALL_IN, flank_via=(30, 10))
    assert o[0]["to"] == [30, 10] and o[1]["to"] == [40, 40] and o[1]["queued"] is True


# -- lessons from match bc8ae082: spend the bank -----------------------------------


def test_money_piling_up_builds_a_second_war_factory():
    eco = {"cash": 0, "ore": 6000, "resource_capacity": 8000, "power_provided": 400, "power_drained": 100}
    five_rax = _ECON_BASE + [{"id": 15, "type": "powr"}, {"id": 16, "type": "fix"}] + \
        [{"id": 20 + i, "type": "tent"} for i in range(4)]
    assert buildplan.next_build_orders(_view(buildings=five_rax, economy=eco)) == [{"cmd": "build", "item": "weap"}]
    rich = dict(eco, ore=4000)
    assert {"cmd": "build", "item": "weap"} not in buildplan.next_build_orders(_view(buildings=five_rax, economy=rich))


def test_even_rich_queues_at_most_two_deep():
    v = _view(buildings=_ECON_BASE, units=_harvs(3), economy={"cash": 0, "ore": 6000},
              production=[{"item": "e1", "queue": "Infantry", "ready": False}] * 2)
    items = [o.get("item") for o in executor._production_orders(v, Mode.BUILD)]
    assert "e1" not in items                               # money finishes what is queued


# -- expansion (expand.py) -------------------------------------------------------

from agent.redalert.expand import Expansion, choose_site  # noqa: E402

# Base [4,4], enemy [40,40]. Ore: our main field by the base, a second field on
# our side at (24, 6), and one on the enemy's side at (36, 34).
_EXP_ORE = ((8, 8), (9, 8), (24, 6), (25, 6), (24, 7), (36, 34), (37, 34), (6, 28), (7, 28))
_EXP_BASE = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 11, "type": "tent", "x": 8, "y": 4},
             {"id": 12, "type": "weap", "x": 4, "y": 8}, {"id": 13, "type": "proc", "x": 8, "y": 9},
             {"id": 14, "type": "proc", "x": 12, "y": 9}, {"id": 15, "type": "fix", "x": 1, "y": 8}]


def _exp_view(t=260.0, units=(), enemies=(), buildings=_EXP_BASE, available=("mcv", "proc", "1tnk"), army=7000, **kw):
    v = _view(_t=t, buildings=list(buildings), units=list(units) + _harvs(4), visible_enemies=list(enemies),
              available_production=list(available), military={"army_value": army}, **kw)
    return dataclasses.replace(v, ore_cells=_EXP_ORE)


def test_expansion_site_is_a_second_field_on_our_side():
    spot, ore = choose_site(_exp_view())
    assert ore in ((24, 6), (25, 6), (24, 7))        # not our main field, not the enemy's
    assert spot not in _EXP_ORE and (spot[0] - 4) ** 2 + (spot[1] - 4) ** 2 < (ore[0] - 4) ** 2 + (ore[1] - 4) ** 2


def test_expansion_waits_for_time_economy_and_army():
    for v in (_exp_view(t=200.0), _exp_view(buildings=_EXP_BASE[:4])):   # army size no longer gates it
        e = Expansion()
        e.update(v, Mode.BUILD)
        assert e.state == "idle"
    e = Expansion()
    e.update(_exp_view(), Mode.DEFEND_BASE)
    assert e.state == "idle"


def test_expansion_scouts_then_trains_moves_and_deploys():
    e = Expansion()
    e.update(_exp_view(), Mode.BUILD)
    assert e.state == "scouting" and e.scout_target == e.site
    assert e.orders(_exp_view()) == [{"cmd": "train", "item": "mcv", "count": 1}]  # built while the scout looks
    seer = [{"id": 60, "type": "e1", "x": e.site[0], "y": e.site[1], "hp": 100}]
    e.update(_exp_view(t=270.0, units=seer), Mode.BUILD)
    assert e.state == "training"
    assert e.orders(_exp_view(t=271.0)) == [{"cmd": "train", "item": "mcv", "count": 1}]
    mcv = [{"id": 70, "type": "mcv", "x": 6, "y": 10, "hp": 100}]
    e.update(_exp_view(t=300.0, units=mcv), Mode.BUILD)
    assert e.state == "moving" and e.escorting and e.stage_override == e.site
    assert e.orders(_exp_view(t=301.0, units=mcv)) == [{"cmd": "move", "units": [70], "to": list(e.site)}]
    there = [{"id": 70, "type": "mcv", "x": e.site[0], "y": e.site[1], "hp": 100}]
    assert e.orders(_exp_view(t=330.0, units=there)) == [{"cmd": "deploy", "units": [70]}]
    site = e.site
    yard = _EXP_BASE + [{"id": 80, "type": "fact", "x": site[0], "y": site[1]}]
    e.update(_exp_view(t=335.0, buildings=yard), Mode.BUILD)
    assert e.state == "idle" and e.sites == [(site[0] + 1, site[1] + 1)]     # done, at the yard's centre
    assert e.refinery_ore(_exp_view(t=336.0, buildings=yard)) is not None    # a refinery by the new ore next


def test_expansion_aborts_when_the_site_is_contested():
    e = Expansion()
    e.update(_exp_view(), Mode.BUILD)
    enemy = [{"id": 90, "type": "3tnk", "x": e.site[0] + 2, "y": e.site[1], "hp": 100}]
    e.update(_exp_view(t=265.0, enemies=enemy), Mode.BUILD)
    assert e.state == "idle" and e.site is None


def test_a_lost_mcv_counts_as_a_failure_and_cools_down():
    e = Expansion()
    e.state, e.site, e.mcv_id = "moving", (20, 8), 70
    e.update(_exp_view(t=320.0), Mode.BUILD)
    assert e.state == "idle" and e._failures == 1
    e.update(_exp_view(t=330.0), Mode.BUILD)
    assert e.state == "idle"                                                  # cooling down


def test_advisor_can_hold_or_hurry_the_expansion():
    wait = ra_advisor.parse_advice({**_ANSWER, "expand": "wait"}, _view(), 0.0)
    now = ra_advisor.parse_advice({**_ANSWER, "expand": "now"}, _view(), 0.0)
    e = Expansion()
    e.update(_exp_view(), Mode.BUILD, wait)
    assert e.state == "idle"
    e.update(_exp_view(t=200.0), Mode.BUILD, now)                             # 3:20 is fine on "now"
    assert e.state == "scouting"


def test_the_army_escorts_the_mcv_instead_of_attacking():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent._expansion.state, agent._expansion.site, agent._expansion.mcv_id = "moving", (20, 8), 70
    agent._expansion._since = 280.0                                          # on the road for 20 s
    tanks = [{"id": 30 + i, "type": "1tnk", "x": 5, "y": 5, "hp": 100, "idle": True} for i in range(4)]
    mcv = [{"id": 70, "type": "mcv", "x": 10, "y": 8, "hp": 100}]
    dec = agent.choose_action(_state(_t=300.0, buildings=BASE, units=tanks + mcv, military={"army_value": 4000}), CTX)
    assert agent._mode is Mode.BUILD
    assert any(o["cmd"] == "attack_move" and (o["to"][0] - 20) ** 2 + (o["to"][1] - 8) ** 2 <= 5 ** 2
               for o in dec.action["orders"])                                # a concave at the site


def test_the_expansion_refinery_goes_beside_the_new_ore():
    yard = _EXP_BASE + [{"id": 80, "type": "fact", "x": 20, "y": 8}]
    v = _exp_view(buildings=yard, production=[{"item": "proc", "queue": "Building", "ready": True}])
    (order,) = buildplan.placement_orders(v, None, (24, 6))
    x, y = order["at"]
    assert (x - 24) ** 2 + (y - 6) ** 2 < (x - 4) ** 2 + (y - 4) ** 2         # by the new field, not home


# -- enemy expansions: scout them, raid them ---------------------------------------

from agent.redalert.expand import enemy_expansion_sites, enemy_expansions  # noqa: E402

# Base [4,4], enemy [40,40]. Ore: theirs at home (38,36), a field they may
# expand to at (40,14) (their side, far from their base), ours at (8,8).
_THEIR_ORE = ((38, 36), (39, 36), (40, 14), (41, 14), (40, 15), (8, 8))


def test_likely_enemy_expansion_sites_are_far_from_their_base_on_their_side():
    v = dataclasses.replace(_view(), ore_cells=_THEIR_ORE)
    sites = enemy_expansion_sites(v)
    assert sites and sites[0] in ((40, 14), (41, 14), (40, 15))
    assert all(s not in ((38, 36), (39, 36), (8, 8)) for s in sites)


def test_scout_watches_enemy_expansion_sites_from_three_thirty():
    early = dataclasses.replace(_view(_t=150.0), ore_cells=_THEIR_ORE)
    late = dataclasses.replace(_view(_t=240.0), ore_cells=_THEIR_ORE)
    site = enemy_expansion_sites(late)[0]
    assert site not in scout.patrol_route(early) and site in scout.patrol_route(late)


def test_a_far_enemy_building_is_an_expansion_preferring_its_refinery():
    blds = [{"id": 80, "type": "powr", "x": 40, "y": 15, "hp": 100}, {"id": 81, "type": "proc", "x": 42, "y": 13, "hp": 100},
            {"id": 82, "type": "fact", "x": 38, "y": 38, "hp": 100}]   # the last is their main base
    assert enemy_expansions(_view(visible_enemy_buildings=blds)) == [(42, 13)]


def test_raid_hits_a_lightly_defended_enemy_expansion():
    blds = [{"id": 81, "type": "proc", "x": 42, "y": 13, "hp": 100}]
    r = Raid()
    v = _view(_t=300.0, buildings=BASE, units=_raid_army(), visible_enemy_buildings=blds)
    r.update(v, Mode.BUILD, None)
    assert r.orders(v)[0] == {"cmd": "attack_move", "units": r.unit_ids, "to": [42, 13]}


def test_a_defended_enemy_expansion_is_left_to_the_main_army():
    blds = [{"id": 81, "type": "proc", "x": 42, "y": 13, "hp": 100},
            {"id": 83, "type": "tsla", "x": 41, "y": 15, "hp": 100}, {"id": 84, "type": "ftur", "x": 43, "y": 15, "hp": 100}]
    r = Raid()
    v = dataclasses.replace(_view(_t=300.0, buildings=BASE, units=_raid_army(), visible_enemy_buildings=blds),
                            ore_cells=_THEIR_ORE)
    r.update(v, Mode.BUILD, None)
    assert r.orders(v)[0]["to"] != [42, 13]


def test_a_visible_harvester_still_comes_first():
    blds = [{"id": 81, "type": "proc", "x": 42, "y": 13, "hp": 100}]
    r = Raid()
    v = _view(_t=300.0, buildings=BASE, units=_raid_army(), visible_enemy_buildings=blds, visible_enemies=[_ENEMY_HARV])
    r.update(v, Mode.BUILD, None)
    assert r.orders(v)[0]["target"] == 90


# -- keep expanding, garrison, lanes (match f9f06e34) ---------------------------------

from agent.redalert.guard import Garrison  # noqa: E402


def _deployed_expansion(t=335.0):
    e = Expansion()
    e.done.append({"site": (20, 6), "ore": (24, 6), "asked_at": None})
    e._last_done_at = t
    return e


def test_the_expansion_refinery_is_asked_for_once_not_every_view():
    e = _deployed_expansion()
    yard = _EXP_BASE + [{"id": 80, "type": "fact", "x": 20, "y": 6}]
    v = _exp_view(t=340.0, buildings=yard, production=[{"item": "proc", "queue": "Building", "ready": False}])
    e.update(v, Mode.BUILD)
    assert e.refinery_ore(_exp_view(t=350.0, buildings=yard)) is None        # just asked: wait
    assert e.refinery_ore(_exp_view(t=440.0, buildings=yard)) == (24, 6)     # 90 s later: try again


def test_refineries_are_capped():
    e = _deployed_expansion()
    many = _EXP_BASE + [{"id": 100 + i, "type": "proc", "x": 30 + 4 * i, "y": 30} for i in range(6)]
    assert e.refinery_ore(_exp_view(buildings=many)) is None                 # 7 refineries: at the cap


def test_a_second_expansion_waits_unless_the_ore_runs_out():
    e = _deployed_expansion(t=335.0)
    e.update(_exp_view(t=400.0), Mode.BUILD)
    assert e.state == "idle"                                                 # 65 s after the first
    idle_harvs = [{"id": 20 + i, "type": "harv", "x": 6, "y": 6, "hp": 100, "idle": True} for i in range(2)]
    e.update(_exp_view(t=400.0, units=idle_harvs), Mode.BUILD)
    e.update(_exp_view(t=435.0, units=idle_harvs), Mode.BUILD)
    assert e.state == "scouting"                                             # harvesters idle 35 s: expand now


def test_a_refused_deploy_tries_spots_around_the_site_not_toward_home():
    e = Expansion()
    e.state, e.site, e.mcv_id = "moving", (20, 8), 70
    e._spots = [(23, 8), (17, 8)]
    there = [{"id": 70, "type": "mcv", "x": 20, "y": 8, "hp": 100}]
    for t in (300.0, 304.0):
        e.orders(_exp_view(t=t, units=there))
    o = e.orders(_exp_view(t=308.0, units=there))
    assert o == [{"cmd": "move", "units": [70], "to": [23, 8]}]


def test_each_expansion_gets_a_turret_on_its_enemy_side():
    e = _deployed_expansion()
    yard = _EXP_BASE + [{"id": 80, "type": "fact", "x": 20, "y": 6}]
    v = _exp_view(buildings=yard, available=("pbox", "mcv"))
    assert e.defense_site(v) == (20, 6)
    assert buildplan.defense_orders(v, under_attack=False, expansion_site=(20, 6)) == [{"cmd": "build", "item": "pbox"}]
    pv = _exp_view(buildings=yard, production=[{"item": "pbox", "queue": "Defense", "ready": True}])
    (order,) = buildplan.placement_orders(pv, None, None, (20, 6))
    x, y = order["at"]
    assert (x - 20) ** 2 + (y - 6) ** 2 <= 100                               # at the expansion, not home


def test_garrison_holds_the_expansion_then_joins_a_push_once_complete():
    e = _deployed_expansion()
    tanks = [{"id": 300 + i, "type": "1tnk", "x": 15, "y": 15, "hp": 100, "idle": True} for i in range(12)]
    yard = _EXP_BASE + [{"id": 80, "type": "fact", "x": 20, "y": 6}]
    g = Garrison()
    g.update(_exp_view(units=tanks, buildings=yard), Mode.BUILD, e, taken=set())
    assert len(g.unit_ids) == 4
    assert g.orders(_exp_view(units=tanks, buildings=yard), e)[0]["cmd"] == "attack_move"
    g.update(_exp_view(units=tanks, buildings=yard), Mode.ALL_IN, e, taken=set())
    assert len(g.unit_ids) == 4                                              # not complete yet: hold it
    done = yard + [{"id": 81, "type": "proc", "x": 24, "y": 8}, {"id": 82, "type": "pbox", "x": 22, "y": 10}]
    g.update(_exp_view(units=tanks, buildings=done), Mode.ALL_IN, e, taken=set())
    assert g.unit_ids == []                                                  # complete: join the push


def test_an_attack_on_an_expansion_yard_is_a_base_attack():
    yard = BASE + [{"id": 80, "type": "fact", "x": 30, "y": 8}]
    near = [{"id": 50, "type": "1tnk", "x": 32, "y": 10, "hp": 100}, {"id": 51, "type": "e1", "x": 33, "y": 9, "hp": 100}]
    v = _view(buildings=yard, visible_enemies=near)
    assert scout.detect_cheese(v) is not None
    o = executor._army_orders(_view(buildings=yard, units=_ARMY, visible_enemies=near), Mode.DEFEND_BASE)
    assert o[0]["to"] in ([32, 10], [33, 9])


def test_buildings_keep_the_road_out_and_factory_exits_clear():
    from agent.redalert.buildplan import _lanes

    blds = [{"id": 10, "type": "fact", "x": 10, "y": 10}, {"id": 11, "type": "weap", "x": 6, "y": 14}]
    v = _view(base_center=[10, 10], buildings=blds)
    lanes = _lanes(v)
    assert (7, 17) in lanes and (8, 18) in lanes                             # below the war factory
    for item in ("tent", "powr", "proc", "fix", "pbox"):
        spot = buildplan.placement_cell(v, item)
        if spot is None:
            continue
        w, h = __import__("agent.redalert.config", fromlist=["x"]).FOOTPRINTS.get(item, (2, 2))
        cells = {(spot[0] + i, spot[1] + j) for i in range(w) for j in range(h)}
        assert not cells & lanes, (item, spot)


# -- lessons from match f9f06e34's end: the tiebreak and the bank ---------------------


def test_protect_lead_only_when_actually_ahead_on_trades():
    behind = _view(buildings=BASE, _r=110.0,
                   military={"army_value": 20000, "visible_enemy_value": 15000, "kills_cost": 25500, "deaths_cost": 28600})
    ahead = _view(buildings=BASE, _r=110.0,
                  military={"army_value": 20000, "visible_enemy_value": 15000, "kills_cost": 29600, "deaths_cost": 26500})
    assert brain.choose_mode(ahead, cheese_since=None) is Mode.PROTECT_LEAD
    assert brain.choose_mode(behind, cheese_since=None) is not Mode.PROTECT_LEAD


def test_behind_at_the_end_the_army_attacks_despite_hold_and_regrowth():
    v = _view(buildings=BASE, _t=1050.0, _r=150.0,
              military={"army_value": 9000, "visible_enemy_value": 8000, "kills_cost": 25500, "deaths_cost": 28600})
    assert brain.choose_mode(v, cheese_since=None, model_pref=Mode.BUILD,
                             retreated_at=900.0, retreat_army=8500.0) is Mode.ALL_IN
    early = _view(buildings=BASE, _t=600.0, _r=600.0,
                  military={"army_value": 9000, "visible_enemy_value": 8000, "kills_cost": 25500, "deaths_cost": 28600})
    assert brain.choose_mode(early, cheese_since=None, model_pref=Mode.BUILD) is Mode.BUILD   # not yet the end


def test_a_big_bank_builds_barracks_first_then_a_second_war_factory():
    eco = {"cash": 0, "ore": 20000, "resource_capacity": 30000, "power_provided": 500, "power_drained": 100}
    base = _ECON_BASE + [{"id": 15, "type": "powr"}, {"id": 16, "type": "fix"}]
    assert buildplan.next_build_orders(_view(buildings=base, economy=eco)) == [{"cmd": "build", "item": "tent"}]
    five = base + [{"id": 20 + i, "type": "tent"} for i in range(4)]
    assert buildplan.next_build_orders(_view(buildings=five, economy=eco)) == [{"cmd": "build", "item": "weap"}]


def _lead_view(remaining, lead):
    return _view(buildings=BASE, _r=remaining,
                 military={"army_value": 9000, "visible_enemy_value": 9000, "kills_cost": 20000 + lead, "deaths_cost": 20000})


def test_protect_lead_windows():
    assert brain.choose_mode(_lead_view(150.0, 5000), cheese_since=None) is not Mode.PROTECT_LEAD  # 2:30 left: keep playing
    assert brain.choose_mode(_lead_view(110.0, 1500), cheese_since=None) is Mode.PROTECT_LEAD      # 2 min, real lead
    assert brain.choose_mode(_lead_view(110.0, 800), cheese_since=None) is not Mode.PROTECT_LEAD   # 2 min, thin lead
    assert brain.choose_mode(_lead_view(50.0, 100), cheese_since=None) is Mode.PROTECT_LEAD        # last minute: any lead
    assert brain.choose_mode(_lead_view(50.0, 0), cheese_since=None) is not Mode.PROTECT_LEAD      # level: go get kills


# -- lessons from match 4b5980f8: the escort must not freeze the army ------------------


def test_a_slow_mcv_deploys_where_it_stands_then_gives_up():
    e = Expansion()
    e.state, e.site, e.mcv_id, e._since = "moving", (40, 8), 70, 300.0
    stuck = [{"id": 70, "type": "mcv", "x": 25, "y": 9, "hp": 100}]
    e.update(_exp_view(t=395.0, units=stuck), Mode.BUILD)
    assert e.state == "moving" and e.site == (25, 9)                         # 95 s: deploy here instead
    assert e.orders(_exp_view(t=396.0, units=stuck)) == [{"cmd": "deploy", "units": [70]}]
    e.update(_exp_view(t=425.0, units=stuck), Mode.BUILD)
    assert e.state == "idle" and not e.escorting                             # 125 s: abandoned, army free


def test_escort_never_blocks_a_must_attack():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent._expansion.state, agent._expansion.site, agent._expansion.mcv_id = "moving", (20, 8), 70
    agent._expansion._since = 1040.0
    tanks = [{"id": 30 + i, "type": "1tnk", "x": 5, "y": 5, "hp": 100, "idle": True} for i in range(10)]
    mcv = [{"id": 70, "type": "mcv", "x": 10, "y": 8, "hp": 100}]
    behind = {"army_value": 30000, "visible_enemy_value": 20000, "kills_cost": 17950, "deaths_cost": 20700}
    agent.choose_action(_state(_t=1050.0, _r=150.0, buildings=BASE, units=tanks + mcv, military=behind), CTX)
    assert agent._mode is Mode.ALL_IN


def test_no_new_expansion_late_in_the_game():
    e = Expansion()
    e.update(_exp_view(t=900.0), Mode.BUILD)
    assert e.state == "idle"


def test_the_regrow_wait_expires():
    v = _view(buildings=BASE, _t=940.0, military={"army_value": 33150, "visible_enemy_value": 5000})
    assert brain.choose_mode(v, cheese_since=None, retreated_at=814.0, retreat_army=27250.0) is Mode.ALL_IN


def test_the_army_targets_a_known_enemy_expansion():
    blds = [{"id": 81, "type": "proc", "x": 42, "y": 13, "hp": 100}, {"id": 82, "type": "powr", "x": 36, "y": 38, "hp": 100}]
    o = executor._army_orders(_view(buildings=BASE, units=_ARMY, visible_enemy_buildings=blds), Mode.ALL_IN)
    assert o[0]["to"] == [42, 13]                                            # their expansion, not their main base


# -- MCVs always find room to unfold (match 4b5980f8) ----------------------------------

from agent.redalert.expand import has_room, roomy_spot  # noqa: E402


def test_an_mcv_needs_the_whole_3x3_block_clear():
    v = dataclasses.replace(_view(), ore_cells=((21, 20),), blocked_cells=frozenset({(30, 31)}))
    assert has_room((25, 25), v)
    assert not has_room((20, 20), v)                       # ore one cell over
    assert not has_room((30, 30), v)                       # a rock in the block
    yard = _view(buildings=[{"id": 10, "type": "fact", "x": 10, "y": 10}])
    assert not has_room((14, 11), yard)                    # would touch our construction yard
    assert roomy_spot((20, 20), v) not in (None, (20, 20))


def test_the_chosen_site_has_room_to_unfold():
    v = dataclasses.replace(_exp_view(), blocked_cells=frozenset())
    spot, _ = choose_site(v)
    assert has_room(spot, v)


def test_a_stuck_mcv_deploys_near_where_it_stopped():
    e = Expansion()
    e.state, e.site, e.mcv_id, e._since = "moving", (40, 8), 70, 300.0
    stopped = [{"id": 70, "type": "mcv", "x": 35, "y": 9, "hp": 100, "idle": True}]   # 5 cells short, stopped
    o = e.orders(_exp_view(t=320.0, units=stopped))
    assert e.site != (40, 8) and (e.site[0] - 35) ** 2 + (e.site[1] - 9) ** 2 <= 25
    assert o == [{"cmd": "deploy", "units": [70]}] or o[0]["cmd"] == "move"


def test_a_moving_mcv_is_not_mistaken_for_stuck():
    e = Expansion()
    e.state, e.site, e.mcv_id, e._since = "moving", (40, 8), 70, 300.0
    driving = [{"id": 70, "type": "mcv", "x": 35, "y": 9, "hp": 100, "idle": False}]
    assert e.orders(_exp_view(t=320.0, units=driving)) == [{"cmd": "move", "units": [70], "to": [40, 8]}]
    assert e.site == (40, 8)


# -- units keep moving and fight back when pressed ----------------------------------------

from agent.redalert.movement import Unstick  # noqa: E402


def test_a_unit_stuck_with_an_order_gets_stopped_to_repath():
    u = Unstick()
    tank = [{"id": 30, "type": "1tnk", "x": 20, "y": 20, "hp": 100, "idle": False}]
    assert u.orders(_view(_t=100.0, units=tank)) == []
    assert u.orders(_view(_t=105.0, units=tank)) == []
    assert u.orders(_view(_t=111.0, units=tank)) == [{"cmd": "stop", "units": [30]}]


def test_moving_fighting_unloading_and_idle_units_are_not_stuck():
    u = Unstick()
    moving = lambda x: [{"id": 30, "type": "1tnk", "x": x, "y": 20, "hp": 100, "idle": False}]  # noqa: E731
    for t, x in ((100.0, 20), (106.0, 22), (112.0, 24)):
        assert u.orders(_view(_t=t, units=moving(x))) == []
    fighting = [{"id": 31, "type": "1tnk", "x": 20, "y": 20, "hp": 100, "idle": False}]
    enemy = [{"id": 90, "type": "e1", "x": 24, "y": 20, "hp": 100}]
    for t in (100.0, 111.0, 122.0):
        assert u.orders(_view(_t=t, units=fighting, visible_enemies=enemy)) == []
    harv = [{"id": 32, "type": "harv", "x": 9, "y": 10, "hp": 100, "idle": False}]
    ref = [{"id": 13, "type": "proc", "x": 8, "y": 9}]
    for t in (100.0, 111.0, 122.0):
        assert u.orders(_view(_t=t, units=harv, buildings=ref)) == []
    idle = [{"id": 33, "type": "1tnk", "x": 20, "y": 20, "hp": 100, "idle": True}]
    for t in (100.0, 111.0, 122.0):
        assert u.orders(_view(_t=t, units=idle)) == []


def test_a_withdrawal_attack_moves_every_unit():
    units = [{"id": 30, "type": "1tnk", "x": 38, "y": 38, "hp": 100}, {"id": 31, "type": "1tnk", "x": 30, "y": 30, "hp": 100}]
    enemy = [{"id": 90, "type": "3tnk", "x": 40, "y": 39, "hp": 100}]
    o = executor._army_orders(_view(buildings=BASE, units=units, visible_enemies=enemy), Mode.RETREAT)
    assert o == [{"cmd": "attack_move", "units": [30, 31], "to": [11, 11]}]   # no plain moves under fire


# -- rule-based micro and Haiku squad leads -------------------------------------------

from agent.redalert import micro  # noqa: E402
from agent.redalert import squadlead  # noqa: E402


def test_focus_fire_on_the_weakest_enemy_in_reach():
    ours = [{"id": 30, "type": "1tnk", "x": 20, "y": 20, "hp": 100}, {"id": 31, "type": "1tnk", "x": 21, "y": 20, "hp": 100}]
    enemies = [{"id": 90, "type": "3tnk", "x": 24, "y": 20, "hp": 80}, {"id": 91, "type": "3tnk", "x": 23, "y": 21, "hp": 25},
               {"id": 92, "type": "harv", "x": 22, "y": 22, "hp": 5}]
    o, ids = micro.plan(_view(units=ours, visible_enemies=enemies), Mode.ALL_IN, frozenset(), (15, 15))
    assert o == [{"cmd": "attack", "units": [30, 31], "target": 91}] and ids == {30, 31}   # the 25% tank, not the harvester


def test_badly_damaged_vehicles_pull_back_from_attacks_not_defense():
    ours = [{"id": 30, "type": "1tnk", "x": 30, "y": 30, "hp": 20}, {"id": 31, "type": "1tnk", "x": 30, "y": 31, "hp": 90}]
    o, ids = micro.plan(_view(units=ours), Mode.ALL_IN, frozenset(), (15, 15))
    assert o == [{"cmd": "attack_move", "units": [30], "to": [15, 15]}] and ids == {30}
    o2, _ = micro.plan(_view(units=ours), Mode.DEFEND_BASE, frozenset(), (15, 15))
    assert o2 == []                                                          # defending: everyone fights


def test_micro_leaves_squads_and_retreats_alone():
    ours = [{"id": 30, "type": "1tnk", "x": 20, "y": 20, "hp": 100}]
    enemy = [{"id": 90, "type": "3tnk", "x": 23, "y": 20, "hp": 50}]
    assert micro.plan(_view(units=ours, visible_enemies=enemy), Mode.ALL_IN, frozenset({30}), (15, 15)) == ([], set())
    assert micro.plan(_view(units=ours, visible_enemies=enemy), Mode.RETREAT, frozenset(), (15, 15)) == ([], set())


def test_micro_units_are_taken_out_of_the_army_orders():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    tanks = [{"id": 30 + i, "type": "1tnk", "x": 20, "y": 20, "hp": 100} for i in range(4)]
    enemy = [{"id": 90, "type": "e1", "x": 22, "y": 21, "hp": 40}]
    dec = agent.choose_action(_state(_t=300.0, buildings=BASE, units=tanks, visible_enemies=enemy,
                                     military={"army_value": 4000}), CTX)
    focus = [o for o in dec.action["orders"] if o["cmd"] == "attack" and o.get("target") == 90]
    others = [o for o in dec.action["orders"] if o["cmd"] in ("attack_move", "move")]
    assert focus and not any(set(o["units"]) & set(focus[0]["units"]) for o in others)


class _FakeLead:
    def __init__(self, answer):
        self.answer = answer

    def complete_structured(self, messages, name, schema, *, timeout, effort):
        assert "squad" in messages[0]["content"] and json_ok(messages[1]["content"])
        return self.answer


def json_ok(text):
    import json
    return isinstance(json.loads(text), dict)


def _settle(lead, view, ids):
    lead.poll(view, ids, "m")
    lead._future.result()
    return lead.poll(view, ids, "m")


def test_squad_lead_directs_its_squad_and_is_validated():
    v = _view(_t=300.0, visible_enemies=[_ENEMY_HARV])
    lead = squadlead.SquadLead("raid", _FakeLead({"action": "attack", "target_id": 90, "cell": None, "reasoning": "kill it"}),
                               log=lambda *_: None, clock=lambda: 0.0, interval=0.0)
    d = _settle(lead, v, [100, 101])
    assert squadlead.squad_orders(v, [100, 101], d, (15, 15)) == [{"cmd": "attack", "units": [100, 101], "target": 90}]
    ghost = squadlead.parse_directive({"action": "attack", "target_id": 999, "cell": None, "reasoning": ""}, v, 300.0)
    assert squadlead.squad_orders(v, [100], ghost, (15, 15)) is None          # unseen target: rules play
    off_map = squadlead.parse_directive({"action": "move", "target_id": None, "cell": [500, 5], "reasoning": ""}, v, 300.0)
    assert off_map.cell is None and squadlead.squad_orders(v, [100], off_map, (15, 15)) is None


def test_squad_lead_attack_moves_and_retreats_home():
    v = _view()
    mv = squadlead.parse_directive({"action": "move", "target_id": None, "cell": [30, 30], "reasoning": ""}, v, 0.0)
    rt = squadlead.parse_directive({"action": "retreat", "target_id": None, "cell": None, "reasoning": ""}, v, 0.0)
    assert squadlead.squad_orders(v, [1], mv, (15, 15)) == [{"cmd": "attack_move", "units": [1], "to": [30, 30]}]
    assert squadlead.squad_orders(v, [1], rt, (15, 15)) == [{"cmd": "attack_move", "units": [1], "to": [15, 15]}]


def test_the_raid_follows_its_lead_and_falls_back_to_rules_on_failure():
    class Boom:
        def complete_structured(self, *a, **k):
            raise RuntimeError("down")

    def leads(answer_provider):
        return lambda log, on_directive=None: {"raid": squadlead.SquadLead("raid", answer_provider, log=log, interval=0.0)}

    good = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0,
                         squad_lead_factory=leads(_FakeLead({"action": "move", "target_id": None, "cell": [33, 33], "reasoning": ""})))
    good._raid.unit_ids = [100, 101, 102]
    v = _view(_t=300.0, buildings=BASE, units=_raid_army(), visible_enemies=[_ENEMY_HARV])
    good._raid_orders(v)
    good._leads["raid"]._future.result()
    assert good._raid_orders(v) == [{"cmd": "attack_move", "units": [100, 101, 102], "to": [33, 33]}]
    bad = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0, squad_lead_factory=leads(Boom()))
    bad._raid.unit_ids = [100, 101, 102]
    bad._raid_orders(v)
    bad._leads["raid"]._future.exception()
    assert bad._raid_orders(v)[0]["target"] == 90                            # the raid rules: hit the harvester


# -- pro-replay features (RAGL/ladder replays) ---------------------------------------------


def test_the_build_order_waits_instead_of_jumping_to_the_second_power_plant():
    # The yard and one power plant stand; barracks are not offered yet (power still placing).
    v = _view(buildings=[{"id": 10, "type": "fact"}, {"id": 11, "type": "powr"}], available_production=["powr", "proc"],
              economy={"power_provided": 100, "power_drained": 0})
    assert buildplan.next_build_orders(v) == []           # wait for barracks, don't queue powr #2


def test_barracks_multiply_after_the_service_depot():
    base = _ECON_BASE + [{"id": 15, "type": "powr"}]
    assert buildplan.barracks_target(_view(buildings=base)) == 1
    assert buildplan.barracks_target(_view(buildings=base + [{"id": 16, "type": "fix"}])) == 2
    assert buildplan.barracks_target(_view(_t=340.0, buildings=base + [{"id": 16, "type": "fix"}])) == 5


def test_refineries_keep_coming_over_time():
    eco = {"cash": 2000, "power_provided": 400, "power_drained": 100}
    built = _ECON_BASE + [{"id": 15, "type": "powr"}, {"id": 16, "type": "fix"}, {"id": 17, "type": "proc"},
                          {"id": 18, "type": "proc"}] + [{"id": 20 + i, "type": "tent"} for i in range(4)]
    v = dataclasses.replace(_view(_t=400.0, buildings=built, economy=eco), ore_cells=((12, 12),))
    assert buildplan.refinery_target(v) == 5
    assert buildplan.next_build_orders(v) == [{"cmd": "build", "item": "proc"}]   # 3 refineries, 5 wanted


def test_the_army_is_infantry_heavy_with_tank_support():
    v = _view(buildings=_ECON_BASE + [{"id": 16, "type": "tent"}], units=_harvs(3), economy={"cash": 2000})
    items = [o.get("item") for o in executor._production_orders(v, Mode.BUILD)]
    assert "e1" in items and "1tnk" in items
    assert [o for o in executor._production_orders(v, Mode.BUILD) if o.get("item") == "e1"][0]["count"] == 1


def test_rally_points_point_production_at_the_staging_point():
    blds = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 11, "type": "tent", "x": 8, "y": 4, "rally": [10, 10]},
            {"id": 12, "type": "weap", "x": 4, "y": 8, "rally": None}, {"id": 13, "type": "proc", "x": 9, "y": 9}]
    v = _view(buildings=blds)
    stage = list(scout.choke_cell(v))                       # the refinery sits on (10, 10): moved off it
    assert stage != [10, 10]
    assert executor._rally_orders(v) == [{"cmd": "rally", "building": b, "to": stage} for b in (11, 12)]


def test_harvesters_go_to_an_expansion_field_once_it_has_a_refinery():
    # One owner of harvester orders: HarvestManager staffs the expansion's
    # field by refinery share (the expansion no longer sends its own trucks).
    harvs = [{"id": 20 + i, "type": "harv", "x": 8 + i, "y": 8, "hp": 100} for i in range(4)]
    yard = _EXP_BASE + [{"id": 80, "type": "fact", "x": 20, "y": 6}, {"id": 81, "type": "proc", "x": 24, "y": 8}]
    v = dataclasses.replace(_view(_t=500.0, buildings=yard, units=harvs), ore_cells=_EXP_ORE)
    o = HarvestManager().orders(v)
    sent = [x for x in o if x["cmd"] == "harvest" and x.get("to") and (x["to"][0] - 24) ** 2 + (x["to"][1] - 6) ** 2 <= 9]
    assert sent, o
    assert not hasattr(Expansion(), "_harvest_orders")


def test_waiting_units_hold_and_attackers_engage():
    units = [{"id": 30, "type": "1tnk", "x": 15, "y": 15, "hp": 100, "stance": "AttackAnything"},
             {"id": 31, "type": "e1", "x": 15, "y": 16, "hp": 100, "stance": "Defend"},
             {"id": 32, "type": "e1", "x": 5, "y": 5, "hp": 100, "stance": "AttackAnything"}]
    hold = executor.stance_orders(_view(units=units), Mode.BUILD, holders=set())
    assert hold == [{"cmd": "set_stance", "units": [30, 32], "stance": "defend"}]
    push = executor.stance_orders(_view(units=units), Mode.ALL_IN, holders={32})
    assert {"cmd": "set_stance", "units": [31], "stance": "attack_anything"} in push
    assert {"cmd": "set_stance", "units": [32], "stance": "defend"} in push   # 32 is a guard: it holds


# -- lessons from match 2035c6b3 -------------------------------------------------------


def test_a_finished_unplaced_power_plant_is_not_built_twice():
    v = _view(buildings=[{"id": 10, "type": "fact"}], available_production=["powr", "tent"],
              economy={"power_provided": 0, "power_drained": 0},
              production=[{"item": "powr", "queue": "Building", "ready": True}])
    assert {"cmd": "build", "item": "powr"} not in buildplan.next_build_orders(v)


def test_the_hunter_checks_the_spawn_first_then_sweeps_and_starts_over():
    h = scout.Hunter()
    v = dataclasses.replace(_view(units=[{"id": 30, "type": "1tnk", "x": 30, "y": 30, "hp": 100}]), ore_cells=_THEIR_ORE)
    assert h.target(v) == (40, 40)                                   # the enemy spawn first
    at_spawn = dataclasses.replace(_view(units=[{"id": 30, "type": "1tnk", "x": 40, "y": 40, "hp": 100}]),
                                   ore_cells=_THEIR_ORE)
    nxt = h.target(at_spawn)
    assert nxt != (40, 40)                                           # checked: on to the next point
    for _ in range(20):                                              # visit everything
        pt = h.target(at_spawn)
        h._checked.add(pt)
    assert h.target(at_spawn) is not None                            # and it starts over, never None


def test_with_nothing_known_the_army_hunts_instead_of_idling_on_ruins():
    units = [{"id": 30 + i, "type": "1tnk", "x": 40, "y": 40, "hp": 100} for i in range(4)]
    o = executor._army_orders(_view(buildings=BASE, units=units), Mode.ALL_IN, hunt=(10, 40))
    assert o[0]["to"] == [10, 40]


def test_garrison_helps_defend_the_main_base_but_holds_an_attacked_expansion():
    e = _deployed_expansion()
    tanks = [{"id": 300 + i, "type": "1tnk", "x": 15, "y": 15, "hp": 100} for i in range(12)]
    yard = _EXP_BASE + [{"id": 80, "type": "fact", "x": 20, "y": 6}]
    at_home = [{"id": 90, "type": "3tnk", "x": 5, "y": 6, "hp": 100}, {"id": 91, "type": "e1", "x": 6, "y": 6, "hp": 100}]
    at_exp = [{"id": 90, "type": "3tnk", "x": 21, "y": 8, "hp": 100}, {"id": 91, "type": "e1", "x": 22, "y": 8, "hp": 100}]
    g = Garrison()
    g.update(_exp_view(units=tanks, buildings=yard), Mode.BUILD, e, taken=set())
    assert len(g.unit_ids) == 4
    g.update(_exp_view(units=tanks, buildings=yard, enemies=at_exp), Mode.DEFEND_BASE, e, taken=set())
    assert len(g.unit_ids) == 4                                      # the expansion is the one attacked: hold it
    g.update(_exp_view(units=tanks, buildings=yard, enemies=at_home), Mode.DEFEND_BASE, e, taken=set())
    assert g.unit_ids == []                                          # the main base is attacked: go help


# -- turrets: an even fan facing the enemy ------------------------------------------------


def _fan_angle(spot, base=(10, 10), enemy=(40, 40)):
    import math
    a = math.degrees(math.atan2(spot[1] - base[1], spot[0] - base[0]) - math.atan2(enemy[1] - base[1], enemy[0] - base[0]))
    return (a + 180) % 360 - 180


_FAN_BASE = [{"id": 10, "type": "fact", "x": 10, "y": 10}, {"id": 11, "type": "powr", "x": 6, "y": 7},
             {"id": 12, "type": "tent", "x": 14, "y": 8}, {"id": 13, "type": "proc", "x": 7, "y": 14},
             {"id": 14, "type": "weap", "x": 14, "y": 13}, {"id": 15, "type": "powr", "x": 6, "y": 11}]


def test_turrets_fill_an_even_fan_facing_the_enemy():
    blds, spots = list(_FAN_BASE), []
    for i in range(5):                       # as many as the fan has free slots (that's all we'd build)
        v = _view(base_center=[11, 11], buildings=blds)
        if buildplan.perimeter_spot(v, (11, 11), (40, 40), "ftur") is None:
            break
        spot = buildplan.placement_cell(v, "ftur")
        spots.append(spot)
        blds = blds + [{"id": 20 + i, "type": "ftur", "x": spot[0], "y": spot[1]}]
    assert len(spots) >= 3
    angles = [_fan_angle(s, base=(11, 11)) for s in spots]
    assert all(abs(a) <= 75 for a in angles), angles             # never to the side or behind
    assert abs(angles[0]) <= 30                                   # the first one straight at the enemy
    assert min(angles) < -15 and max(angles) > 15                 # spread to both sides
    for i, a in enumerate(spots):                                 # and spaced out, not piled up
        assert all((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 >= 9 for b in spots[i + 1:]), spots


def test_turrets_grow_from_the_base_not_from_each_other():
    blds, spots = list(_FAN_BASE), []
    for i in range(5):
        spot = buildplan.placement_cell(_view(base_center=[11, 11], buildings=blds), "pbox")
        spots.append(spot)
        blds = blds + [{"id": 20 + i, "type": "pbox", "x": spot[0], "y": spot[1]}]
    assert all((x - 11) ** 2 + (y - 11) ** 2 <= 14 ** 2 for x, y in spots)   # no chain marching outward


def test_a_raid_from_behind_does_not_pull_turrets_behind_the_base():
    near_back = [{"id": 50, "type": "e1", "x": 4, "y": 6, "hp": 100}, {"id": 51, "type": "e1", "x": 5, "y": 4, "hp": 100}]
    v = _view(visible_enemies=near_back, base_center=[10, 10], buildings=[{"id": 10, "type": "fact", "x": 10, "y": 10}])
    assert abs(_fan_angle(buildplan.placement_cell(v, "pbox"))) <= 75


def test_no_free_slot_no_turret():
    blds = [{"id": 10, "type": "fact", "x": 10, "y": 10}, {"id": 12, "type": "weap", "x": 6, "y": 10}]
    for i in range(5):                                             # fill the fan
        v = _view(base_center=[10, 10], buildings=blds)
        spot = buildplan.perimeter_spot(v, (10, 10), (40, 40), "pbox")
        if spot is None:
            break
        blds = blds + [{"id": 20 + i, "type": "pbox", "x": spot[0], "y": spot[1]}]
    v = _view(base_center=[10, 10], buildings=blds, available_production=["pbox"])
    assert buildplan.perimeter_spot(v, (10, 10), (40, 40), "pbox") is None
    assert buildplan.defense_orders(v, under_attack=True) == []   # full fan (and over the cap): build nothing


def test_an_expansion_turret_is_asked_for_once_per_window():
    e = _deployed_expansion()
    yard = _EXP_BASE + [{"id": 80, "type": "fact", "x": 20, "y": 6}]
    assert e.defense_site(_exp_view(t=340.0, buildings=yard)) == (20, 6)
    e.update(_exp_view(t=341.0, buildings=yard, production=[{"item": "ftur", "queue": "Defense", "ready": False}]), Mode.BUILD)
    assert e.defense_site(_exp_view(t=350.0, buildings=yard)) is None        # asked: wait
    assert e.defense_site(_exp_view(t=440.0, buildings=yard)) == (20, 6)     # 90 s later: try again


# -- harvesters follow the ore (harvest.HarvestManager) ------------------------------------

from agent.redalert.harvest import HarvestManager  # noqa: E402

# Home refinery by ore at (8, 8); expansion refinery by ore at (30, 8).
_FIELDS_ORE = ((8, 8), (9, 8), (30, 8), (31, 8))
_FIELDS_BLDS = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 13, "type": "proc", "x": 7, "y": 10},
                {"id": 80, "type": "fact", "x": 27, "y": 4}, {"id": 81, "type": "proc", "x": 29, "y": 10}]


def _hv(t, harvs):
    return dataclasses.replace(_view(_t=t, buildings=_FIELDS_BLDS, units=harvs), ore_cells=_FIELDS_ORE)


def test_fields_come_from_our_refineries():
    assert sorted(HarvestManager().fields(_hv(100.0, []))) == [(8, 8), (30, 8)]


def test_a_mined_out_home_field_sends_harvesters_to_the_expansion():
    m = HarvestManager()
    idle_home = [{"id": 20, "type": "harv", "x": 9, "y": 9, "hp": 100, "idle": True}]
    m.orders(_hv(100.0, idle_home))
    o = m.orders(_hv(106.0, idle_home))                       # idle 6 s by the home field: it's empty
    assert o == [{"cmd": "harvest", "units": [20], "to": [30, 8]}]


def test_working_harvesters_leave_a_depleted_field_a_couple_at_a_time():
    m = HarvestManager()
    idle_home = [{"id": 20, "type": "harv", "x": 9, "y": 9, "hp": 100, "idle": True}]
    working = [{"id": 21 + i, "type": "harv", "x": 8, "y": 9, "hp": 100, "idle": False} for i in range(4)]
    m.orders(_hv(100.0, idle_home + working))
    o = m.orders(_hv(106.0, idle_home + working))
    moved = [u for order in o for u in order["units"] if u != 20]
    assert len(moved) == 2 and all(order["to"] == [30, 8] for order in o)


def test_a_depleted_field_is_rechecked_later():
    m = HarvestManager()
    idle_home = [{"id": 20, "type": "harv", "x": 9, "y": 9, "hp": 100, "idle": True}]
    m.orders(_hv(100.0, idle_home)); m.orders(_hv(106.0, idle_home))
    fresh = [{"id": 30, "type": "harv", "x": 30, "y": 9, "hp": 100, "idle": True}]
    o = m.orders(_hv(240.0, fresh))                           # 2+ min later: home may have regrown
    assert o and o[0]["to"] in ([8, 8], [30, 8])
    assert (8, 8) in [f for f in m.fields(_hv(240.0, [])) if 240.0 - m._depleted.get(f, -1e9) >= 120.0]


def test_idle_harvesters_spread_by_load():
    m = HarvestManager()
    working_home = [{"id": 21 + i, "type": "harv", "x": 8, "y": 9, "hp": 100, "idle": False} for i in range(3)]
    new = [{"id": 40, "type": "harv", "x": 10, "y": 10, "hp": 100, "idle": True}]
    o = m.orders(_hv(100.0, working_home + new))
    # home has 3, the expansion none: the idle one goes there, and one more evens it out at 2 and 2
    assert o == [{"cmd": "harvest", "units": [40, 21], "to": [30, 8]}]


def test_no_known_fields_falls_back_to_a_bare_harvest():
    assert HarvestManager().orders(_view(units=[{"id": 20, "type": "harv", "x": 9, "y": 9, "hp": 100, "idle": True}])) is None


# -- end game: radar dome, siege units, turtle response ---------------------------------------

_LATE_BASE = _ECON_BASE + [{"id": 15, "type": "powr"}, {"id": 16, "type": "fix"}] + \
    [{"id": 20 + i, "type": "tent"} for i in range(4)] + [{"id": 30 + i, "type": "proc"} for i in range(7)]
_ENEMY_TOWERS = [{"id": 90 + i, "type": "ftur", "x": 36 + i * 2, "y": 36, "hp": 100} for i in range(3)]


def test_a_radar_dome_around_seven_minutes_when_money_allows():
    eco = {"cash": 1500, "power_provided": 600, "power_drained": 300}
    v = _view(_t=430.0, buildings=_LATE_BASE, available_production=["dome", "powr", "proc", "tent", "weap"], economy=eco)
    assert {"cmd": "build", "item": "dome"} in buildplan.next_build_orders(v)
    early = _view(_t=300.0, buildings=_LATE_BASE, available_production=["dome"], economy=eco)
    assert {"cmd": "build", "item": "dome"} not in buildplan.next_build_orders(early)


def test_siege_units_only_once_enemy_defenses_are_known():
    blds = _ECON_BASE + [{"id": 17, "type": "dome"}]
    avail = ["e1", "e3", "3tnk", "harv", "v2rl"]
    quiet = _view(buildings=blds, units=_harvs(3), economy={"cash": 3000}, available_production=avail)
    assert "v2rl" not in [o.get("item") for o in executor._production_orders(quiet, Mode.BUILD)]
    towers = _view(buildings=blds, units=_harvs(3), economy={"cash": 3000}, available_production=avail,
                   visible_enemy_buildings=_ENEMY_TOWERS[:1])
    assert "v2rl" in [o.get("item") for o in executor._production_orders(towers, Mode.BUILD)]
    capped = _view(buildings=blds, economy={"cash": 3000}, available_production=avail, visible_enemy_buildings=_ENEMY_TOWERS[:1],
                   units=_harvs(3) + [{"id": 60 + i, "type": "v2rl", "x": 5, "y": 5, "hp": 100} for i in range(4)])
    assert "v2rl" not in [o.get("item") for o in executor._production_orders(capped, Mode.BUILD)]


def test_siege_units_shell_defenses_first():
    ours = [{"id": 60, "type": "v2rl", "x": 30, "y": 36, "hp": 100}, {"id": 61, "type": "e1", "x": 32, "y": 36, "hp": 100}]
    enemies = [{"id": 70, "type": "e1", "x": 33, "y": 37, "hp": 20}]
    o, ids = micro.plan(_view(units=ours, visible_enemies=enemies, visible_enemy_buildings=_ENEMY_TOWERS[:1]),
                        Mode.ALL_IN, frozenset(), (15, 15))
    assert {"cmd": "attack", "units": [60], "target": 90} in o                 # the V2 hits the tower
    assert {"cmd": "attack", "units": [61], "target": 70} in o                 # the rifleman the weak unit


def test_against_a_turtle_lean_on_tanks_over_infantry():
    blds = _ECON_BASE + [{"id": 16, "type": "tent"}]
    open_map = _view(buildings=blds, units=_harvs(3), economy={"cash": 2000})
    turtle = _view(buildings=blds, units=_harvs(3), economy={"cash": 2000}, visible_enemy_buildings=_ENEMY_TOWERS)
    rifles = lambda v: [o["count"] for o in executor._production_orders(v, Mode.BUILD) if o.get("item") == "e1"]  # noqa: E731
    assert rifles(open_map) == [1] and rifles(turtle) == [1]
    q = [{"item": "1tnk", "queue": "Vehicle", "ready": False}]
    tanks = lambda v: [o for o in executor._production_orders(v, Mode.BUILD) if o.get("item") == "1tnk"]  # noqa: E731
    assert not tanks(dataclasses.replace(open_map, production=q))
    assert tanks(dataclasses.replace(turtle, production=q))                     # a deeper tank queue


# -- expansions get refineries and trucks (match 4bc1ee71) ------------------------------------


def test_a_finished_refinery_goes_to_the_expansion_even_right_after_asking():
    e = _deployed_expansion()
    yard = _EXP_BASE + [{"id": 80, "type": "fact", "x": 20, "y": 6}]
    e.done[0]["asked_at"] = 330.0                                         # asked 10 s ago: throttled
    v = _exp_view(t=340.0, buildings=yard, production=[{"item": "proc", "queue": "Building", "ready": True}])
    assert e.refinery_ore(v) is None and e.needs_refinery(v) == (24, 6)
    (order,) = [o for o in executor.plan(v, Mode.BUILD, expansion=e) if o["cmd"] == "place"]
    x, y = order["at"]
    assert (x - 24) ** 2 + (y - 6) ** 2 < (x - 4) ** 2 + (y - 4) ** 2     # at the expansion, not home


def test_fields_stay_staffed_by_refinery_share_while_home_has_ore():
    m = HarvestManager()
    home = [{"id": 21 + i, "type": "harv", "x": 8, "y": 9, "hp": 100, "idle": False} for i in range(5)]
    o = m.orders(_hv(100.0, home))
    assert o == [{"cmd": "harvest", "units": [21], "to": [30, 8]}]       # one at a time, to the empty field
    assert m.orders(_hv(101.0, home)) == [{"cmd": "harvest", "units": [22], "to": [30, 8]}]
    assert m.last_counts[(30, 8)] >= 1


def test_a_moved_harvester_settles_before_moving_again():
    m = HarvestManager()
    m._moved_at[21] = 100.0
    home = [{"id": 21, "type": "harv", "x": 8, "y": 9, "hp": 100, "idle": False},
            {"id": 22, "type": "harv", "x": 8, "y": 9, "hp": 100, "idle": False},
            {"id": 23, "type": "harv", "x": 8, "y": 9, "hp": 100, "idle": False}]
    o = m.orders(_hv(110.0, home))
    assert all(21 not in order["units"] for order in o)                  # moved 10 s ago: stays


# -- finishing a beaten enemy (match 4bc1ee71) --------------------------------------------------


def test_no_protecting_a_lead_against_a_beaten_enemy():
    beaten = _view(buildings=BASE, _r=100.0,
                   military={"army_value": 27000, "visible_enemy_value": 1200, "kills_cost": 44000, "deaths_cost": 19000})
    assert brain.choose_mode(beaten, cheese_since=None) is Mode.ALL_IN
    even = _view(buildings=BASE, _r=100.0,
                 military={"army_value": 15000, "visible_enemy_value": 12000, "kills_cost": 44000, "deaths_cost": 19000})
    assert brain.choose_mode(even, cheese_since=None) is Mode.PROTECT_LEAD


def test_an_unseen_enemy_is_not_a_beaten_one():
    fog = _view(buildings=BASE, military={"army_value": 27000, "visible_enemy_value": 0})
    assert not brain._overwhelming(fog)


def test_no_extra_turrets_when_broke():
    yard = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 11, "type": "tent", "x": 8, "y": 4},
            {"id": 12, "type": "weap", "x": 4, "y": 8}, {"id": 13, "type": "pbox", "x": 9, "y": 9}]
    broke = _view(buildings=yard, available_production=["pbox"], economy={"cash": 100, "ore": 0})
    rich = _view(buildings=yard, available_production=["pbox"], economy={"cash": 2000})
    assert buildplan.defense_orders(broke, under_attack=True) == []
    assert buildplan.defense_orders(rich, under_attack=True) == [{"cmd": "build", "item": "pbox"}]


def test_an_overwhelming_army_splits_between_two_enemy_clusters():
    units = [{"id": 30 + i, "type": "3tnk", "x": 30, "y": 25 + i, "hp": 100} for i in range(10)]
    blds = [{"id": 81, "type": "proc", "x": 40, "y": 40, "hp": 100}, {"id": 82, "type": "fact", "x": 40, "y": 10, "hp": 100}]
    v = _view(buildings=BASE, units=units, visible_enemy_buildings=blds,
              military={"army_value": 12000, "visible_enemy_value": 1000})
    o = executor._army_orders(v, Mode.ALL_IN)
    assert len(o) == 2 and {tuple(x["to"]) for x in o} == {(40, 40), (40, 10)}
    assert sorted(len(x["units"]) for x in o) == [5, 5]
    even = dataclasses.replace(v, enemy_army_value=10000.0)
    together = executor._army_orders(even, Mode.ALL_IN)                       # not overwhelming: one target...
    centre = together[len(together) // 2]["to"]
    assert all((x["to"][0] - centre[0]) ** 2 + (x["to"][1] - centre[1]) ** 2 <= 6 ** 2
               for x in together)                                             # ...in a semi-concave round it


# -- detecting a mined-out home field and switching to the expansion -----------------------------


def test_harvesters_around_a_field_but_none_mining_marks_it_depleted():
    m = HarvestManager()
    searching = [{"id": 20 + i, "type": "harv", "x": 12, "y": 14 + i, "hp": 100, "idle": False} for i in range(2)]
    for t in (100.0, 120.0, 140.0):
        m.orders(_hv(t, searching))                            # near home (8, 8), but not on its ore
    assert (8, 8) not in m.depleted()
    m.orders(_hv(150.0, searching))                            # 50 s with nobody mining
    assert (8, 8) in m.depleted() and m.home_depleted(_hv(150.0, searching))


def test_harvesters_mining_keep_a_field_live():
    m = HarvestManager()
    mining = [{"id": 20, "type": "harv", "x": 9, "y": 9, "hp": 100, "idle": False}]
    for t in (100.0, 130.0, 160.0):
        m.orders(_hv(t, mining))
    assert m.depleted() == set()


def test_an_unstaffed_new_field_is_not_mistaken_for_depleted():
    m = HarvestManager()
    home = [{"id": 20, "type": "harv", "x": 9, "y": 9, "hp": 100, "idle": False}]
    for t in (100.0, 160.0, 220.0):
        m.orders(_hv(t, home))                                 # nobody near the expansion field yet
    assert (30, 8) not in m.depleted()


def test_new_refineries_go_to_live_ore_not_the_mined_out_home():
    yard = _EXP_BASE + [{"id": 80, "type": "fact", "x": 20, "y": 6}]
    v = dataclasses.replace(_exp_view(buildings=yard, production=[{"item": "proc", "queue": "Building", "ready": True}]),
                            ore_cells=((8, 8), (9, 8), (24, 6), (25, 6)))
    home_field = frozenset({(8, 8)})
    (order,) = buildplan.placement_orders(v, None, None, None, home_field)
    x, y = order["at"]
    assert (x - 24) ** 2 + (y - 6) ** 2 < (x - 8) ** 2 + (y - 8) ** 2  # beside the expansion's ore


def test_no_refinery_is_scheduled_when_every_field_is_mined_out():
    eco = {"cash": 3000, "power_provided": 600, "power_drained": 300}
    v = dataclasses.replace(_view(_t=700.0, buildings=_LATE_BASE[:9], economy=eco,
                                  available_production=["proc", "powr", "tent", "dome"]), ore_cells=((8, 8),))
    assert {"cmd": "build", "item": "proc"} not in buildplan.next_build_orders(v, depleted=frozenset({(8, 8)}))


def test_a_mined_out_home_hurries_the_next_expansion():
    e = _deployed_expansion(t=335.0)
    e.update(_exp_view(t=400.0), Mode.BUILD, ore_running_out=False)
    assert e.state == "idle"                                  # only 65 s after the first
    e.update(_exp_view(t=400.0), Mode.BUILD, ore_running_out=True)
    assert e.state == "scouting"                              # home is dry: go now


# -- the base moves to the expansion once home is mined out; far ore when ahead ------------------


def _shifted():
    e = _deployed_expansion()
    yard = _EXP_BASE + [{"id": 80, "type": "fact", "x": 20, "y": 6}]
    return e, yard


def test_the_active_base_is_the_expansion_once_home_is_mined_out():
    e, yard = _shifted()
    v = _exp_view(buildings=yard)
    assert e.active_base(v, home_depleted=False, depleted=set()) is None          # home still has ore
    assert e.active_base(v, home_depleted=True, depleted={(8, 8)}) == (20, 6)
    assert e.active_base(v, home_depleted=True, depleted={(8, 8), (24, 6)}) is None   # the expansion is dry too


def test_new_buildings_grow_at_the_active_base():
    e, yard = _shifted()
    v = _exp_view(buildings=yard)
    for item in ("powr", "tent", "pbox"):
        x, y = buildplan.placement_cell(v, item, home=(20, 6))
        assert (x - 20) ** 2 + (y - 6) ** 2 <= 12 ** 2, (item, x, y)                # at the expansion, not home


def test_the_active_base_gets_a_full_turret_fan_and_home_gets_none():
    e, yard = _shifted()
    v = _exp_view(buildings=yard + [{"id": 90, "type": "pbox", "x": 24, "y": 9}], available=("pbox",))
    assert buildplan.defense_orders(v, under_attack=False, home=(20, 6)) == [{"cmd": "build", "item": "pbox"}]
    three = yard + [{"id": 90 + i, "type": "pbox", "x": 24, "y": 3 + 4 * i} for i in range(3)]
    assert buildplan.defense_orders(_exp_view(buildings=three, available=("pbox",)), under_attack=False, home=(20, 6)) == []


def test_far_ore_still_gets_a_refinery_when_ahead_and_the_base_crawls_toward_it():
    eco = {"cash": 3000, "power_provided": 600, "power_drained": 300}
    blds = _LATE_BASE[:9] + [{"id": 40, "type": "tent"}, {"id": 41, "type": "dome"},
                             {"id": 42, "type": "proc"}, {"id": 43, "type": "proc"}]   # opening done
    far = ((40, 4), (41, 4))                                                    # beyond refinery reach of the yard
    ahead = dataclasses.replace(_view(_t=700.0, buildings=blds, economy=eco,
                                      available_production=["proc", "powr", "tent", "dome"]), ore_cells=far)
    assert {"cmd": "build", "item": "proc"} in buildplan.next_build_orders(ahead)
    poor = dataclasses.replace(ahead, cash=500.0)
    assert {"cmd": "build", "item": "proc"} not in buildplan.next_build_orders(poor)   # not ahead: wait
    with_cells = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 13, "type": "proc", "x": 8, "y": 4}]
    crawl = dataclasses.replace(_view(buildings=with_cells), ore_cells=far)
    x, y = buildplan.placement_cell(crawl, "proc")
    assert x > 10                                                               # grows out toward (40, 4)


# -- buildings never go on ore: expansion refineries land at the expansion (match 768b2cbd) ----

def _ore_patch(x0, y0, w, h):
    return tuple((x, y) for x in range(x0, x0 + w) for y in range(y0, y0 + h))


def test_no_building_is_placed_on_ore_or_blocked_terrain():
    v = dataclasses.replace(_view(buildings=[{"id": 10, "type": "fact", "x": 10, "y": 10}]),
                            ore_cells=_ore_patch(14, 10, 4, 4), blocked_cells=frozenset({(9, 15)}))
    assert buildplan._clashes("proc", (14, 10), v)                       # on the ore
    assert not set(buildplan._cells("proc", (14, 15))) & buildplan._no_build(v)   # beside it
    assert buildplan._clashes("powr", (8, 14), v)                        # on a rock


def test_an_expansion_refinery_lands_beside_its_ore_at_the_expansion():
    home = [{"id": 10, "type": "fact", "x": 26, "y": 32}, {"id": 13, "type": "proc", "x": 22, "y": 28}]
    expansion_yard = [{"id": 80, "type": "fact", "x": 13, "y": 76}]
    ore = _ore_patch(17, 76, 5, 5) + _ore_patch(20, 26, 4, 4)            # expansion field + home field
    v = dataclasses.replace(_view(buildings=home + expansion_yard, base_center=[27, 33],
                                  production=[{"item": "proc", "queue": "Building", "ready": True}]),
                            ore_cells=ore, bounds=(0, 0, 96, 96))
    (order,) = buildplan.placement_orders(v, None, (17, 78), None)
    x, y = order["at"]
    footprint = {(x + i, y + j) for i in range(3) for j in range(3)}
    assert not footprint & set(ore)                                      # not on the ore
    assert (x - 14) ** 2 + (y - 77) ** 2 <= 12 ** 2                      # at the expansion, not home (26, 32)
    assert min((x + 1 - ox) ** 2 + (y + 1 - oy) ** 2 for ox, oy in ore[:25]) <= 5 ** 2   # and right by its ore


# -- early game: no flanking, watch their base for an early push ---------------------------------


def test_no_flanking_before_five_minutes():
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    center_threat = [{"id": 50 + i, "type": "1tnk", "x": 22, "y": 22, "hp": 100} for i in range(3)]
    their_base = [{"id": 80, "type": "proc", "x": 40, "y": 40, "hp": 100}]
    early = _view(_t=200.0, visible_enemies=center_threat, visible_enemy_buildings=their_base)
    late = _view(_t=400.0, visible_enemies=center_threat, visible_enemy_buildings=their_base)
    assert agent._pick_route(early, None) is None
    assert agent._pick_route(late, None) is not None                       # guarded middle: flank later


def test_an_early_push_is_spotted_and_we_hold_at_home():
    seen = [{"id": 50 + i, "type": "e1", "x": 36, "y": 36, "hp": 100} for i in range(14)]
    v = read(_live_raw(_t=150.0, buildings=BASE, visible_enemies=seen,
                       units=[{"id": 30 + i, "type": "e1", "x": 15, "y": 15, "hp": 100} for i in range(10)]), LIVE_CONFIG)
    assert scout.early_push_warning(v)                                      # 1400 seen vs 1000 ours
    assert brain.choose_mode(v, cheese_since=None, prev_mode=Mode.BUILD) is Mode.BUILD
    assert scout.choke_cell(v) == (6, 6)                                    # mass beside the base
    late = dataclasses.replace(v, game_seconds=400.0)
    assert not scout.early_push_warning(late)                               # only early


def test_an_early_push_unlocks_turrets_and_unit_training():
    seen = [{"id": 50 + i, "type": "e1", "x": 36, "y": 36, "hp": 100} for i in range(14)]
    yard = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 11, "type": "tent", "x": 8, "y": 4},
            {"id": 12, "type": "weap", "x": 4, "y": 8}, {"id": 13, "type": "pbox", "x": 9, "y": 9}]
    v = read(_live_raw(_t=200.0, buildings=yard, visible_enemies=seen, available_production=["pbox", "e1"],
                       economy={"cash": 2000}), LIVE_CONFIG)
    o = executor.plan(v, Mode.BUILD)
    assert {"cmd": "build", "item": "pbox"} in o                            # a second turret, as if attacked


# -- cliffs: harvesting by walking distance, not straight lines (match 0561274e) ----------------

from agent.redalert import config as C, terrain  # noqa: E402
from agent.redalert.observe import _blocked_cells, _legend_blocked_chars  # noqa: E402

_CLIFF = frozenset((x, 13) for x in range(48))          # a cliff wall across the whole map at y = 13


def test_the_map_legend_marks_cliffs_impassable():
    legend = {".": "clear ground", "#": "cliff", "~": "water", "$": "ore", "=": "bridge over water", "M": "ore mine"}
    chars = _legend_blocked_chars(legend)
    assert {"#", "~", "M"} <= chars and "." not in chars and "$" not in chars and "=" not in chars
    layout = {"origin": [0, 0], "rows": ["..#..", "..#.."], "legend": legend}
    assert _blocked_cells("cliffmap-test", layout) == frozenset({(2, 0), (2, 1)})


def test_walking_goes_around_a_wall():
    v = dataclasses.replace(_view(), blocked_cells=frozenset((x, 10) for x in range(0, 30)))
    walk = terrain.walk_distances(v, [(5, 5)], 60)
    assert walk[(5, 15)] > 20                              # round the end of the wall at x = 30, not 10 steps


def test_a_refinery_harvest_field_is_ore_its_trucks_can_walk_to():
    ore = _ore_patch(20, 16, 3, 3) + _ore_patch(30, 4, 3, 3)            # south (across the cliff) and east
    blds = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 13, "type": "proc", "x": 20, "y": 9}]
    v = dataclasses.replace(_view(buildings=blds), ore_cells=ore, blocked_cells=_CLIFF)
    fields = HarvestManager().fields(v)
    assert all(f[1] < 13 for f in fields), fields           # the east ore, not the ore across the cliff


def test_a_refinery_goes_on_the_ores_side_of_a_cliff():
    ore = _ore_patch(20, 16, 3, 3)                          # south of the cliff
    blds = [{"id": 10, "type": "fact", "x": 19, "y": 9},    # north of the cliff, straight-line close
            {"id": 11, "type": "fact", "x": 30, "y": 16}]   # south of the cliff, a little farther
    v = dataclasses.replace(_view(buildings=blds, base_center=[20, 10]), ore_cells=ore, blocked_cells=_CLIFF)
    x, y = buildplan._beside_ore(v, (21, 17), "proc")
    assert y > 13, (x, y)                                   # on the ore's side
    walk = terrain.walk_distances(v, ore, 8)
    assert terrain.steps_from_rect(walk, (x, y, x + 2, y + 2)) <= C.REFINERY_ORE_WALK_STEPS


def test_an_expansion_deploys_where_it_can_walk_to_its_ore():
    ore = _ore_patch(20, 16, 3, 3)
    v = dataclasses.replace(_view(base_center=[20, 4]), ore_cells=ore, blocked_cells=_CLIFF)
    from agent.redalert.expand import walkable_site
    spot = walkable_site(v, (21, 17), (20, 4))
    assert spot is not None and spot[1] > 13                # the ore's side, though our base is north
    assert terrain.walk_distances(v, ore, 8).get(spot, 99) <= C.EXPAND_SITE_WALK_STEPS


def test_a_yard_that_unfolds_off_the_planned_spot_is_still_our_expansion():
    e = Expansion()
    e.state, e.site, e.mcv_id, e._since = "moving", (20, 8), 70, 300.0
    e._yards_before = {10}                                   # the main yard existed already
    off_spot = _EXP_BASE + [{"id": 81, "type": "fact", "x": 22, "y": 10}]   # 3 cells off the plan
    e.update(_exp_view(t=320.0, buildings=off_spot), Mode.BUILD)
    assert e.state == "idle" and e.sites == [(23, 11)] and e._failures == 0


# -- an expansion's refinery goes AT the expansion, and the engine moving it is noticed (0561274e) -

_HOME_NEAR_ORE = [{"id": 10, "type": "fact", "x": 20, "y": 20}, {"id": 11, "type": "powr", "x": 24, "y": 22}]
_FAR_YARD = [{"id": 80, "type": "fact", "x": 38, "y": 12}]        # the expansion, centre (39, 13)
_MID_ORE = _ore_patch(27, 17, 4, 4)                               # its ore: nearer the main base's buildings


def _mid_view(extra=()):
    v = _view(buildings=_HOME_NEAR_ORE + _FAR_YARD + list(extra), base_center=[21, 21],
              production=[{"item": "proc", "queue": "Building", "ready": True}])
    return dataclasses.replace(v, ore_cells=_MID_ORE, bounds=(0, 0, 64, 64))


def test_an_expansion_refinery_is_never_put_beside_the_main_base():
    v = _mid_view()
    home_spot = buildplan._beside_ore(v, (28, 18), "proc")
    assert (home_spot[0] - 21) ** 2 + (home_spot[1] - 21) ** 2 < 6 ** 2         # unrestricted: at home
    (order,) = buildplan.placement_orders(v, None, (28, 18), None, expansion_site=(39, 13))
    x, y = order["at"]
    assert (x + 1 - 39) ** 2 + (y + 1 - 13) ** 2 <= C.EXPAND_REFINERY_RADIUS_CELLS ** 2   # at the expansion
    walk = terrain.walk_distances(v, _MID_ORE, 8)
    assert terrain.steps_from_rect(walk, (x, y, x + 2, y + 2)) <= C.REFINERY_ORE_WALK_STEPS


def _expansion_needing_refinery():
    e = Expansion()
    e.done = [{"site": (39, 13), "ore": (28, 18), "asked_at": None}]
    return e


def test_a_refinery_the_engine_moved_home_marks_its_spot_off_limits():
    e, v = _expansion_needing_refinery(), _mid_view()
    job = e.refinery_job(v)
    assert job is e.done[0]
    spot = buildplan._beside_ore(v, job["ore"], "proc", near=job["site"])
    e.placed_refinery(v, job, spot)
    moved = _mid_view([{"id": 99, "type": "proc", "x": 16, "y": 24}])          # came up beside the main yard
    e.update(dataclasses.replace(moved, game_seconds=v.game_seconds + 1), Mode.BUILD)
    assert e.refineries_moved == 1 and spot in job["avoid"]
    again = buildplan._beside_ore(moved, job["ore"], "proc", near=job["site"], avoid=job["avoid"])
    assert again is not None and not set(buildplan._cells("proc", again)) & job["avoid"]


def test_a_refinery_that_lands_where_asked_teaches_nothing():
    e, v = _expansion_needing_refinery(), _mid_view()
    job = e.refinery_job(v)
    spot = buildplan._beside_ore(v, job["ore"], "proc", near=job["site"])
    e.placed_refinery(v, job, spot)
    landed = _mid_view([{"id": 99, "type": "proc", "x": spot[0], "y": spot[1]}])
    e.update(dataclasses.replace(landed, game_seconds=v.game_seconds + 1), Mode.BUILD)
    assert e.refineries_moved == 0 and "avoid" not in job
    assert e.refinery_job(landed) is None                                      # served now


def test_a_refinery_against_a_cliff_does_not_harvest_across_it():
    ore = _ore_patch(20, 16, 3, 3)                                             # just across the cliff (y = 13)
    blds = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 13, "type": "proc", "x": 20, "y": 10}]
    v = dataclasses.replace(_view(buildings=blds), ore_cells=ore, blocked_cells=_CLIFF)
    assert HarvestManager().fields(v) == {}                                   # footprint touches the cliff
    assert not _expansion_needing_refinery()._served(v, (21, 17))


# -- the natural first; buildings never on rough ground; refineries spread over fields (b7fcbd4e) -

from agent.redalert.expand import choose_site as _choose_site, has_room as _has_room  # noqa: E402
from agent.redalert.observe import _unbuildable_cells  # noqa: E402

_HOME_FIELD = _ore_patch(10, 10, 4, 4)
_NATURAL = _ore_patch(2, 24, 3, 4)          # behind our base, ~18 cells out (past its reach), 12 cells of ore
_RICH_FAR = _ore_patch(28, 6, 6, 6)         # bigger, ~22 cells out toward the middle


def _fields_view(buildings=(), ore=_HOME_FIELD + _NATURAL + _RICH_FAR, **kw):
    blds = [{"id": 10, "type": "fact", "x": 5, "y": 5}] + list(buildings)
    v = _view(buildings=blds, base_center=[6, 6], **kw)
    return dataclasses.replace(v, ore_cells=ore)


def test_the_first_expansion_takes_the_natural_not_the_richest_field():
    spot, ore = _choose_site(_fields_view())
    assert ore in _NATURAL
    assert terrain.walk_distances(_fields_view(), _NATURAL, 8).get(spot, 99) <= C.EXPAND_SITE_WALK_STEPS


def test_a_natural_our_refinery_already_mines_is_skipped():
    spot, ore = _choose_site(_fields_view([{"id": 13, "type": "proc", "x": 5, "y": 25}]))
    assert ore in _RICH_FAR


def test_a_straggler_is_not_a_natural():
    straggler = _ore_patch(4, 18, 1, 2)                     # 2 cells, nearer than the real fields
    spot, ore = _choose_site(_fields_view(ore=_HOME_FIELD + straggler + _RICH_FAR))
    assert ore in _RICH_FAR


_LIVE_LEGEND = {".": "clear ground", ",": "rough ground", "=": "road or bridge", "_": "beach",
                "~": "water or river (ground units cannot cross)", "^": "rock or cliff (impassable)",
                "T": "trees (impassable)", "#": "wall, fence, civilian building or other neutral obstacle",
                "$": "ore at the start of the match", "M": "ore mine (ore regrows around it)",
                "S": "spawn point: a start position (rules.maps.<id>.spawns)"}


def test_rough_ground_beaches_and_roads_take_no_buildings():
    layout = {"origin": [0, 0], "rows": [".,_=S$^"], "legend": _LIVE_LEGEND}
    assert _unbuildable_cells("unbuildable-test", layout) == frozenset({(1, 0), (2, 0), (3, 0), (5, 0)})
    rough = frozenset((x, y) for x in range(20, 23) for y in range(20, 23))
    v = dataclasses.replace(_view(buildings=[{"id": 10, "type": "fact", "x": 16, "y": 20}]), unbuildable_cells=rough)
    assert buildplan._clashes("proc", (20, 20), v) and not buildplan._clashes("proc", (24, 20), v)
    assert not _has_room((21, 21), v) and _has_room((26, 21), v)


_HOME_PROCS = [{"id": 13, "type": "proc", "x": 14, "y": 10}, {"id": 14, "type": "proc", "x": 10, "y": 14},
               {"id": 15, "type": "proc", "x": 6, "y": 11}]


def test_a_field_takes_refineries_by_its_size_three_at_most():
    big = _ore_patch(10, 10, 9, 5)                                   # 45 cells: room for 3
    two = _fields_view(_HOME_PROCS[:2], ore=big + _NATURAL)
    assert buildplan.open_ore(two) in big
    three = _fields_view(_HOME_PROCS, ore=big + _NATURAL)
    assert buildplan.open_ore(three) is None                         # full, nothing else in reach...
    assert buildplan.open_ore(three, radius=None) in _NATURAL        # ...so the base crawls to the natural
    small = _fields_view(_HOME_PROCS[:1], ore=_HOME_FIELD + _NATURAL)   # 16 cells: room for 1
    assert buildplan.open_ore(small) is None


def test_refineries_spread_over_the_fields_in_reach_by_load():
    # a2952a6d: a 37-cell field 8 cells out took all three refineries while a
    # 47-cell field 11 cells out got none. Now the second goes to the other field.
    near = _ore_patch(10, 10, 6, 6)                                  # 36 cells, the nearer field
    west = _ore_patch(2, 14, 4, 12)                                  # 48 cells, a little farther
    first = _fields_view(ore=near + west)
    assert buildplan.open_ore(first) in near                         # first: the nearest
    one = _fields_view([{"id": 13, "type": "proc", "x": 16, "y": 12}], ore=near + west)
    assert buildplan.open_ore(one) in west                           # second: the emptier field


def test_no_refinery_is_bought_for_a_full_field():
    v = _fields_view(_HOME_PROCS + [{"id": 16, "type": "weap", "x": 1, "y": 1}], ore=_HOME_FIELD, _t=500.0,
                     economy={"cash": 1000})
    assert buildplan.open_ore(v, radius=None) is None
    assert not any(o.get("item") == "proc" for o in buildplan.next_build_orders(v))


def test_ore_one_cell_apart_is_one_field():
    from agent.redalert.expand import _ore_components
    v = dataclasses.replace(_view(), ore_cells=((1, 1), (3, 1), (9, 9)))
    assert sorted(len(c) for c in _ore_components(v)) == [1, 2]


# -- a razed enemy is hunted to the last building (match 0561274e went to the time limit) ---------

import types  # noqa: E402

_ENEMY_YARD = [{"id": 90, "type": "fact", "x": 40, "y": 40, "hp": 100}]


_ENEMY_BASE = [{"id": 90 + i, "type": "powr", "x": 30 + 3 * (i % 4), "y": 36 + 3 * (i // 4), "hp": 100} for i in range(8)]


def test_the_hunt_starts_once_their_base_is_nearly_razed_and_we_lead():
    h = scout.Hunter()
    assert not h.hunting(_view(military={"kills_cost": 5000}))                       # never saw their base
    assert not h.hunting(_view(visible_enemy_buildings=_ENEMY_YARD, military={"kills_cost": 5000}))  # fog: 1 seen
    assert not h.hunting(_view(visible_enemy_buildings=_ENEMY_BASE, military={"kills_cost": 5000}))  # base stands
    last = _ENEMY_BASE[:1]                                                           # one building left (b7fcbd4e)
    assert h.hunting(_view(visible_enemy_buildings=last, military={"kills_cost": 5000, "deaths_cost": 1000}))
    assert h.hunting(_view(military={"kills_cost": 5000, "deaths_cost": 1000}))      # razed, and ahead
    assert not h.hunting(_view(military={"kills_cost": 1000, "deaths_cost": 5000}))  # razed, but behind


def test_a_stale_advisor_target_does_not_hold_the_army_off_the_last_building():
    units = [{"id": 30 + i, "type": "1tnk", "x": 23, "y": 30, "hp": 100} for i in range(4)]
    dome = [{"id": 99, "type": "dome", "x": 13, "y": 40, "hp": 100}]
    stale = types.SimpleNamespace(army_target=(30, 40))                              # nothing known near it
    o = executor._army_orders(_view(buildings=BASE, units=units, visible_enemy_buildings=dome),
                              Mode.ALL_IN, advice=stale)
    assert o[0]["to"] == [13, 40]
    fresh = types.SimpleNamespace(army_target=(15, 40))                              # beside a known building
    o = executor._army_orders(_view(buildings=BASE, units=units, visible_enemy_buildings=dome),
                              Mode.ALL_IN, advice=fresh)
    assert o[0]["to"] == [15, 40]


def test_hunting_beats_protecting_the_lead_and_a_model_hold():
    units = [{"id": 30 + i, "type": "1tnk", "x": 20, "y": 20, "hp": 100} for i in range(10)]
    v = _view(buildings=BASE, units=units, _r=60.0,
              military={"army_value": 9000, "kills_cost": 9000, "deaths_cost": 6000})    # ahead, not dominant
    assert brain.choose_mode(v, cheese_since=None) is Mode.PROTECT_LEAD              # without the hunt
    assert brain.choose_mode(v, cheese_since=None, hunting=True) is Mode.ALL_IN
    assert brain.choose_mode(v, cheese_since=None, model_pref=Mode.BUILD, hunting=True) is Mode.ALL_IN


def test_a_blind_army_follows_the_hunter_not_a_stale_advisor_target():
    units = [{"id": 30 + i, "type": "1tnk", "x": 30, "y": 30, "hp": 100} for i in range(4)]
    stale = types.SimpleNamespace(army_target=(40, 40))                              # the razed enemy base
    o = executor._army_orders(_view(buildings=BASE, units=units), Mode.ALL_IN, advice=stale, hunt=(10, 40))
    assert o[0]["to"] == [10, 40]


def test_a_big_hunting_army_sweeps_several_points_at_once():
    units = [{"id": 30 + i, "type": "1tnk", "x": 24, "y": 24, "hp": 100} for i in range(18)]
    points = [(40, 40), (10, 40), (40, 10)]
    o = executor._army_orders(_view(buildings=BASE, units=units), Mode.ALL_IN, hunt=points[0], hunt_points=points)
    assert sorted(tuple(x["to"]) for x in o) == sorted(points)
    assert sorted(len(x["units"]) for x in o) == [6, 6, 6]
    assert len({u for x in o for u in x["units"]}) == 18


def test_an_expansion_whose_refineries_keep_landing_at_home_is_given_up():
    e, v = _expansion_needing_refinery(), _mid_view()
    e.done[0]["moved"] = C.EXPAND_REFINERY_MAX_MOVES
    assert e.refinery_job(v) is None and e.needs_refinery(v) is None and e.refinery_ore(v) is None


# -- fixing b7fcbd4e's mistakes: no early trade, guard the harvesters, never walk away under fire, sell to recover

def test_no_all_in_before_the_economy_is_up_even_against_a_small_seen_army():
    v = _view(_t=65.0, buildings=BASE, military={"army_value": 2300, "visible_enemy_value": 500})   # 1:05, "4.6x"
    assert brain.choose_mode(v, cheese_since=None) is Mode.BUILD
    assert brain.choose_mode(v, cheese_since=None, model_pref=Mode.ALL_IN) is Mode.BUILD
    with_factory = _view(_t=65.0, buildings=BASE + [{"id": 12, "type": "weap", "x": 4, "y": 8}],
                         military={"army_value": 2300, "visible_enemy_value": 500})
    assert brain.choose_mode(with_factory, cheese_since=None) is Mode.ALL_IN
    later = _view(_t=200.0, buildings=BASE, military={"army_value": 2300, "visible_enemy_value": 500})
    assert brain.choose_mode(later, cheese_since=None) is Mode.ALL_IN


def test_the_massing_point_sits_between_the_harvesters_and_the_enemy():
    v = _fields_view()                                          # base [6,6], home ore around (10..13, 10..13)
    stage = scout.choke_cell(v)
    base_d = (stage[0] - 6) ** 2 + (stage[1] - 6) ** 2
    assert 6 ** 2 <= base_d <= 14 ** 2                          # just in front of the base, not a third of the map out
    assert (stage[0] - 40) ** 2 + (stage[1] - 40) ** 2 < (10 - 40) ** 2 + (10 - 40) ** 2   # on the enemy side of the ore


def test_new_units_rally_to_the_massing_point():
    blds = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 11, "type": "tent", "x": 8, "y": 4, "rally": None}]
    v = _view(buildings=blds)
    assert executor._rally_orders(v) == [{"cmd": "rally", "building": 11, "to": list(scout.choke_cell(v))}]


def test_no_combat_order_is_a_plain_move():
    units = [{"id": 30 + i, "type": "1tnk", "x": 38, "y": 38, "hp": 15 if i == 0 else 100, "idle": True} for i in range(6)]
    enemy = [{"id": 90, "type": "3tnk", "x": 39, "y": 39, "hp": 100}]
    for mode in Mode:
        v = _view(_t=400.0, buildings=BASE, units=units, visible_enemies=enemy)
        o = executor._army_orders(v, mode) + micro.plan(v, mode, frozenset(), scout.choke_cell(v))[0]
        assert all(x["cmd"] != "move" for x in o), (mode, o)


def test_broke_with_no_harvester_sells_a_spare_building():
    blds = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 12, "type": "weap", "x": 4, "y": 8},
            {"id": 13, "type": "proc", "x": 9, "y": 9}, {"id": 14, "type": "dome", "x": 12, "y": 4},
            {"id": 15, "type": "barr", "x": 16, "y": 4}]
    broke = _view(buildings=blds, economy={"cash": 200})
    assert buildplan.sell_for_income(broke) == {"cmd": "sell", "building": 14}       # the dome, not production
    assert ra_orders.order_ok(buildplan.sell_for_income(broke))
    harv = [{"id": 40, "type": "harv", "x": 9, "y": 12, "hp": 100}]
    assert buildplan.sell_for_income(_view(buildings=blds, units=harv, economy={"cash": 0})) is None
    assert buildplan.sell_for_income(_view(buildings=blds, economy={"cash": 1500})) is None   # can buy one
    bare = [b for b in blds if b["type"] not in ("dome",)]
    assert buildplan.sell_for_income(_view(buildings=bare, economy={"cash": 0})) is None      # last barracks kept


# -- never ask for a building where units stand (a1329ceb: two expansion refineries went home) -----

def test_an_expansion_refinery_avoids_a_spot_trucks_are_driving_through():
    v = _mid_view()
    clear = buildplan._beside_ore(v, (28, 18), "proc", near=(39, 13))
    truck = [{"id": 40, "type": "harv", "x": clear[0] + 1, "y": clear[1] + 1, "hp": 100}]
    busy = dataclasses.replace(v, units=read(_raw(units=truck)).units)
    other = buildplan._beside_ore(busy, (28, 18), "proc", near=(39, 13))
    assert other != clear and (other[0] + 1, other[1] + 1) != (clear[0] + 1, clear[1] + 1)
    assert not {(clear[0] + 1, clear[1] + 1)} & set(buildplan._cells("proc", other))


def test_a_grown_building_skips_a_slot_with_units_on_it():
    blds = [{"id": 10, "type": "fact", "x": 10, "y": 10}]
    v = _view(buildings=blds)
    first = buildplan.grow_spot(v, (1.0, 0.0), "powr")
    tank = [{"id": 30, "type": "1tnk", "x": first[0], "y": first[1], "hp": 100}]
    second = buildplan.grow_spot(_view(buildings=blds, units=tank), (1.0, 0.0), "powr")
    assert second is not None and second != first


def test_a_moved_expansion_refinery_is_logged_with_where_it_landed():
    e, v = _expansion_needing_refinery(), _mid_view()
    job = e.refinery_job(v)
    spot = buildplan._beside_ore(v, job["ore"], "proc", near=job["site"])
    e.placed_refinery(v, job, spot)
    moved = _mid_view([{"id": 99, "type": "proc", "x": 16, "y": 24}])
    e.update(dataclasses.replace(moved, game_seconds=v.game_seconds + 1), Mode.BUILD)
    assert e.moves == [{"t": round(v.game_seconds + 1), "asked": list(spot), "landed": [16, 24]}]


# -- broke but trading even: expand again sooner (a1329ceb, b7fcbd4e: no second expansion by 10:00) --

def _broke(t, kills=5000, deaths=5000, **kw):
    return dataclasses.replace(_exp_view(t=t, economy={"cash": 100}, **kw), kills_cost=kills, deaths_cost=deaths)


def test_broke_and_trading_even_expands_again_a_minute_after_the_last():
    e = _deployed_expansion(t=335.0)
    for t in (340.0, 380.0, 394.0):
        e.update(_broke(t), Mode.BUILD)
    assert e.state == "idle"                                  # 59 s after the first: not yet
    e.update(_broke(396.0), Mode.BUILD)
    assert e.state == "scouting"                              # broke 56 s, trades even, 61 s on: go


def test_broke_but_losing_trades_waits_the_normal_interval():
    e = _deployed_expansion(t=335.0)
    for t in (340.0, 380.0, 396.0, 420.0):
        e.update(_broke(t, kills=2000, deaths=6000), Mode.BUILD)
    assert e.state == "idle"


def test_broke_and_trading_even_overrides_an_advisor_wait():
    e = _deployed_expansion(t=335.0)
    wait = types.SimpleNamespace(expand="wait")
    for t in (340.0, 380.0, 396.0):
        e.update(_broke(t), Mode.BUILD, wait)
    assert e.state == "scouting"


def test_the_expansion_mcv_is_pre_funded():
    units = [{"id": 30 + i, "type": "e1", "x": 10, "y": 10, "hp": 100} for i in range(8)]
    v = _view(buildings=BASE, units=units, production=[{"item": "mcv", "queue": "Vehicle", "progress": 0.2}])
    assert buildplan.build_reserve(v) >= 0.8 * 2500                      # what is left of the MCV is set aside


# -- 654f1b2e: finish a dominated enemy; no MCV for a field our base reaches ------------------------

def test_a_dominant_lead_finishes_the_enemy_instead_of_protecting_the_lead():
    units = [{"id": 30 + i, "type": "1tnk", "x": 20, "y": 20, "hp": 100} for i in range(10)]
    v = _view(buildings=BASE, units=units, _r=100.0,
              military={"army_value": 22600, "kills_cost": 59650, "deaths_cost": 25950})   # 8 buildings left
    assert brain.choose_mode(v, cheese_since=None) is Mode.ALL_IN
    assert brain.choose_mode(v, cheese_since=None, model_pref=Mode.BUILD) is Mode.ALL_IN
    strong = dataclasses.replace(v, enemy_army_value=20000)                            # a real army left
    assert brain.choose_mode(strong, cheese_since=None) is Mode.PROTECT_LEAD           # then keep the lead


def test_no_expansion_mcv_for_a_field_our_base_already_reaches():
    near_only = _HOME_FIELD + _ore_patch(2, 18, 3, 4)          # a second field 13 cells out: refineries' job
    assert _choose_site(_fields_view(ore=near_only)) is None
    spot, ore = _choose_site(_fields_view(ore=near_only + _RICH_FAR))
    assert ore in _RICH_FAR


def test_every_placement_is_audited_with_where_it_landed():
    from agent.redalert.agent import RedAlertAgent
    a = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    logged = []
    a._matchlog.placement = logged.append
    v = _view(buildings=BASE)
    a._note_placements(v, [{"cmd": "place", "item": "powr", "at": [10, 10]}])
    a._audit_placements(_view(buildings=BASE + [{"id": 99, "type": "powr", "x": 11, "y": 11}]))
    assert logged == [{"t": 100, "item": "powr", "asked": [10, 10], "landed": [11, 11]}]


# -- into turrets only with an infantry mass or armor; a pro opening keeps a small guard -------------

_TOWERS = [{"id": 80 + i, "type": "ftur", "x": 38 + i, "y": 38, "hp": 100} for i in range(3)]


def _attack_view(units):
    return _view(_t=400.0, buildings=BASE + [{"id": 12, "type": "weap", "x": 4, "y": 8}], units=units,
                 visible_enemy_buildings=_TOWERS + [{"id": 90, "type": "fact", "x": 40, "y": 40, "hp": 100}],
                 military={"army_value": 9000, "visible_enemy_value": 1000})


def test_no_attack_into_turrets_with_a_small_mixed_army():
    mixed = [{"id": 30 + i, "type": "e1" if i < 8 else "1tnk", "x": 10, "y": 10, "hp": 100} for i in range(12)]
    v = _attack_view(mixed)                                       # 8 infantry, 4 tanks vs 3 flame towers
    assert not brain.can_break_defenses(v)
    assert brain.choose_mode(v, cheese_since=None, prev_mode=Mode.BUILD) is Mode.BUILD


def test_an_infantry_mass_or_armor_may_attack_into_turrets():
    mass = [{"id": 30 + i, "type": "e1", "x": 10, "y": 10, "hp": 100} for i in range(18)]
    assert brain.choose_mode(_attack_view(mass), cheese_since=None, prev_mode=Mode.BUILD) is Mode.ALL_IN
    armor = [{"id": 30 + i, "type": "3tnk" if i < 5 else "v2rl", "x": 10, "y": 10, "hp": 100} for i in range(7)]
    assert brain.can_break_defenses(_attack_view(armor))          # 5 tanks + 2 V2 (x2) = 9 = 3 per tower


def test_an_undefended_target_needs_no_mass():
    few = [{"id": 30 + i, "type": "e1", "x": 10, "y": 10, "hp": 100} for i in range(4)]
    v = _view(_t=400.0, buildings=BASE, units=few, visible_enemy_buildings=[{"id": 90, "type": "proc", "x": 40, "y": 40, "hp": 100}])
    assert brain.can_break_defenses(v)


def test_no_infantry_cap_only_the_build_order_reserve():
    many = [{"id": 30 + i, "type": "e1", "x": 10, "y": 10, "hp": 100} for i in range(14)]
    rich = _view(buildings=BASE, units=many, economy={"cash": 2500})
    assert any(x.get("item") == "e1" for x in executor._production_orders(rich, Mode.BUILD))   # 14 and counting
    rush = executor._production_orders(_view(buildings=BASE, units=many, economy={"cash": 150}), Mode.DEFEND_RUSH)
    assert any(x.get("item") == "e1" for x in rush)                # a rush is answered whatever the reserve


# -- one army: commit together or retreat together (fd507a58: the garrison pulled troops out of an attack)

def test_squads_never_take_units_from_the_front():
    front = [{"id": 30 + i, "type": "1tnk" if i < 4 else "e1", "x": 30, "y": 30, "hp": 100} for i in range(10)]
    g = Garrison()
    e = _deployed_expansion()
    v = _view(_t=400.0, buildings=BASE, units=front)
    g.update(v, Mode.BUILD, e, taken=set())
    assert g.unit_ids == []                                        # 30 cells out: those are fighting
    home = [{**u, "x": 8, "y": 8} for u in front]
    g.update(_view(_t=400.0, buildings=BASE, units=home), Mode.BUILD, e, taken=set())
    assert len(g.unit_ids) == C.GARRISON_SIZE


def test_no_squad_claims_during_an_attack_and_raiders_join_it():
    home = [{"id": 30 + i, "type": "e1", "x": 8, "y": 8, "hp": 100} for i in range(10)]
    g = Garrison()
    v = _view(_t=400.0, buildings=BASE, units=home)
    g.update(v, Mode.ALL_IN, _deployed_expansion(), taken=set())
    assert g.unit_ids == []
    r = Raid()
    r.unit_ids, r._formed_size = [30, 31, 32], 3
    r.update(dataclasses.replace(v, enemy_units=read(_raw(visible_enemies=[_ENEMY_HARV])).enemy_units), Mode.ALL_IN, None)
    assert r.unit_ids == [30, 31, 32]                               # the raid stays: it is the second prong
    assert r.orders(v, main_fight=(30, 30)) == [{"cmd": "attack_move", "units": [30, 31, 32], "to": [30, 30]}]


# -- the observation reports a building's centre; placement takes its top-left (fd507a58 audit) -----

def test_our_buildings_are_read_at_their_top_left_cell():
    raw = _raw()
    raw["observation"]["buildings"] = [{"id": 10, "type": "fact", "x": 24, "y": 45},      # as reported (centre)
                                       {"id": 11, "type": "powr", "x": 21, "y": 45},
                                       {"id": 12, "type": "pbox", "x": 30, "y": 30}]
    cells = {b.type: b.cell for b in read(raw).buildings}
    assert cells == {"fact": (23, 44), "powr": (20, 44), "pbox": (30, 30)}           # asked [20,44] -> seen [21,45]


def test_a_spot_right_of_a_building_is_one_real_cell_away():
    v = dataclasses.replace(_view(buildings=[{"id": 10, "type": "fact", "x": 10, "y": 10}]), ore_cells=_ore_patch(17, 10, 3, 3))
    x, y = buildplan._beside_ore(v, (17, 11), "proc")
    assert x == 14                                         # yard covers x 10-12 for real: one free column (13)


def test_a_turret_is_never_the_anchor_for_a_new_building():
    blds = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 11, "type": "pbox", "x": 20, "y": 5}]
    v = dataclasses.replace(_view(buildings=blds), ore_cells=_ore_patch(23, 4, 3, 3))
    x, y = buildplan._beside_ore(v, (24, 5), "proc")
    assert x <= 9                                          # beside the yard, not out beside the lone pillbox


def test_the_massing_point_moves_off_buildings_and_rocks():
    v = _view()
    stage = scout.choke_cell(v)
    on_it = [{"id": 20, "type": "weap", "x": stage[0] - 1, "y": stage[1] - 1}]
    moved = scout.choke_cell(_view(buildings=on_it))
    assert moved != stage and (moved[0] - stage[0]) ** 2 + (moved[1] - stage[1]) ** 2 <= C.STAGE_MOVE_SEARCH_CELLS ** 2
    rocks = frozenset((stage[0] + dx, stage[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    assert scout.choke_cell(dataclasses.replace(v, blocked_cells=rocks)) != stage


def test_nothing_is_built_on_the_massing_point():
    v = _view(buildings=[{"id": 10, "type": "fact", "x": 4, "y": 4}])
    stage = scout.choke_cell(v)
    assert buildplan._clashes("powr", (stage[0], stage[1]), v)


# -- units on a building's spot step aside before it is placed ---------------------------------------

def _crowded(t, extra_units=()):
    blds = [{"id": 10, "type": "fact", "x": 10, "y": 10}]
    crowd = [{"id": 100 + i * 40 + j, "type": "e1", "x": 4 + i, "y": 4 + j, "hp": 100}
             for i in range(20) for j in range(20) if not (10 <= 4 + i <= 12 and 10 <= 4 + j <= 12)]
    return _view(_t=t, buildings=blds, units=crowd + list(extra_units),
                 production=[{"item": "powr", "queue": "Building", "ready": True}])


def test_units_on_the_spot_are_moved_off_before_placing():
    clearing = {}
    o = buildplan.placement_orders(_crowded(100.0), clearing=clearing)
    assert not any(x["cmd"] == "place" for x in o)                        # wait: the spot is occupied
    moves = [x for x in o if x["cmd"] == "attack_move"]                    # fighters step aside, still fighting
    assert moves and clearing == {"powr": 100.0}
    at_spot = moves[0]["to"]
    o2 = buildplan.placement_orders(_crowded(106.0), clearing=clearing)
    assert any(x["cmd"] == "place" for x in o2) and clearing == {}         # waited long enough: place anyway
    assert at_spot is not None


def test_trucks_on_the_spot_get_a_plain_move_off_it():
    v = _view(buildings=[{"id": 10, "type": "fact", "x": 10, "y": 10}],
              units=[{"id": 40, "type": "harv", "x": 14, "y": 10, "hp": 100}, {"id": 41, "type": "e1", "x": 15, "y": 11, "hp": 100}])
    o = buildplan.clear_spot_orders(v, "powr", (14, 10))
    cmds = {x["cmd"]: x for x in o}
    assert cmds["move"]["units"] == [40] and cmds["attack_move"]["units"] == [41]
    dest = tuple(cmds["move"]["to"])
    assert dest not in set(buildplan._cells("powr", (14, 10)))             # off the footprint


def test_a_free_spot_is_placed_at_once():
    v = _view(buildings=[{"id": 10, "type": "fact", "x": 10, "y": 10}],
              production=[{"item": "powr", "queue": "Building", "ready": True}])
    clearing = {}
    o = buildplan.placement_orders(v, clearing=clearing)
    assert [x["cmd"] for x in o] == ["place"] and clearing == {}


def test_units_stepping_aside_are_left_out_of_army_orders():
    out = executor.plan(_crowded(100.0), Mode.BUILD, clearing={})
    moved = [x for x in out if x["cmd"] in ("attack_move", "move")]
    first = moved[0]["units"]                                               # the step-aside order comes first
    later = [i for x in moved[1:] for i in x["units"]]
    assert first and not set(first) & set(later), (first, later)            # nobody walks them back


# -- TEST ONLY: the self-play sparring rusher (agent/redalert/testrush.py; remove with it) -----------

from agent.redalert import testrush  # noqa: E402


def _ctx(seat, tournament=None, session="spar-1"):
    return DecisionContext(session_id=session, tournament_id=tournament, game_type="red_alert",
                           agent_id="me", seat_position=seat)


def test_the_rusher_wakes_only_against_ourselves_in_a_test_match(monkeypatch, tmp_path):
    monkeypatch.setenv("RA_TEST_RUSH", "1")
    monkeypatch.setattr(testrush, "_MARKER_DIR", tmp_path)
    seat0, seat1 = testrush.SelfPlayRusher(), testrush.SelfPlayRusher()
    assert not seat1.active(_ctx(1))                       # alone so far: not known to be self-play
    assert not seat0.active(_ctx(0))                       # seat 0 sees seat 1: self-play, but it defends
    assert seat1.active(_ctx(1))                           # seat 1 sees seat 0: it rushes
    other = testrush.SelfPlayRusher()
    assert not other.active(_ctx(1, tournament="t-1", session="spar-1"))   # never in a tournament game


def test_the_rusher_can_be_turned_off_or_moved(monkeypatch, tmp_path):
    monkeypatch.setattr(testrush, "_MARKER_DIR", tmp_path)
    monkeypatch.setenv("RA_TEST_RUSH", "0")
    a, b = testrush.SelfPlayRusher(), testrush.SelfPlayRusher()
    a.active(_ctx(0, session="spar-2")); assert not b.active(_ctx(1, session="spar-2"))
    monkeypatch.setenv("RA_TEST_RUSH", "1")
    monkeypatch.setenv("RA_TEST_RUSH_SEAT", "0")
    c, d = testrush.SelfPlayRusher(), testrush.SelfPlayRusher()
    d.active(_ctx(1, session="spar-3"))
    assert c.active(_ctx(0, session="spar-3"))


def test_the_rusher_masses_a_ball_then_pushes_until_it_is_spent():
    rifles = [{"id": 30 + i, "type": "e1", "x": 10, "y": 10, "hp": 100} for i in range(12)]
    v = _view(_t=130.0, buildings=BASE, units=rifles, military={"army_value": 1200})
    r = testrush.SelfPlayRusher()
    assert r.mode(v, None) is Mode.BUILD                                       # 1200: still massing
    assert r.mode(dataclasses.replace(v, army_value=2600), None) is Mode.ALL_IN   # the ball launches
    assert r.mode(dataclasses.replace(v, army_value=1300), None) is Mode.ALL_IN   # pushing: 43% of 2600 left
    assert r.mode(dataclasses.replace(v, army_value=900), None) is Mode.BUILD     # spent: re-mass
    assert r.mode(dataclasses.replace(v, army_value=1600), None) is Mode.BUILD    # not the next ball yet...
    late = dataclasses.replace(v, game_seconds=215.0, army_value=1600)
    assert r.mode(late, None) is Mode.ALL_IN                                   # ...but by 3:30 it goes anyway
    assert brain.choose_mode(v, cheese_since=None) is Mode.BUILD               # our own agent holds


def test_the_rusher_produces_without_pause_and_builds_barracks():
    rifles = [{"id": 30 + i, "type": "e1", "x": 10, "y": 10, "hp": 100} for i in range(12)]
    q2 = [{"item": "e1", "queue": "Infantry", "ready": False}] * 2
    v = _view(_t=130.0, buildings=BASE, units=rifles, production=q2)
    assert any(x.get("item") == "e1" for x in executor._production_orders(v, Mode.BUILD, rusher=True))   # 3 deep
    assert not any(x.get("item") == "e1" for x in executor._production_orders(v, Mode.BUILD))            # ours: capped
    blds = BASE + [{"id": 13, "type": "proc", "x": 9, "y": 9}, {"id": 14, "type": "powr", "x": 1, "y": 1}]
    o = buildplan.next_build_orders(_view(buildings=blds, economy={"cash": 2000}), rusher=True)
    assert o == [{"cmd": "build", "item": "tent"}]                             # a second barracks right away


# -- c2fec761: legal growth spots, no stall on stale targets, memory follows a moving yard, home responders

from agent.redalert.memory import EnemyMemory  # noqa: E402

_NO_AREA_T = {"pbox", "hbox", "gun", "ftur", "tsla", "agun", "sam"}


def _real_gap(v, item, spot):
    nw, nh = C.FOOTPRINTS.get(item, (2, 2))
    r = (spot[0], spot[1], spot[0] + nw - 1, spot[1] + nh - 1)
    gaps = []
    for b in v.buildings:
        if b.type in _NO_AREA_T:
            continue
        w, h = C.FOOTPRINTS.get(b.type, (2, 2))
        q = (b.cell[0], b.cell[1], b.cell[0] + w - 1, b.cell[1] + h - 1)
        gaps.append(max(max(q[0] - r[2], r[0] - q[2]) - 1, max(q[1] - r[3], r[1] - q[3]) - 1))
    return min(gaps)


def test_grown_buildings_are_always_one_cell_from_a_building_even_on_a_diagonal():
    blds = [{"id": 10, "type": "fact", "x": 30, "y": 30}, {"id": 11, "type": "powr", "x": 27, "y": 30}]
    v = _view(buildings=blds, base_center=[31, 31])
    for bearing in ((-0.707, -0.707), (0.707, 0.707), (1.0, 0.0), (0.0, -1.0), (-0.6, 0.8)):
        for item in ("barr", "weap", "powr", "fix"):
            spot = buildplan.grow_spot(v, bearing, item)
            assert spot is not None and _real_gap(v, item, spot) == 1, (bearing, item, spot)


def test_a_stalled_attack_forgets_stale_buildings_and_drops_the_advisor_target():
    from agent.redalert.agent import RedAlertAgent
    a = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    a._mode = Mode.ALL_IN
    army = [{"id": 30 + i, "type": "3tnk", "x": 22, "y": 44, "hp": 100} for i in range(6)]
    ghost = [{"id": 90, "type": "powr", "x": 29, "y": 44, "hp": 100}]           # destroyed, still remembered
    a._memory.update(_view(_t=500.0, visible_enemy_buildings=ghost))
    for t in (520.0, 545.0):
        a._break_stall(_view(_t=t, units=army, military={"kills_cost": 42900}))
    assert [b.id for b in a._memory.buildings] == [90]                           # not stalled long enough yet
    a._break_stall(_view(_t=562.0, units=army, military={"kills_cost": 42900}))
    assert a._memory.buildings == []                                             # 42 s, no kills: forgotten
    adv = ra_advisor.parse_advice({**_ANSWER, "army_target": [23, 44]}, _view(), 100.0)
    assert a._army_advice(adv, False, _view(_t=570.0)).army_target is None       # the hunter leads a while
    assert a._army_advice(adv, False, _view(_t=700.0)).army_target is not None
    assert a._army_advice(adv, True, _view(_t=700.0)).army_target is None        # finishing: rules aim


def test_memory_keeps_a_building_where_we_last_saw_it_and_follows_a_packed_up_yard():
    m = EnemyMemory()
    m.update(_view(_t=100.0, visible_enemy_buildings=[{"id": 90, "type": "fact", "x": 40, "y": 40, "hp": 100}]))
    m.update(_view(_t=110.0, visible_enemy_buildings=[{"id": 90, "type": "fact", "x": 41, "y": 40, "hp": 100}]))
    assert [b.cell for b in m.buildings] == [(41, 40)]                          # refreshed to where we saw it now
    m.update(_view(_t=120.0, visible_enemies=[{"id": 95, "type": "mcv", "x": 42, "y": 41, "hp": 100}]))
    assert m.buildings == [] and [u.type for u in m.units] == ["mcv"]           # packed up: follow the MCV
    v = dataclasses.replace(_view(buildings=BASE), known_enemy_units=m.units)
    assert executor._attack_target(v) == (42, 41)                               # and hunt it down
    m.update(_view(_t=130.0, visible_enemy_buildings=[{"id": 99, "type": "fact", "x": 43, "y": 41, "hp": 100}]))
    assert [u.type for u in m.units] == [] and [b.id for b in m.buildings] == [99]   # deployed again


def test_units_at_home_kill_a_small_raid_while_the_army_is_out():
    ranger = [{"id": 70, "type": "jeep", "x": 9, "y": 10, "hp": 20}]
    home = [{"id": 30, "type": "e3", "x": 10, "y": 10, "hp": 100}, {"id": 31, "type": "e1", "x": 11, "y": 11, "hp": 100}]
    away = [{"id": 40 + i, "type": "3tnk", "x": 38, "y": 38, "hp": 100} for i in range(6)]
    v = _view(_t=600.0, buildings=BASE, units=home + away, visible_enemies=ranger)
    o, ids = executor.home_response(v, Mode.ALL_IN, None, frozenset())
    assert o == [{"cmd": "attack", "units": [30, 31], "target": 70}] and ids == {30, 31}
    army = executor._army_orders(v, Mode.ALL_IN, reserved=frozenset(ids))
    assert not any(set(x["units"]) & ids for x in army)                         # the army leaves them to it
    big = [{"id": 70 + i, "type": "3tnk", "x": 9, "y": 10, "hp": 100} for i in range(6)]       # 1800 > 1500
    assert executor.home_response(_view(buildings=BASE, units=home, visible_enemies=big), Mode.ALL_IN, None, frozenset()) == ([], set())


def test_a_mid_game_bank_buys_a_second_war_factory_and_a_deeper_tank_queue():
    v = _view(_t=400.0, buildings=_ECON_BASE, units=_harvs(6), economy={"cash": 400, "ore": 3000},
              available_production=["powr", "tent", "proc", "weap", "e1", "e3", "1tnk", "harv"])
    assert any(o.get("item") == "weap" for o in buildplan.next_build_orders(v))
    early = dataclasses.replace(v, game_seconds=300.0)
    assert not any(o.get("item") == "weap" for o in buildplan.next_build_orders(early))
    q = [{"item": "1tnk", "queue": "Vehicle", "ready": False}]
    assert any(o.get("item") == "1tnk" for o in executor._production_orders(dataclasses.replace(v, production=q), Mode.BUILD))


# -- 5aea2632 (vs Ryanxhuang): raid tactics, two-pronged attack, scout survival, early guard ---------

def _raid_with(units, enemies, t=400.0):
    r = Raid()
    r.unit_ids, r._formed_size = [u["id"] for u in units], len(units)
    return r, _view(_t=t, buildings=BASE, units=units, visible_enemies=enemies)


_RAIDERS = [{"id": 30 + i, "type": "e3", "x": 30, "y": 30, "hp": 100} for i in range(4)]
_THEIR_HARV = {"id": 90, "type": "harv", "x": 31, "y": 31, "hp": 100}


def test_the_raid_kills_defenders_before_harvesters():
    rifle = {"id": 80, "type": "e1", "x": 32, "y": 30, "hp": 40}
    r, v = _raid_with(_RAIDERS, [_THEIR_HARV, rifle])
    assert r.situation(v)[0] == "fight"
    assert r.orders(v) == [{"cmd": "attack", "units": [30, 31, 32, 33], "target": 80}]
    r2, v2 = _raid_with(_RAIDERS, [_THEIR_HARV])
    assert r2.orders(v2) == [{"cmd": "attack", "units": [30, 31, 32, 33], "target": 90}]   # undefended: the truck


def test_an_outmatched_raid_pulls_back_fighting_and_marks_their_army_distracted():
    tanks = [{"id": 80 + i, "type": "3tnk", "x": 32, "y": 31, "hp": 100} for i in range(6)]
    r, v = _raid_with(_RAIDERS, [_THEIR_HARV] + tanks)
    assert r.situation(v)[0] == "outmatched"
    o = r.orders(v)
    assert o[0]["cmd"] == "attack_move" and o[0]["to"] == list(scout.choke_cell(v))
    assert r.distracting(v) and not r.distracting(dataclasses.replace(v, game_seconds=440.0))


def test_a_distracted_enemy_army_opens_the_centre_for_our_main_army():
    ours = [{"id": 40 + i, "type": "3tnk", "x": 10, "y": 10, "hp": 100} for i in range(8)]
    theirs = [{"id": 80 + i, "type": "3tnk", "x": 30, "y": 30, "hp": 100} for i in range(6)]
    v = _view(_t=400.0, buildings=BASE + [{"id": 12, "type": "weap", "x": 4, "y": 8}], units=ours,
              visible_enemies=theirs, military={"army_value": 2400, "visible_enemy_value": 2000})
    assert brain.choose_mode(v, cheese_since=None, prev_mode=Mode.BUILD) is Mode.BUILD        # edge 1.2: hold
    assert brain.choose_mode(v, cheese_since=None, prev_mode=Mode.BUILD, distraction=True) is Mode.ALL_IN
    weaker = dataclasses.replace(v, units=v.units[:4])
    assert brain.choose_mode(weaker, cheese_since=None, prev_mode=Mode.BUILD, distraction=True) is Mode.BUILD


# -- intel, the scout corps, the build-order reserve, semi-concave attacks, look before committing ---

from agent.redalert.intel import EnemyIntel  # noqa: E402
from agent.redalert.scouting import ScoutCorps  # noqa: E402


def _their_base(*types):
    return [{"id": 80 + i, "type": t, "x": 36 + (i % 3) * 3, "y": 36 + (i // 3) * 3, "hp": 100} for i, t in enumerate(types)]


def test_one_refinery_and_a_war_factory_early_reads_as_a_rush():
    i = EnemyIntel()
    i.update(_view(_t=110.0, visible_enemy_buildings=_their_base("fact", "powr", "tent", "proc", "weap")))
    assert i.opening() == "rush" and i.rushing(_view(_t=110.0)) and not i.rushing(_view(_t=400.0))
    j = EnemyIntel()
    j.update(_view(_t=110.0, visible_enemy_buildings=_their_base("fact", "powr", "tent", "proc", "proc")))
    assert j.opening() == "economy"
    k = EnemyIntel()
    k.update(_view(_t=110.0, visible_enemy_buildings=_their_base("fact", "powr")))
    assert k.opening() is None                                               # haven't seen enough yet


def test_two_early_scouts_round_the_back_of_their_base_left_and_right():
    corps = ScoutCorps()
    v = dataclasses.replace(_view(_t=60.0), bounds=(0, 0, 96, 96))         # base [4,4], enemy [40,40], room behind
    left, right = corps.circuit(v, 1), corps.circuit(v, -1)
    assert left[0] != right[0] and left[1] != right[1]                      # one each side
    for route in (left, right):
        back = route[-1]
        assert 8 ** 2 <= (back[0] - 40) ** 2 + (back[1] - 40) ** 2 <= 13 ** 2       # round their base...
        assert (back[0] - 4) ** 2 + (back[1] - 4) ** 2 > (40 - 4) ** 2 + (40 - 4) ** 2 - 300   # ...at its far side
    rifles = [{"id": 30 + i, "type": "e1", "x": 5, "y": 5, "hp": 100, "idle": True} for i in range(3)]
    v = _view(_t=60.0, units=rifles)
    corps.update(v)
    assert set(corps.roles) == {"left", "right"} and corps._circling == {"left", "right"}
    near = [{"id": 80, "type": "e1", "x": 6, "y": 6, "hp": 100}]
    o = corps.orders(dataclasses.replace(v, enemy_units=read(_raw(visible_enemies=near)).enemy_units))
    assert {tuple(x["to"]) for x in o} == {left[0], right[0]}                # no backing off on the circuit


def test_after_the_circuit_a_scout_holds_a_picket_post_and_backs_off_from_fighters():
    corps = ScoutCorps()
    corps.roles = {"left": 30}
    post = corps.post(_view(_t=400.0), "left")
    at_post = [{"id": 30, "type": "e1", "x": post[0], "y": post[1], "hp": 100, "idle": True}]
    near = [{"id": 80, "type": "e1", "x": post[0] + 2, "y": post[1] + 2, "hp": 100}]
    o = corps.orders(_view(_t=400.0, units=at_post, visible_enemies=near))
    assert o and o[0]["cmd"] == "move" and o[0]["to"] != list(post)          # backs off toward home
    posts = [corps.post(_view(_t=400.0), r) for r in ("left", "p0", "p1", "right")]
    assert len(set(posts)) == 4                                              # spread across the approach


def test_mid_game_pickets_and_a_regular_base_check():
    corps = ScoutCorps()
    many = [{"id": 30 + i, "type": "e1", "x": 5, "y": 5, "hp": 100, "idle": True} for i in range(14)]
    corps.update(_view(_t=400.0, units=many))
    assert set(corps.roles) == {"left", "right", "p0", "p1"}
    assert corps._circling                                                    # a base check started
    corps._circling.clear()
    corps.update(_view(_t=450.0, units=many))
    assert not corps._circling                                                # not again until 150 s later
    corps.update(_view(_t=551.0, units=many))
    assert corps._circling


def test_a_scout_killed_out_front_raises_the_alarm():
    corps = ScoutCorps()
    corps.roles = {"left": 30}
    corps.update(_view(_t=100.0, units=[{"id": 30, "type": "e1", "x": 30, "y": 30, "hp": 100}]))
    corps.update(_view(_t=105.0))
    assert corps.alarm(_view(_t=110.0)) and not corps.alarm(_view(_t=140.0))


def test_a_kennel_follows_the_war_factory_when_dogs_exist():
    done = [{"id": 10, "type": "fact"}, {"id": 11, "type": "powr"}, {"id": 12, "type": "barr"}, {"id": 13, "type": "proc"},
            {"id": 14, "type": "proc"}, {"id": 15, "type": "powr"}, {"id": 16, "type": "weap"}, {"id": 17, "type": "proc"},
            {"id": 18, "type": "fix"}]
    v = _view(buildings=done, economy={"cash": 300}, available_production=["powr", "barr", "proc", "weap", "kenn", "dog", "e1"])
    assert buildplan.next_build_orders(v) == [{"cmd": "build", "item": "kenn"}]


def test_the_reserve_covers_what_is_owed_and_the_next_step():
    q = [{"item": "proc", "queue": "Building", "ready": False, "progress": 0.5}]
    v = _view(buildings=[{"id": 10, "type": "fact"}, {"id": 11, "type": "powr"}, {"id": 12, "type": "tent"}], production=q)
    assert buildplan.build_reserve(v) == 700 + 1400                           # half the refinery + the second refinery


def test_the_army_attacks_in_a_semi_concave_of_groups():
    units = [{"id": 30 + i, "type": "e1", "x": 20 + (i % 4), "y": 20 + i // 4, "hp": 100} for i in range(9)]
    v = _view(_t=400.0, buildings=BASE, units=units, visible_enemy_buildings=[{"id": 90, "type": "fact", "x": 40, "y": 40, "hp": 100}])
    o = executor._army_orders(v, Mode.ALL_IN)
    assert len(o) == 3 and all(x["cmd"] == "attack_move" for x in o)
    spots = [tuple(x["to"]) for x in o]
    assert len(set(spots)) == 3 and sorted(len(x["units"]) for x in o) == [3, 3, 3]


def test_an_attack_call_on_even_trades_is_checked_again_before_committing():
    tanks = [{"id": 30 + i, "type": "3tnk", "x": 10, "y": 10, "hp": 100} for i in range(6)]
    ahead = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    ahead.choose_action(_state(_t=400.0, buildings=BASE, units=tanks,
                               military={"army_value": 5000, "visible_enemy_value": 1000, "kills_cost": 3000}), CTX)
    assert ahead._mode is Mode.ALL_IN                                         # 3000 up on trades: no second look
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    strong = {"army_value": 5000, "visible_enemy_value": 1000}
    dec = agent.choose_action(_state(_t=400.0, buildings=BASE, units=tanks, military=strong), CTX)
    assert agent._mode is not Mode.ALL_IN and agent._confirm_at == 400.0      # looking first
    probe = agent._probe_point(read(_raw(_t=400.0, buildings=BASE, units=tanks)))
    assert probe is not None and any(x["cmd"] == "attack_move" and abs(x["to"][0] - probe[0]) <= 4
                                     for x in dec.action["orders"])           # edging forward to see
    worse = {"army_value": 5000, "visible_enemy_value": 4800}                 # the fog lifts: it's even
    agent.choose_action(_state(_t=402.0, buildings=BASE, units=tanks, military=worse), CTX)
    assert agent._mode is not Mode.ALL_IN and agent._confirm_at is None       # called off
    agent.choose_action(_state(_t=440.0, buildings=BASE, units=tanks, military=strong), CTX)   # past the mode dwell
    agent.choose_action(_state(_t=444.5, buildings=BASE, units=tanks, military=strong), CTX)
    assert agent._mode is Mode.ALL_IN                                         # still strong 4 s on: go


# -- simplification step 1: corrections ---------------------------------------------------------

def test_squad_units_dont_count_when_checking_the_army_regrouped():
    stage = scout.choke_cell(_view())
    home = [{"id": 30 + i, "type": "e1", "x": stage[0], "y": stage[1], "hp": 100} for i in range(3)]
    away = [{"id": 40 + i, "type": "e1", "x": 40, "y": 40, "hp": 100} for i in range(4)]       # scouts, far out
    v = _view(units=home + away)
    assert not brain._regrouped(v, None)
    assert brain._regrouped(v, None, frozenset(u["id"] for u in away))


def test_the_mcv_escort_beats_the_attack_look_ahead_for_the_massing_point():
    class Escort:
        stage_override = (20, 8)
        def refinery_job(self, view): return None
        def needs_defense(self, view): return None
        def refinery_ore(self, view): return None
        def defense_site(self, view): return None
        def orders(self, view): return []
    tanks = [{"id": 30 + i, "type": "1tnk", "x": 5, "y": 5, "hp": 100, "idle": True} for i in range(4)]
    out = executor.plan(_view(_t=400.0, buildings=BASE, units=tanks), Mode.BUILD, expansion=Escort(), probe=(30, 30))
    moves = [o for o in out if o["cmd"] == "attack_move"]
    assert moves and all((o["to"][0] - 20) ** 2 + (o["to"][1] - 8) ** 2 <= 6 ** 2 for o in moves)


def test_a_confirmation_made_during_the_anti_flip_window_starts_the_attack_when_it_ends():
    tanks = [{"id": 30 + i, "type": "3tnk", "x": 10, "y": 10, "hp": 100} for i in range(6)]
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent._mode, agent._mode_since = Mode.BUILD, 400.0                      # just changed: window until 425
    strong = {"army_value": 5000, "visible_enemy_value": 1000}
    for t in (405.0, 410.0, 420.0):
        agent.choose_action(_state(_t=t, buildings=BASE, units=tanks, military=strong), CTX)
        assert agent._mode is Mode.BUILD
    assert agent._confirm_at == 405.0                                       # confirmed, held by the window
    agent.choose_action(_state(_t=426.0, buildings=BASE, units=tanks, military=strong), CTX)
    assert agent._mode is Mode.ALL_IN                                       # at once, no second look


def test_the_test_rusher_opens_with_one_refinery():
    assert testrush.RUSH_BUILD_ORDER.count("proc") == 1


# -- simplification step 2: the commander owns the decision -------------------------------------

from agent.redalert.commander import Commander  # noqa: E402


def _strong(t, **kw):
    tanks = [{"id": 30 + i, "type": "3tnk", "x": 10, "y": 10, "hp": 100} for i in range(6)]
    return _view(_t=t, buildings=BASE + [{"id": 12, "type": "weap", "x": 4, "y": 8}], units=tanks,
                 military={"army_value": 5000, "visible_enemy_value": 1000, **kw})


def test_the_commander_confirms_on_even_trades_and_goes_at_once_when_ahead():
    c = Commander()
    c.mode = Mode.BUILD
    assert c.decide(_strong(400.0)) is Mode.BUILD and c.confirming           # looking first
    assert c.decide(_strong(404.0)) is Mode.ALL_IN and not c.confirming       # held: committed
    ahead = Commander()
    ahead.mode = Mode.BUILD
    assert ahead.decide(_strong(400.0, kills_cost=3000)) is Mode.ALL_IN       # clearly ahead: no look


def test_the_commander_escort_and_forced_modes():
    c = Commander()
    c.mode = Mode.BUILD
    edge = _strong(400.0, kills_cost=3000, army_value=2000)                   # 2:1, not overwhelming
    assert c.decide(edge, escorting=True) is Mode.BUILD                       # the MCV first
    r = Commander()
    r.mode = Mode.BUILD
    assert r.decide(_view(_t=100.0, buildings=BASE), rushing=True, forced=Mode.ALL_IN) is Mode.ALL_IN


def test_the_commander_keeps_the_bookkeeping():
    c = Commander()
    c.mode = Mode.BUILD
    c.decide(_strong(400.0, kills_cost=3000))
    assert c.changed and c.mode_since == 400.0 and c.attack_army == 5000 and c.attacks == 1
    behind = _view(_t=430.0, buildings=BASE, units=[{"id": 30, "type": "3tnk", "x": 10, "y": 10, "hp": 100}],
                   military={"army_value": 800, "visible_enemy_value": 5000, "kills_cost": 3000})
    assert c.decide(behind) is Mode.RETREAT and c.retreated_at == 430.0 and c.retreat_army == 800


# -- the advisor is aligned with the strategy -----------------------------------------------------

def test_the_advisor_schema_has_only_fields_the_rules_use():
    props = set(ra_advisor.ADVICE_SCHEMA["properties"])
    assert props == {"posture", "attack_route", "expand", "army_target", "raid_target_id", "raid_target_cell",
                     "defense_at", "plan", "reasoning"}
    for word in ("double-refinery", "rush", "concave", "second prong", "DEFEND", "FINISH", "4-second look"):
        assert word in ra_advisor.SYSTEM_PROMPT, word


def test_the_advisor_sees_what_the_rules_see():
    seen = {}

    class Spy:
        calls = 0
        def poll(self, view, situation):
            seen.update(situation)
            return None
        def close(self):
            pass

    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0, advisor_factory=lambda log, on_advice: Spy())
    agent.choose_action(_state(_t=300.0, buildings=BASE, units=_raid_army(2, 6)), CTX)
    for key in ("mode", "confirming_attack", "massing_point", "our_unit_value", "enemy_unit_value_known",
                "enemy_turrets_known", "raid_squad", "scouts", "intel"):
        assert key in seen, key
    assert set(seen["intel"]) == {"enemy_opening", "threat_alarm", "their_base_seen_at"}


# -- placement keeps off the trucks' roads and away from units (a2952a6d) -------------------------

def test_no_building_on_the_road_from_a_refinery_to_its_ore():
    blds = [{"id": 10, "type": "fact", "x": 4, "y": 4}, {"id": 13, "type": "proc", "x": 10, "y": 4}]
    v = dataclasses.replace(_view(buildings=blds), ore_cells=_ore_patch(20, 4, 3, 3))
    road = buildplan._truck_roads(v)
    assert all(any((x, y) in road for y in (4, 5, 6)) for x in range(13, 20))   # spans the gap to the ore
    assert buildplan._clashes("powr", (15, 4), v)


def test_a_spot_with_units_beside_it_loses_to_a_clear_one():
    blds = [{"id": 10, "type": "fact", "x": 10, "y": 10}]
    v = _view(buildings=blds)
    first = buildplan.grow_spot(v, (1.0, 0.0), "powr")
    truck = [{"id": 40, "type": "harv", "x": first[0] - 1, "y": first[1], "hp": 100}]   # beside, not on it
    second = buildplan.grow_spot(_view(buildings=blds, units=truck), (1.0, 0.0), "powr")
    assert second != first
