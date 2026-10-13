"""Red Alert on real maps: replays of our own bases from match logs.

Each fixture (tests/fixtures/redalert/*.json) holds a map picture and legend
from a real match (Sidestep, Trapped from both sides), our base centre, the
enemy spawn, and our buildings at three moments ("cells": top-left or as
reported). Views are built through the real parser (observe.read), from the
cells the server would report.

Placements are judged by OpenRA's rule as our placement audit measured it,
written out here independently of buildplan.py: the footprint lies on
buildable ground, overlaps no building, and lies within one free cell of a
building that extends the buildable area (not a turret), gaps counted
between solid cells: OpenRA's IsCloseEnoughToBase with the shapes from
mods/ra/rules/structures.yaml (proc _X_ / xxx / X== / ===, fix _+_ / +++ / _+_).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agent.redalert import buildplan, scout, terrain
from agent.redalert import config as C
from agent.redalert.observe import centre_offset, read
from agent.redalert.scouting import ScoutCorps

FIXTURES = sorted(pathlib.Path(__file__).parent.glob("fixtures/redalert/*.json"))
NO_AREA = {"pbox", "hbox", "gun", "ftur", "tsla", "agun", "sam"}


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _view(fx: dict, state: dict, name: str):
    pic = fx["picture"]
    ox, oy = pic["origin"]
    bounds = [ox, oy, len(pic["rows"][0]), len(pic["rows"])]
    map_id = f"fixture-{name}"
    raw = {
        "phase": "playing", "is_terminal": False, "state_version": 1, "pacing": {"mode": "realtime"},
        "observation": {
            "tick": int(state["t"] * 25), "economy": {"cash": 2000, "power_provided": 300, "power_drained": 100},
            "military": {"army_value": 0, "visible_enemy_value": 0, "kills_cost": 0, "deaths_cost": 0},
            "units": [], "visible_enemies": [], "visible_enemy_buildings": [], "production": [], "events": [],
            "buildings": [_as_reported(fx, i, t, x, y) for i, (t, x, y) in enumerate(state["buildings"])],
            "available_production": ["powr", "barr", "tent", "proc", "weap", "fix", "dome", "ftur", "pbox", "e1"],
            "base_center": fx["base_center"],
        },
        "time": {"game_seconds": state["t"], "remaining_seconds": 1200 - state["t"]},
        "map": {"id": map_id, "bounds": bounds},
    }
    config = {"rules": {"maps": {map_id: {"layout": {**pic, "legend": fx["legend"]}, "bounds": bounds,
                                          "spawns": [fx["base_center"], fx["enemy_spawn"]]}}}}
    return read(raw, config)


def _as_reported(fx: dict, i: int, kind: str, x: int, y: int) -> dict:
    """The building as the server reports it: its centre cell. Fixtures from
    logs written after the centre-cell fix hold top-left cells ("cells":
    "top_left"); older ones the reported cells."""
    if fx.get("cells") == "top_left":
        dx, dy = centre_offset(kind)
        x, y = x + dx, y + dy
    return {"id": i + 1, "type": kind, "x": x, "y": y}


def _rect(kind, cell):
    w, h = C.FOOTPRINTS.get(kind, C.DEFAULT_FOOTPRINT)
    return cell[0], cell[1], cell[0] + w - 1, cell[1] + h - 1


def _legal(view, item: str, spot) -> str | None:
    """Why OpenRA would move ``item`` placed at ``spot``, or None if legal."""
    cells = _solid(item, spot)               # empty corners aren't checked (CanPlaceBuilding)
    bad = view.unbuildable_cells | view.blocked_cells | frozenset(view.ore_cells)
    if any(c in bad for c in cells):
        return "unbuildable ground"
    near = False
    mine = _solid(item, spot)
    for b in view.buildings:
        theirs = _solid(b.type, b.cell)
        gap = min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for p in mine for q in theirs) - 1
        if gap < 0:
            return f"overlaps {b.type}"
        if b.type not in NO_AREA and gap <= 1:
            near = True
    return None if near else "out of reach"


def _solid(kind, cell):
    r = _rect(kind, cell)
    empty = {"proc": {(0, 0), (2, 0)}, "fix": {(0, 0), (2, 0), (0, 2), (2, 2)}}.get(kind, set())
    empty = {(r[0] + i, r[1] + j) for i, j in empty}
    return [(x, y) for x in range(r[0], r[2] + 1) for y in range(r[1], r[3] + 1) if (x, y) not in empty]


CASES = [(path, i) for path in FIXTURES for i in range(3)]


@pytest.mark.parametrize("path,i", CASES, ids=[f"{p.stem}@{i}" for p, i in CASES])
def test_every_role_places_legally_on_real_maps(path, i):
    fx = _load(path)
    view = _view(fx, fx["states"][i], path.stem)
    rax = "barr" if fx["faction"] in ("russia", "ukraine") else "tent"
    for item in ("powr", rax, "proc", "weap", "fix", "dome", "ftur"):
        spot = buildplan.placement_cell(view, item)
        assert spot is not None, (item, "no spot")
        assert _legal(view, item, spot) is None, (item, spot, _legal(view, item, spot))


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_refineries_on_real_maps_can_walk_to_their_ore(path):
    fx = _load(path)
    checked = 0
    for i, state in enumerate(fx["states"]):
        view = _view(fx, state, path.stem)
        ore = buildplan.open_ore(view)
        if ore is None:
            continue                     # every field in reach is full: the base crawls outward instead
        if not buildplan.refinery_room(view):
            continue                     # no spot the trucks walk from (base boxed in): no refinery now
        spot = buildplan.refinery_spot(view)
        assert _legal(view, "proc", spot) is None, (i, spot)
        field = terrain.field_cells(view, ore, C.EXPAND_CLUSTER_CELLS * 1.5)
        steps = terrain.steps_from_rect(terrain.walk_distances(view, field, 12), _rect("proc", spot))
        assert steps is not None and steps <= C.REFINERY_ORE_WALK_STEPS, (i, spot, steps)
        checked += 1
    assert checked >= 1


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_the_massing_point_is_open_ground_toward_the_enemy(path):
    fx = _load(path)
    view = _view(fx, fx["states"][2], path.stem)
    stage = scout.choke_cell(view)
    base, enemy = fx["base_center"], fx["enemy_spawn"]
    assert stage not in view.blocked_cells
    assert all(_rect(b.type, b.cell)[0] > stage[0] or _rect(b.type, b.cell)[2] < stage[0]
               or _rect(b.type, b.cell)[1] > stage[1] or _rect(b.type, b.cell)[3] < stage[1] for b in view.buildings)
    d = lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5  # noqa: E731
    assert d(stage, enemy) < d(base, enemy)                         # in front of the base
    assert stage in terrain.walk_distances(view, [tuple(base)], 60)  # and the army can walk there


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_scout_circuits_and_posts_are_reachable_on_real_maps(path):
    fx = _load(path)
    view = _view(fx, fx["states"][0], path.stem)
    walk = terrain.walk_distances(view, [tuple(fx["base_center"])], 250)
    corps = ScoutCorps()
    for side in (1, -1):
        for radius in (C.CIRCUIT_OUTER_CELLS, C.CIRCUIT_INNER_CELLS):
            for point in corps.circuit(view, side, radius):
                assert point in walk, (side, radius, point)
    for role in ("left", "p0", "p1", "right"):
        assert corps.post(view, role) in walk, role


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_turrets_stand_on_the_line_to_the_enemy(path):
    import math

    fx = _load(path)
    state = dict(fx["states"][2])
    state["buildings"] = [b for b in state["buildings"] if b[0] not in NO_AREA]   # no turret yet
    view = _view(fx, state, path.stem)
    base = tuple(fx["base_center"])
    enemy = tuple(fx["enemy_spawn"])
    bearing = lambda c: math.degrees(math.atan2(c[1] - base[1], c[0] - base[0]))  # noqa: E731
    guard = buildplan.factory_guard_spot(view, "ftur")
    if guard is not None:                                  # the first guards the war factory, on its outward side
        weap = min((tuple(b[1:3]) for b in state["buildings"] if b[0] == "weap"),
                   key=lambda c: (c[0] - guard[0]) ** 2 + (c[1] - guard[1]) ** 2)
        assert _legal(view, "ftur", guard) is None and buildplan.placement_cell(view, "ftur") == guard
        assert (guard[0] - weap[0]) ** 2 + (guard[1] - weap[1]) ** 2 <= C.FACTORY_GUARD_CELLS ** 2
        assert (guard[0] - base[0]) ** 2 + (guard[1] - base[1]) ** 2 > (weap[0] - base[0]) ** 2 + (weap[1] - base[1]) ** 2
        state["buildings"] = state["buildings"] + [["ftur", guard[0], guard[1]]]
        view = _view(fx, state, path.stem)
    for _ in range(5):                                                            # every turret we'd build on the line
        if buildplan.perimeter_spot(view, base, enemy, "ftur") is None:
            break
        spot = buildplan.placement_cell(view, "ftur")
        assert _legal(view, "ftur", spot) is None
        assert abs((bearing(spot) - bearing(enemy) + 180) % 360 - 180) <= 25, spot   # on the line to the enemy
        state["buildings"] = state["buildings"] + [["ftur", spot[0], spot[1]]]
        view = _view(fx, state, path.stem)


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_chokes_are_found_from_the_terrain_on_real_maps(path):
    fx = _load(path)
    view = _view(fx, fx["states"][0], path.stem)
    found = terrain.chokes(view, tuple(fx["enemy_spawn"]), tuple(fx["base_center"]))
    assert found and min(w for _, w in found) <= C.CHOKE_MAX_WIDTH         # the middle passages
    for cell, _ in found:
        assert cell not in view.blocked_cells


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_the_concave_holds_the_mouth_of_the_choke_on_real_maps(path):
    from agent.redalert import executor

    fx = _load(path)
    view = _view(fx, fx["states"][1], path.stem)
    base = tuple(fx["base_center"])
    if scout.choke_stage(view) is None:                                           # open ground (Shadowfiend)
        point, facing = scout.concave_point(view)
        assert point not in view.blocked_cells
        slots = executor.concave_slots(view, point, facing, 20)
        assert len(slots) >= 15 and not any(s in view.blocked_cells for s in slots)
        return
    choke = scout.choke_stage(view)[1]
    assert terrain.passage_width(view, choke) <= C.CHOKE_STAGE_MAX_WIDTH         # Sidestep's gap, Trapped's passage
    point, facing = scout.concave_point(view)
    assert facing == choke
    d = lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5              # noqa: E731
    assert d(point, choke) < d(scout.choke_cell(view), choke) - 3                   # up at the choke, not back
    ux, uy = (choke[0] - point[0]) / d(point, choke), (choke[1] - point[1]) / d(point, choke)
    kill = (point[0] + ux * C.FORMATION_FOCUS_AHEAD, point[1] + uy * C.FORMATION_FOCUS_AHEAD)
    assert d(kill, choke) <= 3                                                    # the kill point at its mouth
    assert d(point, base) < d(choke, base)                                        # on our side of it
    slots = executor.concave_slots(view, point, facing, 20)
    assert len(slots) >= 15 and not any(s in view.blocked_cells for s in slots)


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_scouts_and_the_army_go_round_the_choke_and_the_front_of_their_base(path):
    from agent.redalert.observe import enemy_spawn_guess

    fx = _load(path)
    view = _view(fx, fx["states"][0], path.stem)
    base, enemy = tuple(fx["base_center"]), tuple(enemy_spawn_guess(view))
    corps = ScoutCorps()
    avoid, zone = corps.avoid(view), terrain.choke_zone(view)
    assert enemy not in zone and base not in zone
    assert zone or "shadowfiend" in path.stem                                   # open ground has none
    for role in ("left", "right", "p0", "p1"):                                   # posts and looks stand outside it
        assert corps.post(view, role) not in avoid, role
    for role in ("left", "right"):
        assert corps.peek_point(view, role) not in avoid, role
    for side, radius in ((1, C.CIRCUIT_OUTER_CELLS), (-1, C.CIRCUIT_INNER_CELLS)):   # each circuit, walked in legs
        here, walked = base, []
        for wp in corps.circuit(view, side, radius):
            route = terrain.route_around(view, here, wp, avoid, C.SCOUT_MAX_DETOUR)
            for leg in (terrain.legs(route, C.SCOUT_LEG_CELLS) if route else [wp]):
                walked += terrain.walk_path(view, here, leg)
                here = leg
        assert sum(1 for c in walked if c in zone) <= 3, (side, [c for c in walked if c in zone])
    stage = scout.choke_cell(view)
    route = terrain.route_around(view, stage, enemy, zone, C.DETOUR_MAX_RATIO)
    if "trapped" in path.stem:                                                  # Trapped: round the gap, 1.42x
        assert route and not any(c in zone for c in route[3:-3])


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_nothing_of_ours_walks_down_the_choke_into_their_concave(path):
    """ae03adbd (Sidestep): scouts walked the gap to their army's spot and
    died every 15-30 s, each hit sent two more units to look, and their
    concave at the far mouth pulled the whole army in (6250 lost in 30 s)."""
    import dataclasses

    from agent.redalert import executor

    fx = _load(path)
    view = _view(fx, fx["states"][1], path.stem)
    stage = scout.enemy_staging(view)
    corps = ScoutCorps()
    point = scout.concave_point(view)[0]
    zone = terrain.choke_zone(view)
    unit = type("U", (), {"id": 7, "cell": point, "idle": True})()
    moves, arrived = corps._go(view, unit, stage)
    if stage in corps.unreachable:
        assert moves == [] and arrived                                            # nowhere safe: it doesn't go
    else:
        legs_ = corps._legs[7][1]
        walked, here = [], point
        for leg in legs_:
            walked += terrain.walk_path(view, here, leg)
            here = leg
        deep = [c for c in walked if c in zone and (c[0] - legs_[-1][0]) ** 2 + (c[1] - legs_[-1][1]) ** 2
                > C.SCOUT_CHOKE_ENTRY_CELLS ** 2 and (c[0] - point[0]) ** 2 + (c[1] - point[1]) ** 2 > 9]
        assert len(deep) <= 4, deep[:8]                       # round (its legs may clip the zone's padded edge), not down the gap
    # Their army at the far mouth is not the army's fight at home.
    foes = tuple(type("A", (), {"id": 900 + k, "type": "3tnk", "cell": stage, "wreck": False, "hp": 100})()
                 for k in range(4))
    ours = [type("F", (), {"id": 100 + k, "type": "e1", "cell": point})() for k in range(6)]
    there = dataclasses.replace(view, enemy_units=foes)
    assert executor.army_contact(there, ours, defending=False) == []



@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_the_attack_marches_on_one_road_in_short_legs(path):
    """b9145367 (Trapped): half the army, re-aimed mid-march, walked round the
    other side of the rocks. Every route is the walking road in legs a few
    steps apart, so no unit can take a road of its own."""
    import dataclasses

    from agent.redalert.agent import RedAlertAgent
    from agent.redalert.modes import Mode
    from agent.redalert.observe import enemy_spawn_guess

    fx = _load(path)
    view = _view(fx, fx["states"][1], path.stem)
    enemy = tuple(enemy_spawn_guess(view))
    fact = type("B", (), {"id": 900, "type": "fact", "cell": enemy, "wreck": False, "hp": 100})()
    there = dataclasses.replace(view, known_enemy_buildings=[fact])
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent._mode = Mode.ALL_IN
    for flank in (False, True):
        legs_ = agent._pick_route(there, None, flank=flank) + [enemy]
        here = tuple(fx["base_center"])
        for leg in legs_:
            step = terrain.walk_path(there, here, leg)
            assert step and len(step) <= C.ARMY_LEG_CELLS + 4, (flank, here, leg, len(step))
            here = leg


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_attack_groups_spread_only_where_they_walk_without_a_detour(path):
    import dataclasses

    from agent.redalert import executor
    from agent.redalert.observe import read as _rd

    fx = _load(path)
    view = _view(fx, fx["states"][1], path.stem)
    target = scout.enemy_staging(view)
    far = scout.concave_point(view)[0]
    units = _rd({"phase": "playing", "is_terminal": False, "state_version": 1, "pacing": {"mode": "realtime"},
                 "observation": {"tick": 1, "units": [{"id": 100 + k, "type": "e1", "x": far[0], "y": far[1], "hp": 100}
                                                      for k in range(12)]},
                 "time": {"game_seconds": 600}}, None).units
    v = dataclasses.replace(view, units=units)
    ids = [u.id for u in units]
    points = executor._attack_points(v, ids, target)
    for p in points:
        route = terrain.walk_path(v, far, p)
        assert route and len(route) <= C.ATTACK_POINT_MAX_DETOUR * ((p[0] - far[0]) ** 2 + (p[1] - far[1]) ** 2) ** 0.5 + 4


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_an_attack_from_our_concave_takes_the_lane_not_the_corridor(path):
    """d1615ac6 (Sidestep): from our concave at the corridor's mouth every
    lane looked long, so the 17:31 attack went down the gap into their
    concave. With a way round between the bases, the march takes it."""
    import dataclasses

    from agent.redalert.agent import RedAlertAgent
    from agent.redalert.modes import Mode
    from agent.redalert.observe import enemy_spawn_guess
    from agent.redalert.observe import read as _rd

    fx = _load(path)
    view = _view(fx, fx["states"][1], path.stem)
    zone = terrain.choke_zone(view)
    if not zone or terrain.choke_between(view):
        return                                                                   # no corridor, or no way round it
    enemy, point = tuple(enemy_spawn_guess(view)), scout.concave_point(view)[0]
    units = _rd({"phase": "playing", "is_terminal": False, "state_version": 1, "pacing": {"mode": "realtime"},
                 "observation": {"tick": 1, "units": [{"id": 100 + k, "type": "e1", "x": point[0], "y": point[1], "hp": 100}
                                                      for k in range(10)]},
                 "time": {"game_seconds": 600}}, None).units
    fact = type("B", (), {"id": 900, "type": "fact", "cell": enemy, "wreck": False, "hp": 100})()
    there = dataclasses.replace(view, units=units, known_enemy_buildings=[fact])
    agent = RedAlertAgent(log=lambda *_: None, clock=lambda: 0.0)
    agent._mode = Mode.ALL_IN
    walked, here = [], point
    for leg in agent._pick_route(there, None, flank=False) + [enemy]:
        walked += terrain.walk_path(view, here, leg)
        here = leg
    d2 = lambda a, b: (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2                     # noqa: E731
    deep = [c for c in walked if c in zone and d2(c, point) > 64 and d2(c, enemy) > 20 ** 2]
    assert len(deep) <= 4, deep[:8]
