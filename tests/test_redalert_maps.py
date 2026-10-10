"""Red Alert on real maps: replays of our own bases from match logs.

Each fixture (tests/fixtures/redalert/*.json) holds a map picture and legend
from a real match (Sidestep, Trapped from both sides), our base centre, the
enemy spawn, and our buildings at three moments ("cells": top-left or as
reported). Views are built through the real parser (observe.read), from the
cells the server would report.

Placements are judged by OpenRA's rule as our placement audit measured it,
written out here independently of buildplan.py: the footprint lies on
buildable ground, overlaps no building, and lies within one free cell of a
building that extends the buildable area (not a turret).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agent.redalert import buildplan, scout, terrain
from agent.redalert import config as C
from agent.redalert.observe import read
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
        w, h = C.FOOTPRINTS.get(kind, C.DEFAULT_FOOTPRINT)
        x, y = x + w // 2, y + h // 2
    return {"id": i + 1, "type": kind, "x": x, "y": y}


def _rect(kind, cell):
    w, h = C.FOOTPRINTS.get(kind, C.DEFAULT_FOOTPRINT)
    return cell[0], cell[1], cell[0] + w - 1, cell[1] + h - 1


def _legal(view, item: str, spot) -> str | None:
    """Why OpenRA would move ``item`` placed at ``spot``, or None if legal."""
    r = _rect(item, spot)
    cells = [(x, y) for x in range(r[0], r[2] + 1) for y in range(r[1], r[3] + 1)]
    bad = view.unbuildable_cells | view.blocked_cells | frozenset(view.ore_cells)
    if any(c in bad for c in cells):
        return "unbuildable ground"
    gaps = []
    for b in view.buildings:
        q = _rect(b.type, b.cell)
        gap = max(max(q[0] - r[2], r[0] - q[2]) - 1, max(q[1] - r[3], r[1] - q[3]) - 1)
        if gap < 0:
            return f"overlaps {b.type}"
        if b.type not in NO_AREA:
            gaps.append(gap)
    if not gaps or min(gaps) > 1:
        return f"out of reach (nearest gap {min(gaps) if gaps else None})"
    return None


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
        spot = buildplan._beside_ore(view, ore, "proc")
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
        for point in corps.circuit(view, side):
            assert point in walk, (side, point)
    for role in ("left", "p0", "p1", "right"):
        assert corps.post(view, role) in walk, role
