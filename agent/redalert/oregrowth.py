"""Where ore will be: OpenRA's ore mines seed ore all game.

Every mine (``M`` on the map picture) seeds every 75 ticks (3 s): a random
walk from it skips full ore cells and adds ore to the first other cell
(SeedsResource, ResourceLayer). No field starts full, so it fills, then
creeps out at its edge. A Monte Carlo of exactly those rules on Shadowfiend
II, Sidestep and Trapped (harvesting ignored: an upper bound) put new ore
1 cell past the starting field by 5:00, 2 by 10:00-15:00 and rarely 3 by
20:00 — fields without a mine never grow. A building (or an MCV's yard)
asked for on ore is dropped by the platform next to the main yard instead,
so expansion spots keep clear of this ring (``ORE_GROWTH``).
"""

from __future__ import annotations

from . import config as C
from .observe import View

_CACHE: dict = {}


def growth_margin(seconds: float) -> int:
    """Cells past the starting field that may be ore by ``seconds``."""
    margin = 0
    for at, cells in C.ORE_GROWTH:
        if seconds >= at:
            margin = cells
    return margin


def mine_cells(view: View) -> frozenset:
    """The ore mines on the map picture."""
    pic = view.map_picture or {}
    rows, origin = pic.get("rows"), pic.get("origin")
    if not isinstance(rows, list) or not isinstance(origin, (list, tuple)) or len(origin) != 2:
        return frozenset()
    key = ("mines", id(rows))
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is rows:
        return hit[1]
    ox, oy = origin
    cells = frozenset((ox + j, oy + i) for i, row in enumerate(rows) if isinstance(row, str)
                      for j, ch in enumerate(row) if ch == "M")
    _CACHE[key] = (rows, cells)
    return cells


def predicted_ore(view: View, ahead: float = 0.0) -> frozenset:
    """Cells not ore at the start that may be by ``ahead`` seconds from now:
    within :func:`growth_margin` of a field that has a mine."""
    k = growth_margin(view.game_seconds + ahead)
    mines = mine_cells(view)
    if k <= 0 or not mines or not view.ore_cells:
        return frozenset()
    key = ("ring", id(view.ore_cells), id(view.map_picture.get("rows")), k)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is view.ore_cells:
        return hit[1]
    ore = set(view.ore_cells)
    seen: set = set()
    ring: set = set()
    for start in ore:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:                                  # one 8-connected field
            c = stack.pop()
            comp.append(c)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in ore and n not in seen:
                        seen.add(n)
                        stack.append(n)
        if not any(c in mines for c in comp):
            continue                                  # no mine: it never grows
        for c in comp:
            for dx in range(-k, k + 1):
                for dy in range(-k, k + 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n not in ore:
                        ring.add(n)
    out = frozenset(ring)
    if len(_CACHE) > 32:
        _CACHE.clear()
    _CACHE[key] = (view.ore_cells, out)
    return out
