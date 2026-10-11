"""Walking distance on the map grid, around cliffs, rock, water and trees.

Straight-line distance lies near cliffs: a refinery "3 cells from ore" with a
cliff between them can't be served, and its harvester wanders off. So the
checks that matter for harvesting — where an expansion deploys, where its
refinery goes, which ore a refinery's trucks work — use the number of steps a
unit actually walks (8 directions, blocked cells from the map picture).
"""

from __future__ import annotations

from collections import deque

from .observe import View

_CACHE: dict[tuple, dict] = {}
_CACHE_MAX = 64
_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def walk_distances(view: View, sources, max_steps: int) -> dict[tuple[int, int], int]:
    """{cell: steps} for every cell within ``max_steps`` walking steps of any
    source cell (sources count as step 0 even if blocked, e.g. an ore mine)."""
    srcs = tuple(sorted(set(sources)))
    key = (id(view.blocked_cells), view.bounds, srcs, max_steps)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is view.blocked_cells:   # the same set, not a reused id
        return hit[1]
    blocked = view.blocked_cells
    if view.bounds is not None:
        bx, by, bw, bh = view.bounds
        inside = lambda c: bx <= c[0] < bx + bw and by <= c[1] < by + bh  # noqa: E731
    else:
        inside = lambda c: True  # noqa: E731
    dist = {c: 0 for c in srcs}
    queue = deque(srcs)
    while queue:
        cell = queue.popleft()
        d = dist[cell]
        if d >= max_steps:
            continue
        for dx, dy in _STEPS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in dist or nxt in blocked or not inside(nxt):
                continue
            dist[nxt] = d + 1
            queue.append(nxt)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = (view.blocked_cells, dist)
    return dist


def field_cells(view: View, ore, radius: float) -> list[tuple[int, int]]:
    """The ore cells of the field around ``ore`` (straight-line ``radius``)."""
    r2 = radius * radius
    return [c for c in view.ore_cells if (c[0] - ore[0]) ** 2 + (c[1] - ore[1]) ** 2 <= r2]


def ring(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """The cells touching a footprint (x0, y0, x1, y1) from outside."""
    x0, y0, x1, y1 = rect
    return [(x, y) for x in range(x0 - 1, x1 + 2) for y in range(y0 - 1, y1 + 2)
            if not (x0 <= x <= x1 and y0 <= y <= y1)]


def steps_from_rect(dist: dict, rect) -> int | None:
    """Fewest steps from a footprint's edge to the sources of ``dist``."""
    best = [dist[c] for c in ring(rect) if c in dist]
    return min(best) if best else None


def walk_path(view: View, start, goal, max_steps: int = 400) -> list[tuple[int, int]]:
    """The shortest walking path from ``start`` to ``goal`` (both cells), or []."""
    walk = walk_distances(view, [tuple(goal)], max_steps)
    if not walk:
        return []
    cur = tuple(start)
    if cur not in walk:
        cur = min(walk, key=lambda c: (c[0] - start[0]) ** 2 + (c[1] - start[1]) ** 2)
    path = [cur]
    while walk[cur] > 0:
        cur = min(((cur[0] + dx, cur[1] + dy) for dx, dy in _STEPS if (cur[0] + dx, cur[1] + dy) in walk),
                  key=lambda c: walk[c])
        path.append(cur)
    return path


def passage_width(view: View, cell, limit: int = 12) -> int:
    """How narrow the way is at ``cell``: the shortest walkable span through it
    (horizontal, vertical or diagonal, impassable ground or the map edge at
    both ends). A gap in a cliff line is narrow across at least one of them;
    open ground is wide in all four."""
    best = None
    for ax, ay in ((1, 0), (0, 1), (1, 1), (1, -1)):
        width = 1
        for sign in (1, -1):
            for k in range(1, limit + 1):
                c = (cell[0] + sign * ax * k, cell[1] + sign * ay * k)
                if c in view.blocked_cells:
                    break
                if view.bounds is not None:
                    x, y, w, h = view.bounds
                    if not (x <= c[0] < x + w and y <= c[1] < y + h):
                        break
                width += 1
        best = width if best is None else min(best, width)
    return best


def chokes(view: View, start, goal) -> list[tuple[tuple[int, int], int]]:
    """(cell, width) along the walking path from ``start`` to ``goal``: the
    narrowest points are chokes (:func:`passage_width`)."""
    path = walk_path(view, start, goal)
    return [(c, passage_width(view, c)) for c in path[3:-3]]
