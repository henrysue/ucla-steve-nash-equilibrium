"""Walking distance on the map grid, around cliffs, rock, water and trees.

Straight-line distance lies near cliffs: a refinery "3 cells from ore" with a
cliff between them can't be served, and its harvester wanders off. So the
checks that matter for harvesting — where an expansion deploys, where its
refinery goes, which ore a refinery's trucks work — use the number of steps a
unit actually walks (8 directions, blocked cells from the map picture).
"""

from __future__ import annotations

from collections import deque

from .observe import View, dist2

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


def choke_zone(view: View) -> frozenset:
    """The chokes on the road between the bases, padded: every cell within
    ``CHOKE_AVOID_CELLS`` of a passage ``CHOKE_STAGE_MAX_WIDTH`` across or
    narrower on the walking path from our base to theirs, at least
    ``CHOKE_AVOID_BASE_CELLS`` from either base (a base's own exit isn't one).
    Armies and scouts go round it when a way round isn't much longer (match
    82bdca3e: the 6:12 attack went through Trapped's gap into their concave
    and lost 4600 in 50 s; the scouts' left route ran through it too).
    Cached per map and spawns."""
    from . import config as C
    from .observe import enemy_spawn_guess

    base, enemy = view.base_center, enemy_spawn_guess(view)
    if base is None or enemy is None or not view.blocked_cells:
        return frozenset()
    key = ("zone", id(view.blocked_cells), tuple(base), tuple(enemy))
    hit = _ZONES.get(key)
    if hit is not None and hit[0] is view.blocked_cells:
        return hit[1]
    far2 = C.CHOKE_AVOID_BASE_CELLS ** 2
    narrow = [c for c in walk_path(view, base, enemy)
              if dist2(c, base) > far2 and dist2(c, enemy) > far2
              and passage_width(view, c) <= C.CHOKE_STAGE_MAX_WIDTH]
    r = C.CHOKE_AVOID_CELLS
    zone = frozenset((c[0] + i, c[1] + j) for c in narrow for i in range(-r, r + 1) for j in range(-r, r + 1))
    if len(_ZONES) > 16:
        _ZONES.clear()
    _ZONES[key] = (view.blocked_cells, zone)
    return zone


_ZONES: dict = {}


def choke_between(view: View) -> bool:
    """The only road between the bases runs through a choke: there is a
    choke zone on it and no way round within ``DETOUR_MAX_RATIO``. An
    attack then walks out of the choke into their concave, one column at a
    time. Cached per map and spawns."""
    from . import config as C
    from .observe import enemy_spawn_guess

    zone = choke_zone(view)
    base, enemy = view.base_center, enemy_spawn_guess(view)
    if not zone or base is None or enemy is None:
        return False
    key = ("between", id(view.blocked_cells), tuple(base), tuple(enemy))
    hit = _ZONES.get(key)
    if hit is not None and hit[0] is view.blocked_cells:
        return hit[1]
    direct = walk_path(view, tuple(base), tuple(enemy))
    through = bool(direct) and any(c in zone for c in direct) and \
        route_around(view, base, enemy, zone, C.DETOUR_MAX_RATIO) is None
    _ZONES[key] = (view.blocked_cells, through)
    return through


def route_around(view: View, start, goal, avoid, max_ratio: float) -> list[tuple[int, int]] | None:
    """The walking path from ``start`` to ``goal`` that keeps out of
    ``avoid`` (cells within 3 of either end excepted), when the straight road
    runs through it and the way round is at most ``max_ratio`` as long; None
    when the straight road is clear (go straight) or there is no acceptable
    way round."""
    start, goal = tuple(start), tuple(goal)
    direct = walk_path(view, start, goal)
    if not direct or not any(c in avoid for c in direct):
        return None
    start, goal = direct[0], direct[-1]
    if view.bounds is not None:
        bx, by, bw, bh = view.bounds
        inside = lambda c: bx <= c[0] < bx + bw and by <= c[1] < by + bh  # noqa: E731
    else:
        inside = lambda c: True  # noqa: E731
    near = lambda c: dist2(c, start) <= 9 or dist2(c, goal) <= 9  # noqa: E731
    limit = int(len(direct) * max_ratio) + 2
    prev = {start: None}
    depth = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        if depth[cur] >= limit:
            continue
        for dx, dy in _STEPS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if (nxt in prev or nxt in view.blocked_cells or not inside(nxt)
                    or (nxt in avoid and not near(nxt))):
                continue
            prev[nxt] = cur
            depth[nxt] = depth[cur] + 1
            queue.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def legs(path, every: int) -> list[tuple[int, int]]:
    """Waypoints along ``path`` every ``every`` steps, ending at its last cell
    (each leg short enough that a unit's own pathing can't stray back onto
    the road we avoided)."""
    if not path:
        return []
    points = [tuple(path[i]) for i in range(every, len(path) - 1, every)]
    return points + [tuple(path[-1])]
