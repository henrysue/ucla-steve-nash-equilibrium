"""Scouting and map positions: threats at our base, the staging point, the
roaming scout's patrol, and the hunt for the enemy's last buildings.

Threat detection is deliberately conservative (two enemies near a yard, or
an enemy MCV): a false alarm costs tempo. It expires when the threat goes.
"""

from __future__ import annotations

from . import config as C
from .observe import View, dist2

# Enemy unit types that are NOT a threat worth panicking over (a lone harvester
# or a single scout wandering by is normal, not cheese).
_NON_THREAT = set(C.HARVESTER)


def enemy_threat_near_base(view: View) -> list:
    """Enemy combat actors within ``CHEESE_NEAR_BASE_CELLS`` of our base.

    Counts units and a forward enemy MCV (a base-move/MCV-rush tell). Harvesters
    are ignored. Returns the offending actors so the caller can size the threat.
    """
    yards = our_yards(view)
    if not yards:
        return []
    radius2 = C.CHEESE_NEAR_BASE_CELLS ** 2

    def close(cell):
        return any(dist2(cell, y) <= radius2 for y in yards)

    near = [a for a in view.enemy_units if a.type not in _NON_THREAT and close(a.cell)]
    # A visible enemy MCV / construction yard close to us is itself cheese.
    near += [a for a in view.enemy_buildings if a.type in (C.MCV, C.CONSTRUCTION_YARD) and close(a.cell)]
    return near


def our_yards(view: View) -> list[tuple[int, int]]:
    """The base centre and every construction yard of ours (expansions
    included): an attack on any of them is an attack on the base."""
    yards = [view.base_center] if view.base_center is not None else []
    yards += [b.cell for b in view.buildings if b.type == C.CONSTRUCTION_YARD and b.cell is not None and not b.wreck]
    return yards


def early_under_attack(view: View) -> bool:
    """An ``under_attack`` event before we have a real army is a cheese tell."""
    if len(view.combat_units) >= 4:
        return False
    for event in view.events:
        kind = str(event.get("kind") or event.get("type") or "").lower()
        tick = event.get("tick")
        fresh = not isinstance(tick, int) or view.tick - tick <= C.UNDER_ATTACK_FRESH_TICKS
        if fresh and ("under_attack" in kind or "attacked" in kind):
            return True
    return False


def detect_cheese(view: View) -> dict | None:
    """A conservative cheese verdict, or ``None`` when the board looks normal.

    Fires only when enough enemy combat value is near our base, or a forward
    enemy MCV is seen, or we are taking damage before we have an army. The
    thresholds are intentionally high to avoid tempo-wrecking false alarms.
    """
    near = enemy_threat_near_base(view)
    forward_mcv = any(a.type in (C.MCV, C.CONSTRUCTION_YARD) for a in near)
    if (not forward_mcv and view.game_seconds >= C.CHEESE_WINDOW_SECONDS
            and not _base_needs_army(view, near)):
        return None   # a raid the home units and turrets can handle: the army stays on task
    value = sum(view.unit_costs.get(a.type, C.KNOWN_COSTS.get(a.type, C.DEFAULT_UNIT_COST)) for a in near)
    if forward_mcv or (len(near) >= C.CHEESE_MIN_ENEMY_UNITS and value >= C.RUSH_MIN_VALUE):
        return {
            "kind": "forward_mcv" if forward_mcv else "unit_rush",
            "near_base": len(near),
            "detected_at": view.game_seconds,
        }
    if early_under_attack(view):
        return {"kind": "early_damage", "near_base": len(near), "detected_at": view.game_seconds}
    return None


def _base_needs_army(view: View, near: list) -> bool:
    """After the opening, call the army home only for a real threat: worth
    ``BASE_DEFEND_MIN_VALUE`` and more than what already defends the base
    (our units near it and our turrets). Two stragglers used to recall a
    winning attack every time (match feba5f26: attack, recall, park 40-120 s,
    attack again, six times against a rusher's stream of reinforcements)."""
    cost = lambda a: view.unit_costs.get(a.type, C.DEFAULT_UNIT_COST)  # noqa: E731
    threat = sum(cost(a) for a in near)
    if threat < C.BASE_DEFEND_MIN_VALUE:
        return False
    yards = our_yards(view)
    r2 = C.CHEESE_NEAR_BASE_CELLS ** 2
    home = sum(cost(u) for u in view.combat_units
               if u.cell is not None and any(dist2(u.cell, y) <= r2 for y in yards))
    turrets = sum(cost(b) for b in view.buildings if b.type in C.DEFENSES and not b.wreck)
    return threat > C.BASE_DEFEND_HOME_RATIO * (home + turrets)


def _open_cell(view: View, cell) -> tuple[int, int]:
    """``cell`` rounded, kept inside the map (inset) and off impassable terrain."""
    x, y = round(cell[0]), round(cell[1])
    if view.bounds is not None:
        bx, by, bw, bh = view.bounds
        m = min(C.SCOUT_INSET_CELLS // 2, bw // 4, bh // 4)
        x, y = min(max(x, bx + m), bx + bw - 1 - m), min(max(y, by + m), by + bh - 1 - m)
    if (x, y) in view.blocked_cells:
        near = [(x + i, y + j) for i in range(-4, 5) for j in range(-4, 5) if (x + i, y + j) not in view.blocked_cells]
        if near:
            x, y = min(near, key=lambda c: dist2(c, (x, y)))
    return x, y


def early_push_warning(view: View) -> bool:
    """Early in the game, a known enemy army (seen recently, see memory.py)
    big enough to threaten us: before ``EARLY_PUSH_WINDOW_SECONDS``, at least
    ``EARLY_PUSH_MIN_VALUE`` and ``EARLY_PUSH_RATIO`` x our army."""
    return (view.game_seconds < C.EARLY_PUSH_WINDOW_SECONDS
            and view.enemy_army_value >= C.EARLY_PUSH_MIN_VALUE
            and view.enemy_army_value >= C.EARLY_PUSH_RATIO * view.army_value)


_STAGE: list = [None, None]   # [view, massing point]: computed once per view


def choke_cell(view: View) -> tuple[int, int] | None:
    if _STAGE[0] is view:
        return _STAGE[1]
    stage = _massing_point(view)
    _STAGE[0], _STAGE[1] = view, stage
    return stage


def _massing_point(view: View) -> tuple[int, int] | None:
    """Where the army masses, new units rally and the army regroups:
    ``STAGE_FORWARD_CELLS`` in front of the base toward the enemy (never past
    ``STAGE_FRACTION`` of the way): ready to attack, between them and us.
    (Measured from the midpoint of the base and its nearest ore, it leaned
    toward the ore side — match a2952a6d massed away from the enemy; home
    responders guard the harvesters.) If known enemies are near it,
    ``STAGE_FALLBACK_CELLS`` only: never mass, or send reinforcements one by
    one, into ground the enemy holds."""
    base = view.base_center
    toward = _enemy_side(view)
    if base is None or toward is None:
        return None
    anchor = base
    stage = _forward(view, anchor, toward, C.STAGE_FORWARD_CELLS, base)
    if early_push_warning(view) or not stage_is_safe(view, stage):
        stage = _forward(view, anchor, toward, C.STAGE_FALLBACK_CELLS, base)
    return _clear_stage(view, stage, toward)


def _clear_stage(view: View, stage, toward) -> tuple[int, int]:
    """``stage`` if no building of ours or impassable ground lies within
    ``STAGE_CLEAR_CELLS`` of it, else the nearest such cell (within
    ``STAGE_MOVE_SEARCH_CELLS``, ties toward the enemy): the army needs room
    to stand, and units on a building spot get it moved home (159f4f11)."""
    from .buildplan import _rect

    r = C.STAGE_CLEAR_CELLS
    taken = set(view.blocked_cells)
    for b in view.buildings:
        if b.cell is not None and not b.wreck:
            x0, y0, x1, y1 = _rect(b.type, b.cell)
            taken.update((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))
    if not taken:
        return stage

    def clear(c):
        if view.bounds is not None:
            bx, by, bw, bh = view.bounds
            if not (bx <= c[0] < bx + bw and by <= c[1] < by + bh):
                return False
        return not any((c[0] + dx, c[1] + dy) in taken
                       for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= r * r)

    if clear(stage):
        return stage
    m = C.STAGE_MOVE_SEARCH_CELLS
    near = [(stage[0] + dx, stage[1] + dy) for dx in range(-m, m + 1) for dy in range(-m, m + 1)
            if 0 < dx * dx + dy * dy <= m * m]
    near.sort(key=lambda c: (dist2(c, stage), dist2(c, toward)))
    return next((c for c in near if clear(c)), stage)


def _forward(view: View, anchor, toward, cells: float, base) -> tuple[int, int]:
    """``cells`` from ``anchor`` toward ``toward`` (at most ``STAGE_FRACTION``
    of the base-to-enemy distance), on an open cell inside the map."""
    dx, dy = toward[0] - anchor[0], toward[1] - anchor[1]
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    step = min(cells, C.STAGE_FRACTION * dist2(base, toward) ** 0.5)
    x, y = round(anchor[0] + dx / length * step), round(anchor[1] + dy / length * step)
    if view.bounds is not None:
        bx, by, bw, bh = view.bounds
        x, y = min(max(x, bx + 1), bx + bw - 2), min(max(y, by + 1), by + bh - 2)
    if (x, y) in view.blocked_cells:
        open_ = [(x + i, y + j) for i in range(-3, 4) for j in range(-3, 4) if (x + i, y + j) not in view.blocked_cells]
        if open_:
            x, y = min(open_, key=lambda c: dist2(c, (x, y)))
    return x, y


def stage_is_safe(view: View, cell) -> bool:
    """No known enemy combat unit or defense within
    ``STAGE_DANGER_RADIUS_CELLS`` of ``cell``."""
    r2 = C.STAGE_DANGER_RADIUS_CELLS ** 2
    threats = [a for a in list(view.enemy_units) + list(view.known_enemy_units) if a.type not in _NON_THREAT]
    threats += [a for a in view.known_enemy_buildings if a.type in C.DEFENSES]
    return not any(a.cell is not None and dist2(a.cell, cell) <= r2 for a in threats)


def stage_is_ours(view: View, cell) -> bool:
    """``cell`` is on our side: within ``STAGE_MAX_FRACTION`` of the way to the enemy."""
    base, toward = view.base_center, _enemy_side(view)
    if base is None or toward is None:
        return False
    return dist2(base, cell) <= (C.STAGE_MAX_FRACTION ** 2) * dist2(base, toward)


def _enemy_side(view: View):
    from .observe import enemy_spawn_guess

    enemy = enemy_spawn_guess(view)
    if enemy is not None:
        return enemy
    if view.bounds is not None:
        x, y, w, h = view.bounds
        return (x + w // 2, y + h // 2)
    return None


def approach_cell(view: View):
    """Where a frontal attack reaches our base: the cell about
    ``APPROACH_RING_CELLS`` from it that is the shortest walk from the enemy
    spawn — the road in, not the straight line across cliffs. The first
    turret faces it (it catches units sent in one by one). Falls back to the
    enemy side."""
    from .terrain import walk_distances

    base, enemy = view.base_center, _enemy_side(view)
    if base is None or enemy is None or not view.blocked_cells:
        return enemy
    walk = walk_distances(view, [tuple(enemy)], C.APPROACH_MAX_STEPS)
    r = C.APPROACH_RING_CELLS
    ring = [c for c in walk if (r - 1) ** 2 <= dist2(c, base) <= (r + 1) ** 2]
    return min(ring, key=lambda c: (walk[c], dist2(c, enemy))) if ring else enemy


def _along(a, b, fraction: float) -> tuple[int, int]:
    return (round(a[0] + (b[0] - a[0]) * fraction), round(a[1] + (b[1] - a[1]) * fraction))


# -- the roaming scout ---------------------------------------------------------


def patrol_route(view: View) -> list[tuple[int, int]]:
    """A clockwise ring of waypoints around the map, starting at the one
    nearest our base: the four corners and four edge midpoints, inset from the
    playable edge. Empty when the map bounds are unknown."""
    if view.bounds is None:
        return []
    x, y, w, h = view.bounds
    inset = max(1, min(C.SCOUT_INSET_CELLS, w // 4, h // 4))
    left, top, right, bottom = x + inset, y + inset, x + w - 1 - inset, y + h - 1 - inset
    midx, midy = (left + right) // 2, (top + bottom) // 2
    ring = [(left, top), (midx, top), (right, top), (right, midy),
            (right, bottom), (midx, bottom), (left, bottom), (left, midy)]
    if view.base_center is not None:
        start = min(range(len(ring)), key=lambda i: dist2(ring[i], view.base_center))
        ring = ring[start:] + ring[:start]
    # Visit the enemy's ore field on every loop, right after the ring point
    # nearest to it: that is where their harvesters (raid targets) work.
    from .observe import enemy_ore_field

    stops = [enemy_ore_field(view)]
    # From 3:30, also the ore fields the enemy is likeliest to expand to.
    if view.game_seconds >= C.ENEMY_EXPANSION_WATCH_AFTER_SECONDS:
        from .expand import enemy_expansion_sites

        stops += enemy_expansion_sites(view)[:C.ENEMY_EXPANSION_SCOUT_SITES]
    for stop in stops:
        if stop is not None and stop not in ring:
            after = min(range(len(ring)), key=lambda i: dist2(ring[i], stop))
            ring.insert(after + 1, stop)
    return ring


class Hunter:
    """Find the enemy's last buildings when we know of none.

    Destroying every enemy building (and the MCV) wins outright; at the
    time limit only the tiebreak counts. When the army attacks and no enemy
    building or unit is known, it sweeps search points instead of standing
    on the ruins of the enemy base (match 2035c6b3: a 30k army idled there
    for 8 minutes and the game went to the time limit): the enemy spawn,
    their likely expansion sites, their ore fields, then the map's corners.
    A point is checked once one of our units has been within
    ``HUNT_CHECKED_CELLS``; when all are checked, the sweep starts over.
    """

    def __init__(self) -> None:
        self._checked: set[tuple[int, int]] = set()
        self._most_buildings = 0       # the most enemy buildings we have known at once
        self._ever: list[tuple[int, int]] = []   # every cell where we knew an enemy building

    def hunting(self, view: View) -> bool:
        """Their base is (nearly) razed: we once knew at least twice
        ``FINISH_MAX_BUILDINGS`` enemy buildings (not a base half seen through
        fog), know of at most ``FINISH_MAX_BUILDINGS`` now, and lead on trades.
        Time to finish them (match 0561274e: an 80-unit army stood on the
        ruins for 6 minutes; b7fcbd4e: a 40k army spent its last 5 minutes
        10 cells from the enemy's last building, a radar dome)."""
        seen = [a for a in list(view.enemy_buildings) + list(view.known_enemy_buildings) if not a.wreck]
        known = {a.id for a in seen}
        self._most_buildings = max(self._most_buildings, len(known))
        for a in seen:
            if a.cell is not None and all(dist2(a.cell, c) > 9 for c in self._ever):
                self._ever.append(a.cell)
        return (self._most_buildings >= 2 * C.FINISH_MAX_BUILDINGS and len(known) <= C.FINISH_MAX_BUILDINGS
                and view.kills_cost > view.deaths_cost)

    def target(self, view: View) -> tuple[int, int] | None:
        points = self.targets(view, 1)
        return points[0] if points else None

    def targets(self, view: View, k: int) -> list[tuple[int, int]]:
        """Up to ``k`` unchecked search points: the enemy spawn first while it
        is unchecked, then the points nearest the army. Every cell where we
        once knew an enemy building is one (match c4caa45e: a barracks our
        memory had dropped survived 80 s of a 100-unit hunt that searched
        the spawn, ore fields and corners)."""
        points = self._points(view)
        for c in self._ever:
            if all(dist2(c, p) > 36 for p in points):
                points.append(c)
        if not points:
            return []
        r2 = C.HUNT_CHECKED_CELLS ** 2
        ours = [u.cell for u in view.units if u.cell is not None and not u.wreck]
        for p in points:
            if p not in self._checked and any(dist2(c, p) <= r2 for c in ours):
                self._checked.add(p)
        left = [p for p in points if p not in self._checked]
        if not left:
            self._checked.clear()
            left = points
        first = [left[0]] if left[0] == points[0] else []   # the enemy spawn first, while unchecked
        rest = [p for p in left if p not in first]
        army = [u.cell for u in view.combat_units if u.cell is not None]
        if army:
            cx = sum(c[0] for c in army) / len(army)
            cy = sum(c[1] for c in army) / len(army)
            rest.sort(key=lambda p: dist2(p, (cx, cy)))
        return (first + rest)[:max(1, k)]

    @staticmethod
    def _points(view: View) -> list[tuple[int, int]]:
        from .expand import _ore_fields, enemy_expansion_sites
        from .observe import enemy_spawn_guess

        pts: list[tuple[int, int]] = []
        spawn = enemy_spawn_guess(view)
        if spawn is not None:
            pts.append(spawn)
        pts += enemy_expansion_sites(view)
        base = view.base_center
        for _, centre, ore in _ore_fields(view):
            if base is None or dist2(ore, base) > C.ENEMY_EXPANSION_MIN_CELLS ** 2:
                pts.append(ore)
        if view.bounds is not None:
            x, y, w, h = view.bounds
            m = C.SCOUT_INSET_CELLS
            pts += [(x + m, y + m), (x + w - 1 - m, y + m), (x + m, y + h - 1 - m), (x + w - 1 - m, y + h - 1 - m)]
        out: list[tuple[int, int]] = []
        for p in pts:
            if all(dist2(p, q) > 36 for q in out):
                out.append(p)
        return out
