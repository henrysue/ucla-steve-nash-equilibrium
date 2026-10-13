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


def enemy_threat_near_base(view: View, main_only: bool = False) -> list:
    """Enemy combat actors within ``CHEESE_NEAR_BASE_CELLS`` of our base
    (every yard of ours; ``main_only``: the main base alone — the army's
    business: expansions have their outpost, turret and nearby units).

    Counts units and a forward enemy MCV (a base-move/MCV-rush tell). Harvesters
    are ignored. Returns the offending actors so the caller can size the threat.
    """
    yards = ([view.base_center] if view.base_center is not None else []) if main_only else our_yards(view)
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
    """An ``under_attack`` event on something of ours at the main base
    (within ``CHEESE_NEAR_BASE_CELLS``) before we have a real army is a cheese
    tell. Our scouts shot at their base are not (82bdca3e: the sparring
    seat's two scouts, hit at the enemy's massing point, flipped it into rush
    defense at 2:47). The attacked actor is found by the event's id, else
    its x/y; an event we can't place doesn't count."""
    if len(view.combat_units) >= 4 or view.base_center is None:
        return False
    ours = {a.id: a.cell for a in list(view.units) + list(view.buildings)
            if a.id is not None and a.cell is not None and not a.wreck}
    r2 = C.CHEESE_NEAR_BASE_CELLS ** 2
    for event in view.events:
        kind = str(event.get("kind") or event.get("type") or "").lower()
        tick = event.get("tick")
        fresh = not isinstance(tick, int) or view.tick - tick <= C.UNDER_ATTACK_FRESH_TICKS
        if not (fresh and ("under_attack" in kind or "attacked" in kind)):
            continue
        cell = ours.get(event.get("id", event.get("actor_id")))
        if cell is None and isinstance(event.get("x"), int) and isinstance(event.get("y"), int):
            cell = (event["x"], event["y"])
        if cell is not None and dist2(cell, view.base_center) <= r2:
            return True
    return False


def detect_cheese(view: View) -> dict | None:
    """A conservative cheese verdict, or ``None`` when the board looks normal.

    Fires only when enough enemy combat value is near our base, or a forward
    enemy MCV is seen, or we are taking damage before we have an army. The
    thresholds are intentionally high to avoid tempo-wrecking false alarms.
    """
    near = enemy_threat_near_base(view, main_only=True)
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
    """After the opening, call the army home only for a real threat at the
    main base: worth more than a small raid (``SMALL_RAID_VALUE``) and not
    beaten by what already defends it — our units near it and our turrets,
    judged like every fight (strength.odds under ``BASE_DEFEND_HOME_ODDS``).
    Two stragglers used to recall a winning attack every time (match
    feba5f26: attack, recall, park 40-120 s, attack again, six times)."""
    from . import strength

    theirs = [a.type for a in near if a.type not in C.ALL_BUILDING_TYPES and a.type != C.MCV]
    if strength.value(view, theirs) <= C.SMALL_RAID_VALUE:
        return False
    base = view.base_center
    if base is None:
        return True
    r2 = C.CHEESE_NEAR_BASE_CELLS ** 2
    home = [u.type for u in view.combat_units if u.cell is not None and dist2(u.cell, base) <= r2]
    home += [b.type for b in view.buildings if b.type in C.TURRETS and not b.wreck and b.cell is not None
             and dist2(b.cell, base) <= C.DEFENSE_MAIN_RADIUS_CELLS ** 2]
    return strength.odds(view, home, theirs) < C.BASE_DEFEND_HOME_ODDS


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
    big enough to threaten us: before ``EARLY_PUSH_WINDOW_SECONDS``, worth
    at least ``EARLY_PUSH_MIN_VALUE`` and our army's odds against it no
    better than ``EARLY_PUSH_RATIO`` (strength.army_odds)."""
    from . import strength

    return (view.game_seconds < C.EARLY_PUSH_WINDOW_SECONDS
            and view.enemy_army_value >= C.EARLY_PUSH_MIN_VALUE
            and strength.army_odds(view) <= C.EARLY_PUSH_RATIO)


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
    responders guard the harvesters.) It stays there: the army up front
    defends. It used to pull back into the base whenever an enemy unit was
    remembered nearby (for 90 s after a scout passed), and the rally points
    and the concave went with it; the army now fights what comes near it and
    holds the main line by its turrets when it can't win (executor._army_home)."""
    base = view.base_center
    toward = _enemy_side(view)
    if base is None or toward is None:
        return None
    # Farther out on a big map (a share of the way to the enemy): the army
    # meets attacks sooner and leaves room for buildings (Shadowfiend II).
    forward = max(C.STAGE_FORWARD_CELLS, C.STAGE_SHARE * dist2(base, toward) ** 0.5)
    stage = _forward(view, base, toward, forward, base)
    return _clear_stage(view, stage, toward)


_CHOKE: dict = {}


def choke_stage(view: View, base=None, enemy=None):
    """(massing point, facing) for a concave that abuses the choke on the
    enemy's walking path to our base: the narrowest passage
    (``CHOKE_STAGE_MAX_WIDTH`` across or less) between ``STAGE_FORWARD_CELLS``
    from the base and ``CHOKE_STAGE_MAX_FRACTION`` of the way to them. The
    concave's kill point (``FORMATION_FOCUS_AHEAD`` in front of the massing
    point) sits where the way opens up on our side of it, so every unit in
    the arc fires into the mouth while they come through a few at a time
    (Sidestep's middle gap, Trapped's island passage: the concave used to
    stand 7-9 cells short of them, and they spread out before contact).
    Facing the choke itself. None without such a choke (the distance rule
    then places it). From the terrain alone: cached per map and spawns."""
    from .observe import enemy_spawn_guess
    from .terrain import passage_width, walk_path

    if base is None and enemy is None:                 # ours by default; theirs with the bases swapped
        base, enemy = view.base_center, enemy_spawn_guess(view)
    if base is None or enemy is None or not view.blocked_cells:
        return None
    key = (id(view.blocked_cells), tuple(base), tuple(enemy))
    hit = _CHOKE.get(key)
    if hit is not None and hit[0] is view.blocked_cells:
        return hit[1]
    result = None
    path = walk_path(view, tuple(enemy), tuple(base))           # from them to us
    lo2, hi2 = C.STAGE_FORWARD_CELLS ** 2, C.CHOKE_STAGE_MAX_FRACTION ** 2 * dist2(base, enemy)
    widths = {i: passage_width(view, c) for i, c in enumerate(path) if lo2 <= dist2(c, base) <= hi2}
    narrow = [(w, -i) for i, w in widths.items() if w <= C.CHOKE_STAGE_MAX_WIDTH]
    if narrow:
        width, back = min(narrow)                                # the narrowest; on a tie the one nearest us
        i = -back
        mouth = i
        while (mouth + 1 < len(path) and mouth - i < C.CHOKE_MOUTH_MAX_STEPS
               and passage_width(view, path[mouth]) <= width + 2):
            mouth += 1
        # The massing point FORMATION_FOCUS_AHEAD behind the mouth, in line
        # with the choke: the concave's kill point lands on the mouth.
        (cx, cy), (mx, my) = path[i], path[mouth]
        d = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5 or 1.0
        stage = _open_cell(view, (mx + (mx - cx) / d * C.FORMATION_FOCUS_AHEAD,
                                  my + (my - cy) / d * C.FORMATION_FOCUS_AHEAD))
        result = (stage, path[i])
    if len(_CHOKE) > 16:
        _CHOKE.clear()
    _CHOKE[key] = (view.blocked_cells, result)
    return result


_CONCAVE: list = [None, None]   # [view, (point, facing)]: computed once per view


def concave_point(view: View):
    """(point, facing) where the waiting army forms its concave: at the mouth
    of the choke on their road in, facing it (:func:`choke_stage`), else at
    the massing point facing their side. Only the concave moves: rally
    points, the attack probe and everything else keep the massing point."""
    if _CONCAVE[0] is view:
        return _CONCAVE[1]
    held = choke_stage(view)
    if held is not None:
        out = (_clear_stage(view, held[0], held[1], terrain=False), held[1])
    else:
        stage = choke_cell(view)
        out = (stage, _enemy_side(view)) if stage is not None else None
    _CONCAVE[0], _CONCAVE[1] = view, out
    return out


def _clear_stage(view: View, stage, toward, terrain: bool = True) -> tuple[int, int]:
    """``stage`` if no building of ours (or, with ``terrain``, impassable
    ground) lies within ``STAGE_CLEAR_CELLS`` of it, else the nearest such
    cell (within ``STAGE_MOVE_SEARCH_CELLS``, ties toward the enemy): the
    army needs room to stand, and units on a building spot get it moved home
    (159f4f11). At a choke the cliffs are the point: only buildings count."""
    from .buildplan import _rect

    r = C.STAGE_CLEAR_CELLS
    taken = set(view.blocked_cells) if terrain else set()
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
        return c not in view.blocked_cells and not any(
            (c[0] + dx, c[1] + dy) in taken for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= r * r)

    if clear(stage):
        return stage
    m = C.STAGE_MOVE_SEARCH_CELLS
    near = [(stage[0] + dx, stage[1] + dy) for dx in range(-m, m + 1) for dy in range(-m, m + 1)
            if 0 < dx * dx + dy * dy <= m * m]
    near.sort(key=lambda c: (dist2(c, stage), dist2(c, toward)))
    return next((c for c in near if clear(c)), stage)


def enemy_staging(view: View) -> tuple[int, int] | None:
    """Where their army most likely masses: our own rule from their side —
    the mouth of the choke on their road to us when there is one
    (:func:`choke_stage` with the bases swapped), else the massing distance
    from their base. A scout looks there before a blind attack (9ff91807
    looked at the open-ground spot, saw nothing, and attacked into 7300 at
    their choke)."""
    from .observe import enemy_spawn_guess

    enemy, base = enemy_spawn_guess(view), view.base_center
    if enemy is None or base is None:
        return None
    held = choke_stage(view, tuple(enemy), tuple(base))
    if held is not None:
        return held[0]
    forward = max(C.STAGE_FORWARD_CELLS, C.STAGE_SHARE * dist2(enemy, base) ** 0.5)
    return _forward(view, enemy, base, forward, enemy)


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


def _enemy_side(view: View):
    from .observe import enemy_spawn_guess

    enemy = enemy_spawn_guess(view)
    if enemy is not None:
        return enemy
    if view.bounds is not None:
        x, y, w, h = view.bounds
        return (x + w // 2, y + h // 2)
    return None


def _along(a, b, fraction: float) -> tuple[int, int]:
    return (round(a[0] + (b[0] - a[0]) * fraction), round(a[1] + (b[1] - a[1]) * fraction))


# -- the roaming scout ---------------------------------------------------------


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
        self._razed = False                      # their main base fell: hunt until the end (latched)
        self._grid: tuple | None = None          # (key, cells): the map's search grid
        self._aimed: dict = {}                   # search point -> when it was first aimed at

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
        if len(known) > 2 * C.FINISH_MAX_BUILDINGS or view.kills_cost <= view.deaths_cost:
            self._razed = False              # a real base found after all, or no longer ahead: a normal attack
        elif (self._most_buildings >= 2 * C.FINISH_MAX_BUILDINGS and len(known) <= C.FINISH_MAX_BUILDINGS
                and view.kills_cost > view.deaths_cost):
            self._razed = True               # stays on while they rebuild a building or two
        return self._razed

    def targets(self, view: View, k: int) -> list[tuple[int, int]]:
        """Up to ``k`` unchecked search points: the enemy spawn first while it
        is unchecked, then the points nearest the army. Every cell where we
        once knew an enemy building is one (match c4caa45e: a barracks our
        memory had dropped survived 80 s of a 100-unit hunt that searched
        the spawn, ore fields and corners)."""
        points = self._points(view)
        if not points:
            return []
        r2 = C.HUNT_CHECKED_CELLS ** 2
        ours = [u.cell for u in view.units if u.cell is not None and not u.wreck]
        t = view.game_seconds
        for p in points:
            # Reached — or aimed at for HUNT_POINT_GIVEUP_SECONDS without a unit
            # getting near (sea, a plateau): checked, so the sweep moves on
            # (the audit: an unreachable point held the hunt for good).
            given_up = t - self._aimed.get(p, t) >= C.HUNT_POINT_GIVEUP_SECONDS
            if p not in self._checked and (given_up or any(dist2(c, p) <= r2 for c in ours)):
                self._checked.add(p)
                self._aimed.pop(p, None)
        left = [p for p in points if p not in self._checked]
        if not left:
            self._checked.clear()
            self._aimed.clear()
            left = points
        first = [left[0]] if left[0] == points[0] else []   # the enemy spawn first, while unchecked
        rest = [p for p in left if p not in first]
        # A stable order — nearest their spawn first — not by the army's
        # moving centroid, which re-chose the set each view as groups spread.
        spawn = points[0]
        rest.sort(key=lambda p: dist2(p, spawn))
        chosen = (first + rest)[:max(1, k)]
        for p in chosen:
            self._aimed.setdefault(p, t)
        return chosen

    def _points(self, view: View) -> list[tuple[int, int]]:
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
        pts += self._ever                 # where we knew buildings, before the grid's nearby cells
        pts += self.grid(view)
        out: list[tuple[int, int]] = []
        for p in pts:
            if all(dist2(p, q) > 36 for q in out):
                out.append(p)
        return out

    def grid(self, view: View) -> list[tuple[int, int]]:
        """Search cells ``HUNT_GRID_CELLS`` apart over the whole map, each on
        ground our units can reach (a scout sees ~5 cells: the hunt covers the
        map, not just spawns, fields and corners)."""
        if view.bounds is None:
            return []
        key = (view.bounds, id(view.blocked_cells))
        if self._grid is not None and self._grid[0] == key:
            return self._grid[1]
        from .scouting import _reachable

        x, y, w, h = view.bounds
        step, m = C.HUNT_GRID_CELLS, C.HUNT_GRID_CELLS // 2
        cells = []
        for gx in range(x + m, x + w - 1, step):
            for gy in range(y + m, y + h - 1, step):
                c = _reachable(view, (gx, gy))
                if c not in view.blocked_cells and all(dist2(c, q) > 16 for q in cells):
                    cells.append(c)
        self._grid = (key, cells)
        return cells


_WATCH: dict = {}


def watch_point(view: View, yard) -> tuple[int, int] | None:
    """The choke a base's first tower should watch: the narrowest point
    (``CHOKE_MAX_WIDTH`` across or less) on the enemy's walking path to
    ``yard``, ``CHOKE_MIN_CELLS``-``CHOKE_MAX_CELLS`` from it on their side,
    so a tower there sees units coming one by one. From the terrain alone:
    any map. None when the way in is open."""
    from .observe import enemy_spawn_guess
    from .terrain import chokes

    enemy = enemy_spawn_guess(view)
    if yard is None or enemy is None or not view.blocked_cells:
        return None
    key = (id(view.blocked_cells), tuple(yard), tuple(enemy))
    hit = _WATCH.get(key)
    if hit is not None and hit[0] is view.blocked_cells:
        return hit[1]
    lo2, hi2 = C.CHOKE_MIN_CELLS ** 2, C.CHOKE_MAX_CELLS ** 2
    near = [(c, w) for c, w in chokes(view, enemy, yard)
            if w <= C.CHOKE_MAX_WIDTH and lo2 <= dist2(c, yard) <= hi2 and dist2(c, enemy) < dist2(yard, enemy)]
    point = min(near, key=lambda r: (r[1], -dist2(r[0], yard)))[0] if near else None
    if len(_WATCH) > 32:
        _WATCH.clear()
    _WATCH[key] = (view.blocked_cells, point)
    return point


_INTERCEPT: dict = {"at": None, "facing": None, "group": [], "seen": -1e9}
_LINE: dict = {}


def _defensive_line(view: View, yard, threat) -> tuple[int, int] | None:
    """2 cells behind our turret nearest ``threat`` (within the build radius
    of ``yard``), else 2 cells in front of our building nearest it, else part
    of the way out from the yard toward it."""
    r2 = C.BUILD_RADIUS_CELLS ** 2

    def step(frm, to, cells):
        dx, dy = to[0] - frm[0], to[1] - frm[1]
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        return round(frm[0] + dx / d * cells), round(frm[1] + dy / d * cells)

    mine = [b for b in view.buildings if b.cell is not None and not b.wreck and dist2(b.cell, yard) <= r2]
    towers = [b.cell for b in mine if b.type in C.TURRETS]
    if towers:
        tower = min(towers, key=lambda c: dist2(c, threat))
        return step(tower, yard, 2)
    if mine:
        front = min((b.cell for b in mine), key=lambda c: dist2(c, threat))
        return step(front, threat, 2)
    d = dist2(yard, threat) ** 0.5
    return step(yard, threat, min(0.5 * d, C.STAGE_FORWARD_CELLS)) if d > 0 else None


def home_line(view: View, toward=None) -> tuple[int, int] | None:
    """The one defensive line, at the main base, facing ``toward`` (default:
    the enemy's side): just behind our turret nearest it (the turrets fire
    first), else just in front of our building nearest it (buildings, new
    units and repairs at our back), else part of the way out from the base.
    The army readies here for a push it sees coming, meets a push here when
    it can't win in the open, and regroups here after a retreat."""
    base = view.base_center
    toward = tuple(toward) if toward is not None else _enemy_side(view)
    if base is None or toward is None:
        return None
    key = (id(view), toward)
    if _LINE.get("view") is not view:
        _LINE.clear()
        _LINE["view"] = view
    if key not in _LINE:
        point = _defensive_line(view, base, toward)
        _LINE[key] = None if point is None else _open_cell(view, point)   # by the turret: no clearance needed
    return _LINE[key]


def incoming_push(view: View):
    """(set-up point, facing, the enemy group) for an enemy army worth
    ``INTERCEPT_MIN_VALUE`` we see in our half: the army gets ready at the
    main base's line facing it (:func:`home_line`) — never out at an
    expansion, which its outpost, turret and nearby units hold. Held
    ``INTERCEPT_HOLD_SECONDS`` after the last sighting so the army doesn't
    swing back and forth; seen units only (a remembered one may be long
    gone). None when nothing is coming."""
    from . import strength
    from .observe import enemy_spawn_guess

    base, enemy = view.base_center, enemy_spawn_guess(view)
    t = view.game_seconds
    if base is None or enemy is None:
        return None
    ours = [a for a in view.enemy_units
            if a.id is not None and a.cell is not None and not a.wreck and a.type not in C.HARVESTER
            and a.type != C.MCV and dist2(a.cell, base) < dist2(a.cell, enemy)]
    group = []
    if ours:
        lead = min(ours, key=lambda a: dist2(a.cell, base))   # the one nearest us
        group = [a for a in ours if dist2(a.cell, lead.cell) <= 12 ** 2]
    if group and strength.value(view, [a.type for a in group]) >= C.INTERCEPT_MIN_VALUE:
        facing = (round(sum(a.cell[0] for a in group) / len(group)), round(sum(a.cell[1] for a in group) / len(group)))
        point = home_line(view, facing)
        if point is not None:
            _INTERCEPT.update(at=point, facing=facing, group=group, seen=t)
    if _INTERCEPT["at"] is not None and t - _INTERCEPT["seen"] <= C.INTERCEPT_HOLD_SECONDS and t >= _INTERCEPT["seen"]:
        return _INTERCEPT["at"], _INTERCEPT["facing"], _INTERCEPT["group"]
    _INTERCEPT.update(at=None, facing=None, group=[], seen=-1e9)
    return None
