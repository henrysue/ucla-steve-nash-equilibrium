"""Buildings: what to build next and where to place it. Stateless: decided
from what is built and queued each view, so refusals and losses self-correct.

Next build, first match wins: deploy the MCV; power (and cancel what blocks
a power plant); the expansion's refinery; more barracks / a second war
factory; more refineries; a refinery when harvesters saturate; a silo; then
the build order (waiting at the first unmet step). Placement grows the base
legally from its frontmost building in each role's direction, keeping lanes
and factory exits clear.
"""

from __future__ import annotations

from . import config as C
from . import orders
from .observe import View, centre_offset, dist2, enemy_spawn_guess


def _available(view: View, candidates) -> str | None:
    cands = (candidates,) if isinstance(candidates, str) else candidates
    return C.resolve(cands, view.available)


def _count_built_and_queued(view: View, type_ids) -> int:
    ids = (type_ids,) if isinstance(type_ids, str) else tuple(type_ids)
    built = sum(1 for b in view.buildings if b.type in ids and not b.wreck)
    queued = sum(view.queued_count(i) for i in ids)
    # A finished building waiting to be placed is neither built nor queued;
    # missing it queued a second copy (match 9a9cd98e: 2 war factories).
    ready = sum(1 for p in view.production if p.get("ready") and str(p.get("item") or "").lower() in ids)
    return built + queued + ready


def next_build_orders(view: View, expansion_ore: tuple[int, int] | None = None,
                      depleted: frozenset = frozenset(), sparring=None,
                      units_first: bool = False) -> list[dict]:
    """Zero or one production order that advances the build order this view.

    Returns a list (0 or 1 orders) so the executor can concatenate it with
    other work. Placement of finished buildings is handled separately by
    :func:`placement_orders`.
    """
    # Interrupt 1: deploy the MCV into a construction yard.
    if not view.has_building(C.CONSTRUCTION_YARD):
        mcvs = view.units_of(C.MCV)
        if mcvs and mcvs[0].id is not None:
            return [orders.deploy(mcvs[0].id)]
        return []  # no yard and no MCV visible yet: nothing to do

    # Interrupt 2: power margin thin -> queue power before anything else.
    if view.power_margin < C.POWER_MARGIN:
        powr = _available(view, C.POWER_PLANT) or _available(view, C.ADV_POWER_PLANT)
        # A finished plant waiting to be placed is power on the way: counting
        # it as missing queued a second plant before the barracks (match
        # 2035c6b3: barracks at 0:47 against the pros' 0:24).
        pending = powr and (view.queued_count(powr) or view.ready_in_queue(powr))
        if powr and not pending:
            return [orders.build(powr)]
        # The Building queue is one-at-a-time: a power plant queued behind a
        # big building waits for it, and that building crawls at low power.
        # (Match e4caa28c: a war factory sat ahead of the power plant for 5
        # minutes at 9/130 power.) When we are actually underpowered, cancel
        # whatever is ahead of the power plant; the build order re-queues it.
        blocker = _ahead_of_power(view, powr)
        if blocker is not None and view.power_margin < 0:
            return [orders.cancel(blocker)]
        # If power is queued/placing, fall through — don't stall other logic.

    # Interrupt 2z: no war factory past the opening (lost to an attack):
    # rebuild it before anything else — tanks, harvesters and the MCV all
    # come from it — jumping ahead of optional buildings in the queue.
    if factory_lost(view):
        weap = _available(view, C.WAR_FACTORY)
        if weap and not view.queued_count(weap) and not view.ready_in_queue(weap):
            return [orders.build(weap)]
        blocker = _ahead_of(view, weap)
        if blocker in C.OPTIONAL_BUILDINGS:
            return [orders.cancel(blocker)]

    if sparring is not None:   # sparring (TEST ONLY): the rusher's build, the macro's extra refineries
        own = sparring.build(view, depleted)
        if own is not None:
            return own

    # Interrupt 2a: a new expansion yard without a refinery -> one by its ore.
    if expansion_ore is not None:
        proc = _available(view, C.REFINERY)
        if proc and view.queued_count(proc) == 0:
            return [orders.build(proc)]

    # Interrupt 2b: production to spend on, the way pros do it — more
    # barracks (infantry-heavy army), and a second war factory only with a
    # big bank. Barracks follow BARRACKS_SCHEDULE; a big bank skips it.
    money = view.cash + view.ore
    if view.has_building(C.WAR_FACTORY) and not recovering(view):
        very = money >= C.FUNDS_VERY_RICH
        rax = _available(view, C.BARRACKS)
        if rax and money >= C.BARRACKS_ADD_FUNDS and not any(view.queued_count(b) for b in C.BARRACKS):
            if _count_built_and_queued(view, C.BARRACKS) < (C.MAX_BARRACKS if very else barracks_target(view)):
                return [orders.build(rax)]
        weap = _available(view, C.WAR_FACTORY)
        # A second war factory once money piles up mid-game: the bank is
        # production we failed to build (match c2fec761 sat on 3000-5700).
        banked = money >= C.FUNDS_RICH and view.game_seconds >= C.SECOND_FACTORY_SECONDS
        cap = C.MAX_WAR_FACTORIES_VERY_RICH if (very or banked) else C.MAX_WAR_FACTORIES
        if weap and _count_built_and_queued(view, (C.WAR_FACTORY,)) < cap:
            return [orders.build(weap)]

    # Interrupt 2d: radar dome for the end game (unlocks siege units).
    from .brain import outgunned
    if (view.game_seconds >= C.DOME_AFTER_SECONDS and money >= C.DOME_ADD_FUNDS and not outgunned(view) and not units_first
            and not recovering(view)
            and view.has_building(C.SERVICE_DEPOT) and not _count_built_and_queued(view, (C.RADAR_DOME,))):
        dome = _available(view, C.RADAR_DOME)
        if dome:
            return [orders.build(dome)]

    # Past the build order, no more refineries while outgunned, or when the
    # advisor reads that our army needs the money (balance "more_army").
    short = outgunned(view) or units_first

    # Interrupt 2c: more refineries over time (each comes with a harvester).
    if (not short and money >= C.REFINERY_ADD_FUNDS
            and _count_built_and_queued(view, (C.REFINERY,)) < refinery_target(view)):
        proc = _available(view, C.REFINERY)
        # Only where a field has room for one more (MAX_REFINERIES_PER_FIELD);
        # far ore doesn't rule it out when we're ahead: the base crawls toward it.
        if proc and view.queued_count(proc) == 0 and (
                open_ore(view, depleted) is not None
                or (_ahead(view, money) and open_ore(view, depleted, radius=None) is not None)):
            return [orders.build(proc)]

    # Interrupt 3: harvesters saturating the refineries (3 each) -> another
    # refinery, not more trucks ("never spam trucks without drop-off points").
    refineries = sum(1 for b in view.buildings if b.type == C.REFINERY and not b.wreck)
    harvesters = len(view.units_of(C.HARVESTER))
    if not short and refineries and harvesters >= C.MAX_HARVESTERS_PER_REFINERY * refineries \
            and open_ore(view, depleted, radius=None) is not None:
        proc = _available(view, C.REFINERY)
        if proc and not _count_built_and_queued(view, (C.REFINERY,)) > refineries:
            return [orders.build(proc)]

    # Interrupt 4: ore storage nearly full -> a silo, or harvesters stop unloading.
    if view.resource_capacity > 0 and view.ore >= C.SILO_FILL_RATIO * view.resource_capacity:
        silo = _available(view, C.SILO)
        if silo and view.queued_count(silo) == 0:
            return [orders.build(silo)]

    step = walk_build_order(view, _build_order(view))
    if step:
        return step
    # A kennel once the war factory is up (Soviet: dogs, the best scouts).
    kennel = _available(view, C.KENNEL)
    if kennel and view.has_building(C.WAR_FACTORY) and not recovering(view) and not _count_built_and_queued(view, (C.KENNEL,)):
        return [orders.build(kennel)]
    return []


def _build_order(view: View) -> tuple:
    """The opening, then its follow-up unless outgunned (units first); while
    recovering trucks, only its refinery (it brings one)."""
    from .brain import outgunned

    if opening_done(view) and outgunned(view):
        return C.BUILD_ORDER
    if opening_done(view) and recovering(view):
        return C.BUILD_ORDER + (C.REFINERY,)
    return C.BUILD_ORDER + C.FOLLOW_UP


def factory_lost(view: View) -> bool:
    """Past the opening (``WAR_FACTORY_REBUILD_AFTER_SECONDS``, a barracks and
    a refinery up) with no war factory standing: it was destroyed (or never
    finished). Rebuilding it comes first."""
    return (view.game_seconds >= C.WAR_FACTORY_REBUILD_AFTER_SECONDS
            and view.has_building(C.CONSTRUCTION_YARD) and view.has_building(C.BARRACKS)
            and view.has_building(C.REFINERY) and not view.has_building(C.WAR_FACTORY))


def factory_reserve(view: View) -> float:
    """Money set aside for a lost war factory's rebuild (what is left to pay
    on it), kept even while defending: units must not starve it."""
    if not factory_lost(view):
        return 0.0
    weap = _available(view, C.WAR_FACTORY) or C.WAR_FACTORY
    cost = view.unit_costs.get(weap) or C.KNOWN_COSTS.get(weap, C.DEFAULT_UNIT_COST)
    for p in view.production:
        if str(p.get("item") or "").lower() == weap and not p.get("ready"):
            progress = p.get("progress")
            done = float(progress) if isinstance(progress, (int, float)) and not isinstance(progress, bool) else 0.0
            return cost * max(0.0, 1.0 - done)
    return cost


def recovering(view: View) -> bool:
    """Fewer harvesters than refineries once the war factory is up: income
    first — no depot, kennel, dome or extra production until the trucks are
    back (match 6b457914 paid for a refinery, a kennel, the depot and a turret
    on one harvester, and its replacement waited 90 s)."""
    refineries = sum(1 for b in view.buildings if b.type == C.REFINERY and not b.wreck)
    return view.has_building(C.WAR_FACTORY) and len(view.units_of(C.HARVESTER)) < refineries


def opening_done(view: View) -> bool:
    """Every step of the standard build order stands (built, not queued)."""
    have: dict = {}
    for step in C.BUILD_ORDER:
        ids = _ids(step)
        have[ids] = have.get(ids, 0) + 1
        built = sum(1 for b in view.buildings if b.type in ids and not b.wreck)
        if built < have[ids]:
            return False
    return True


def build_reserve(view: View) -> float:
    """Money the economy still needs, which units mustn't spend: what is left
    to pay on buildings, harvesters and an MCV in production, plus the next
    build-order step not started yet. The pros' build order runs on time
    because its money is set aside; the rest goes into units (no infantry
    cap)."""
    from .brain import outgunned

    cost = lambda item: view.unit_costs.get(item) or C.KNOWN_COSTS.get(item, C.DEFAULT_UNIT_COST)  # noqa: E731
    owed = 0.0
    short = outgunned(view)
    for p in view.production:
        item = str(p.get("item") or "").lower()
        if p.get("ready") or not item:
            continue
        if item == C.MCV and short:
            continue                   # outgunned: units before the MCV (it is cancelled)
        if _is_building(view, item) or item in C.HARVESTER or item == C.MCV:
            progress = p.get("progress")
            done = float(progress) if isinstance(progress, (int, float)) and not isinstance(progress, bool) else 0.0
            owed += cost(item) * max(0.0, 1.0 - done)
    nxt = 0.0
    seen: dict = {}
    for step in _build_order(view):
        seen[_ids(step)] = want = seen.get(_ids(step), 0) + 1
        if _count_built_and_queued(view, _ids(step)) >= want:
            continue
        resolved = _available(view, step) or (step if isinstance(step, str) else step[0])
        nxt = cost(resolved)
        break
    weap = _available(view, C.WAR_FACTORY) or C.WAR_FACTORY
    if factory_lost(view) and not view.queued_count(weap):
        nxt = max(nxt, cost(weap))     # a lost war factory, not yet queued: its money waits
    return owed + nxt


def walk_build_order(view: View, build_order) -> list[dict]:
    """Queue the first step of ``build_order`` we don't yet have enough of.
    The k-th time an id appears, we want k of it (two REFINERY steps = two)."""
    seen: dict = {}
    for step in build_order:
        seen[_ids(step)] = want = seen.get(_ids(step), 0) + 1
        resolved = _available(view, step)
        if resolved is None:
            # Not offered yet (its prerequisite is still building): wait for it
            # rather than jump ahead — skipping queued our second power plant
            # before the barracks and cost ~10 s on the whole opening.
            if _count_built_and_queued(view, _ids(step)) < want:
                return []
            continue
        if _count_built_and_queued(view, _ids(step)) < want:
            if view.queued_count(resolved) == 0:  # don't double-queue
                return [orders.build(resolved)]
            return []  # it's queued; wait for it
    return []


def placement_orders(
    view: View, defense_at: tuple[int, int] | None = None, expansion_ore: tuple[int, int] | None = None,
    expansion_defense: tuple[int, int] | None = None, depleted: frozenset = frozenset(),
    home: tuple[int, int] | None = None, expansion_site: tuple[int, int] | None = None,
    clearing: dict | None = None,
) -> list[dict]:
    """Place every finished-but-unplaced building (ready in the queue), at a
    spot chosen by role (:func:`placement_cell`). ``defense_at`` (the model
    advisor's pick) steers which side a defense goes on. ``expansion_site``
    keeps the expansion's refinery among that expansion's own buildings (never
    at home).

    Our units standing on the chosen spot are moved off it first (OpenRA
    moves a building placed over units to the main yard), and the placement
    waits for them — at most ``PLACE_CLEAR_WAIT_SECONDS``, tracked in
    ``clearing`` ({item: first game time it was blocked}), then it goes ahead.
    """
    out = []
    seen: set[str] = set()
    for entry in view.production:
        item = str(entry.get("item") or "").lower()
        if entry.get("ready") and item and item not in seen and _is_building(view, item, entry.get("queue")):
            if item == C.REFINERY and expansion_ore is not None:
                # The expansion's refinery: beside its ore (never on it), else
                # grown from the expansion's own buildings toward the ore.
                at = (_beside_ore(view, expansion_ore, item, near=expansion_site)
                      or _crawl_from_nearest(view, expansion_ore, item, near=expansion_site))
            elif item == C.REFINERY:
                # Beside a field with room (not a mined-out or full one), else
                # grow the base toward the nearest such field.
                live = open_ore(view, depleted)
                at = _beside_ore(view, live, item) if live is not None else \
                    _crawl_toward(view, open_ore(view, depleted, radius=None), item)
            elif item in C.DEFENSES and expansion_defense is not None:
                # The expansion's turret: watching a choke in range, else on its enemy-facing edge.
                at = (_watch_spot(view, item, expansion_defense, near=expansion_defense)
                      or perimeter_spot(view, expansion_defense, scout_watch(view, expansion_defense)
                                        or enemy_spawn_guess(view), item, near=expansion_defense, strict=False))
            else:
                at = placement_cell(view, item, defense_at, home)
            seen.add(item)
            moves = clear_spot_orders(view, item, at)
            if moves and clearing is not None:
                since = clearing.setdefault(item, view.game_seconds)
                if view.game_seconds - since < C.PLACE_CLEAR_WAIT_SECONDS:
                    out += moves             # step aside first; place once the spot is clear
                    continue
            if clearing is not None:
                clearing.pop(item, None)
            out += moves
            out.append(orders.place(item, at))
    return out


def clear_spot_orders(view: View, item: str, at) -> list[dict]:
    """Move our units off ``item``'s footprint at ``at`` (fighters
    attack-move, so they still fight; trucks and the MCV just move) to the
    nearest open cell clear of it. Empty when nobody stands there."""
    if at is None:
        return []
    footprint = set(_cells(item, at))
    on_it = [u for u in view.units if u.id is not None and u.cell in footprint and not u.wreck]
    if not on_it:
        return []
    x0, y0, x1, y1 = _rect(item, at)
    margin = {(x, y) for x in range(x0 - 1, x1 + 2) for y in range(y0 - 1, y1 + 2)}
    others = set()
    for b in view.buildings:
        if b.cell is not None and not b.wreck:
            others.update(_cells(b.type, b.cell))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ring = [(x, y) for x in range(x0 - 4, x1 + 5) for y in range(y0 - 4, y1 + 5)
            if (x, y) not in margin and (x, y) not in view.blocked_cells and (x, y) not in others
            and (view.bounds is None or (view.bounds[0] <= x < view.bounds[0] + view.bounds[2]
                                         and view.bounds[1] <= y < view.bounds[1] + view.bounds[3]))]
    if not ring:
        return []
    # Away from the base's centre if we can (out of the way), else nearest.
    base = view.base_center or (cx, cy)
    spot = min(ring, key=lambda c: (dist2(c, (cx, cy)), -dist2(c, base)))
    walkers = [u.id for u in on_it if u.type in C.HARVESTER or u.type == C.MCV]
    fighters = [u.id for u in on_it if u.id not in walkers]
    out = []
    if fighters:
        out.append(orders.attack_move(fighters[:orders.MAX_UNITS_PER_ORDER], list(spot)))
    if walkers:
        out.append(orders.move(walkers[:orders.MAX_UNITS_PER_ORDER], list(spot)))
    return out


def placement_cell(view: View, item: str, defense_at: tuple[int, int] | None = None,
                   home: tuple[int, int] | None = None) -> tuple[int, int] | None:
    """A legal top-left cell for ``item``, chosen by role, growing the base
    the way good players do (OpenRA wiki guides; OpenRA forum "Multiplayer
    Guide 2022"; Sharoma's Red Alert guide; cited in PLAN.md).

    OpenRA only lets a building go within a cell or two of an existing one
    ("you can construct buildings at a distance of one cell"); a hint farther
    out is silently replaced by a spot next to the construction yard. (Match
    28e54539: almost every building, and all three of one seat's pillboxes,
    ended up piled within ~4 cells of the yard.) So every spot here is just
    outside the frontmost building in the wanted direction (:func:`grow_spot`),
    and the base grows that way building by building:

    - refinery: beside the nearest ore if it is within reach, else grow toward it;
    - barracks: toward the enemy (a wall, and a rally point);
    - war factory and service depot: toward the front, turned off the ore side;
    - power plants and silos: toward the back, off the ore side ("Place the
      Power plant next to the Con yard so it allows you to place your
      Refinery next to the ore supply");
    - defenses: on the enemy-facing edge, spread out (:func:`perimeter_spot`).
    """
    # ``home``: the active base once the main base's ore is mined out (an
    # expansion); new buildings grow from its buildings instead.
    base = home or view.base_center
    if base is None:
        return None
    if item in C.DEFENSES:
        from .scout import enemy_threat_near_base

        threats = [a.cell for a in enemy_threat_near_base(view) if a.cell is not None]
        prefer = min(threats, key=lambda c: dist2(c, base)) if threats else None
        # Watching the choke on the enemy's way in when one is in range (any
        # map: from the terrain), else facing the road in; the advisor may pick
        # the side, but only one facing the enemy.
        from .scout import approach_cell

        watch = _watch_spot(view, item, base, near=home)
        if watch is not None:
            return watch
        enemy = scout_watch(view, base) or (approach_cell(view) if home is None else enemy_spawn_guess(view))
        if defense_at is not None and _within_degrees(base, defense_at, enemy, C.DEFENSE_SIDE_MAX_DEGREES):
            prefer = defense_at
        return perimeter_spot(view, base, enemy, item, prefer=prefer, strict=False, near=home)
    ore_cell = _nearest_ore(view, base)
    ore = _unit(base, ore_cell)
    if item == C.REFINERY:
        live = open_ore(view)
        if live is None:   # no field with room in reach: grow the base toward the nearest one
            return _crawl_toward(view, open_ore(view, radius=None), item)
        return _beside_ore(view, live, item) or grow_spot(view, _unit(base, live), item)
    enemy = _unit(base, enemy_spawn_guess(view))
    if enemy is None:
        return None
    # Barracks, war factory and depot grow out to the side, a little forward:
    # the front is where the army masses (match 5aea2632: four barracks
    # asked for beside the massing point, all moved home — units on the spot).
    side = (-enemy[1], enemy[0])
    if ore is not None and side[0] * ore[0] + side[1] * ore[1] > 0:
        side = (-side[0], -side[1])                 # the side away from the ore
    flank = _blend(side, enemy, C.PRODUCTION_FORWARD_BIAS)
    if item in C.BARRACKS:
        return grow_spot(view, flank, item, near=home)
    if item in (C.WAR_FACTORY, C.SERVICE_DEPOT):
        return grow_spot(view, flank, item, near=home)
    if item in (C.POWER_PLANT, C.ADV_POWER_PLANT, C.SILO, C.RADAR_DOME, C.KENNEL):
        back = (-enemy[0], -enemy[1])
        return grow_spot(view, _blend(back, ore, -1.0) if ore else back, item, near=home)
    return None


def _within_degrees(origin, a, b, limit: float) -> bool:
    """Whether bearings origin->a and origin->b are within ``limit`` degrees
    (False when either is unknown or zero-length)."""
    ua, ub = _unit(origin, a), _unit(origin, b)
    if ua is None or ub is None:
        return False
    return ua[0] * ub[0] + ua[1] * ub[1] >= __import__("math").cos(__import__("math").radians(limit))


def _unit(origin, target) -> tuple[float, float] | None:
    if origin is None or target is None:
        return None
    dx, dy = target[0] - origin[0], target[1] - origin[1]
    length = (dx * dx + dy * dy) ** 0.5
    return (dx / length, dy / length) if length else None


def _blend(a, b, weight: float) -> tuple[float, float]:
    """``a + weight * b``, normalized (falls back to ``a``)."""
    x, y = a[0] + weight * b[0], a[1] + weight * b[1]
    length = (x * x + y * y) ** 0.5
    return (x / length, y / length) if length > 1e-6 else a


def _nearest_ore(view: View, base) -> tuple[int, int] | None:
    r2 = C.REFINERY_ORE_SEARCH_CELLS ** 2
    near = [c for c in view.ore_cells if dist2(c, base) <= r2]
    return min(near, key=lambda c: dist2(c, base)) if near else None


# -- footprints and growing the base -------------------------------------------


def _footprint(item: str) -> tuple[int, int]:
    return C.FOOTPRINTS.get(item, C.DEFAULT_FOOTPRINT)


def _rect(item: str, cell) -> tuple[int, int, int, int]:
    w, h = _footprint(item)
    return cell[0], cell[1], cell[0] + w - 1, cell[1] + h - 1


def _cells(item: str, cell) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = _rect(item, cell)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


_TERRAIN_CACHE: dict[tuple, tuple] = {}


def _no_build(view: View) -> frozenset:
    """Cells no building may cover: ore, impassable terrain, and walkable
    terrain buildings can't stand on (rough ground, beach). OpenRA refuses a
    building there and silently drops it next to the main yard instead
    (matches 768b2cbd, b7fcbd4e: expansion refineries ended up at home)."""
    key = (id(view.ore_cells), id(view.blocked_cells), id(view.unbuildable_cells))
    hit = _TERRAIN_CACHE.get(key)
    sources = (view.ore_cells, view.blocked_cells, view.unbuildable_cells)
    if hit is not None and all(a is b for a, b in zip(hit[0], sources)):   # the same objects, not reused ids
        return hit[1]
    cells = frozenset(view.ore_cells) | view.blocked_cells | view.unbuildable_cells
    _TERRAIN_CACHE.clear()
    _TERRAIN_CACHE[key] = (sources, cells)
    return cells


_NO_AREA = frozenset(("pbox", "hbox", "gun", "ftur", "tsla", "agun", "sam"))   # don't extend the buildable area


def _occupied(view: View) -> frozenset:
    """Cells units stand on right now (ours and visible enemies'). OpenRA
    won't put a building over a unit and drops it at the main yard instead
    (match a1329ceb: an expansion refinery went home twice from a clear spot
    where a power plant stood later — trucks were driving through)."""
    return frozenset(u.cell for u in list(view.units) + list(view.enemy_units)
                     if u.cell is not None and not u.wreck)


def _clashes(item: str, cell, view: View) -> bool:
    """``item`` at top-left ``cell`` covers ore or impassable terrain,
    overlaps one of our buildings, leaves no free cell beside it, or sits on
    a kept-clear lane (:func:`_lanes`): units need paths through the base,
    not mazes around it."""
    footprint = _solid(item, cell)             # empty corners aren't checked (BuildingUtils.CanPlaceBuilding)
    no_build = _no_build(view)
    if any(c in no_build for c in footprint):
        return True
    lanes = _lanes(view)
    if any(c in lanes for c in footprint):
        return True
    for b in view.buildings:
        if b.cell is None or b.wreck:
            continue
        if _gap(item, cell, b.type, b.cell) < 1:
            return True
    return False


def _solid(item: str, cell) -> list[tuple[int, int]]:
    """The cells that are part of the building (its footprint less the
    ``EMPTY_CELLS`` corners)."""
    empty = C.EMPTY_CELLS.get(item, ())
    return [c for c in _cells(item, cell) if (c[0] - cell[0], c[1] - cell[1]) not in empty]


def _gap(a: str, a_cell, b: str, b_cell) -> int:
    """Free cells (Chebyshev) between two buildings' solid cells; -1 = overlap."""
    if a not in C.EMPTY_CELLS and b not in C.EMPTY_CELLS:
        ax0, ay0, ax1, ay1 = _rect(a, a_cell)
        bx0, by0, bx1, by1 = _rect(b, b_cell)
        return max(bx0 - ax1, ax0 - bx1, by0 - ay1, ay0 - by1) - 1
    sa, sb = _solid(a, a_cell), _solid(b, b_cell)
    return min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for p in sa for q in sb) - 1


_LANES: list = [None, None]   # [view, lanes]: one view's lanes (computed once per view)


def _lanes(view: View) -> frozenset:
    if _LANES[0] is view:
        return _LANES[1]
    lanes = _compute_lanes(view)
    _LANES[0], _LANES[1] = view, lanes
    return lanes


def _compute_lanes(view: View) -> frozenset:
    """Cells kept free of buildings so units can move:

    - a corridor ``LANE_WIDTH_CELLS`` wide from the base out toward the
      staging point (the army's road out, and reinforcements' road in);
    - ``EXIT_CLEAR_ROWS`` rows in front of (below) each war factory and
      barracks, where new units come out;
    - ``STAGE_CLEAR_CELLS`` around the massing point, where the army stands
      (match 159f4f11: barracks asked for there four times, moved home each
      time — units standing on the spot);
    - the harvesters' road from each refinery to the ore it serves (match
      a2952a6d: a war factory, depot and power plant asked for on truck
      routes were all moved home — trucks on the spot — and buildings there
      would box the trucks in).
    """
    from .scout import choke_cell

    cells: set = set()
    base, stage = view.base_center, choke_cell(view)
    if base is not None and stage is not None:
        steps = max(abs(stage[0] - base[0]), abs(stage[1] - base[1]), 1)
        half = C.LANE_WIDTH_CELLS // 2
        for i in range(steps + 1):
            x = round(base[0] + (stage[0] - base[0]) * i / steps)
            y = round(base[1] + (stage[1] - base[1]) * i / steps)
            if dist2((x, y), base) <= C.LANE_KEEP_CLEAR_CELLS ** 2 and dist2((x, y), base) > 4:
                cells.update((x + dx, y + dy) for dx in range(-half, half + 1) for dy in range(-half, half + 1))
    if stage is not None:
        r = C.STAGE_CLEAR_CELLS
        cells.update((stage[0] + dx, stage[1] + dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                     if dx * dx + dy * dy <= r * r)
    for b in view.buildings:
        if b.cell is not None and not b.wreck and (b.type == C.WAR_FACTORY or b.type in C.BARRACKS):
            x0, _, x1, y1 = _rect(b.type, b.cell)
            cells.update((x, y) for x in range(x0, x1 + 1) for y in range(y1 + 1, y1 + 1 + C.EXIT_CLEAR_ROWS))
    cells |= _truck_roads(view)
    return frozenset(cells)


def _truck_roads(view: View) -> set:
    """The straight road from each refinery to the nearest ore its trucks can
    walk to (within ``FIELD_WALK_STEPS``), one cell wide."""
    from .terrain import ring, walk_distances

    if not view.ore_cells:
        return set()
    ore = set(view.ore_cells)
    roads: set = set()
    for b in view.buildings:
        if b.type != C.REFINERY or b.cell is None or b.wreck:
            continue
        rect = _rect(b.type, b.cell)
        starts = [c for c in ring(rect) if c not in view.blocked_cells]
        walk = walk_distances(view, starts, C.FIELD_WALK_STEPS)
        near = [c for c in walk if c in ore]
        if not near:
            continue
        target = min(near, key=lambda c: walk[c])
        cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        steps = max(abs(target[0] - cx), abs(target[1] - cy), 1)
        for i in range(int(steps) + 1):
            cell = (round(cx + (target[0] - cx) * i / steps), round(cy + (target[1] - cy) * i / steps))
            if not (rect[0] <= cell[0] <= rect[2] and rect[1] <= cell[1] <= rect[3]):
                roads.add(cell)
    return roads


def _in_bounds(view: View, item: str, cell) -> bool:
    if view.bounds is None:
        return True
    bx, by, bw, bh = view.bounds
    x0, y0, x1, y1 = _rect(item, cell)
    return bx <= x0 and by <= y0 and x1 <= bx + bw - 1 and y1 <= by + bh - 1


_SPOTS: dict = {}


def legal_spots(view: View, item: str, near=None, skip_types=()) -> list[tuple[int, int]]:
    """Every top-left cell where ``item`` may go — the one source of placement
    candidates. A spot sits exactly ``BUILDING_GAP_CELLS`` free cells
    (Chebyshev) from one of our buildings that extends the buildable area
    (not a turret; within ``EXPAND_REFINERY_RADIUS_CELLS`` of ``near`` when
    given, to grow one particular base), inside the map, on buildable ground,
    off the lanes and the massing area, touching no other building
    (:func:`_clashes`). OpenRA's rule as the placement audit measured it
    (matches fd507a58, c2fec761): such spots are placed exactly as asked.
    Cached per view."""
    nw, nh = _footprint(item)
    key = (id(view), item, near, tuple(skip_types))     # by item: same-size shapes differ (proc, fix)
    hit = _SPOTS.get(key)
    if hit is not None and hit[0] is view:     # the same view object, not a reused id
        return hit[1]
    anchors = [(b.type, b.cell) for b in view.buildings
               if b.cell is not None and not b.wreck and b.type not in skip_types and b.type not in _NO_AREA]
    if near is not None:
        anchors = [b for b in anchors if dist2(b[1], near) <= C.EXPAND_REFINERY_RADIUS_CELLS ** 2]
    gap = C.BUILDING_GAP_CELLS
    found: set = set()
    for kind, cell in anchors:
        x0, y0, x1, y1 = _rect(kind, cell)
        # Gaps count solid cells only (a refinery's or depot's empty corners
        # don't reach): OpenRA's IsCloseEnoughToBase.
        for sx in range(x0 - nw - gap, x1 + gap + 2):
            for sy in range(y0 - nh - gap, y1 + gap + 2):
                if _gap(item, (sx, sy), kind, cell) == gap:
                    found.add((sx, sy))
    # The build radius (the lobby's "buildradius"): the building's centre
    # within BUILD_RADIUS_CELLS of one of our construction yards. Placements
    # farther out were all moved home (placement audit over six games:
    # beyond 16 cells of a yard every one, 14-16 about half).
    yards = []
    for b in view.buildings:
        if b.type == C.CONSTRUCTION_YARD and b.cell is not None and not b.wreck:
            dx, dy = centre_offset(b.type)
            yards.append((b.cell[0] + dx, b.cell[1] + dy))
    if near is not None and yards:
        # Growing one base: inside ITS yard's radius, not the main base's
        # (anchors near the expansion can reach back toward home).
        yards = [min(yards, key=lambda y: dist2(y, near))]
    r2 = C.BUILD_RADIUS_CELLS ** 2

    def in_radius(s):
        c = (s[0] + (nw - 1) / 2, s[1] + (nh - 1) / 2)
        return not yards or any(dist2(c, y) <= r2 for y in yards)

    failed = view.failed_cells
    spots = sorted(s for s in found if in_radius(s) and _in_bounds(view, item, s) and not _clashes(item, s, view)
                   and not (failed and any(c in failed for c in _cells(item, s))))
    if len(_SPOTS) > 64:
        _SPOTS.clear()
    _SPOTS[key] = (view, spots)                # holding the view keeps its id from being reused
    return spots


def _pick(view: View, item: str, spots, score, slack: float | None = None) -> tuple[int, int] | None:
    """The best spot by ``score`` (lower is better; a tuple ranks tiers
    first), a spot with no unit on it or beside it beating one with units
    there within the same first tier: OpenRA moves a building placed over a
    unit to the main yard, and units beside a spot step onto it. ``None``
    when there is none. ``slack``: units in the way only decide among spots
    scoring within this of the best (turrets: the direction matters more,
    and units on the chosen spot are cleared off it)."""
    if not spots:
        return None
    if slack is not None:
        best = min(score(s) for s in spots)
        spots = [s for s in spots if score(s) <= best + slack]
    occupied = _occupied(view)
    nw, nh = _footprint(item)

    def in_the_way(spot):
        # a unit on the footprint or right beside it (it may step onto it)
        return any((x, y) in occupied for x in range(spot[0] - 1, spot[0] + nw + 1)
                   for y in range(spot[1] - 1, spot[1] + nh + 1))

    def rank(spot):
        s = score(spot)
        tier, rest = (s[0], s[1:]) if isinstance(s, tuple) else (0, (s,))
        return (tier, in_the_way(spot)) + rest

    return min(spots, key=rank)


def grow_spot(view: View, bearing, item: str, near=None, skip_types=(), side_weight: float = 0.5,
              slack: float | None = None) -> tuple[int, int] | None:
    """A legal spot (:func:`legal_spots`) as far along ``bearing`` (a unit
    vector) as possible without drifting far to the side: the base grows in
    the wanted direction building by building. ``None`` (engine default)
    when nothing fits."""
    if bearing is None:
        return None
    spots = legal_spots(view, item, near, skip_types)
    if not spots:
        return None
    ux, uy = bearing
    nw, nh = _footprint(item)
    origin = near or view.base_center
    if origin is None:
        anchors = [b.cell for b in view.buildings if b.cell is not None and not b.wreck]
        origin = (sum(c[0] for c in anchors) / len(anchors), sum(c[1] for c in anchors) / len(anchors))

    def ahead(spot):
        return (spot[0] + (nw - 1) / 2 - origin[0]) * ux + (spot[1] + (nh - 1) / 2 - origin[1]) * uy

    def score(spot):
        ox, oy = spot[0] + (nw - 1) / 2 - origin[0], spot[1] + (nh - 1) / 2 - origin[1]
        return -(ox * ux + oy * uy - side_weight * abs(oy * ux - ox * uy))

    # Only the half of the base facing the bearing, when it has a spot: units
    # standing in front (the massing army) must not push a building behind
    # the base (match 08560d9b: the first flame tower went up at the back).
    # Units on the chosen spot are cleared off it (executor._keep_clear).
    front = [s for s in spots if ahead(s) > 0]
    return _pick(view, item, front or spots, score, slack)


def _beside_ore(view: View, ore, item: str, near=None) -> tuple[int, int] | None:
    """A legal refinery spot (:func:`legal_spots`) nearest ``ore`` whose
    trucks can WALK to that field's ore in ``REFINERY_ORE_WALK_STEPS`` — not
    across a cliff; with no walkable spot, the nearest spot anyway (better a
    refinery than none). ``near``: only beside one expansion's buildings (a
    spot beside the main base put the expansion's refinery at home — match
    0561274e)."""
    from .terrain import field_cells, steps_from_rect, walk_distances

    if ore is None:
        return None
    field = field_cells(view, ore, C.EXPAND_CLUSTER_CELLS * 1.5) or [ore]
    walk = walk_distances(view, field, C.REFINERY_ORE_WALK_STEPS + 2)
    nw, nh = _footprint(item)
    spots = legal_spots(view, item, near)

    def score(spot):
        steps = steps_from_rect(walk, _rect(item, spot))
        walkable = steps is not None and steps <= C.REFINERY_ORE_WALK_STEPS
        return (not walkable, dist2((spot[0] + (nw - 1) / 2, spot[1] + (nh - 1) / 2), ore))

    return _pick(view, item, spots, score)


def scout_watch(view: View, base) -> tuple[int, int] | None:
    from .scout import watch_point
    return watch_point(view, base)


def _watch_spot(view: View, item: str, base, near=None) -> tuple[int, int] | None:
    """A legal spot within ``WATCH_SPOT_CELLS`` of the base's choke, while no
    defense of ours stands there yet (the screenshot "potential-towers": a
    forward tower at each gap on the enemy's way in)."""
    watch = scout_watch(view, base)
    if watch is None:
        return None
    r2 = C.WATCH_SPOT_CELLS ** 2
    if any(b.type in C.DEFENSES and b.cell is not None and not b.wreck and dist2(b.cell, watch) <= r2
           for b in view.buildings):
        return None                    # it has its watch tower
    spots = [s for s in legal_spots(view, item, near) if dist2(s, watch) <= r2]
    return _pick(view, item, spots, lambda s: dist2(s, watch)) if spots else None


def perimeter_spot(view: View, base: tuple[int, int], toward, item: str = "pbox", near=None,
                   prefer=None, strict: bool = True) -> tuple[int, int] | None:
    """A turret slot on the edge of the base, in a fan facing the enemy.

    The slots sit at ``DEFENSE_FAN_DEGREES`` from the bearing to ``toward``
    (the enemy) — straight ahead, then evenly to either side, never behind —
    each just outside our frontmost building that way (:func:`grow_spot`).
    The first slot with no defense within ``DEFENSE_MIN_GAP_CELLS`` wins, so
    turrets form an even front line instead of piling up wherever the last
    attack came from. ``prefer`` (a threat, or the advisor's pick) only
    reorders the slots: the free slot nearest it first. With every slot taken,
    ``strict`` returns None (build nothing); otherwise the nearest slot anyway.
    """
    import math

    if toward is None:
        return None
    base_angle = math.atan2(toward[1] - base[1], toward[0] - base[0])
    slots = list(C.DEFENSE_FAN_DEGREES)
    if prefer is not None:
        rel = math.degrees(math.atan2(prefer[1] - base[1], prefer[0] - base[0]) - base_angle)
        rel = (rel + 180) % 360 - 180
        slots.sort(key=lambda d: abs(d - rel))
    defenses = [b.cell for b in view.buildings if b.type in C.DEFENSES and b.cell is not None and not b.wreck]
    gap2 = C.DEFENSE_MIN_GAP_CELLS ** 2
    fallback = None
    for deg in slots:
        angle = base_angle + math.radians(deg)
        # Grow from base buildings only: growing from the last turret chained
        # turrets out toward the enemy one past another.
        # Close to the slot's line beats far forward: a turret beside the
        # road in covers it, one far out to the side covers nothing.
        spot = grow_spot(view, (math.cos(angle), math.sin(angle)), item, near=near, skip_types=C.DEFENSES,
                         side_weight=C.DEFENSE_SIDE_WEIGHT, slack=C.DEFENSE_UNIT_SLACK)
        if spot is None:
            continue
        fallback = fallback or spot
        if all(dist2(spot, c) >= gap2 for c in defenses):
            return spot
    return None if strict else fallback


def defense_orders(view: View, under_attack: bool, expansion_site: tuple[int, int] | None = None,
                   home: tuple[int, int] | None = None, attacked: bool = True) -> list[dict]:
    """Zero or one ``build`` for a base defense (its own Defense queue, so it
    never holds up the Building queue).

    One standing defense on the enemy's approach once the war factory is up;
    up to ``MAX_DEFENSES`` while the base is under attack — the guide's advice
    is to have one ready and put it down "right as the enemy is attacking".
    Kept few and cheap: the match is short, and money spent on concrete is
    money not spent on harvesters and tanks.
    """
    defense = _available(view, C.DEFENSES)
    if defense is None:
        return []
    if any(view.queued_count(d) for d in C.DEFENSES) or any(
            p.get("ready") and str(p.get("item") or "").lower() in C.DEFENSES for p in view.production):
        return []
    enemy = enemy_spawn_guess(view)
    # An expansion still without its turret — only if its fan has a free slot.
    if expansion_site is not None and perimeter_spot(view, expansion_site, enemy, defense, near=expansion_site):
        return [orders.build(defense)]
    # Once the main base's ore is gone, its turrets stop and the active base
    # (``home``, an expansion) gets a full fan instead.
    base = home or view.base_center
    if home is None:
        from .scout import approach_cell

        enemy = approach_cell(view)
    if base is None:
        return []
    r2 = C.DEFENSE_MAIN_RADIUS_CELLS ** 2
    have = sum(1 for b in view.buildings if b.type in C.DEFENSES and b.cell is not None
               and not b.wreck and dist2(b.cell, base) <= r2)
    cap = C.MAX_DEFENSES if under_attack else (C.STANDING_DEFENSES if view.has_building(C.WAR_FACTORY) else 0)
    if under_attack and not attacked and not view.has_building(C.WAR_FACTORY):
        # A warning, not an attack: one turret, then the war factory
        # (0ee93a05 put three pillboxes ahead of it; it came at 3:16, not 2:40).
        cap = min(cap, 1)
    if home is not None:
        cap = max(cap, C.ACTIVE_BASE_DEFENSES)
    if have >= 1 and view.cash + view.ore < C.DEFENSE_ADD_FUNDS:
        return []   # broke: the last money goes to units (match 4bc1ee71 fed 6 pillboxes to an attack)
    if have >= cap or perimeter_spot(view, base, enemy, defense, near=home) is None:
        return []   # enough, or no free slot in the fan: build nothing rather than misplace it
    return [orders.build(defense)]


# -- helpers -----------------------------------------------------------------


# Spare buildings, sold first to last when broke with no income. Never the
# yard, a refinery, the last war factory or barracks, or power.
_SELL_ORDER = ("dome", "fix", "silo", "tsla", "gun", "ftur", "pbox", "hbox", "agun", "sam", "barr", "tent", "weap")


def sell_for_income(view: View) -> dict | None:
    """With no harvester (none queued) and not enough money for one, sell a
    spare building to pay for it — a base with no income just waits to die
    (match b7fcbd4e: 0 harvesters and 0 cash for the last 10 minutes)."""
    if view.units_of(C.HARVESTER) or any(view.queued_count(h) for h in C.HARVESTER):
        return None
    if not view.has_building(C.REFINERY) or not view.has_building(C.WAR_FACTORY):
        return None   # a harvester can't be had anyway
    cost = view.unit_costs.get(C.HARVESTER[0], 1100.0)
    if view.cash + view.ore >= cost:
        return None
    for kind in _SELL_ORDER:
        mine = [b for b in view.buildings if b.type == kind and b.id is not None and not b.wreck]
        keep = 1 if kind in C.BARRACKS or kind == C.WAR_FACTORY else 0
        if len(mine) > keep:
            return orders.sell(mine[-1].id)
    return None


def open_ore(view: View, depleted: frozenset = frozenset(),
             radius: float | None = C.REFINERY_ORE_SEARCH_CELLS) -> tuple[int, int] | None:
    """Where the next refinery goes: the ore cell, nearest our yards, of the
    field with the most room — fewest refineries for its size — among fields
    within ``radius`` of a yard (None: any distance) that are not mined out.

    A field's capacity is one refinery per ``ORE_CELLS_PER_REFINERY`` ore
    cells, at least 1 and at most ``MAX_REFINERIES_PER_FIELD``. Spreading by
    load, not distance: the nearest field used to take every refinery (match
    a2952a6d: three around a 37-cell field 8 cells out, none at the 47-cell
    field 11 cells out, while the trucks drove there; b7fcbd4e: five on one
    field)."""
    from .expand import _ore_components
    from .harvest import refinery_fields
    from .scout import our_yards

    yards = our_yards(view)
    if not yards or not view.ore_cells:
        return None
    comps = _ore_components(view)
    owner = {c: i for i, comp in enumerate(comps) for c in comp}
    load = [0] * len(comps)
    for key, n in refinery_fields(view).items():
        if key in owner:
            load[owner[key]] += n
    field2 = (C.EXPAND_CLUSTER_CELLS * 1.5) ** 2
    r2 = radius * radius if radius is not None else None
    best = None
    for i, comp in enumerate(comps):
        cap = max(1, min(C.MAX_REFINERIES_PER_FIELD, len(comp) // C.ORE_CELLS_PER_REFINERY))
        if load[i] >= cap:
            continue
        live = [c for c in comp if not any(dist2(c, f) <= field2 for f in depleted)]
        if not live:
            continue
        d, cell = min((min(dist2(c, y) for y in yards), c) for c in live)
        if r2 is not None and d > r2:
            continue
        rank = (load[i] / cap, d)
        if best is None or rank < best[0]:
            best = (rank, cell)
    return best[1] if best else None


def live_ore_near_yards(view: View, depleted: frozenset = frozenset()) -> tuple[int, int] | None:
    """The ore cell nearest one of our construction yards that is not in a
    mined-out field (``depleted`` holds field cells from harvest.py), within
    reach of a refinery; None if there is none."""
    from .scout import our_yards

    field2 = (C.EXPAND_CLUSTER_CELLS * 1.5) ** 2
    reach2 = C.REFINERY_ORE_SEARCH_CELLS ** 2
    yards = our_yards(view)
    live = [c for c in view.ore_cells if not any(dist2(c, f) <= field2 for f in depleted)]
    near = [(min(dist2(c, y) for y in yards), c) for c in live if yards and any(dist2(c, y) <= reach2 for y in yards)]
    return min(near)[1] if near else None


def nearest_live_ore(view: View, depleted: frozenset = frozenset()) -> tuple[int, int] | None:
    """The live ore cell nearest any of our yards, however far."""
    from .scout import our_yards

    field2 = (C.EXPAND_CLUSTER_CELLS * 1.5) ** 2
    yards = our_yards(view)
    live = [c for c in view.ore_cells if not any(dist2(c, f) <= field2 for f in depleted)]
    if not live or not yards:
        return None
    return min(live, key=lambda c: min(dist2(c, y) for y in yards))


def _crawl_toward(view: View, target, item: str) -> tuple[int, int] | None:
    """A spot one step from our frontmost building toward ``target``: the
    base grows that way building by building ("crawl your base toward the new
    resource field")."""
    if target is None or view.base_center is None:
        return None
    from .scout import our_yards

    yard = min(our_yards(view), key=lambda y: dist2(y, target))
    return grow_spot(view, _unit(yard, target), item)


def _crawl_from_nearest(view: View, target, item: str, near=None) -> tuple[int, int] | None:
    """Grow from the buildings nearest ``target`` (e.g. an expansion's) toward
    it; ``near``: only from buildings around it (one expansion's)."""
    ours = [b.cell for b in view.buildings if b.cell is not None and not b.wreck and b.type not in _NO_AREA
            and (near is None or dist2(b.cell, near) <= C.EXPAND_REFINERY_RADIUS_CELLS ** 2)]
    if not ours:
        return None
    anchor = min(ours, key=lambda c: dist2(c, target))
    return grow_spot(view, _unit(anchor, target), item, near=anchor)


def _ahead(view: View, money: float) -> bool:
    """Ahead enough to keep growing outward: winning on trades, or rich."""
    return view.kills_cost - view.deaths_cost > 0 or money >= C.FUNDS_RICH


def barracks_target(view: View) -> int:
    """1 barracks until the opening is done (service depot up: pros add the
    2nd around 4:00-4:20, after it); then 2, and more by game time."""
    if not view.has_building(C.SERVICE_DEPOT):
        return 1
    want = 2
    for at, n in C.BARRACKS_SCHEDULE:
        if view.game_seconds >= at:
            want = max(want, n)
    return min(want, C.MAX_BARRACKS)


def refinery_target(view: View) -> int:
    """Refineries wanted by game time (REFINERY_SCHEDULE), capped."""
    want = 0
    for at, n in C.REFINERY_SCHEDULE:
        if view.game_seconds >= at:
            want = n
    return min(want, C.MAX_REFINERIES)


def _ahead_of(view: View, item: str | None) -> str | None:
    """The unfinished item at the head of the Building queue when ``item`` is
    waiting behind it, else ``None``."""
    if item is None:
        return None
    waiting = [
        str(p.get("item") or "").lower() for p in view.production
        if str(p.get("queue") or "").lower() == "building" and not p.get("ready")
    ]
    if waiting and waiting[0] != item and item in waiting:
        return waiting[0]
    return None


def _ahead_of_power(view: View, powr: str | None) -> str | None:
    """The unfinished item at the head of the Building queue when a power
    plant is waiting behind it, else ``None``."""
    if powr is None:
        return None
    waiting = [
        str(p.get("item") or "").lower() for p in view.production
        if str(p.get("queue") or "").lower() == "building" and not p.get("ready")
    ]
    if waiting and waiting[0] != powr and powr in waiting:
        return waiting[0]
    return None


def _ids(step):
    return (step,) if isinstance(step, str) else tuple(step)


def _is_building(view: View, item: str, queue: str | None = None) -> bool:
    """Whether a ready queue item is a building (it waits to be placed) vs a
    unit (it leaves the factory by itself).

    Anything finished in the Building or Defense queue is a building, and so
    is any known building id. A finished building nobody places blocks its
    queue for good (match 14a78c43: a kennel, missing from a hand-kept list,
    sat unplaced from 4:05 and stalled power, barracks, refineries and the war
    factory for the rest of the game with 9600 in the bank)."""
    if str(queue or "").lower() in ("building", "defense"):
        return True
    return item in C.ALL_BUILDING_TYPES
