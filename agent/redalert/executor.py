"""The executor: one batch of orders per view, from the view and the mode.

In priority order: place and build (buildplan), defenses, economy (idle
harvesters, repairs, rally points), unit production, squad orders, army
orders. No model call; the batch is capped by ``orders.batch``.
"""

from __future__ import annotations

from . import buildplan, orders, scout
from . import config as C
from .modes import AGGRESSIVE, DEFENSIVE, Mode
from .observe import View, dist2, enemy_spawn_guess


def plan(
    view: View, mode: Mode, *, scout_id: int | None = None, scout_orders: list[dict] = (),
    raid_ids: list[int] = (), raid_orders: list[dict] = (), advice=None,
    flank_via: tuple[int, int] | None = None,
    expansion=None,
    hunt: tuple[int, int] | None = None,
    hunt_points: list | None = None,
    harvest_orders: list[dict] | None = None,
    depleted: frozenset = frozenset(),
    home: tuple[int, int] | None = None,
    clearing: dict | None = None,
    sparring=None,   # sparring (TEST ONLY): an archetype's build, turrets and production
    threat: bool = False,
    probe: tuple[int, int] | None = None,
    warning: bool = False,
    fan: bool = False,
) -> list[dict]:
    """All orders for this view, concatenated in priority order (pre-cap).

    ``scout_id`` is the roaming scout (kept out of the army; ``None`` means we
    have none, so one is trained) and ``scout_orders`` its patrol leg.
    ``raid_ids`` / ``raid_orders`` are the harvester-raid squad (raid.py) and the
    home guard (guard.py), also
    kept out of the army. ``advice`` is the model advisor's fresh advice, if
    any (advisor.py): targets, route, raid target and turret side, within the rules' limits.
    """
    if not view.ok:
        return []
    out: list[dict] = []
    # Requests are throttled (asked for once per retry window); placement is
    # not: a finished refinery or turret goes to the expansion that lacks one.
    exp_ore = expansion.refinery_ore(view) if expansion is not None else None
    exp_def = expansion.defense_site(view) if expansion is not None else None
    job = expansion.refinery_job(view) if expansion is not None else None
    place_def = expansion.needs_defense(view) if expansion is not None else None
    placed = buildplan.placement_orders(
        view, getattr(advice, "defense_at", None), job["ore"] if job else None, place_def, depleted, home,
        expansion_site=job["site"] if job else None, clearing=clearing)
    # Units stepping off a building's spot sit out the army's orders this view
    # (the army code would walk them straight back).
    clearing_ids = {i for o in placed if o.get("cmd") in ("move", "attack_move") for i in o.get("units") or ()}
    out += placed
    out += buildplan.next_build_orders(view, exp_ore, depleted, sparring=sparring,   # advance the build order / power
                                       units_first=getattr(advice, "balance", None) == "more_army")
    out += expansion.orders(view) if expansion is not None else []
    # warning (intel.suspect: one refinery seen early) only adds the warning turret.
    under_attack = mode in (Mode.DEFEND_RUSH, Mode.DEFEND_BASE) or scout.early_push_warning(view) or threat or warning
    # The advisor reads an aggressive enemy: turrets up on the approach (we win on macro).
    under_attack = under_attack or getattr(advice, "enemy_style", None) == "aggressive"
    spar_def = sparring.defense(view, under_attack) if sparring is not None else None
    if spar_def is not None:   # sparring: the rusher's none, the turtler's ring
        out += spar_def
    else:
        out += buildplan.defense_orders(view, under_attack=under_attack,
                                        expansion_site=exp_def if expansion is not None else None, home=home,
                                        attacked=mode in (Mode.DEFEND_RUSH, Mode.DEFEND_BASE))
    out += _economy_orders(view, harvest_orders)
    out += _rally_orders(view)
    out += _production_orders(view, mode, sparring=sparring, threat=threat,
                              units_first=getattr(advice, "balance", None) == "more_army")
    out += list(scout_orders)
    out += list(raid_orders)
    response, responders = home_response(view, mode, scout_id, frozenset(raid_ids) | clearing_ids)
    out += response
    # The massing point, one priority: escort a travelling MCV first, else
    # edge forward while an attack call is being confirmed.
    stage = (expansion.stage_override if expansion is not None else None) or probe
    out += _army_orders(view, mode, scout_id, reserved=frozenset(raid_ids) | clearing_ids | responders, advice=advice,
                        flank_via=flank_via, stage_override=stage, hunt=hunt, hunt_points=hunt_points, fan=fan)
    return _keep_clear(out, placed, clearing_ids)


def _keep_clear(out: list[dict], placed: list[dict], clearing_ids: set) -> list[dict]:
    """Units stepping off a building's spot get no other order this view —
    not from harvesting, a squad, micro or the army — or they walk straight
    back onto it and the building is moved home (match 14a78c43: expansion
    refineries and a silo spot moved home again and again)."""
    if not clearing_ids:
        return out
    keep = {id(o) for o in placed}
    cleaned = []
    for o in out:
        units = o.get("units")
        if id(o) in keep or not isinstance(units, list) or not set(units) & clearing_ids:
            cleaned.append(o)
            continue
        rest = [u for u in units if u not in clearing_ids]
        if rest:
            cleaned.append({**o, "units": rest})
    return cleaned


# -- economy -----------------------------------------------------------------


def _economy_orders(view: View, harvest_orders: list[dict] | None = None) -> list[dict]:
    """Harvest orders (from harvest.HarvestManager: harvesters steered to
    fields that still have ore), else a bare ``harvest`` for idle harvesters;
    plus repairs."""
    if harvest_orders is None:
        idle = [h.id for h in view.units_of(C.HARVESTER) if h.idle and h.id is not None]
        harvest_orders = [orders.harvest(idle)] if idle else []
    return list(harvest_orders) + _repair_orders(view)


def _rally_orders(view: View) -> list[dict]:
    """Point every barracks' and war factory's rally at the staging point, so
    new units walk to the front by themselves (pros set rally points 11-46
    times a game)."""
    stage = scout.choke_cell(view)
    if stage is None:
        return []
    r2 = C.RALLY_TOLERANCE_CELLS ** 2
    return [
        orders.rally(b.id, stage) for b in view.buildings
        if b.id is not None and not b.wreck and (b.type in C.BARRACKS or b.type == C.WAR_FACTORY)
        and (b.rally is None or dist2(b.rally, stage) > r2)
    ]


def stance_orders(view: View, mode: Mode, holders: set) -> list[dict]:
    """Waiting and guarding units hold (``STANCE_HOLD``: fight in range, don't
    chase into traps); attacking and base-defending units engage freely
    (``STANCE_PUSH``). ``holders`` (home guard, garrison) always hold. Only
    units whose reported stance differs get an order."""
    push = mode in (Mode.ALL_IN, Mode.DEFEND_BASE, Mode.DEFEND_RUSH)
    want: dict[str, list[int]] = {}
    for u in view.combat_units:
        if u.id is None or u.stance is None:
            continue
        target = C.STANCE_HOLD if (u.id in holders or not push) else C.STANCE_PUSH
        if u.stance != target:
            want.setdefault(target, []).append(u.id)
    return [orders.set_stance(ids[:orders.MAX_UNITS_PER_ORDER], stance) for stance, ids in want.items()]


def _repair_orders(view: View) -> list[dict]:
    """Repair damaged key buildings (yard, power, refinery, war factory):
    repairing costs far less than rebuilding. ``hp`` is a percentage."""
    return [
        orders.repair(b.id) for b in view.buildings
        if b.type in C.REPAIR_TYPES and b.id is not None and not b.wreck
        and not b.repairing and 0 < b.hp < C.REPAIR_BELOW_HP
    ]


# -- army production ---------------------------------------------------------


def harvester_target(view: View) -> int:
    """How many harvesters we want: 2.5 per refinery (rounded up), at least 4
    once the war factory is up ("4-5 harvesters as quickly as possible"), at
    most 3 per refinery (more just queue to unload), capped overall."""
    refineries = max(1, sum(1 for b in view.buildings if b.type == C.REFINERY and not b.wreck))
    # Pros: trucks come with refineries; train an early pair (and replacements).
    want = refineries + C.EXTRA_HARVESTERS
    target = min(C.MAX_HARVESTERS, C.MAX_HARVESTERS_PER_REFINERY * refineries, want)
    if view.ore_running_dry:
        # The fields are running out: no more trucks (the money goes to the
        # army) until the next expansion's refinery opens fresh ore.
        target = min(target, len(view.units_of(C.HARVESTER)))
    return target


def funds(view: View) -> float:
    """Spendable money: cash plus stored ore."""
    return view.cash + view.ore


def _production_orders(view: View, mode: Mode, *, sparring=None, threat: bool = False,
                       units_first: bool = False) -> list[dict]:
    """Train units: harvesters first (unblocked if stuck behind tanks), then
    the scout vehicle, then an infantry-heavy army with tank support. Broke:
    one queue, one deep. DEFEND_RUSH: rifle spam. Every mode keeps training
    (units count toward assets)."""

    has_barracks = view.has_building(C.BARRACKS)
    has_factory = view.has_building(C.WAR_FACTORY)
    out: list[dict] = []

    rifle = C.resolve((C.RIFLE,), view.available)
    rocket = C.resolve((C.ROCKET,), view.available)
    tank = C.resolve(C.LIGHT_TANKS + C.HEAVY_TANKS, view.available)

    if mode is Mode.DEFEND_RUSH:
        # Cheapest attack units, as fast as possible, to kill the cheese.
        if has_barracks and rifle and view.queued_count(rifle) < C.MAX_QUEUE_DEPTH:
            out.append(orders.train(rifle, count=1))
        return out

    # Economy first. Harvesters share the vehicle queue with tanks, so while we
    # are short the factory makes harvesters only (and no tank spends the cash).
    harv = C.resolve(C.HARVESTER, view.available)
    short_harvesters = (
        harv is not None and has_factory
        and len(view.units_of(C.HARVESTER)) + view.queued_count(harv) < harvester_target(view)
    )
    if short_harvesters and view.queued_count(harv) == 0:
        out.append(orders.train(harv, count=1))
    # A harvester we need, queued behind tanks in the one-at-a-time Vehicle
    # queue, waits for them — minutes at zero cash (match 28e54539: 200 s on
    # one harvester, and the economy never recovered). Cancel what is ahead
    # (the money comes back); the tanks are queued again afterwards.
    if harv is not None and len(view.units_of(C.HARVESTER)) < harvester_target(view):
        blocker = ahead_in_queue(view, "vehicle", harv)
        if blocker is not None:
            out.insert(0, orders.cancel(blocker))

    if sparring is not None:   # sparring (TEST ONLY): the rusher's nonstop units, the macro's token army
        own = sparring.production(view, mode)
        if own is not None:
            return out + own

    # No infantry cap: units spend what the economy doesn't need. The build
    # order (and buildings, harvesters, an MCV in production) is pre-funded —
    # unless a rush is on its way or we are defending: then everything goes
    # into the army.
    # A lost war factory's rebuild keeps its money even then: it is where
    # the tanks (and harvesters) come from.
    reserve = (buildplan.factory_reserve(view) if (threat or mode in DEFENSIVE)
               else buildplan.build_reserve(view, units_first=units_first))
    spare = funds(view) - reserve
    unit_cost = lambda item: view.unit_costs.get(item, C.DEFAULT_UNIT_COST)  # noqa: E731

    # Short on money: one unit queue at a time, one item deep — harvesters
    # first (above), then tanks, then infantry — so money finishes something.
    broke = funds(view) < C.FUNDS_LOW
    if broke and (short_harvesters or view.queued_count(harv or "harv")):
        return out
    rich = funds(view) >= C.FUNDS_RICH
    very_rich = funds(view) >= C.FUNDS_VERY_RICH
    # Never more than MAX_QUEUE_DEPTH queued per unit type, one at a time:
    # deep queues spread the money over many half-built units and kept cash
    # at zero all game (matches b7fcbd4e, 654f1b2e, 159f4f11).
    depth = 1 if broke else min(C.MAX_QUEUE_DEPTH, C.FUNDS_VERY_RICH_DEPTH if very_rich else 2)

    # Infantry-heavy, like the pro replays: riflemen and rocket soldiers (~3:1)
    # from every barracks, tanks as support from one war factory. More
    # barracks build faster, so the infantry queue runs deeper with each.
    barracks = sum(1 for b in view.buildings if b.type in C.BARRACKS and not b.wreck)
    # Against a turtle (several known defenses, e.g. flame towers that shred
    # infantry): fewer riflemen per batch, a deeper tank queue, siege units.
    turtle = len(enemy_defenses(view)) >= C.TURTLE_DEFENSES
    if has_barracks and rifle and spare >= unit_cost(rifle):
        ratio = C.RIFLE_TO_ROCKET if funds(view) >= C.ROCKET_RATIO_CREDIT_SWITCH else C.RIFLE_TO_ROCKET_EARLY
        if view.queued_count(rifle) < depth:
            out.append(orders.train(rifle, count=1))
        if not broke and rocket and _should_add_rocket(view, ratio) and view.queued_count(rocket) < 1 \
                and spare >= unit_cost(rifle) + unit_cost(rocket):
            out.append(orders.train(rocket, count=1))
    if has_factory and tank and not short_harvesters and not broke and spare >= unit_cost(tank):
        if view.queued_count(tank) < min(C.MAX_QUEUE_DEPTH, (2 if rich else 1) + (1 if turtle else 0)):
            out.append(orders.train(tank, count=1))
    # Siege units (V2 / artillery) once defenses are known: they outrange them.
    siege = C.resolve(C.SIEGE_UNITS, view.available)
    if (siege and has_factory and not short_harvesters and not broke and spare >= unit_cost(siege)
            and len(enemy_defenses(view)) >= C.SIEGE_MIN_ENEMY_DEFENSES
            and len(view.units_of(C.SIEGE_UNITS)) + view.queued_count(siege) < C.SIEGE_MAX
            and view.queued_count(siege) == 0):
        out.append(orders.train(siege, count=1))
    return out


def enemy_defenses(view: View) -> list:
    """Enemy defenses we can see or remember."""
    seen = {}
    for a in list(view.enemy_buildings) + list(view.known_enemy_buildings):
        if a.type in C.ENEMY_DEFENSE_TYPES and a.cell is not None and not a.wreck:
            seen[a.id if a.id is not None else a.cell] = a
    return list(seen.values())


def ahead_in_queue(view: View, queue: str, item: str) -> str | None:
    """The unfinished item at the head of ``queue`` when ``item`` is waiting
    behind it, else ``None``."""
    waiting = [
        str(p.get("item") or "").lower() for p in view.production
        if str(p.get("queue") or "").lower() == queue and not p.get("ready")
    ]
    if waiting and waiting[0] != item and item in waiting:
        return waiting[0]
    return None


def _should_add_rocket(view: View, ratio: int) -> bool:
    """Add a rocket when our rifle:rocket mix is below the target ratio."""
    rifles = len(view.units_of(C.RIFLE))
    rockets = len(view.units_of(C.ROCKET))
    return rockets == 0 or rifles >= ratio * (rockets + 1)


# -- army command ------------------------------------------------------------


def _fighters(view: View, scout_id: int | None, reserved: frozenset = frozenset()):
    """Our combat units, minus the roaming scout and the raid squad."""
    return [u for u in view.combat_units if u.id is not None and u.id != scout_id and u.id not in reserved]


def _army_orders(
    view: View, mode: Mode, scout_id: int | None = None, reserved: frozenset = frozenset(), advice=None,
    flank_via: tuple[int, int] | None = None, stage_override: tuple[int, int] | None = None,
    hunt: tuple[int, int] | None = None, hunt_points: list | None = None, fan: bool = False,
) -> list[dict]:
    """Move the army: rally home when defensive, attack-move when aggressive,
    stage at the forward choke while massing.

    Coarse grouping only (tanks issued in the same attack-move as infantry; the
    engine keeps them roughly together) — no continuous concave micro, which
    would be too slow for real time.
    """
    combat = [u.id for u in _fighters(view, scout_id, reserved)]
    if not combat:
        return []

    if mode in DEFENSIVE:
        return _rally_home(view, combat)

    if mode is Mode.RETREAT:
        return _retreat(view, scout_id, reserved)

    if mode in AGGRESSIVE and fan:
        return _fan_out(view, combat, hunt_points or ([hunt] if hunt else []))

    if mode in AGGRESSIVE:
        # With no enemy known the hunter aims the army, not the advisor (it
        # kept naming the razed enemy base for 5 minutes in match 0561274e),
        # and a big army sweeps several search points at once.
        blind = not (view.enemy_buildings or view.enemy_units or view.known_enemy_buildings
                     or any(a.type == C.MCV for a in view.known_enemy_units))
        if blind and hunt is not None:
            units = combat[:orders.MAX_UNITS_PER_ORDER]
            k = min(len(hunt_points or [hunt]), C.HUNT_GROUPS, len(units) // C.HUNT_GROUP_MIN_UNITS)
            if flank_via is None and k >= 2:
                return _groups(view, units, (hunt_points or [hunt])[:k])
            advice = None
        target = _advised_target(view, advice) or _attack_target(view, hunt)
        if target is not None:
            waiting: list[dict] = []
            # attack_move (not attack-on-id) so the army fights whatever it
            # meets on the way instead of walking past it to one building.
            combat, waiting = _gather_stragglers(view, combat, target, stage_override or scout.choke_cell(view))
            units = combat[:orders.MAX_UNITS_PER_ORDER]
            if waiting and not units:
                return waiting
            if flank_via is not None:
                # Flank (or a detour round a choke): round the side first, then on to the target.
                return waiting + [orders.attack_move(units, list(flank_via)),
                                  orders.attack_move(units, list(target), queued=True)]
            # The advisor's own pick is a single point; otherwise split into
            # groups (two clusters, or a semi-concave across the target).
            points = [target] if _advised_target(view, advice) is not None else _attack_points(view, units, target)
            return waiting + _groups(view, units, points)
        # Blind and no spawns known: fall back to the server's ready-made order
        # (without the scout, which the server counts as a combat unit).
        if view.attack_now:
            units = [u for u in view.attack_now.get("units") or [] if u != scout_id and u not in reserved]
            return [{**view.attack_now, "units": units}] if units else []
        return []

    return _stage_forward(view, scout_id, reserved, advice, stage_override)  # OPEN / BUILD: mass at the choke, not in the base


def _gather_stragglers(view: View, units: list[int], target, stage) -> tuple[list[int], list[dict]]:
    """In an attack, units far behind the front (``REINFORCE_GAP_CELLS``, nearer
    our massing point than the front) wait at the massing point until
    ``REINFORCE_GROUP_UNITS`` can go together, instead of walking to the
    fight one by one (match 5e840776: 36 at the front, 22 strung out across
    the map in ones and twos, single file through the choke).
    Returns (the units that attack, the orders for the ones that wait)."""
    cells = {u.id: u.cell for u in view.combat_units if u.cell is not None}
    have = [i for i in units if i in cells]
    if stage is None or len(have) < 2 * C.REINFORCE_GROUP_UNITS:
        return units, []
    near_half = sorted(have, key=lambda i: dist2(cells[i], target))[:max(1, len(have) // 2)]
    front = (sum(cells[i][0] for i in near_half) / len(near_half), sum(cells[i][1] for i in near_half) / len(near_half))
    gap2 = C.REINFORCE_GAP_CELLS ** 2
    hold2 = C.REINFORCE_HOLD_CELLS ** 2
    # Only units still at the massing point wait; one already on its way keeps
    # going (match 906dbb32: infantry falling behind the tanks were sent back
    # and forth between the push and the massing point).
    behind = [i for i in have if dist2(cells[i], front) > gap2 and dist2(cells[i], stage) <= hold2]
    if not behind or len(behind) >= C.REINFORCE_GROUP_UNITS:
        return units, []                      # none behind, or enough to go as a group
    wait = [i for i in behind if dist2(cells[i], stage) > C.STAGING_RADIUS_CELLS ** 2]
    rest = [i for i in units if i not in set(behind)]
    return rest, ([orders.attack_move(wait[:orders.MAX_UNITS_PER_ORDER], list(stage))] if wait else [])


def _fan_out(view: View, units: list[int], points: list) -> list[dict]:
    """Their main base is razed: find and kill the rest. Each building
    cluster we know of (or an MCV) gets the nearest ``FOUND_SQUAD_UNITS``;
    everyone else fans out over the map's unsearched cells in groups of
    ``FAN_GROUP_UNITS``, one group per cell, nearest first (match c4caa45e: a
    100-unit army in 3 groups missed a barracks for 80 s; b90e492f razed four
    expansions one or two at a time)."""
    cells = {u.id: u.cell for u in view.combat_units if u.cell is not None}
    left = [i for i in units if i in cells]
    targets: list[tuple[int, int]] = []
    found = [b.cell for b in list(view.enemy_buildings) + list(view.known_enemy_buildings)
             if b.cell is not None and not b.wreck]
    found += [a.cell for a in list(view.enemy_units) + list(view.known_enemy_units)
              if a.type == C.MCV and a.cell is not None and not a.wreck]
    for c in found:
        if all(dist2(c, t) > C.ENEMY_CLUSTER_CELLS ** 2 for t in targets):
            targets.append(c)
    out: list[dict] = []
    for t in targets:
        if not left:
            break
        squad = sorted(left, key=lambda i: dist2(cells[i], t))[:C.FOUND_SQUAD_UNITS]
        out.append(orders.attack_move(squad, list(t)))
        left = [i for i in left if i not in squad]
    if left and points:
        k = min(len(points), max(1, len(left) // C.FAN_GROUP_UNITS))
        out += _groups(view, left, points[:k])
    return out


def _rally_home(view: View, combat: list[int]) -> list[dict]:
    """Defend the base: attack-move onto the nearest enemy near it, else
    gather at the base centre.

    Going to the centre alone left the army idle while the enemy shot
    buildings at the edge of the base, out of its range (match 9a9cd98e: 200 s
    "defending" with 10-16 units while three war factories died).
    """
    if view.base_center is None:
        return []
    threats = [a for a in scout.enemy_threat_near_base(view) if a.cell is not None]
    if threats:
        yards = scout.our_yards(view)   # the threat nearest any of our yards (expansions too)
        rally = min(threats, key=lambda a: min(dist2(a.cell, y) for y in yards)).cell
    else:
        rally = view.base_center
    return [orders.attack_move(combat[:orders.MAX_UNITS_PER_ORDER], list(rally))]


def _retreat(view: View, scout_id: int | None, reserved: frozenset = frozenset()) -> list[dict]:
    """Pull every fighter not yet back to the staging point (else home): a
    fighting withdrawal. Attack-move only: a plain ``move`` let units be shot
    in the back by longer-ranged enemies without answering.
    """
    point = scout.choke_cell(view) or view.base_center
    if point is None:
        return []
    r2 = C.REGROUP_RADIUS_CELLS ** 2
    away = [u.id for u in _fighters(view, scout_id, reserved) if dist2(u.cell, point) > r2]
    return [orders.attack_move(away[:orders.MAX_UNITS_PER_ORDER], list(point))] if away else []


def _stage_forward(
    view: View, scout_id: int | None = None, reserved: frozenset = frozenset(), advice=None,
    stage_override: tuple[int, int] | None = None,
) -> list[dict]:
    """While massing, gather at the massing point (scout.choke_cell; an
    override while escorting an MCV or edging forward to confirm an attack)
    in a concave facing the enemy base (:func:`concave_slots`); a handful of
    units just attack-move there.

    The rules own staging: the advisor's ``stage_at`` and ``stage_facing``
    are not used (match a2952a6d: it massed the army back toward our own ore
    field and turned the concave to face our base)."""
    stage = stage_override or scout.choke_cell(view)
    if stage is None:
        return []
    fighters = _fighters(view, scout_id, reserved)
    # More than a small raid near the massing army: it fights instead of
    # standing in formation (08560d9b lost 5100 to 1500 in BUILD while tanks
    # and artillery shot the concave from outside its range). Measured from
    # the fighters: the massing point itself falls back when enemies come.
    # A small raid is the home response's (two riflemen).
    cells = [u.cell for u in fighters if u.cell is not None] or [stage]
    # A real push at any of our bases (yards, refineries) is the army's fight
    # too, not just one near the army: match c7a982ac's 75 units watched a
    # refinery die 17 cells away (they counted as "home defenders", so the
    # defend rule never fired; the raid response stops at 1500).
    cells += [b.cell for b in view.buildings if b.cell is not None and not b.wreck
              and b.type in (C.CONSTRUCTION_YARD, C.REFINERY)]
    target = fight_near(view, cells)
    if fighters and target is not None:
        ids = [u.id for u in fighters][:orders.MAX_UNITS_PER_ORDER]
        ours = sum(view.unit_costs.get(u.type, C.KNOWN_COSTS.get(u.type, C.DEFAULT_UNIT_COST)) for u in fighters)
        if ours < C.ENGAGE_MIN_RATIO * _value(view, _enemies_near(view, cells)) and view.base_center is not None:
            # Outnumbered: back to our turret nearest them (else the base
            # centre), where the turrets, buildings and new units fight with
            # us (match 6b457914 charged 22 rushers with 9 units in front of
            # the base and lost 1300 to 300).
            return [orders.attack_move(ids, list(_fallback_point(view, _enemies_near(view, cells))))]
        return [orders.attack_move(ids, list(target))]
    # Massing (and holding at home) is always a concave facing the enemy's
    # approach: every unit can fire into the middle where they arrive.
    if len(fighters) >= 3:
        return _concave_orders(view, fighters, stage, enemy_spawn_guess(view))
    r2 = C.STAGING_RADIUS_CELLS ** 2
    movers = [u.id for u in fighters if u.idle and dist2(u.cell, stage) > r2]
    return [orders.attack_move(movers[:orders.MAX_UNITS_PER_ORDER], list(stage))] if movers else []


def fight_near(view: View, cells) -> tuple[int, int] | None:
    """A real fight at these cells: enemy fighters worth more than a small
    raid within ``STAGE_ENGAGE_CELLS`` of any of them. The cell of the one
    nearest their middle, else None (a raid is the home response's)."""
    cells = [c for c in cells if c is not None]
    if not cells:
        return None
    near = _enemies_near(view, cells)
    if _value(view, near) <= C.SMALL_RAID_VALUE:
        return None
    centre = (sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells))
    return min(near, key=lambda a: dist2(a.cell, centre)).cell


def _enemies_near(view: View, cells) -> list:
    """Enemy fighters within ``STAGE_ENGAGE_CELLS`` of any of ``cells``."""
    r2 = C.STAGE_ENGAGE_CELLS ** 2
    return [a for a in view.enemy_units if a.cell is not None and not a.wreck and a.type not in C.HARVESTER
            and any(dist2(a.cell, c) <= r2 for c in cells if c is not None)]


def _fallback_point(view: View, enemies) -> tuple[int, int]:
    """Where an outnumbered army stands: beside our turret nearest the
    attackers (within the main base), else the base centre."""
    base = view.base_center
    r2 = C.DEFENSE_MAIN_RADIUS_CELLS ** 2
    towers = [b.cell for b in view.buildings if b.type in C.DEFENSES and b.cell is not None and not b.wreck
              and base is not None and dist2(b.cell, base) <= r2]
    cells = [a.cell for a in enemies if a.cell is not None]
    if not towers or not cells:
        return base
    mid = (sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells))
    return min(towers, key=lambda c: dist2(c, mid))


def _value(view: View, actors) -> float:
    return sum(view.unit_costs.get(a.type, C.KNOWN_COSTS.get(a.type, C.DEFAULT_UNIT_COST)) for a in actors)


def concave_slots(view: View, center, facing, n: int) -> list[tuple[int, int]]:
    """``n`` cells on an arc through ``center``, across the line to ``facing``,
    whose ends curve toward ``facing``: a concave, so every unit can fire into
    the middle where the enemy arrives. Clamped to the map, each moved to
    the nearest open cell (not a cliff, tree or one of our buildings: units
    sent there stopped wherever they got stuck), de-duplicated."""
    if facing is None or n <= 0:
        return [tuple(center)] if n > 0 else []
    dx, dy = facing[0] - center[0], facing[1] - center[1]
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / length, dy / length          # toward the enemy
    px, py = -uy, ux                           # along the line
    from .buildplan import _cells
    taken = set(view.blocked_cells)
    for b in view.buildings:
        if b.cell is not None and not b.wreck:
            taken.update(_cells(b.type, b.cell))
    slots: list[tuple[int, int]] = []
    for i in range(n):
        k = (i - (n - 1) / 2) * C.FORMATION_SPACING_CELLS
        forward = C.FORMATION_CURVE * k * k
        x = round(center[0] + px * k + ux * forward)
        y = round(center[1] + py * k + uy * forward)
        if view.bounds is not None:
            bx, by, bw, bh = view.bounds
            x, y = min(max(x, bx), bx + bw - 1), min(max(y, by), by + bh - 1)
        x, y = _open_slot(view, (x, y), taken)
        if (x, y) not in slots:
            slots.append((x, y))
    return slots


def _open_slot(view: View, cell, taken: set) -> tuple[int, int]:
    """``cell``, or the nearest cell within 3 that no cliff, tree or building
    of ours covers (``taken``)."""
    if cell not in taken:
        return cell
    near = [(cell[0] + i, cell[1] + j) for i in range(-3, 4) for j in range(-3, 4)
            if (cell[0] + i, cell[1] + j) not in taken]
    return min(near, key=lambda c: dist2(c, cell)) if near else cell


def _concave_orders(view: View, fighters, center, facing) -> list[dict]:
    """Attack-move each unit to its concave slot (assigned in order along the line,
    so paths don't cross); units already on their slot get no order. Units
    beyond ``FORMATION_MAX_UNITS`` gather at the centre behind the line."""
    units = [u for u in fighters if u.cell is not None]
    if not units:
        return []
    in_line = units[:C.FORMATION_MAX_UNITS]
    slots = concave_slots(view, center, facing, len(in_line))
    if facing is not None:
        dx, dy = facing[0] - center[0], facing[1] - center[1]
        along = lambda c: -dy * (c[0] - center[0]) + dx * (c[1] - center[1])  # noqa: E731
        in_line.sort(key=lambda u: along(u.cell))
        slots.sort(key=along)
    # attack_move, not move: a unit walking to its slot still fights what it meets.
    out = [orders.attack_move([u.id], list(slot)) for u, slot in zip(in_line, slots) if dist2(u.cell, slot) > 2]
    rest = [u.id for u in units[C.FORMATION_MAX_UNITS:] if u.idle and dist2(u.cell, center) > C.STAGING_RADIUS_CELLS ** 2]
    if rest:
        out.append(orders.attack_move(rest[:orders.MAX_UNITS_PER_ORDER], list(center)))
    return out


def flank_waypoint(view: View, target, side: int) -> tuple[int, int] | None:
    """The waypoint for an attack on ``target`` from the ``side`` (-1 left,
    +1 right, 0 center): halfway there, offset sideways by
    ``FLANK_OFFSET_FRACTION`` of the distance, snapped to the nearest passable
    cell of the map picture and kept inside the map."""
    base = view.base_center
    if base is None or target is None:
        return None
    dx, dy = target[0] - base[0], target[1] - base[1]
    mx, my = base[0] + dx / 2, base[1] + dy / 2
    x, y = round(mx - dy * C.FLANK_OFFSET_FRACTION * side), round(my + dx * C.FLANK_OFFSET_FRACTION * side)
    if view.bounds is not None:
        bx, by, bw, bh = view.bounds
        x, y = min(max(x, bx + 1), bx + bw - 2), min(max(y, by + 1), by + bh - 2)
    if (x, y) in view.blocked_cells:
        free = [(x + i, y + j) for i in range(-6, 7) for j in range(-6, 7) if (x + i, y + j) not in view.blocked_cells]
        if free:
            x, y = min(free, key=lambda c: dist2(c, (x, y)))
    return x, y


def choose_flank(view: View, target, attacks: int) -> tuple[int, tuple[int, int] | None]:
    """(side, waypoint) for the next attack: the route whose waypoint has the
    fewest known enemies (units and defenses) near it; ties rotate with the
    attack count so the enemy can't just wait in the middle."""
    r2 = C.FLANK_DANGER_RADIUS_CELLS ** 2
    enemies = [a for a in list(view.enemy_units) + list(view.known_enemy_units) + list(view.known_enemy_buildings)
               if a.cell is not None and a.type not in C.HARVESTER]
    order = [(0, 1, -1), (1, -1, 0), (-1, 0, 1)][attacks % 3]
    best = None
    for side in order:
        via = flank_waypoint(view, target, side)
        if via is None:
            continue
        danger = sum(1 for a in enemies if dist2(a.cell, via) <= r2)
        if best is None or danger < best[0]:
            best = (danger, side, via)
    if best is None:
        return 0, None
    return best[1], (None if best[1] == 0 else best[2])


def _advised_target(view: View, advice) -> tuple[int, int] | None:
    """The advisor's army target, only while a known enemy (visible, or a
    remembered building) stands within ``ADVICE_TARGET_SUPPORT_CELLS`` of it:
    a stale pick must not hold the army. (Match b7fcbd4e: it named the razed
    enemy main for 5 minutes while their last building stood 10 cells away.)"""
    cell = getattr(advice, "army_target", None)
    if cell is None:
        return None
    r2 = C.ADVICE_TARGET_SUPPORT_CELLS ** 2
    known = list(view.enemy_buildings) + list(view.enemy_units) + list(view.known_enemy_buildings)
    if any(a.cell is not None and not a.wreck and dist2(a.cell, cell) <= r2 for a in known):
        return tuple(cell)
    return None


def home_response(view: View, mode: Mode, scout_id: int | None, reserved: frozenset) -> tuple[list[dict], set]:
    """A small raid on our base or harvesters (enemies worth at most
    ``SMALL_RAID_VALUE`` within ``HOME_RAID_RADIUS_CELLS`` of a yard or a
    harvester of ours): the units at home — new ones included — kill it, the
    weakest raider first, instead of walking past to the front. (Match
    c2fec761: a low-health ranger harassed the base unanswered.) Bigger
    threats are the defensive modes' job. Returns (orders, unit ids used)."""
    from .guard import _at_home

    if mode in DEFENSIVE or mode is Mode.RETREAT:
        return [], set()
    anchors = [b.cell for b in view.buildings if b.type == C.CONSTRUCTION_YARD and b.cell is not None]
    anchors += [h.cell for h in view.units_of(C.HARVESTER) if h.cell is not None]
    if view.base_center is not None:
        anchors.append(view.base_center)
    r2 = C.HOME_RAID_RADIUS_CELLS ** 2
    raiders = [a for a in view.enemy_units if a.cell is not None and not a.wreck and a.id is not None
               and a.type not in C.HARVESTER and any(dist2(a.cell, c) <= r2 for c in anchors)]
    if not raiders:
        return [], set()
    cost = lambda a: view.unit_costs.get(a.type, C.DEFAULT_UNIT_COST)  # noqa: E731
    value = sum(cost(a) for a in raiders)
    if value > C.SMALL_RAID_VALUE:
        return [], set()
    target = min(raiders, key=lambda a: (a.hp if a.hp else 100))
    # The nearest couple of infantry (then anything else): the rest of the
    # army keeps its position, a scout costs us two riflemen's time.
    home = sorted((u for u in _fighters(view, scout_id, reserved) if _at_home(view, u.cell)),
                  key=lambda u: (u.type in C.VEHICLE_TYPES, dist2(u.cell, target.cell)))
    picked, sent = [], 0.0
    for u in home:
        if sent >= C.RESPONSE_VALUE_RATIO * value and len(picked) >= C.RESPONSE_MIN_UNITS:
            break
        picked.append(u.id)
        sent += cost(u)
    if not picked:
        return [], set()
    return [orders.attack(picked[:orders.MAX_UNITS_PER_ORDER], target.id)], set(picked)


def _groups(view: View, units: list[int], points: list) -> list[dict]:
    """The one way to split the army: each unit attack-moves to the nearest
    point whose group still has room (groups of about equal size, nearest
    units placed first). One point: everyone goes there. The hunt, a split
    between two enemy clusters and the semi-concave are just different
    points (:func:`_attack_points`)."""
    points = [tuple(p) for p in points]
    if not units or not points:
        return []
    if len(points) == 1:
        return [orders.attack_move(units, list(points[0]))]
    cells = {u.id: u.cell for u in view.combat_units if u.cell is not None}
    room = {p: -(-len(units) // len(points)) for p in points}
    groups: dict = {p: [] for p in points}
    for uid in sorted(units, key=lambda i: min(dist2(cells.get(i, points[0]), p) for p in points)):
        here = cells.get(uid, points[0])
        p = min((p for p in points if room[p] > 0), key=lambda p: dist2(here, p))
        groups[p].append(uid)
        room[p] -= 1
    return [orders.attack_move(ids, list(p)) for p, ids in groups.items() if ids]


def _attack_points(view: View, units: list[int], target) -> list[tuple[int, int]]:
    """Where the attack groups go.

    - Overwhelming, with two or more enemy building clusters known: the two
      nearest clusters at once (match 4bc1ee71 took a beaten enemy's base and
      expansions one at a time and never won).
    - Else, with ``ATTACK_GROUP_MIN_UNITS`` or more: ``ATTACK_GROUPS`` points
      across the target, ``ATTACK_GROUP_SPREAD_CELLS`` apart, so the army
      arrives as a rough semi-concave and every gun fires.
    - Else the target itself."""
    from .brain import _overwhelming

    if len(units) >= C.SPLIT_MIN_UNITS and _overwhelming(view) and view.base_center is not None:
        clusters: list[tuple[int, int]] = []
        for b in list(view.enemy_buildings) + list(view.known_enemy_buildings):
            if b.cell is not None and not b.wreck and all(dist2(b.cell, c) > C.ENEMY_CLUSTER_CELLS ** 2 for c in clusters):
                clusters.append(b.cell)
        k = min(len(clusters), C.SPLIT_MAX_CLUSTERS, max(2, len(units) // C.SPLIT_UNITS_PER_CLUSTER))
        if len(clusters) >= 2:
            return sorted(clusters, key=lambda c: dist2(c, view.base_center))[:k]
    if len(units) < C.ATTACK_GROUP_MIN_UNITS:
        return [tuple(target)]
    cells = [u.cell for u in view.combat_units if u.id in set(units) and u.cell is not None]
    if not cells:
        return [tuple(target)]
    cx, cy = sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells)
    dx, dy = target[0] - cx, target[1] - cy
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    px, py = -dy / length, dx / length
    k = C.ATTACK_GROUPS
    out = []
    for g in range(k):
        side = (g - (k - 1) / 2) / max(1, (k - 1) / 2)          # -1 .. 1
        x, y = round(target[0] + px * side * C.ATTACK_GROUP_SPREAD_CELLS), round(target[1] + py * side * C.ATTACK_GROUP_SPREAD_CELLS)
        if view.bounds is not None:
            bx, by, bw, bh = view.bounds
            x, y = min(max(x, bx), bx + bw - 1), min(max(y, by), by + bh - 1)
        out.append((x, y))
    return out


def _attack_target(view: View, hunt: tuple[int, int] | None = None) -> tuple[int, int] | None:
    """The cell to attack-move to: a visible enemy building, else the nearest
    visible enemy unit, else an enemy building we remember, else the enemy
    base (server hint or spawn guess).

    Among buildings, the priority follows the OpenRA wiki ("Take out the power
    first ... Next take out any construction buildings"): within
    ``ATTACK_FOCUS_RADIUS_CELLS`` of the nearest building, power beats
    production beats refineries beats the rest — without dragging the army
    across the whole enemy base for it.
    """
    if view.base_center is None:
        return enemy_spawn_guess(view)
    # A known enemy expansion first: away from their army, lightly defended,
    # and their income (match 4b5980f8: a 33k army sat idle while red's
    # expansion was known for the last 2.5 minutes).
    from .expand import enemy_expansions

    expansions = enemy_expansions(view)
    if expansions:
        return min(expansions, key=lambda c: dist2(c, view.base_center))
    for group in (view.enemy_buildings, view.enemy_units, view.known_enemy_buildings):
        seen = [a for a in group if a.cell is not None]
        if not seen:
            continue
        nearest = min(seen, key=lambda a: dist2(a.cell, view.base_center))
        if group is view.enemy_units:
            return nearest.cell
        r2 = C.ATTACK_FOCUS_RADIUS_CELLS ** 2
        cluster = [a for a in seen if dist2(a.cell, nearest.cell) <= r2]
        return min(cluster, key=lambda a: (_priority(a.type), dist2(a.cell, view.base_center))).cell
    # An enemy MCV we remember (a yard that packed up and drove off): the
    # game isn't won until it dies too.
    mcvs = [a for a in view.known_enemy_units if a.type == C.MCV and a.cell is not None]
    if mcvs:
        return min(mcvs, key=lambda a: dist2(a.cell, view.base_center)).cell
    # Nothing known: hunt for the last buildings (scout.Hunter) — not just
    # the enemy spawn, which may be ruins by now.
    return hunt or enemy_spawn_guess(view)


def _priority(kind: str) -> int:
    for rank, kinds in enumerate(C.TARGET_PRIORITY):
        if kind in kinds:
            return rank
    return len(C.TARGET_PRIORITY)
