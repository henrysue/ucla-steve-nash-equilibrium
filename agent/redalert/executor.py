"""The executor: one batch of orders per view, from the view and the mode.

In priority order: place and build (buildplan), defenses, economy (idle
harvesters, repairs, rally points), unit production, squad orders, army
orders. No model call; the batch is capped by ``orders.split``.
"""

from __future__ import annotations

from . import buildplan, orders, scout, strength
from . import config as C
from .modes import AGGRESSIVE, DEFENSIVE, Mode
from .observe import View, dist2, enemy_spawn_guess


def plan(
    view: View, mode: Mode, *, scout_id: int | None = None, scout_orders: list[dict] = (),
    raid_ids: list[int] = (), raid_orders: list[dict] = (), advice=None,
    flank_via=None,
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
    attacked: list | None = None,
    responding: dict | None = None,
    home_memory: dict | None = None,
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
    # The advisor's "more_army" puts units before refineries and the MCV —
    # not against a passive enemy, which we out-expand (5a4bc83f: "more_army"
    # from 5:47 on, its turtle took 9 harvesters to our 8).
    units_first = getattr(advice, "balance", None) == "more_army" and not view.enemy_passive
    # Requests are throttled (asked for once per retry window); placement is
    # not: a finished refinery or turret goes to the expansion that lacks one.
    exp_ore = expansion.refinery_ore(view) if expansion is not None else None
    exp_def = expansion.defense_site(view) if expansion is not None else None
    job = expansion.refinery_job(view) if expansion is not None else None
    place_def = expansion.needs_defense(view) if expansion is not None else None
    placed = buildplan.placement_orders(
        view, getattr(advice, "defense_at", None), job["ore"] if job else None, place_def, depleted, home,
        expansion_site=job["site"] if job else None, clearing=clearing)
    # The expansion's yard spot while its MCV drives there: whoever stands on
    # it steps off, and no order sends anyone onto it (_off_the_yard).
    keep_out = expansion.keep_out() if hasattr(expansion, "keep_out") else frozenset()
    placed = placed + (expansion.clear_orders(view) if hasattr(expansion, "clear_orders") else [])
    # Units stepping off a building's spot sit out the army's orders this view
    # (the army code would walk them straight back).
    clearing_ids = {i for o in placed if o.get("cmd") in ("move", "attack_move") for i in o.get("units") or ()}
    out += placed
    out += buildplan.next_build_orders(view, exp_ore, depleted, sparring=sparring,   # advance the build order / power
                                       units_first=units_first)
    out += expansion.orders(view) if expansion is not None else []
    # warning (intel.suspect: one refinery seen early) only adds the warning turret.
    under_attack = mode in (Mode.DEFEND_RUSH, Mode.DEFEND_BASE) or scout.early_push_warning(view) or threat or warning
    # The advisor reads an aggressive enemy: turrets up on the approach (we win on macro).
    under_attack = under_attack or (getattr(advice, "enemy_style", None) == "aggressive" and not view.enemy_passive
                                    and getattr(advice, "confidence", 1.0) >= C.ADVISOR_ACT_CONFIDENCE)
    spar_def = sparring.defense(view, under_attack) if sparring is not None else None
    if spar_def is not None:   # sparring: the rusher's none, the turtler's ring
        out += spar_def
    else:
        out += buildplan.defense_orders(view, under_attack=under_attack,
                                        expansion_site=exp_def if expansion is not None else None, home=home,
                                        attacked=mode in (Mode.DEFEND_RUSH, Mode.DEFEND_BASE))
    out += _economy_orders(view, harvest_orders)
    out += _rally_orders(view, mode)
    out += _production_orders(view, mode, sparring=sparring, threat=threat,
                              units_first=units_first)
    out += list(scout_orders)
    out += list(raid_orders)
    # The base's defense, one handler: idle units answer what comes near them;
    # the army's own fight, a push coming and massing are _army_home's.
    response, responders = respond(view, mode, scout_id, frozenset(raid_ids) | clearing_ids, attacked or [], advice,
                                   responding)
    out += response
    # The massing point stays at the main choke (an MCV on the road gets the
    # outpost, not the army); it edges forward while an attack is confirmed.
    out += _army_orders(view, mode, scout_id, reserved=frozenset(raid_ids) | clearing_ids | responders, advice=advice,
                        flank_via=flank_via, stage_override=probe, hunt=hunt, hunt_points=hunt_points, fan=fan,
                        home_memory=home_memory)
    return _off_the_yard(view, _keep_clear(out, placed, clearing_ids), keep_out,
                         getattr(expansion, "mcv_id", None))


def _off_the_yard(view: View, out: list[dict], keep: frozenset, mcv_id) -> list[dict]:
    """Every order sending our units (or a rally point) into ``keep`` — the
    expansion's yard spot while its MCV drives there — sent to the nearest
    cell outside it instead: scout posts, concave slots and rally points
    near a natural would otherwise stand on it (d2d8dc27)."""
    if not keep:
        return out
    from .expand import outside

    fixed = []
    for o in out:
        to = o.get("to")
        if (o.get("cmd") in ("move", "attack_move", "rally") and isinstance(to, list) and len(to) == 2
                and tuple(to) in keep and o.get("units") != [mcv_id]):
            spot = outside(view, tuple(to), keep)
            if spot is not None:
                o = {**o, "to": list(spot)}
        fixed.append(o)
    return fixed


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


def _rally_orders(view: View, mode: Mode | None = None) -> list[dict]:
    """Point every barracks' and war factory's rally at the staging point, so
    new units walk to the front by themselves (pros set rally points 11-46
    times a game) — but at the army's line at home (scout.home_line) while
    their units stand at the staging point, a push is coming or we defend:
    new units join the army instead of walking out one by one into theirs
    (2fe84e1f at 4:05: riflemen fresh from the barracks shot one at a time
    at the massing point, where the push stood)."""
    stage = scout.choke_cell(view)
    if stage is None:
        return []
    near = _enemies_near(view, [stage])
    push = scout.incoming_push(view)
    if push is not None:
        stage = push[0]
    elif near:
        stage = scout.home_line(view, _centre(near)) or stage
    elif mode in DEFENSIVE:
        stage = scout.home_line(view) or stage
    r2 = C.RALLY_TOLERANCE_CELLS ** 2
    return [
        orders.rally(b.id, stage) for b in view.buildings
        if b.id is not None and not b.wreck and (b.type in C.BARRACKS or b.type == C.WAR_FACTORY)
        and (b.rally is None or dist2(b.rally, stage) > r2)
    ]


def stance_orders(view: View, mode: Mode, holders: set, prefer: str | None = None) -> list[dict]:
    """Waiting and guarding units hold (``STANCE_HOLD``: fight in range, don't
    chase into traps); attacking and base-defending units engage freely
    (``STANCE_PUSH``). ``holders`` (home guard, garrison) always hold.
    ``prefer``: the advisor's stance for the waiting army (one of
    ``ARMY_STANCES``; "auto" or None keeps the rule). Only units whose
    reported stance differs get an order."""
    push = mode in (Mode.ALL_IN, Mode.DEFEND_BASE, Mode.DEFEND_RUSH)
    waiting = prefer if prefer in C.ARMY_STANCES else C.STANCE_HOLD
    want: dict[str, list[int]] = {}
    for u in view.combat_units:
        if u.id is None or u.stance is None:
            continue
        target = C.STANCE_HOLD if u.id in holders else (C.STANCE_PUSH if push else waiting)
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
    """How many harvesters we want: one per refinery (each comes with one)
    plus ``EXTRA_HARVESTERS``, at most ``MAX_HARVESTERS_PER_REFINERY`` per
    refinery (more just queue to unload), capped overall; no more while the
    fields run dry."""
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
    tank = C.resolve(C.TANK_PREFERENCE, view.available)

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
    reserve = (buildplan.threat_reserve(view) if (threat or mode in DEFENSIVE)
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

    # The pros' mix: riflemen and rocket soldiers (~3:1) and about half the
    # army's value in main battle tanks.
    # Against a turtle (several known defenses, e.g. flame towers that shred
    # infantry): fewer riflemen per batch, a deeper tank queue, siege units.
    turtle = len(enemy_defenses(view)) >= C.TURTLE_DEFENSES
    # The pros' mix: about half the army's value in main battle tanks. While
    # tanks lag (and a main battle tank is offered), the tank queue runs full
    # and infantry wait unless there is money for both (52a77fb1: riflemen
    # and light tanks against a medium-tank army).
    # Against a turtle, siege units first (about one per turret): they share
    # the vehicle queue with tanks, so tanks wait while the siege is short.
    siege = C.resolve(C.SIEGE_UNITS, view.available)
    siege_want = (min(C.SIEGE_MAX_TURTLE, max(C.SIEGE_ATTACK_MIN, len(enemy_defenses(view))))
                  if view.enemy_turtling else 0)
    siege_first = (bool(siege) and has_factory and not short_harvesters
                   and len(view.units_of(C.SIEGE_UNITS)) + view.queued_count(siege or "") < siege_want)
    lag = tanks_lag(view, tank) and has_factory and not short_harvesters and not siege_first
    hold_infantry = (lag and not threat and mode not in DEFENSIVE
                     and spare < unit_cost(tank) + unit_cost(rifle or "e1"))
    if has_barracks and rifle and spare >= unit_cost(rifle) and not hold_infantry:
        ratio = C.RIFLE_TO_ROCKET if funds(view) >= C.ROCKET_RATIO_CREDIT_SWITCH else C.RIFLE_TO_ROCKET_EARLY
        # The counter: more rockets as their army turns to armor (a rifleman
        # does 125/s to a tank, a rocket soldier 2500/s).
        theirs = list(view.enemy_mix) or strength.enemy_kinds(view)   # the last few minutes' sightings
        if len(theirs) >= C.MATCHUP_MIN_UNITS:
            ratio = min(ratio, strength.rifles_per_rocket(theirs))
        if view.queued_count(rifle) < depth:
            out.append(orders.train(rifle, count=1))
        if not broke and rocket and _should_add_rocket(view, ratio) and view.queued_count(rocket) < 1 \
                and spare >= unit_cost(rifle) + unit_cost(rocket):
            out.append(orders.train(rocket, count=1))
    if siege_first:
        if view.queued_count(siege) == 0:
            out.append(orders.train(siege, count=1))
    elif has_factory and tank and not short_harvesters and (lag or (not broke and spare >= unit_cost(tank))):
        if view.queued_count(tank) < min(C.MAX_QUEUE_DEPTH, (2 if rich or lag else 1) + (1 if turtle else 0)):
            out.append(orders.train(tank, count=1))
    # Siege units (V2 / artillery): part of the mix (SIEGE_VALUE_SHARE of
    # combat value) once the dome offers them, and once defenses are known
    # (they outrange them).
    if (siege and not siege_first and has_factory and not short_harvesters and not broke and spare >= unit_cost(siege)
            and (len(enemy_defenses(view)) >= C.SIEGE_MIN_ENEMY_DEFENSES or siege_short(view))
            and len(view.units_of(C.SIEGE_UNITS)) + view.queued_count(siege) < C.SIEGE_MAX
            and view.queued_count(siege) == 0):
        out.append(orders.train(siege, count=1))
    return out


def siege_short(view: View) -> bool:
    """Siege units (queued counted) under ``SIEGE_VALUE_SHARE`` of our
    combat value."""
    cost = lambda item: view.unit_costs.get(item, C.DEFAULT_UNIT_COST)  # noqa: E731
    have = sum(cost(u.type) for u in view.combat_units if u.type in C.SIEGE_UNITS)
    have += sum(cost(s) for s in C.SIEGE_UNITS for _ in range(view.queued_count(s)))
    total = sum(cost(u.type) for u in view.combat_units) + sum(
        cost(s) for s in C.SIEGE_UNITS for _ in range(view.queued_count(s)))
    return have < C.SIEGE_VALUE_SHARE * total


def enemy_defenses(view: View) -> list:
    """Enemy defenses we can see or remember."""
    seen = {}
    for a in list(view.enemy_buildings) + list(view.known_enemy_buildings):
        if a.type in C.TURRETS and a.cell is not None and not a.wreck:
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


def tanks_lag(view: View, tank: str | None) -> bool:
    """A main battle tank is offered and tanks hold less than
    ``TANK_VALUE_SHARE`` of our combat value (``TANK_VALUE_SHARE_PASSIVE``
    against a passive enemy: we out-build it; queued tanks count)."""
    if tank not in C.MAIN_BATTLE_TANKS:
        return False
    cost = lambda item: view.unit_costs.get(item, C.DEFAULT_UNIT_COST)  # noqa: E731
    tanks = sum(cost(u.type) for u in view.combat_units if u.type in C.VEHICLE_TYPES)
    tanks += sum(cost(t) for t in C.TANK_PREFERENCE for _ in range(view.queued_count(t)))
    total = sum(cost(u.type) for u in view.combat_units) + tanks - sum(
        cost(u.type) for u in view.combat_units if u.type in C.VEHICLE_TYPES)
    share = C.TANK_VALUE_SHARE_PASSIVE if view.enemy_passive else C.TANK_VALUE_SHARE
    return total <= 0 or tanks < share * total


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
    flank_via=None, stage_override: tuple[int, int] | None = None,
    hunt: tuple[int, int] | None = None, hunt_points: list | None = None, fan: bool = False,
    home_memory: dict | None = None,
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
            if not flank_via and k >= 2:
                return _groups(view, units, (hunt_points or [hunt])[:k])
            advice = None
        # Their buildings or base (route_goal), not whichever unit of theirs is
        # nearest — a harvester coming into sight by our expansion turned
        # the army round (attack-move fights what it meets on the way anyway).
        target = route_goal(view, advice, hunt)
        if target is not None:
            waiting: list[dict] = []
            # attack_move (not attack-on-id) so the army fights whatever it
            # meets on the way instead of walking past it to one building.
            combat, waiting = _gather_stragglers(view, combat, target, stage_override or scout.choke_cell(view))
            units = combat[:orders.MAX_UNITS_PER_ORDER]
            if waiting and not units:
                return waiting
            if flank_via:
                # The route's waypoints (round the chokes, a flank point), then on to the target.
                route = [flank_via] if isinstance(flank_via[0], int) else list(flank_via)
                return waiting + [orders.attack_move(units, list(route[0]))] + [
                    orders.attack_move(units, list(p), queued=True) for p in route[1:] + [target]]
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

    return _army_home(view, mode, scout_id, reserved, stage_override, home_memory)   # OPEN / BUILD / DEFEND / PROTECT_LEAD


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
    # Only idle ones (not already walking out) wait, where they stand: a
    # group released and on its way was cut short and ordered back when its
    # tanks pulled ahead (the rest fell under REINFORCE_GROUP_UNITS).
    idle = {u.id for u in view.combat_units if u.idle}
    behind = [i for i in have if i in idle and dist2(cells[i], front) > gap2 and dist2(cells[i], stage) <= hold2]
    if not behind or len(behind) >= C.REINFORCE_GROUP_UNITS:
        return units, []                      # none behind, or enough to go as a group
    rest = [i for i in units if i not in set(behind)]
    return rest, []


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


def _retreat(view: View, scout_id: int | None, reserved: frozenset = frozenset()) -> list[dict]:
    """Pull every fighter not yet back to the main base's line
    (scout.home_line) — an attack-move, firing at their army as it chases
    (a plain move gave every chaser free shots: a4a78f76 lost 1400 more on
    the way back at 5:54). At the line our turrets fight with us."""
    point = scout.home_line(view) or view.base_center
    if point is None:
        return []
    r2 = C.REGROUP_RADIUS_CELLS ** 2
    away = [u.id for u in _fighters(view, scout_id, reserved) if dist2(u.cell, point) > r2]
    return [orders.attack_move(away[:orders.MAX_UNITS_PER_ORDER], list(point))] if away else []


# -- the base's defense: one handler -------------------------------------------
#
# Everything that comes at us is answered in one place, by one rule each:
#
# - the army at home (massing, defending, protecting a lead): a fight in
#   contact with it or the main base is the army's (army_contact) — it attacks
#   when the odds favour it, else holds the main line with the turrets; a push
#   seen coming has it ready at that line (scout.incoming_push); otherwise the
#   concave at the massing point (_army_home);
# - anything else near us — a small group, a poke at the concave, a raid on the
#   trucks, a spot we were hit at with no attacker in sight — draws the idle
#   units nearby (respond).
#
# Every fight is judged the same (strength.odds). The army never walks out to
# an expansion: its outpost, turret and nearby units hold it.


def enemy_groups(view: View) -> list[list]:
    """The enemy fighters we see (no harvesters), grouped: each within
    ``GROUP_CELLS`` of another member of its group."""
    foes = [a for a in view.enemy_units if a.cell is not None and not a.wreck and a.type not in C.HARVESTER]
    groups: list[list] = []
    r2 = C.GROUP_CELLS ** 2
    for a in foes:
        joined = [g for g in groups if any(dist2(a.cell, b.cell) <= r2 for b in g)]
        merged = [a] + [b for g in joined for b in g]
        groups = [g for g in groups if g not in joined] + [merged]
    return groups


def _centre(actors) -> tuple[int, int]:
    cells = [a.cell for a in actors if a.cell is not None]
    return round(sum(c[0] for c in cells) / len(cells)), round(sum(c[1] for c in cells) / len(cells))


def _main_cells(view: View) -> list[tuple[int, int]]:
    """The main base: its centre, its buildings and the harvesters near it."""
    base = view.base_center
    if base is None:
        return []
    r2 = C.CHEESE_NEAR_BASE_CELLS ** 2
    cells = [base] + [b.cell for b in view.buildings if b.cell is not None and not b.wreck
                      and dist2(b.cell, base) <= C.BUILD_RADIUS_CELLS ** 2]
    return cells + [h.cell for h in view.units_of(C.HARVESTER) if h.cell is not None and dist2(h.cell, base) <= r2]


def army_contact(view: View, fighters, defending: bool = False) -> list:
    """The army's own fight: enemy fighters on our half of the map within
    ``STAGE_ENGAGE_CELLS`` of the army or the main base (buildings, trucks)
    worth more than a small raid — any of them while ``defending`` (a defend
    mode: the threat is real). Empty otherwise (a small group is the idle
    units' job: respond). Their army standing on their side — at the far
    mouth of a choke — is not our fight: the army at home doesn't walk out
    to it (ae03adbd: their concave came within 12 cells of our own
    stragglers in Sidestep's gap and the whole army attack-moved into it,
    6250 lost in 30 s)."""
    enemy_home, base = enemy_spawn_guess(view), view.base_center
    near = _enemies_near(view, [u.cell for u in fighters if u.cell is not None] + _main_cells(view))
    if enemy_home is not None and base is not None:
        near = [a for a in near if dist2(a.cell, enemy_home) >= dist2(a.cell, base)]
    return near if near and (defending or _value(view, near) > C.SMALL_RAID_VALUE) else []


def respond(view: View, mode: Mode, scout_id: int | None, reserved: frozenset, hits=(), advice=None,
            responding: dict | None = None) -> tuple[list[dict], set]:
    """The idle units answer what comes near them, one attack-move per
    target:

    - while the army is home, an enemy group near the concave (within
      ``RESPOND_RADIUS_CELLS`` of it) draws nobody: the concave fires from its slots (its stance) and a
      real push is the army's fight (army_contact) — chasing pokes ran our
      idle units through Sidestep's gap into their massed army for ten
      minutes (e2bc988e lost 10000 more than it killed that way);
    - anything else — a scout or raiders in the base or at the trucks
      (a small group at the main base may also draw idle units at home), a
      spot we were hit at with no attacker in sight (agent.DamageScan) — draws
      only the nearest idle units that beat it clearly
      (``RESPOND_ODDS``, strength.odds; at least ``RESPONSE_MIN_UNITS``; a
      hit with nobody in sight gets ``RESPONSE_MIN_UNITS``). (9ff91807: a lone
      rifleman at the back of the base pulled the whole concave after it,
      twice, and hits on an expansion drew the army north while their
      40-unit push came from the south-west.)

    While the army is home, its own fight (army_contact) is left to it.
    ``advice.guard`` "off" keeps the concave's units out of it.
    ``responding`` (the agent's, kept between views): the units already sent
    and where. Returns (orders, unit ids used)."""
    from .guard import _at_home
    from .scout import our_yards

    if mode is Mode.RETREAT:
        return [], set()
    idle = [u for u in view.combat_units if u.idle and u.id is not None and u.cell is not None
            and u.id != scout_id and u.id not in reserved]
    if not idle:
        return [], set()
    held = scout.concave_point(view)
    if getattr(advice, "guard", "on") == "off" and held is not None:   # the concave's units stay put
        idle = [u for u in idle if dist2(u.cell, held[0]) > C.STAGING_RADIUS_CELLS ** 2 * 4]
    army_fight: set = set()
    if mode not in AGGRESSIVE:
        army_fight = {a.id for a in army_contact(view, _fighters(view, scout_id, reserved), mode in DEFENSIVE)}
    r2 = C.RESPOND_RADIUS_CELLS ** 2
    main = _main_cells(view)
    targets: list[tuple[tuple[int, int], bool, list]] = []    # (cell, small group at the main base, the group)
    groups = [g for g in enemy_groups(view) if not any(a.id in army_fight for a in g)]
    enemy_home = enemy_spawn_guess(view)
    base = view.base_center

    def their_half(cell) -> bool:
        return base is not None and enemy_home is not None and dist2(cell, enemy_home) < dist2(cell, base)

    # Our buildings and trucks, wherever they stand: a group at them is ours
    # to answer even over the half line (an expansion's field across it).
    main_assets = [a.cell for a in list(view.buildings) + list(view.units_of(C.HARVESTER))
                   if a.cell is not None and not a.wreck]
    half_skipped: list = []
    yards = [y for y in our_yards(view) if y is not None]
    push = scout.incoming_push(view)
    push_at = push[1] if push is not None else None      # where the push stands (held a while out of sight)

    for g in sorted(groups, key=lambda g: -_value(view, g)):
        if mode not in AGGRESSIVE and held is not None and any(dist2(a.cell, held[0]) <= r2 for a in g):
            continue                                       # at the concave (the army home): its own fire
        near_ours = any(dist2(a.cell, c) <= C.HOME_RAID_RADIUS_CELLS ** 2 for a in g for c in main_assets)
        if their_half(_centre(g)) and not near_ours:
            half_skipped.append(g)
            continue                                       # on their side: not ours to chase (into their concave)
        centre = _centre(g)
        cell = min(g, key=lambda a: dist2(a.cell, centre)).cell
        # In the base: by its buildings or trucks, or anywhere near a yard —
        # 2fe84e1f's second scout walked round the back of the base, 13 cells
        # from our buildings, and nobody went for it.
        small_home = (_value(view, g) <= C.SMALL_RAID_VALUE
                      and (any(dist2(a.cell, c) <= C.HOME_RAID_RADIUS_CELLS ** 2 for a in g for c in main)
                           or any(dist2(a.cell, y) <= C.BASE_INTRUDER_CELLS ** 2 for a in g for y in yards)))
        targets.append((cell, small_home, g))
    # A hit spot with a group by it is that group's (the concave's own fire
    # included) — but not a group on their side we leave alone.
    seen = [a.cell for g in enemy_groups(view) if g not in half_skipped for a in g]
    # A hit counts where it is ours to answer: by our buildings or trucks —
    # not a scout shot at their concave (ae03adbd: each scout hit at the far
    # mouth of Sidestep's gap sent two more units down it to "look").
    assets = [a.cell for a in list(view.buildings) + list(view.units_of(C.HARVESTER)) if a.cell is not None and not a.wreck]
    h2 = C.HOME_RAID_RADIUS_CELLS ** 2
    hits = [s for s in hits if any(dist2(s, c) <= h2 for c in assets)]
    for spot in hits:
        if all(dist2(spot, c) > C.GROUP_CELLS ** 2 for c in seen) and all(dist2(spot, t[0]) > 36 for t in targets):
            targets.append((tuple(spot), False, []))
    # Units already sent count toward the answer (until they get there, or
    # RESPONSE_HOLD_SECONDS): forgotten each view, a lone scout drew two more
    # idle units every view until the whole concave stood at the back of the
    # base (c1c37a10, at 7:27 and 9:45).
    sent = {} if responding is None else responding
    t = view.game_seconds
    alive = {u.id: u for u in view.combat_units if u.id is not None and u.cell is not None}
    for i, (c, ids0, at) in list(sent.items()):
        u = alive.get(i)
        if u is None or not 0 <= t - at <= C.RESPONSE_HOLD_SECONDS or (u.idle and dist2(u.cell, c) <= 9):
            del sent[i]
    out, used = [], set()
    for cell, small_home, group in targets:
        gids = frozenset(a.id for a in group if a.id is not None)
        # Already sent at this group: by its enemy ids, or to where it stands
        # (units sent to a hit with nobody in sight, before the shooter
        # showed; another member of the group in sight).
        already = [alive[i] for i, (c, ids0, _) in sent.items() if i not in used
                   and ((gids and gids & ids0) or dist2(c, cell) <= C.GROUP_CELLS ** 2)]
        used.update(u.id for u in already)
        near = [u for u in idle if u.id not in used
                and (dist2(u.cell, cell) <= r2 or (small_home and _at_home(view, u.cell)))]
        near.sort(key=lambda u: dist2(u.cell, cell))       # just enough, nearest first
        theirs = [a.type for a in group]
        # A push is the army's, whole (army_contact, incoming_push): never
        # answered in packets — nor the few of it in sight while it is on
        # (2fe84e1f sent 5-6 riflemen at two of a 15-unit push by its war
        # factory, and they died together).
        if theirs and (_value(view, group) > C.SMALL_RAID_VALUE
                       or (push_at is not None and dist2(_centre(group), push_at) <= C.STAGE_ENGAGE_CELLS ** 2)):
            continue
        picked: list = list(already)
        for u in near:
            if len(picked) >= C.RESPONSE_MIN_UNITS and (
                    not theirs or strength.odds(view, [p.type for p in picked], theirs) >= C.RESPOND_ODDS):
                break
            picked.append(u)
        near = [u for u in picked if u not in already]
        if not near:
            continue
        ids = [u.id for u in near][:orders.MAX_UNITS_PER_ORDER]
        mark = [a for a in group if a.id is not None]
        if small_home and mark:
            # In our base: after it by id — an attack-move to where a scout
            # stood lost it as it walked on (2fe84e1f: 40 s shooting our
            # power plant and trucks). RESPONSE_HOLD_SECONDS bounds the chase.
            target = min(mark, key=lambda a: dist2(a.cell, near[0].cell))
            out.append(orders.attack(ids, target.id))
        else:
            out.append(orders.attack_move(ids, list(cell)))
        used.update(ids)
        for i in ids:
            sent[i] = (tuple(cell), gids, t)
    return out, used


def _army_home(view: View, mode: Mode, scout_id: int | None = None, reserved: frozenset = frozenset(),
               stage_override: tuple[int, int] | None = None, memory: dict | None = None) -> list[dict]:
    """The army at home — massing, defending the base, protecting a lead —
    one decision, first match:

    1. In a fight (:func:`army_contact`): attack-move at them when the odds
       favour us (``ENGAGE_MIN_EDGE``, strength.odds), else hold the main
       base's line facing them (scout.home_line: just behind our turret),
       where turrets, buildings and new units fight with us (52a77fb1 met a
       push 5 cells in front of its pillboxes and lost 6400 to 2200).
    2. A push seen coming (scout.incoming_push): ready at that line, facing it.
    3. Otherwise the concave: at the mouth of the choke on their road in
       when there is one (scout.concave_point), else at the massing point
       (``stage_override``: the point it edges to while an attack call is
       confirmed) — at the main line while defending the base (a defend mode).

    The rules own staging (match a2952a6d: the advisor massed the army toward
    our own ore and faced it at our base)."""
    fighters = [u for u in _fighters(view, scout_id, reserved) if u.cell is not None]
    if not fighters:
        return []
    defending = mode in DEFENSIVE
    contact = army_contact(view, fighters, defending)
    if contact:
        centre = _centre(contact)
        # Joined at ENGAGE_MIN_EDGE, kept above ENGAGE_KEEP_EDGE for at least
        # ENGAGE_HOLD_SECONDS: the odds of what is in sight flicker as their
        # units step in and out of vision, and the army charged and fell back
        # to the turret line with them (the audit).
        mem = memory if memory is not None else {}
        t = view.game_seconds
        joined = mem.get("engaged_until", float("-inf")) >= t
        bar = C.ENGAGE_KEEP_EDGE if joined else C.ENGAGE_MIN_EDGE
        # Judged against the whole push as last seen (scout.incoming_push
        # holds it a while), not the few of it in sight: 2fe84e1f at 3:51
        # went at two of a 15-unit push by its war factory, met the rest
        # and lost them all.
        push = scout.incoming_push(view)
        theirs = {a.id: a for a in contact}
        if push is not None:
            theirs.update({a.id: a for a in push[2] if a.id is not None})
        if strength.odds(view, [u.type for u in fighters], [a.type for a in theirs.values()]) >= bar:
            mem["engaged_until"] = t + C.ENGAGE_HOLD_SECONDS
            base = view.base_center or centre
            target = min(contact, key=lambda a: dist2(a.cell, base)).cell
            return [orders.attack_move([u.id for u in fighters][:orders.MAX_UNITS_PER_ORDER], list(target))]
        line = scout.home_line(view, centre)
        if line is not None:
            return _form(view, fighters, line, centre, everyone=True)
    push = scout.incoming_push(view)
    if push is not None:
        point, facing, _ = push
        return _form(view, fighters, point, facing, everyone=True)
    if defending:
        line = scout.home_line(view)
        return _form(view, fighters, line, enemy_spawn_guess(view), everyone=True) if line is not None else []
    if stage_override is not None:                 # edging forward to confirm an attack: everyone goes
        return _form(view, fighters, stage_override, enemy_spawn_guess(view), everyone=True)
    # Early, the army waits at the main base's line by its turret: a rush
    # that passes the forward concave hits the buildings behind it
    # (5a03e4d2 at 3:20). Not against a passive enemy (no rush comes).
    if view.game_seconds < C.EARLY_HOME_DEFENSE_SECONDS and not view.enemy_passive:
        line = scout.home_line(view)
        if line is not None:
            return _form(view, fighters, line, enemy_spawn_guess(view))
    held = scout.concave_point(view)       # at the choke's mouth when there is one
    return _form(view, fighters, held[0], held[1]) if held is not None else []


def _form(view: View, fighters, point, facing, everyone: bool = False) -> list[dict]:
    """The concave at ``point`` facing ``facing``; a handful just walk there
    (the idle ones, or ``everyone`` when the army must be there now)."""
    if len(fighters) >= 3:
        return _concave_orders(view, fighters, point, facing, everyone)
    r2 = C.STAGING_RADIUS_CELLS ** 2
    movers = [u.id for u in fighters if (everyone or u.idle) and dist2(u.cell, point) > r2]
    return [orders.attack_move(movers[:orders.MAX_UNITS_PER_ORDER], list(point))] if movers else []


def _enemies_near(view: View, cells) -> list:
    """Enemy fighters within ``STAGE_ENGAGE_CELLS`` of any of ``cells``."""
    r2 = C.STAGE_ENGAGE_CELLS ** 2
    return [a for a in view.enemy_units if a.cell is not None and not a.wreck and a.type not in C.HARVESTER
            and any(dist2(a.cell, c) <= r2 for c in cells if c is not None)]


def _value(view: View, actors) -> float:
    return strength.value(view, [a.type for a in actors])


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
    import math

    # Ranks on arcs round the kill point, each within rifle range of it,
    # filled from the middle outward at a fixed spacing: the first n slots
    # never move when the army grows or shrinks (the layout used to spread
    # over n, so every new unit shifted every slot and the line reshuffled).
    fx, fy = center[0] + ux * C.FORMATION_FOCUS_AHEAD, center[1] + uy * C.FORMATION_FOCUS_AHEAD
    span = math.radians(C.FORMATION_ARC_DEGREES)
    slots: list[tuple[int, int]] = []
    for radius in C.FORMATION_RANKS:
        if len(slots) >= n:
            break
        step = C.FORMATION_SPACING_CELLS / radius            # radians between neighbours on this arc
        half = int(span / step)
        for k in [0] + [s * j for j in range(1, half + 1) for s in (1, -1)]:
            a = k * step
            # from the kill point back toward us, turned by a
            dx_, dy_ = -ux * math.cos(a) + px * math.sin(a), -uy * math.cos(a) + py * math.sin(a)
            x, y = round(fx + dx_ * radius), round(fy + dy_ * radius)
            if view.bounds is not None:
                bx, by, bw, bh = view.bounds
                x, y = min(max(x, bx), bx + bw - 1), min(max(y, by), by + bh - 1)
            x, y = _open_slot(view, (x, y), taken)
            if (x, y) not in slots:
                slots.append((x, y))
    along = lambda c: px * (c[0] - center[0]) + py * (c[1] - center[1])  # noqa: E731
    return sorted(slots[:n], key=along)                    # left to right along the line


def _open_slot(view: View, cell, taken: set) -> tuple[int, int]:
    """``cell``, or the nearest cell within 3 that no cliff, tree or building
    of ours covers (``taken``)."""
    if cell not in taken:
        return cell
    near = [(cell[0] + i, cell[1] + j) for i in range(-3, 4) for j in range(-3, 4)
            if (cell[0] + i, cell[1] + j) not in taken]
    return min(near, key=lambda c: dist2(c, cell)) if near else cell


_SLOTS: dict = {"key": None, "of": {}}   # the concave's (centre, facing) and unit id -> its slot


def _assign_slots(units, slots, key) -> dict:
    """Each unit keeps its slot while the concave stays put and the slot
    still exists; free slots go to the nearest unplaced units. (Re-sorting
    every view by where units stood swapped them between slots 6 cells apart
    as they walked: two riflemen got 250 orders in 160 s, back and forth.)"""
    if _SLOTS["key"] != key:
        _SLOTS.update(key=key, of={})
    present = {u.id for u in units}
    have = set(slots)
    of = {uid: s for uid, s in _SLOTS["of"].items() if uid in present and s in have}
    free = have - set(of.values())
    for _, uid, s in sorted((dist2(u.cell, s), u.id, s) for u in units if u.id not in of for s in free):
        if uid not in of and s in free:
            of[uid] = s
            free.discard(s)
    _SLOTS["of"] = of
    return of


def _bucketed(center, facing):
    """``facing`` turned to the nearest ``CONCAVE_FACING_BUCKET_DEGREES`` step
    round ``center``, at the same distance."""
    import math

    if facing is None or center is None:
        return facing
    dx, dy = facing[0] - center[0], facing[1] - center[1]
    d = (dx * dx + dy * dy) ** 0.5
    if d == 0:
        return facing
    step = math.radians(C.CONCAVE_FACING_BUCKET_DEGREES)
    a = round(math.atan2(dy, dx) / step) * step
    return (round(center[0] + math.cos(a) * d), round(center[1] + math.sin(a) * d))


def _concave_orders(view: View, fighters, center, facing, everyone: bool = False) -> list[dict]:
    """Attack-move each unit to its concave slot; a unit keeps its slot
    (:func:`_assign_slots`), and only idle units get slot orders — one with
    an order (walking to its slot, answering a poke, fighting) is left to
    finish it — unless ``everyone`` must form here now (a fight, a push
    coming, defending the base). Units beyond the slots gather just behind
    the line."""
    units =[u for u in fighters if u.cell is not None]
    if not units:
        return []
    # The facing in CONCAVE_FACING_BUCKET_DEGREES steps: a moving enemy
    # centroid re-keyed the slots every view and swapped everyone round.
    facing = _bucketed(center, facing)
    slots = concave_slots(view, center, facing, min(len(units), C.FORMATION_MAX_UNITS))
    of = _assign_slots(units, slots, (tuple(center), tuple(facing) if facing is not None else None))
    # attack_move, not move: a unit walking to its slot still fights what it meets.
    # At most CONCAVE_ORDERS_PER_VIEW slot orders a view, the farthest first;
    # everyone else far from the centre walks there in one order, sent FIRST
    # (52a77fb1: one order per unit overflowed the 20-order batch, the
    # walk-to-the-centre order came last and was cut every view, and new
    # units stood at the barracks doors through a fight).
    moves = sorted(((u, of[u.id]) for u in units
                    if u.id in of and (everyone or u.idle) and dist2(u.cell, of[u.id]) > 2),
                   key=lambda p: (not p[0].idle, -dist2(p[0].cell, p[1])))
    sent = moves[:C.CONCAVE_ORDERS_PER_VIEW]
    r2 = C.STAGING_RADIUS_CELLS ** 2
    # Only units beyond the formation walk to the centre: an in-line unit
    # waits for its slot order (sending it to the centre and back to its
    # slot had the edges of the line wandering to and fro, 3bd6e450 12:00+).
    line = set(of)
    rear = center
    if facing is not None:
        ddx, ddy = facing[0] - center[0], facing[1] - center[1]
        dl = (ddx * ddx + ddy * ddy) ** 0.5 or 1.0
        back = max(C.FORMATION_RANKS) - C.FORMATION_FOCUS_AHEAD + 1.5
        rear = _open_slot(view, (round(center[0] - ddx / dl * back), round(center[1] - ddy / dl * back)),
                          set(view.blocked_cells))
    rest = [u.id for u in units if u.id not in line and u.idle and dist2(u.cell, rear) > r2]
    out = [orders.attack_move(rest[:orders.MAX_UNITS_PER_ORDER], list(rear))] if rest else []
    return out + [orders.attack_move([u.id], list(slot)) for u, slot in sent]


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
    enemy main for 5 minutes while their last building stood 10 cells away.)
    The cell returned is that enemy's — a building first, the nearest to the
    pick — not the pick itself: an attack-move to an empty spot stops there
    with the building out of range (a4a78f76: the army stood at (74, 53), the
    middle of their razed base, 9 cells from their radar dome, until the end)."""
    cell = getattr(advice, "army_target", None)
    if cell is None:
        return None
    r2 = C.ADVICE_TARGET_SUPPORT_CELLS ** 2
    for group in (list(view.enemy_buildings) + list(view.known_enemy_buildings), list(view.enemy_units)):
        near = [a.cell for a in group if a.cell is not None and not a.wreck and dist2(a.cell, cell) <= r2]
        if near:
            return tuple(min(near, key=lambda c: dist2(c, cell)))
    return None


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
    # Far off, the army goes as one to the target (the route's legs bring
    # it); it spreads only near it (b9145367: groups aimed across a target on
    # Trapped walked round opposite sides of the rocks).
    if (target[0] - cx) ** 2 + (target[1] - cy) ** 2 > C.ATTACK_SPREAD_NEAR_CELLS ** 2:
        return [tuple(target)]
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
    # Each group's point must be walkable without a detour from where the
    # army stands, else the army goes as one.
    from .terrain import walk_path

    start = min(cells, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
    for p in out:
        if p in view.blocked_cells:
            return [tuple(target)]
        path = walk_path(view, start, p)
        if not path or len(path) > C.ATTACK_POINT_MAX_DETOUR * dist2(start, p) ** 0.5 + 4:
            return [tuple(target)]
    return out


def route_goal(view: View, advice=None, hunt: tuple[int, int] | None = None) -> tuple[int, int] | None:
    """Where the attack's route leads: the advisor's target, else their
    buildings (an expansion first) or their base — never whichever of their
    units is nearest, which jumps about the map (d1615ac6: the route was
    re-planned toward each one, north lane, back, south lane, and the army
    zig-zagged across the middle for 90 s before it arrived piecemeal)."""
    import dataclasses

    return _advised_target(view, advice) or _attack_target(dataclasses.replace(view, enemy_units=()), hunt)


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
    # Away from where our raid squads hold part of their army (attack_avoid),
    # when anything else is known: the attack goes at the rest.
    avoid = view.attack_avoid
    if avoid is not None:
        import dataclasses

        r2 = C.RAID_SPREAD_CELLS ** 2

        def far(actors):
            return [a for a in actors if a.cell is not None and dist2(a.cell, avoid) > r2]

        away = dataclasses.replace(view, attack_avoid=None, enemy_buildings=far(view.enemy_buildings),
                                   enemy_units=far(view.enemy_units),
                                   known_enemy_buildings=far(view.known_enemy_buildings))
        target = _attack_target(away, hunt)
        if target is not None and dist2(target, avoid) > r2:
            return target
    # A known enemy expansion first: away from their army, lightly defended,
    # and their income (match 4b5980f8: a 33k army sat idle while red's
    # expansion was known for the last 2.5 minutes).
    from .raid import remote_expansions

    # The same remote expansions the strike and the raids judge (25+ cells
    # from their main), not their main base's edge.
    expansions = remote_expansions(view)
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
