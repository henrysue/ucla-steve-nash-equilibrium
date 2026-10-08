"""The deterministic executor — the fast path, no model call.

Given the current :class:`View` and the brain's current :class:`Mode`, produce
one batch of orders. Every view, in priority order:

1. **Production/build order** (``buildplan``) — advance the base, place finished
   buildings, keep power healthy.
2. **Economy** — keep harvesters on the nearest ore (re-issue to idle ones).
3. **Army production** — train the composition the mode wants (cheap infantry,
   or tanks once the war factory is up), within what we can afford.
4. **Army command** — rally home (defensive modes) or attack-move at the enemy
   (aggressive modes), with a coarse tanks-front grouping.

The executor is a pure function of (view, mode): same inputs, same orders. It
always returns a legal batch or ``None`` (meaning "nothing to do -> WAIT"), so
the game never waits on the brain. Order count is capped by ``orders.batch``.
"""

from __future__ import annotations

from . import buildplan, orders
from . import config as C
from .modes import AGGRESSIVE, DEFENSIVE, Mode
from .observe import View, enemy_spawn_guess


def plan(view: View, mode: Mode) -> list[dict]:
    """All orders for this view, concatenated in priority order (pre-cap)."""
    if not view.ok:
        return []
    out: list[dict] = []
    out += buildplan.placement_orders(view)      # place anything finished first
    out += buildplan.next_build_orders(view)     # advance the build order / power
    out += _economy_orders(view)
    out += _production_orders(view, mode)
    out += _army_orders(view, mode)
    return out


# -- economy -----------------------------------------------------------------


def _economy_orders(view: View) -> list[dict]:
    """Keep harvesters working: re-issue harvest to any idle/stopped harvester.

    OpenRA auto-harvests the nearest ore when given a bare ``harvest``; we only
    need to nudge idle ones (strategy hole #2: an idle harvester is dead
    economy). We do not micro ore cells — "nearest" is the engine default.
    """
    idle = [h.id for h in view.units_of(C.HARVESTER) if h.idle and h.id is not None]
    return [orders.harvest(idle)] if idle else []


# -- army production ---------------------------------------------------------


def _production_orders(view: View, mode: Mode) -> list[dict]:
    """Train the army. Infantry needs a barracks; tanks need a war factory.

    DEFEND_RUSH spams the cheapest unit (fast). PROTECT_LEAD stops spending into
    risky army (preserve assets for the tiebreak). Otherwise: infantry bridge
    early, tanks as the core once the war factory exists.
    """
    if mode is Mode.PROTECT_LEAD:
        return []  # coast on current assets; don't convert cash into losable army

    has_barracks = view.has_building(C.BARRACKS)
    has_factory = view.has_building(C.WAR_FACTORY)
    out: list[dict] = []

    rifle = C.resolve((C.RIFLE,), view.available)
    rocket = C.resolve((C.ROCKET,), view.available)
    tank = C.resolve(C.LIGHT_TANKS + C.HEAVY_TANKS, view.available)

    if mode is Mode.DEFEND_RUSH:
        # Cheapest attack units, as fast as possible, to kill the cheese.
        if has_barracks and rifle and view.queued_count(rifle) < 2:
            out.append(orders.train(rifle, count=3))
        return out

    # Normal composition. Keep a couple of harvesters coming if we are short.
    harv = C.resolve(C.HARVESTER, view.available)
    if harv and has_factory and len(view.units_of(C.HARVESTER)) < C.TARGET_HARVESTERS and view.queued_count(harv) == 0:
        out.append(orders.train(harv, count=1))

    if has_factory and tank:
        # Tank-centric deathball once the factory is up.
        if view.queued_count(tank) < 2:
            out.append(orders.train(tank, count=1))
        # Keep a trickle of rockets for anti-armor/air alongside tanks.
        if rocket and has_barracks and view.queued_count(rocket) == 0:
            out.append(orders.train(rocket, count=1))
    elif has_barracks and rifle:
        # Infantry bridge before the factory: rifles with a few rockets.
        ratio = C.RIFLE_TO_ROCKET if view.cash >= C.ROCKET_RATIO_CREDIT_SWITCH else C.RIFLE_TO_ROCKET_EARLY
        if view.queued_count(rifle) < 2:
            out.append(orders.train(rifle, count=3))
        if rocket and _should_add_rocket(view, ratio) and view.queued_count(rocket) == 0:
            out.append(orders.train(rocket, count=1))
    return out


def _should_add_rocket(view: View, ratio: int) -> bool:
    """Add a rocket when our rifle:rocket mix is below the target ratio."""
    rifles = len(view.units_of(C.RIFLE))
    rockets = len(view.units_of(C.ROCKET))
    return rockets == 0 or rifles >= ratio * (rockets + 1)


# -- army command ------------------------------------------------------------


def _army_orders(view: View, mode: Mode) -> list[dict]:
    """Move the army: rally home when defensive, attack-move when aggressive.

    Coarse grouping only (tanks issued in the same attack-move as infantry; the
    engine keeps them roughly together) — no continuous concave micro, which
    would be too slow for real time.
    """
    combat = [u.id for u in view.combat_units if u.id is not None]
    if not combat:
        return []

    if mode in DEFENSIVE:
        return _rally_home(view, combat)

    if mode in AGGRESSIVE:
        target = _attack_target(view)
        if target is None:
            return []
        if isinstance(target, int):          # a visible enemy actor id
            return [orders.attack(combat, target)]
        return [orders.attack_move(combat, target)]  # a cell (enemy base guess)

    return []  # OPEN / BUILD: hold position, keep massing


def _rally_home(view: View, combat: list[int]) -> list[dict]:
    """Pull the army to a defensive point near our base (or the choke)."""
    rally = view.base_center
    if rally is None:
        return []
    return [orders.attack_move(combat, list(rally))]


def _attack_target(view: View):
    """Prefer the nearest visible enemy; else attack-move to the enemy spawn."""
    # Nearest visible enemy building (economy/base first), then unit.
    candidates = view.enemy_buildings + view.enemy_units
    with_ids = [a for a in candidates if a.id is not None]
    if with_ids and view.base_center is not None:
        from .observe import dist2
        nearest = min(with_ids, key=lambda a: dist2(a.cell, view.base_center))
        return nearest.id
    return enemy_spawn_guess(view)
