"""TEST ONLY — sparring archetypes for self-play. Never active in a tournament.

When we play ourselves in a test match, one seat can play a fixed archetype
instead of our game plan, so the other seat's reads and answers get tested:

- ``rusher``: plays like Yuriko (replay henry-vs-yuriko, Sidestep): a lean
  build — power, one barracks, two refineries, power, war factory, power,
  depot (a third refinery ~10:00, a second war factory ~11:30), three
  pillboxes on the approach by 2:30, no expansion. Production never pauses:
  riflemen (a rocket soldier per few), then main battle tanks once the depot
  stands. It attacks in waves: the first at ~3:00 with ~20 riflemen
  (``RUSH_FIRST_WAVE_*``), each later one once the army has grown past the
  last launch (``RUSH_WAVE_GROWTH``, or ``RUSH_WAVE_MAX_GAP_SECONDS`` after
  regrouping); a wave pushes until it is down to ``RUSH_REGROUP_SHARE`` of its
  launch value, then everyone regroups at home.
- ``yuriko``: Yuriko's current build (replays yuriko-vs-abu, Trapped and
  Sidestep, October 13; she won both by ~6:45): power, one barracks, two
  refineries, power, war factory (placed 2:39), depot (3:20), power; six
  riflemen at 0:32-0:42, then nothing until the war factory, a few more
  riflemen, pillboxes at 2:40 and 3:35; from 4:05 (the depot up),
  medium/heavy tanks nonstop with rocket soldiers and riflemen. The push from
  ``YURIKO_PUSH_SECONDS`` with ``YURIKO_PUSH_TANKS`` tanks, reinforced until
  spent (``YURIKO_REGROUP_SHARE``), then the next; a second war factory at
  5:54, a third refinery at 6:05 and no expansion (checked against
  yuriko-vs-abu-3 too).
- ``turtler``: our build plus a growing ring of turrets (``TURTLE_DEFENSES``
  over time, up to ``TURTLE_MAX_DEFENSES``), the army at home, no raids. It
  only attacks late (``TURTLE_PUSH_SECONDS`` with an edge) or with a huge army
  (``TURTLE_PUSH_VALUE``).
- ``macro``: fast economy — extra refineries early (``MACRO_REFINERIES``), the
  first expansion as early as allowed, only a token army until
  ``MACRO_ARMY_FROM_SECONDS``, no raids. One big push once the army is worth
  ``MACRO_PUSH_VALUE`` (or ``MACRO_PUSH_SECONDS`` with an edge).

Every archetype still defends its base the usual way (a defend/retreat call
from the rules always stands).

Safety, three locks: off unless ``RA_SPARRING`` names an archetype (the old
``RA_TEST_RUSH=1`` means ``rusher``); never when the context has a
``tournament_id``; and only once both seats of the same session are seen on
this machine (marker files ``<tmp>/redalert_selfplay/<session>-seat<k>``), so
a tournament opponent can never be the other seat. ``RA_SPARRING_SEAT`` picks
the seat (default 1). Remove this module and its call sites (grep
"sparring") to retire it.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from . import config as C
from . import orders
from .modes import DEFENSIVE, Mode

ARCHETYPES = ("rusher", "yuriko", "turtler", "macro")
DEFAULT_SEAT = 1
_MARKER_DIR = Path(tempfile.gettempdir()) / "redalert_selfplay"
_STALE_SECONDS = 3 * 3600

# rusher (Yuriko: 1 barracks, 2 refineries by 1:06, 3 pillboxes by 2:26, war factory 3:54,
# depot 4:38, third refinery 10:17, second war factory 11:36; waves at ~3:00, ~6:30, ~11:30)
RUSH_BUILD_ORDER = ("powr", ("tent", "barr"), "proc", "proc", "powr", "weap", "powr", "fix")
RUSH_LATE_REFINERY_SECONDS = 600.0   # a third refinery then...
RUSH_SECOND_FACTORY_SECONDS = 690.0  # ...and a second war factory
RUSH_PILLBOXES = ((90.0, 1), (140.0, 3))   # (game s, turrets on the approach)
RUSH_PILLBOX_FUNDS = 600
RUSH_FIRST_WAVE_SECONDS = 170.0  # the first wave from 2:50...
RUSH_FIRST_WAVE_UNITS = 18       # ...with this many fighters (Yuriko: 20-28 riflemen at 3:00)
RUSH_WAVE_GROWTH = 1.4           # a later wave once the army is this x the last launch...
RUSH_WAVE_MIN_GAP_SECONDS = 150.0   # ...at least this long after regrouping...
RUSH_WAVE_MAX_GAP_SECONDS = 300.0   # ...or this long after it with at least the last launch's army
RUSH_REGROUP_SHARE = 0.4         # a wave pushes until the army falls below this share of its launch value
RUSH_INFANTRY_DEPTH = 2
RUSH_TANK_DEPTH = 2

# yuriko (current, yuriko-vs-abu x3: war factory 2:39-2:43, depot 3:13-3:20, pillboxes 2:55 and 3:54-3:58,
# 12-16 infantry then tanks from 4:08-4:20, the push 5:00-6:30 with 5-6 medium tanks, 9000 queued by 6:00, a
# second war factory queued 5:54, a third refinery 6:05-6:44, the next push ~8:30 after regrouping)
YURIKO_BUILD_ORDER = ("powr", ("tent", "barr"), "proc", "proc", "powr", "weap", "fix", "powr")
YURIKO_OPENING_RIFLES = 6           # riflemen before the war factory (then the money waits for it)
YURIKO_INFANTRY_BEFORE_TANKS = 12   # infantry while the depot isn't up
YURIKO_TANKS_FROM_SECONDS = 245.0   # tanks from 4:05 (her depot stands ~3:15, the money first goes to a pillbox
                                    # and power; 12-16 infantry queued before the first tank at 4:08-4:20)
YURIKO_PILLBOXES = ((160.0, 1), (215.0, 2))   # (game s, turrets)
YURIKO_PUSH_SECONDS = 270.0         # the push from 4:30...
YURIKO_PUSH_TANKS = 3               # ...with at least this many main battle tanks
YURIKO_REGROUP_SHARE = 0.4          # it pushes until down to this share of its launch value...
YURIKO_NEXT_GROWTH = 1.3            # ...the next once the army is this x the last launch...
YURIKO_NEXT_GAP_SECONDS = 150.0     # ...or this long after regrouping
YURIKO_LATE_REFINERY_SECONDS = 365.0   # a third refinery then (queued 6:05-6:44)...
YURIKO_SECOND_FACTORY_SECONDS = 354.0  # ...and a second war factory (queued 5:54, yuriko-vs-abu-3)
YURIKO_TANK_DEPTH = 2

# turtler
TURTLE_DEFENSES = ((150.0, 2), (240.0, 3), (330.0, 4), (420.0, 5), (510.0, 6), (600.0, 7), (720.0, 8))  # (game s, turrets)
TURTLE_MAX_DEFENSES = 8
TURTLE_DEFENSE_FUNDS = 700       # a turret only with this much money
TURTLE_PUSH_SECONDS = 900.0      # attack from 15:00...
TURTLE_PUSH_EDGE = 1.3           # ...with this edge over the estimate...
TURTLE_PUSH_VALUE = 18000        # ...or any time with an army this big
TURTLE_REGROUP_SHARE = 0.5

# macro
MACRO_REFINERIES = ((45.0, 2), (150.0, 3), (240.0, 4), (330.0, 5), (420.0, 6))   # (game s, refineries)
MACRO_ARMY_FROM_SECONDS = 330.0  # only a token army before this...
MACRO_EARLY_ARMY_VALUE = 1500    # ...worth this much
MACRO_PUSH_VALUE = 12000         # one big push with an army this big...
MACRO_PUSH_SECONDS = 780.0       # ...or from 13:00...
MACRO_PUSH_EDGE = 1.15           # ...with this edge
MACRO_REGROUP_SHARE = 0.5


class Archetype:
    """The hooks an archetype may override; ``None`` means "our game plan"."""

    name = ""
    expands = True
    raids = True
    skip_confirm = False

    def mode(self, view, cheese_since, policy: Mode) -> Mode | None:
        return None

    def build(self, view, depleted=frozenset()) -> list[dict] | None:
        return None

    def defense(self, view, under_attack: bool) -> list[dict] | None:
        return None

    def production(self, view, mode: Mode) -> list[dict] | None:
        return None

    def advice(self, view):
        return None


COUNTER_ODDS = 2.0   # the turtle and the macro counter-attack at this edge (the rules' call)


class _Pusher(Archetype):
    """Masses at home until ``_ready``, then pushes until the army falls
    below ``regroup`` of its launch value. A defend/retreat call stands."""

    regroup = 0.5

    def __init__(self) -> None:
        self._launch_value: float | None = None

    def _ready(self, view) -> bool:
        return False

    def mode(self, view, cheese_since, policy: Mode) -> Mode | None:
        if policy in DEFENSIVE or policy is Mode.RETREAT:
            return policy
        if self._launch_value is not None:
            if view.army_value >= self.regroup * self._launch_value:
                return Mode.ALL_IN
            self._launch_value = None
            return Mode.BUILD
        # It punishes a blunder: when the rules see a 2x edge (our army just
        # thrown away), it counter-attacks — a real opponent would (b9145367:
        # our blind attack left 300 at 5:20 and the turtle sat at home).
        from .brain import army_odds

        if self._ready(view) or (policy is Mode.ALL_IN and army_odds(view) >= COUNTER_ODDS):
            self._launch_value = view.army_value
            return Mode.ALL_IN
        return Mode.BUILD


class Rusher(Archetype):
    name = "rusher"
    expands = False
    raids = False
    skip_confirm = True

    def __init__(self) -> None:
        self._launch_value: float | None = None   # army value when the current wave launched
        self._last_launch = 0.0                    # the last wave's launch value
        self._regrouped_at: float | None = None    # when the last wave was called home

    def mode(self, view, cheese_since, policy: Mode = Mode.BUILD) -> Mode | None:
        """Waves: the first at ``RUSH_FIRST_WAVE_SECONDS`` with
        ``RUSH_FIRST_WAVE_UNITS`` fighters; each pushes until it is down to
        ``RUSH_REGROUP_SHARE`` of its launch value, then regroups at home;
        the next goes once the army has grown ``RUSH_WAVE_GROWTH`` past the
        last launch (or ``RUSH_WAVE_MAX_GAP_SECONDS`` on). Its own base hit:
        it defends. ``None`` before it has a yard and barracks."""
        from .brain import _defend

        if not view.has_building(C.CONSTRUCTION_YARD) or not view.has_building(C.BARRACKS):
            return None
        if cheese_since is not None:
            return _defend(view)
        t, army = view.game_seconds, view.army_value
        if self._launch_value is not None:
            if army >= RUSH_REGROUP_SHARE * self._launch_value:
                return Mode.ALL_IN               # the wave pushes; reinforcements stream in
            self._launch_value, self._regrouped_at = None, t
            return Mode.BUILD                    # spent: everyone home, the next wave masses
        if self._regrouped_at is None:
            go = t >= RUSH_FIRST_WAVE_SECONDS and len(view.combat_units) >= RUSH_FIRST_WAVE_UNITS
        else:
            since = t - self._regrouped_at
            go = ((since >= RUSH_WAVE_MIN_GAP_SECONDS and army >= RUSH_WAVE_GROWTH * self._last_launch)
                  or (since >= RUSH_WAVE_MAX_GAP_SECONDS and army >= self._last_launch))
        if go:
            self._launch_value = self._last_launch = army
            return Mode.ALL_IN
        return Mode.BUILD

    def build(self, view, depleted=frozenset()) -> list[dict]:
        """The lean build order; a third refinery and a second war factory late."""
        from . import buildplan

        step = buildplan.walk_build_order(view, RUSH_BUILD_ORDER)
        if step:
            return step
        money = view.cash + view.ore
        t = view.game_seconds
        late = []
        if t >= RUSH_LATE_REFINERY_SECONDS:
            late.append(("proc", 3))
        if t >= RUSH_SECOND_FACTORY_SECONDS:
            late.append(("weap", 2))
        for item, want in late:
            avail = buildplan._available(view, item)
            if (avail and money >= C.KNOWN_COSTS.get(item, 1000) and not view.queued_count(avail)
                    and buildplan._count_built_and_queued(view, (item,)) < want):
                return [orders.build(avail)]
        return []

    def defense(self, view, under_attack: bool) -> list[dict] | None:
        """Three pillboxes on the approach by 2:30 (the usual turrets while attacked)."""
        from . import buildplan

        if under_attack:
            return None
        defense = buildplan._available(view, C.BUILD_TURRETS)
        if defense is None or any(view.queued_count(d) for d in C.TURRETS) or any(
                p.get("ready") and str(p.get("item") or "").lower() in C.TURRETS for p in view.production):
            return []
        have = sum(1 for b in view.buildings if b.type in C.TURRETS and not b.wreck)
        want = _scheduled(RUSH_PILLBOXES, view.game_seconds, 0)
        if have < want and view.cash + view.ore >= RUSH_PILLBOX_FUNDS:
            return [orders.build(defense)]
        return []

    def production(self, view, mode: Mode) -> list[dict]:
        """Units without pause: riflemen (a rocket soldier per three), then
        main battle tanks once the depot stands."""
        out = []
        rifle = C.resolve((C.RIFLE,), view.available)
        rocket = C.resolve((C.ROCKET,), view.available)
        tank = C.resolve(C.TANK_PREFERENCE, view.available)
        if rifle and view.has_building(C.BARRACKS) and view.queued_count(rifle) < RUSH_INFANTRY_DEPTH:
            out.append(orders.train(rifle, count=1))
        if rocket and view.has_building(C.BARRACKS) and view.queued_count(rocket) < 1 \
                and len(view.units_of(C.ROCKET)) * 3 < len(view.units_of(C.RIFLE)):
            out.append(orders.train(rocket, count=1))
        if tank and view.has_building(C.WAR_FACTORY) and view.queued_count(tank) < RUSH_TANK_DEPTH:
            out.append(orders.train(tank, count=1))
        return out


class Yuriko(Archetype):
    name = "yuriko"
    expands = False
    raids = False
    skip_confirm = True

    def __init__(self) -> None:
        self._launch_value: float | None = None
        self._last_launch = 0.0
        self._regrouped_at: float | None = None

    def mode(self, view, cheese_since, policy: Mode = Mode.BUILD) -> Mode | None:
        """The push from ``YURIKO_PUSH_SECONDS`` once ``YURIKO_PUSH_TANKS``
        main battle tanks stand, reinforced until down to
        ``YURIKO_REGROUP_SHARE`` of its launch value; the next at
        ``YURIKO_NEXT_GROWTH`` x the last launch or ``YURIKO_NEXT_GAP_SECONDS``
        after regrouping. A defend/retreat call stands."""
        from .brain import _defend

        if not view.has_building(C.CONSTRUCTION_YARD) or not view.has_building(C.BARRACKS):
            return None
        if cheese_since is not None:
            return _defend(view)
        if policy in DEFENSIVE:
            return policy
        t, army = view.game_seconds, view.army_value
        if self._launch_value is not None:
            if army >= YURIKO_REGROUP_SHARE * self._launch_value:
                return Mode.ALL_IN
            self._launch_value, self._regrouped_at = None, t
            return Mode.BUILD
        tanks = len(view.units_of(C.MAIN_BATTLE_TANKS))
        if self._regrouped_at is None:
            go = t >= YURIKO_PUSH_SECONDS and tanks >= YURIKO_PUSH_TANKS
        else:
            since = t - self._regrouped_at
            go = army >= YURIKO_NEXT_GROWTH * self._last_launch or (since >= YURIKO_NEXT_GAP_SECONDS
                                                                    and army >= self._last_launch)
        if go:
            self._launch_value = self._last_launch = army
            return Mode.ALL_IN
        return Mode.BUILD

    def build(self, view, depleted=frozenset()) -> list[dict]:
        """Her build order; a third refinery at 6:40, a second war factory later."""
        from . import buildplan

        step = buildplan.walk_build_order(view, YURIKO_BUILD_ORDER)
        if step:
            return step
        money, t = view.cash + view.ore, view.game_seconds
        late = []
        if t >= YURIKO_LATE_REFINERY_SECONDS:
            late.append(("proc", 3))
        if t >= YURIKO_SECOND_FACTORY_SECONDS:
            late.append(("weap", 2))
        for item, want in late:
            avail = buildplan._available(view, item)
            if (avail and money >= C.KNOWN_COSTS.get(item, 1000) and not view.queued_count(avail)
                    and buildplan._count_built_and_queued(view, (item,)) < want):
                return [orders.build(avail)]
        return []

    def defense(self, view, under_attack: bool) -> list[dict] | None:
        """Pillboxes on her schedule (the usual turrets while attacked)."""
        from . import buildplan

        if under_attack:
            return None
        defense = buildplan._available(view, C.BUILD_TURRETS)
        if defense is None or any(view.queued_count(d) for d in C.TURRETS) or any(
                p.get("ready") and str(p.get("item") or "").lower() in C.TURRETS for p in view.production):
            return []
        have = sum(1 for b in view.buildings if b.type in C.TURRETS and not b.wreck)
        if have < _scheduled(YURIKO_PILLBOXES, view.game_seconds, 0):
            return [orders.build(defense)]
        return []

    def production(self, view, mode: Mode) -> list[dict]:
        """Six riflemen, then nothing until the war factory; a few more
        riflemen until the depot; then tanks nonstop with rocket soldiers and
        riflemen alternating (harvesters come first, as ours do)."""
        out = []
        rifle = C.resolve((C.RIFLE,), view.available)
        rocket = C.resolve((C.ROCKET,), view.available)
        tank = C.resolve(C.TANK_PREFERENCE, view.available)
        rax, weap = view.has_building(C.BARRACKS), view.has_building(C.WAR_FACTORY)
        infantry = len(view.units_of(C.RIFLE)) + len(view.units_of(C.ROCKET))
        queued_inf = (view.queued_count(rifle) if rifle else 0) + (view.queued_count(rocket) if rocket else 0)
        if not rax:
            return out
        if not weap:
            if rifle and len(view.units_of(C.RIFLE)) + view.queued_count(rifle) < YURIKO_OPENING_RIFLES:
                out.append(orders.train(rifle, count=1))
            return out
        if not view.has_building(C.SERVICE_DEPOT) or view.game_seconds < YURIKO_TANKS_FROM_SECONDS:
            cap = YURIKO_INFANTRY_BEFORE_TANKS + (4 if view.has_building(C.SERVICE_DEPOT) else 0)   # 12-16
            if rifle and infantry + queued_inf < cap and view.queued_count(rifle) < 1:
                out.append(orders.train(rifle, count=1))
            return out
        if tank and view.queued_count(tank) < YURIKO_TANK_DEPTH:
            out.append(orders.train(tank, count=1))
        if queued_inf < 1:
            rockets, rifles = len(view.units_of(C.ROCKET)), len(view.units_of(C.RIFLE))
            pick = rocket if rocket and rockets <= rifles else rifle
            if pick:
                out.append(orders.train(pick, count=1))
        return out


class Turtler(_Pusher):
    name = "turtler"
    raids = False
    regroup = TURTLE_REGROUP_SHARE

    def _ready(self, view) -> bool:
        from .brain import enemy_estimate

        est = enemy_estimate(view)
        late = view.game_seconds >= TURTLE_PUSH_SECONDS and view.army_value >= TURTLE_PUSH_EDGE * max(est, 1.0)
        return late or view.army_value >= TURTLE_PUSH_VALUE

    def defense(self, view, under_attack: bool) -> list[dict] | None:
        """One more turret whenever the ring is short of ``TURTLE_DEFENSES``
        for the game time (placed on the front by the usual rules); else the
        usual turrets (expansions, under attack)."""
        from . import buildplan

        defense = buildplan._available(view, C.BUILD_TURRETS)
        if defense is None or any(view.queued_count(d) for d in C.TURRETS) or any(
                p.get("ready") and str(p.get("item") or "").lower() in C.TURRETS for p in view.production):
            return None
        want = _scheduled(TURTLE_DEFENSES, view.game_seconds, 0)
        have = sum(1 for b in view.buildings if b.type in C.TURRETS and not b.wreck)
        if have < min(want, TURTLE_MAX_DEFENSES) and view.cash + view.ore >= TURTLE_DEFENSE_FUNDS:
            return [orders.build(defense)]
        return None


class Macro(_Pusher):
    name = "macro"
    raids = False
    regroup = MACRO_REGROUP_SHARE

    def _ready(self, view) -> bool:
        from .brain import enemy_estimate

        est = enemy_estimate(view)
        late = view.game_seconds >= MACRO_PUSH_SECONDS and view.army_value >= MACRO_PUSH_EDGE * max(est, 1.0)
        return late or view.army_value >= MACRO_PUSH_VALUE

    def build(self, view, depleted=frozenset()) -> list[dict] | None:
        """Refineries ahead of our schedule (``MACRO_REFINERIES``) where a
        field has room; otherwise our build."""
        from . import buildplan

        want = _scheduled(MACRO_REFINERIES, view.game_seconds, 1)
        if buildplan._count_built_and_queued(view, (C.REFINERY,)) >= max(want, buildplan.refinery_target(view)):
            return None
        proc = buildplan._available(view, C.REFINERY)
        if (proc and view.queued_count(proc) == 0 and view.cash + view.ore >= C.REFINERY_ADD_FUNDS
                and buildplan.open_ore(view, depleted, radius=None) is not None):
            return [orders.build(proc)]
        return None

    def production(self, view, mode: Mode) -> list[dict] | None:
        """A token army only until ``MACRO_ARMY_FROM_SECONDS`` (unless
        defending): the money goes to harvesters, refineries and the MCV."""
        from .brain import our_unit_value

        if mode in DEFENSIVE or view.game_seconds >= MACRO_ARMY_FROM_SECONDS:
            return None
        return None if our_unit_value(view) < MACRO_EARLY_ARMY_VALUE else []

    def advice(self, view):
        from .advisor import Advice

        return Advice(made_at=view.game_seconds, enemy_style="standard", balance="more_economy", expand="now")


_BY_NAME = {"rusher": Rusher, "yuriko": Yuriko, "turtler": Turtler, "macro": Macro}


def _scheduled(schedule, t: float, start: int) -> int:
    want = start
    for at, n in schedule:
        if t >= at:
            want = n
    return want


def requested() -> str | None:
    """The archetype the environment asks for, or None."""
    name = os.environ.get("RA_SPARRING", "").strip().lower()
    if name in _BY_NAME:
        return name
    if os.environ.get("RA_TEST_RUSH", "0").strip() == "1":     # the old switch
        return "rusher"
    return None


def requested_seat() -> str:
    return (os.environ.get("RA_SPARRING_SEAT") or os.environ.get("RA_TEST_RUSH_SEAT") or str(DEFAULT_SEAT)).strip()


class Sparring:
    """Decides, once per match, whether this seat plays an archetype."""

    def __init__(self) -> None:
        self._session: str | None = None
        self._archetype: Archetype | None = None
        self._settled = False
        self._marked = False

    opponent: str | None = None    # the other seat's archetype as an opponent name ("sparring-macro"), self-play only

    def resolve(self, context, your_seat=None) -> Archetype | None:
        """This seat's archetype in a test match against ourselves, else None.
        ``your_seat``: the observation's seat ("Multi1"), used when the
        context has no ``seat_position``."""
        if self._settled:
            return self._archetype
        name = requested()
        if name is None or getattr(context, "tournament_id", None) is not None:
            self._settled = True          # off unless asked for; never in a tournament game
            return None
        session = getattr(context, "session_id", None)
        seat = getattr(context, "seat_position", None)
        if seat is None and your_seat is not None:
            digits = "".join(ch for ch in str(your_seat) if ch.isdigit())
            seat = int(digits) if digits else None
        if not session or seat is None:
            return None
        self._session = str(session)
        try:
            _MARKER_DIR.mkdir(parents=True, exist_ok=True)
            if not self._marked:
                self._clean()
                (_MARKER_DIR / f"{self._session}-seat{seat}").write_text(str(time.time()), encoding="utf-8")
                self._marked = True
            others = [p for p in _MARKER_DIR.glob(f"{self._session}-seat*") if p.name != f"{self._session}-seat{seat}"]
        except OSError:
            return None
        if not others:
            return None                   # not (yet) known to be self-play; ask again next view
        self._settled = True
        want = requested_seat()
        if want.isdigit() and int(want) == int(seat):
            self._archetype = _BY_NAME[name]()
        else:
            self.opponent = f"sparring-{name}"   # we play the archetype: profile it like a tournament opponent
        return self._archetype

    @staticmethod
    def _clean() -> None:
        now = time.time()
        for p in _MARKER_DIR.glob("*-seat*"):
            try:
                if now - p.stat().st_mtime > _STALE_SECONDS:
                    p.unlink()
            except OSError:
                pass
