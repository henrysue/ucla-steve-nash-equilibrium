"""TEST ONLY — sparring archetypes for self-play. Never active in a tournament.

When we play ourselves in a test match, one seat can play a fixed archetype
instead of our game plan, so the other seat's reads and answers get tested:

- ``rusher``: a one-refinery rush build (power, barracks, refinery, barracks,
  power, war factory; up to ``RUSH_BARRACKS`` barracks), no expansion, no
  turrets unless attacked, production that never pauses. It masses a ball and
  launches at ``RUSH_BALL_VALUE`` (or ``RUSH_MIN_ARMY_VALUE`` by
  ``RUSH_FORCE_SECONDS``), keeps pushing with reinforcements streaming in,
  and re-masses only once the army drops below ``RUSH_REGROUP_SHARE`` of its
  launch value.
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

ARCHETYPES = ("rusher", "turtler", "macro")
DEFAULT_SEAT = 1
_MARKER_DIR = Path(tempfile.gettempdir()) / "redalert_selfplay"
_STALE_SECONDS = 3 * 3600

# rusher
RUSH_START_SECONDS = 120.0
RUSH_MIN_ARMY_VALUE = 1500       # launch with at least this...
RUSH_BALL_VALUE = 2500           # ...once the ball is worth this...
RUSH_FORCE_SECONDS = 210.0       # ...or at this time with RUSH_MIN_ARMY_VALUE
RUSH_REGROUP_SHARE = 0.4         # pushing until the army falls below this share of its launch value
RUSH_BARRACKS = 3
RUSH_INFANTRY_DEPTH = 3
RUSH_TANK_DEPTH = 2
RUSH_BUILD_ORDER = ("powr", ("tent", "barr"), "proc", ("tent", "barr"), "powr", "weap")   # one refinery: a true rush

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
        if self._ready(view):
            self._launch_value = view.army_value
            return Mode.ALL_IN
        return Mode.BUILD


class Rusher(Archetype):
    name = "rusher"
    expands = False
    raids = False
    skip_confirm = True

    def __init__(self) -> None:
        self._launch_value: float | None = None

    def mode(self, view, cheese_since, policy: Mode = Mode.BUILD) -> Mode | None:
        """Defend only if its own base is hit; mass a ball and launch it once
        big enough; keep pushing until it is mostly spent, then re-mass.
        ``None`` before it has a yard and barracks."""
        from .brain import _defend

        if not view.has_building(C.CONSTRUCTION_YARD) or not view.has_building(C.BARRACKS):
            return None
        if cheese_since is not None:
            return _defend(view)
        if self._launch_value is not None:
            if view.army_value >= RUSH_REGROUP_SHARE * self._launch_value:
                return Mode.ALL_IN               # keep pushing; reinforcements stream in
            self._launch_value = None            # spent: fall back and mass the next ball
            return Mode.BUILD
        t, army = view.game_seconds, view.army_value
        ball = army >= RUSH_BALL_VALUE or (t >= RUSH_FORCE_SECONDS and army >= RUSH_MIN_ARMY_VALUE)
        if t >= RUSH_START_SECONDS and ball:
            self._launch_value = army
            return Mode.ALL_IN
        return Mode.BUILD

    def build(self, view, depleted=frozenset()) -> list[dict]:
        """More barracks, else the rush build order."""
        from . import buildplan

        money = view.cash + view.ore
        rax = buildplan._available(view, C.BARRACKS)
        if (rax and view.has_building(C.REFINERY) and money >= C.BARRACKS_ADD_FUNDS
                and not any(view.queued_count(b) for b in C.BARRACKS)
                and buildplan._count_built_and_queued(view, C.BARRACKS) < RUSH_BARRACKS):
            return [orders.build(rax)]
        return buildplan.walk_build_order(view, RUSH_BUILD_ORDER)

    def defense(self, view, under_attack: bool) -> list[dict] | None:
        return None if under_attack else []      # turrets only when attacked

    def production(self, view, mode: Mode) -> list[dict]:
        """Units without pause: infantry ``RUSH_INFANTRY_DEPTH`` deep (a rocket
        soldier in every few), tanks ``RUSH_TANK_DEPTH`` deep."""
        out = []
        rifle = C.resolve((C.RIFLE,), view.available)
        rocket = C.resolve((C.ROCKET,), view.available)
        tank = C.resolve(C.LIGHT_TANKS + C.HEAVY_TANKS, view.available)
        if rifle and view.has_building(C.BARRACKS) and view.queued_count(rifle) < RUSH_INFANTRY_DEPTH:
            out.append(orders.train(rifle, count=1))
        if rocket and view.has_building(C.BARRACKS) and view.queued_count(rocket) < 1 \
                and len(view.units_of(C.ROCKET)) * 3 < len(view.units_of(C.RIFLE)):
            out.append(orders.train(rocket, count=1))
        if tank and view.has_building(C.WAR_FACTORY) and view.queued_count(tank) < RUSH_TANK_DEPTH:
            out.append(orders.train(tank, count=1))
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

        defense = buildplan._available(view, C.DEFENSES)
        if defense is None or any(view.queued_count(d) for d in C.DEFENSES) or any(
                p.get("ready") and str(p.get("item") or "").lower() in C.DEFENSES for p in view.production):
            return None
        want = _scheduled(TURTLE_DEFENSES, view.game_seconds, 0)
        have = sum(1 for b in view.buildings if b.type in C.DEFENSES and not b.wreck)
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


_BY_NAME = {"rusher": Rusher, "turtler": Turtler, "macro": Macro}


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
