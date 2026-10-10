"""TEST ONLY — REMOVE BEFORE THE TOURNAMENT. A sparring partner that rushes.

When we play ourselves in a test match (Testing page, ``tournament_id`` is
None, and the other seat of the same match runs on this machine), one seat
(``RUSH_SEAT``, default 1) turns into an early rusher so the other seat's
scouting, early-push warning and defenses get tested:

- a rush build (power, barracks, refinery, barracks, power, refinery, war
  factory; up to ``RUSH_BARRACKS`` barracks), no expansion, no turrets
  unless attacked, no refineries beyond the opening's two;
- production never pauses: infantry ``RUSH_INFANTRY_DEPTH`` deep, tanks
  ``RUSH_TANK_DEPTH`` deep;
- it masses a ball at its staging point and launches when the army is worth
  ``RUSH_BALL_VALUE`` (or ``RUSH_MIN_ARMY_VALUE`` by ``RUSH_FORCE_SECONDS``),
  keeps pushing with reinforcements streaming in, and falls back to re-mass
  only once the army drops below ``RUSH_REGROUP_SHARE`` of its launch value
  — no early hold, turret rule or protect-lead.

Self-play is detected with marker files: each seat writes
``<tmp>/redalert_selfplay/<session>-seat<k>`` and looks for the other seat's.
Environment: ``RA_TEST_RUSH=0`` turns it off; ``RA_TEST_RUSH_SEAT`` picks the
rushing seat. Remove this module and its few call sites (grep "testrush")
to retire it.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

RUSH_SEAT = 1
RUSH_START_SECONDS = 120.0
RUSH_MIN_ARMY_VALUE = 1500       # launch with at least this...
RUSH_BALL_VALUE = 2500           # ...once the ball is worth this...
RUSH_FORCE_SECONDS = 210.0       # ...or at this time with RUSH_MIN_ARMY_VALUE
RUSH_REGROUP_SHARE = 0.4         # pushing until the army falls below this share of its launch value
RUSH_BARRACKS = 3
RUSH_INFANTRY_DEPTH = 3
RUSH_TANK_DEPTH = 2
RUSH_BUILD_ORDER = ("powr", ("tent", "barr"), "proc", ("tent", "barr"), "powr", "proc", "weap")
_MARKER_DIR = Path(tempfile.gettempdir()) / "redalert_selfplay"
_STALE_SECONDS = 3 * 3600


class SelfPlayRusher:
    """Decides, once per match, whether this seat is the test rusher."""

    def __init__(self) -> None:
        self._session: str | None = None
        self._active = False
        self._settled = False
        self._marked = False
        self._launch_value: float | None = None   # army value when the current push launched

    def active(self, context) -> bool:
        """True when this seat should rush (test match against ourselves)."""
        if self._settled:
            return self._active
        if os.environ.get("RA_TEST_RUSH", "1").strip() == "0":
            self._settled = True
            return False
        if getattr(context, "tournament_id", None) is not None:
            self._settled = True          # never in a tournament game
            return False
        session = getattr(context, "session_id", None)
        seat = getattr(context, "seat_position", None)
        if not session or seat is None:
            return False
        self._session = str(session)
        try:
            _MARKER_DIR.mkdir(parents=True, exist_ok=True)
            if not self._marked:
                self._clean()
                (_MARKER_DIR / f"{self._session}-seat{seat}").write_text(str(time.time()), encoding="utf-8")
                self._marked = True
            others = [p for p in _MARKER_DIR.glob(f"{self._session}-seat*") if p.name != f"{self._session}-seat{seat}"]
        except OSError:
            return False
        if not others:
            return False                  # not (yet) known to be self-play; ask again next view
        self._settled = True
        want = os.environ.get("RA_TEST_RUSH_SEAT", str(RUSH_SEAT)).strip()
        self._active = want.isdigit() and int(want) == int(seat)
        return self._active

    @staticmethod
    def _clean() -> None:
        now = time.time()
        for p in _MARKER_DIR.glob("*-seat*"):
            try:
                if now - p.stat().st_mtime > _STALE_SECONDS:
                    p.unlink()
            except OSError:
                pass

    def mode(self, view, cheese_since):
        """The rusher's posture: defend only if its own base is hit; mass a
        ball (BUILD: the army gathers at its staging point) and launch it
        (ALL_IN) once big enough; keep pushing until it is mostly spent, then
        re-mass. ``None`` before it has a yard and barracks."""
        from . import config as C
        from .brain import _defend
        from .modes import Mode

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


def rush_build(view) -> list[dict]:
    """The rusher's next building: more barracks, else the rush build order."""
    from . import buildplan
    from . import config as C
    from . import orders

    money = view.cash + view.ore
    rax = buildplan._available(view, C.BARRACKS)
    if (rax and view.has_building(C.REFINERY) and money >= C.BARRACKS_ADD_FUNDS
            and not any(view.queued_count(b) for b in C.BARRACKS)
            and buildplan._count_built_and_queued(view, C.BARRACKS) < RUSH_BARRACKS):
        return [orders.build(rax)]
    return buildplan.walk_build_order(view, RUSH_BUILD_ORDER)


def rush_production(view) -> list[dict]:
    """Units without pause: infantry ``RUSH_INFANTRY_DEPTH`` deep (a rocket
    soldier in every few), tanks ``RUSH_TANK_DEPTH`` deep once the factory is
    up — one unit per order."""
    from . import config as C
    from . import orders

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
