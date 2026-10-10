"""The commander: the army's one decision each view.

Everything that decides the army's mode lives here, in this order:

1. ``brain.choose_mode``, the policy: DEFEND (a real threat at a base),
   HOLD (an early warning or a rush read), FINISH (razed or dominated),
   RETREAT (our attack is losing), ATTACK (we clearly outweigh them and can
   break their turrets) or MASS (BUILD), plus the opening.
2. A forced mode (the test rusher only; testrush.py).
3. The MCV escort: no attack while an expansion MCV is on the road, unless
   we must attack.
4. Look before committing: while trades are about even
   (``CONFIRM_TRADE_BAND``), a fresh attack call goes ahead only if it still
   holds ``CONFIRM_SECONDS`` later (the army edges forward to see meanwhile).
5. No BUILD <-> ALL_IN flip within ``MODE_MIN_DWELL_SECONDS``; a confirmation
   made meanwhile is kept.

It also keeps the bookkeeping these need: when the mode last changed, the
army at the start of an attack, the trade window, the last retreat and the
threat window. ``agent.py`` only feeds it data and acts on the mode.
"""

from __future__ import annotations

from . import brain
from . import config as C
from .modes import Mode
from .observe import View


class Commander:
    def __init__(self) -> None:
        self.mode: Mode = Mode.OPEN
        self.mode_since = 0.0
        self.changed = False
        self.confirm_at: float | None = None      # when the attack call being confirmed was made
        self.cheese_since: float | None = None    # threat window (brain.update_cheese_since)
        self.retreated_at: float | None = None
        self.retreat_army: float | None = None    # army value at the last retreat
        self.attack_army = 0.0                    # army value when the current attack began
        self.attacks = 0
        self._trades: list[tuple[float, float, float]] = []   # (game seconds, kills, deaths)
        self._building_ids: set | None = None

    def track_threat(self, view: View) -> None:
        """Update the threat window (call first each view). Losing a building
        opens it too, whatever the threat's size: match e2f73d65 lost seven
        buildings in 90 s and never left BUILD (the attackers it could see
        never outweighed the units and turrets "at home")."""
        self.cheese_since = brain.update_cheese_since(view, self.cheese_since)
        ids = {b.id for b in view.buildings if b.id is not None and not b.wreck}
        if self._building_ids is not None and self._building_ids - ids and view.game_seconds > 0:
            self.cheese_since = view.game_seconds
        self._building_ids = ids

    def decide(self, view: View, *, model_pref: Mode | None = None, scout_id: int | None = None,
               squad_ids: frozenset = frozenset(), hunting: bool = False, alarm: bool = False,
               distraction: bool = False, escorting: bool = False, rushing: bool = False,
               forced: Mode | None = None) -> Mode:
        prev = self.mode
        mode = brain.choose_mode(
            view, cheese_since=self.cheese_since, model_pref=model_pref, prev_mode=prev,
            scout_id=scout_id, squad_ids=squad_ids, losing_trades=self.losing_trades(view),
            retreated_at=self.retreated_at, retreat_army=self.retreat_army, hunting=hunting,
            alarm=alarm, distraction=distraction,
        )
        if forced is not None:                     # testrush (TEST ONLY)
            mode = forced
        if escorting and mode is Mode.ALL_IN and not brain.must_attack(view) and not hunting:
            mode = Mode.BUILD                      # defend the haul
        mode = self._confirm(view, prev, mode, skip=rushing or hunting or brain._catching_up(view))
        if ({prev, mode} == {Mode.BUILD, Mode.ALL_IN}
                and view.game_seconds - self.mode_since < C.MODE_MIN_DWELL_SECONDS):
            mode = prev                            # commit: no quick flip (a confirmation is kept)
        if mode is Mode.ALL_IN:
            self.confirm_at = None
        self._record(view, prev, mode)
        return mode

    @property
    def confirming(self) -> bool:
        return self.confirm_at is not None

    def losing_trades(self, view: View) -> bool:
        """Since the attack began (at most ``TRADE_WINDOW_SECONDS`` back): lost
        at least ``TRADE_LOSS_MIN`` and more than ``TRADE_LOSS_RATIO`` x what
        we killed. Losses from before the attack don't count (match 28e54539
        retreated 3 s into an attack over losses taken while massing)."""
        t = view.game_seconds
        self._trades.append((t, view.kills_cost, view.deaths_cost))
        self._trades = [row for row in self._trades if t - row[0] <= C.TRADE_WINDOW_SECONDS]
        if t - self.mode_since < C.MIN_ENGAGEMENT_SECONDS:
            return False  # give the fight a chance: the first volley always costs
        since = [row for row in self._trades if row[0] >= self.mode_since] or self._trades[-1:]
        _, kills0, deaths0 = since[0]
        lost, killed = view.deaths_cost - deaths0, view.kills_cost - kills0
        floor = max(C.TRADE_LOSS_MIN, C.TRADE_LOSS_ARMY_SHARE * self.attack_army)
        return lost >= floor and lost > C.TRADE_LOSS_RATIO * killed

    # -- internals ---------------------------------------------------------------

    def _confirm(self, view: View, prev: Mode, mode: Mode, skip: bool) -> Mode:
        """Hold a fresh attack call (on about even trades) until it has held
        for ``CONFIRM_SECONDS``; drop the call when the policy changes its mind."""
        even = abs(view.kills_cost - view.deaths_cost) <= C.CONFIRM_TRADE_BAND
        if mode is Mode.ALL_IN and prev is not Mode.ALL_IN and even and not skip:
            if self.confirm_at is None:
                self.confirm_at = view.game_seconds
            if view.game_seconds - self.confirm_at < C.CONFIRM_SECONDS:
                return prev
        elif mode is not Mode.ALL_IN:
            self.confirm_at = None
        return mode

    def _record(self, view: View, prev: Mode, mode: Mode) -> None:
        self.changed = mode is not prev
        if self.changed:
            self.mode_since = view.game_seconds
            if mode is Mode.ALL_IN:
                self.attack_army = view.army_value
                self.attacks += 1
        if mode is Mode.RETREAT and prev is not Mode.RETREAT:
            self.retreated_at = view.game_seconds
            self.retreat_army = view.army_value
        self.mode = mode
