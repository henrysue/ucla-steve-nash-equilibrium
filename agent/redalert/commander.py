"""The commander: the army's one decision each view.

Everything that decides the army's mode lives here, in this order:

1. ``brain.choose_mode``, the policy: DEFEND (a real threat at a base),
   HOLD (an early warning or a rush read), FINISH (razed or dominated),
   RETREAT (our attack is losing), ATTACK (we clearly outweigh them and can
   break their turrets) or MASS (BUILD), plus the opening.
2. A forced mode (the sparring archetypes only; sparring.py).
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

from . import brain, scout
from . import config as C
from .modes import Mode
from .observe import View


class Commander:
    def __init__(self) -> None:
        self.mode: Mode = Mode.OPEN
        self.mode_since = 0.0
        self.changed = False
        self.confirm_at: float | None = None      # when the attack call being confirmed was made
        self.wants_look = False                    # a blind attack call: a scout looks at their army first
        self._ledger: list[tuple[float, float, float]] = []   # (game seconds, army kills, army deaths), TRADE_LEAD_WINDOW_SECONDS
        self._army_trades = [0.0, 0.0]            # kills and deaths while the main army was in a fight
        self._last_trades: tuple[float, float] | None = None
        self._contact = False                     # the main army was in a fight last view
        self.cheese_since: float | None = None    # threat window (brain.update_cheese_since)
        self.retreated_at: float | None = None
        self.retreat_army: float | None = None    # army value at the last retreat
        self.attack_army = 0.0                    # army value when the current attack began
        self.attack_avoid: tuple | None = None    # where the raid held their army when this attack began
        self.attacks = 0
        self._trades: list[tuple[float, float, float]] = []   # (game seconds, kills, deaths)
        self._building_ids: set | None = None
        self._held_since: float | None = None      # the advisor's "hold" has stopped an attack the rules call since

    def track_threat(self, view: View) -> None:
        """Update the threat window (call first each view). Losing a building
        at the main base to a push bigger than a small raid opens it too, however many units
        are "at home": match e2f73d65 lost seven buildings in 90 s and never
        left BUILD. A small raid is the home units' job (executor.respond)."""
        self.cheese_since = brain.update_cheese_since(view, self.cheese_since)
        ids = {b.id for b in view.buildings if b.id is not None and not b.wreck}
        if (self._building_ids is not None and self._building_ids - ids and view.game_seconds > 0
                and _push_value(view) > C.SMALL_RAID_VALUE):
            self.cheese_since = view.game_seconds
        self._building_ids = ids

    def decide(self, view: View, *, model_pref: Mode | None = None, scout_id: int | None = None,
               squad_ids: frozenset = frozenset(), hunting: bool = False, alarm: bool = False,
               distraction=None, escorting: bool = False, rushing: bool = False,
               forced: Mode | None = None) -> Mode:
        prev = self.mode
        losing = self.losing_trades(view)
        up = self.traded_up(view, squad_ids)
        # A base race: our attack is winning, so it goes on while their push
        # hits our base (the home units still fight it). In the opening a
        # rush always calls the army home.
        base_race = (prev is Mode.ALL_IN and not losing and brain.army_odds(view) >= 1.0
                     and view.game_seconds >= C.CHEESE_WINDOW_SECONDS)
        winning = self.winning_trades(view) if prev is Mode.ALL_IN else False

        def choose(pref):
            return brain.choose_mode(
                view, cheese_since=None if base_race else self.cheese_since, model_pref=pref, prev_mode=prev,
                scout_id=scout_id, squad_ids=squad_ids, losing_trades=losing, winning_trades=winning,
                traded_up=up,
                retreated_at=self.retreated_at, retreat_army=self.retreat_army, hunting=hunting,
                alarm=alarm, distraction=distraction,
                attack_seconds=view.game_seconds - self.mode_since if prev is Mode.ALL_IN else None,
            )

        mode = choose(model_pref)
        # The advisor's "hold" may delay an attack the rules call, not veto it:
        # after ADVISOR_HOLD_MAX_SECONDS the rules go (c1c37a10 held 13
        # minutes at 1.4-1.9x odds — their real army was a third of ours at
        # 9:14 — while a turtle built 28800 behind its pillboxes).
        blocking = model_pref is Mode.BUILD and mode is not Mode.ALL_IN and choose(None) is Mode.ALL_IN
        if not blocking:
            self._held_since = None
        else:
            if self._held_since is None:
                self._held_since = view.game_seconds
            if view.game_seconds - self._held_since >= C.ADVISOR_HOLD_MAX_SECONDS:
                mode = Mode.ALL_IN
        if forced is not None:                     # sparring (TEST ONLY): a mode, or a function of the policy's
            mode = (forced(mode) or mode) if callable(forced) else forced
        if escorting and mode is Mode.ALL_IN and prev is not Mode.ALL_IN and not brain.must_attack(view) and not hunting:
            mode = Mode.BUILD                      # defend the haul: no new attack (one under way goes on)
        # A strike on an expansion needs no look at their army: we know what
        # stands there (the raid or a scout just saw it).
        strike = mode is Mode.ALL_IN and prev is not Mode.ALL_IN and brain.expansion_strike(view) is not None
        mode = self._confirm(view, prev, mode, skip=rushing or hunting or up or brain._catching_up(view) or strike)
        if ({prev, mode} == {Mode.BUILD, Mode.ALL_IN}
                and view.game_seconds - self.mode_since < C.MODE_MIN_DWELL_SECONDS):
            mode = prev                            # commit: no quick flip (a confirmation is kept)
        if mode is Mode.ALL_IN:
            self.confirm_at = None
        self._record(view, prev, mode, distraction)
        return mode

    @property
    def confirming(self) -> bool:
        return self.confirm_at is not None

    def traded_up(self, view: View, squad_ids: frozenset = frozenset()) -> bool:
        """In the last ``TRADE_LEAD_WINDOW_SECONDS`` the main army's fights
        killed ``TRADE_LEAD_MIN`` more than they lost and ``TRADE_LEAD_RATIO``
        x it: their army is down by that much — go now (3bd6e450 won a fight
        16600 to 11150 at 8:15 and sat for three minutes). Only kills while
        the main army was in contact count: a raid's refinery and harvester
        say nothing about their army."""
        t = view.game_seconds
        now = (view.kills_cost, view.deaths_cost)
        contact = _army_in_contact(view, squad_ids)
        if self._last_trades is not None and (contact or self._contact):
            self._army_trades[0] += max(0.0, now[0] - self._last_trades[0])
            self._army_trades[1] += max(0.0, now[1] - self._last_trades[1])
        self._last_trades, self._contact = now, contact
        self._ledger.append((t, *self._army_trades))
        self._ledger = [r for r in self._ledger if t - r[0] <= C.TRADE_LEAD_WINDOW_SECONDS]
        _, k0, d0 = self._ledger[0]
        killed, lost = self._army_trades[0] - k0, self._army_trades[1] - d0
        return killed - lost >= C.TRADE_LEAD_MIN and killed >= C.TRADE_LEAD_RATIO * max(lost, 1.0)

    def winning_trades(self, view: View) -> bool:
        """Since this attack began we have killed at least ``TRADE_WIN_MIN``
        more than we lost (losing_trades keeps the history)."""
        since = [row for row in self._trades if row[0] >= self.mode_since]
        if not since:
            return False
        _, kills0, deaths0 = since[0]
        return (view.kills_cost - kills0) - (view.deaths_cost - deaths0) >= C.TRADE_WIN_MIN

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
        # Scaled by the bigger of the army we attacked with and the one we have
        # now (0ee93a05 retreated an 8500 army over 1500 lost, its floor set by
        # the 3900 it started the attack with).
        floor = max(C.TRADE_LOSS_MIN, C.TRADE_LOSS_ARMY_SHARE * max(self.attack_army, view.army_value))
        return lost >= floor and lost > C.TRADE_LOSS_RATIO * killed

    # -- internals ---------------------------------------------------------------

    def _confirm(self, view: View, prev: Mode, mode: Mode, skip: bool) -> Mode:
        """Hold a fresh attack call (on about even trades) until it has held
        for ``CONFIRM_SECONDS``; drop the call when the policy changes its mind."""
        even = abs(view.kills_cost - view.deaths_cost) <= C.CONFIRM_TRADE_BAND
        # A call made knowing too little of their army waits (up to
        # CONFIRM_BLIND_MAX_SECONDS) for a scout to look where their army
        # masses — from the first attack, not only late: with no guess in the
        # estimate, little seen reads as a big edge.
        blind = brain.blind_estimate(view)
        self.wants_look = False
        if mode is Mode.ALL_IN and prev is not Mode.ALL_IN and (even or blind) and not skip:
            if self.confirm_at is None:
                self.confirm_at = view.game_seconds
            waited = view.game_seconds - self.confirm_at
            # Blind, it waits for a look however long it takes (only behind at
            # the end does it go without): the 40 s expiry sent blind attacks
            # that lost the army (b9145367 at 4:41: 3100 into "seen 0", 300
            # left; 5a4bc83f at 7:06: 7350 lost).
            if waited < C.CONFIRM_SECONDS or (blind and not brain._catching_up(view)):
                self.wants_look = blind
                return prev
        elif mode is not Mode.ALL_IN:
            self.confirm_at = None
        return mode

    def _record(self, view: View, prev: Mode, mode: Mode, distraction=None) -> None:
        self.changed = mode is not prev
        if self.changed:
            self.mode_since = view.game_seconds
            if mode is Mode.ALL_IN:
                self.attack_army = view.army_value
                self.attacks += 1
                self.attack_avoid = distraction[2] if distraction else None
        if mode is not Mode.ALL_IN:
            self.attack_avoid = None
        if mode is Mode.RETREAT and prev is not Mode.RETREAT:
            self.retreated_at = view.game_seconds
            self.retreat_army = view.army_value
        self.mode = mode


def _army_in_contact(view: View, squad_ids: frozenset) -> bool:
    """An enemy fighter within ``STAGE_ENGAGE_CELLS`` of a main-army unit."""
    r2 = C.STAGE_ENGAGE_CELLS ** 2
    army = [u.cell for u in view.combat_units if u.cell is not None and u.id not in squad_ids]
    foes = [a.cell for a in view.enemy_units if a.cell is not None and not a.wreck and a.type not in C.HARVESTER]
    return any((f[0] - c[0]) ** 2 + (f[1] - c[1]) ** 2 <= r2 for f in foes for c in army)


def _push_value(view: View) -> float:
    """The enemy combat units at our main base (scout.enemy_threat_near_base), priced."""
    return sum(view.unit_costs.get(a.type, C.DEFAULT_UNIT_COST) for a in scout.enemy_threat_near_base(view, main_only=True))
