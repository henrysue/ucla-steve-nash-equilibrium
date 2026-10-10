"""The Red Alert agent: one ``RedAlertAgent`` per match.

Real time, so ``choose_action`` never waits on a model. Every view: read the
state (observe), remember enemies (memory), pick a Mode (brain), update the
squads (scout, raid, home guard, garrison, expansion), add micro, and send
one order batch (executor). The model advisor and Haiku squad leads run on
background threads; their newest advice is used when ready. Refused orders are
held back briefly; unchanged army orders aren't resent every view. Without a
model the rules play a complete match. Platform rules:
https://api.altruagent-game.com/skill/redalert.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter
import time
from collections.abc import Callable
from typing import Any

from altruagent import WAIT, DecisionContext, GameState, WithReasoning
from altruagent.console import print_line

from . import brain, buildplan, executor, micro, orders, scout, squadlead
from . import testrush  # TEST ONLY (self-play sparring rusher): remove before the tournament
from . import intel, scouting
from .commander import Commander
from .expand import Expansion
from .expand import enemy_expansions as _enemy_expansions
from .guard import Garrison
from .raid import Raid
from . import config as C
from .matchlog import MatchLog
from .memory import EnemyMemory
from .harvest import HarvestManager
from .movement import Unstick
from .modes import Mode
from .observe import dist2, enemy_ore_field, read


def create_agent():
    from .advisor import advisor_from_env
    from .squadlead import squad_leads_from_env

    return RedAlertAgent(advisor_factory=advisor_from_env, squad_lead_factory=squad_leads_from_env)


class RedAlertAgent:
    def __init__(
        self,
        *,
        log: Callable[[str], None] = print_line,
        clock: Callable[[], float] = time.monotonic,
        advisor_factory: Callable[..., Any] | None = None,
        squad_lead_factory: Callable[..., Any] | None = None,
        log_dir: str | None = None,
    ) -> None:
        self._log = log
        self._clock = clock
        self._commander = Commander()   # the army's one decision each view (commander.py)
        self._opening_done = False      # the standard build order stands (buildplan.opening_done)
        self._failed_cells: dict = {}   # cell -> game time the engine moved a building off it
        # The batch play_action answers next; refused order key -> blocked
        # until (clock); army (cmd, destination) -> (last sent, unit ids).
        self._last_sent: list[dict] = []
        self._blocked: dict[str, float] = {}
        self._army_sent: dict[str, tuple[float, frozenset]] = {}
        # build/train item -> when we last sent it (until production shows it).
        self._production_sent: dict[str, float] = {}
        self._patrol = scouting.ScoutCorps()
        self._intel = intel.EnemyIntel()
        self._memory = EnemyMemory()
        self._raid = Raid()
        self._expansion = Expansion()
        self._garrison = Garrison()
        self._unstick = Unstick()
        self._hunter = scout.Hunter()
        self._placing: dict = {}   # item -> {"asked", "t", "known"}: where we asked to place it
        self._sold_at = float("-inf")
        self._kills_seen = (0.0, 0.0)               # (kills_cost, game time it last changed)
        self._ignore_advice_until = float("-inf")   # advisor's army target ignored until (stall breaker)
        self._rusher = testrush.SelfPlayRusher()   # testrush: TEST ONLY
        self._models_checked = False                # REDALERT_MODELS_OFF_SEAT applied yet
        self._clearing: dict = {}   # building item -> game time its spot was first found occupied
        self._harvest = HarvestManager()
        # Per-match review log (off unless a log directory is configured).
        self._matchlog = MatchLog(log, directory=log_dir)
        self._game_seconds: float | None = None
        # Flank route for the current attack (executor.choose_flank).
        self._flank_via: tuple[int, int] | None = None
        # The model advisor (None: rules only). Built after the match log so
        # each piece of advice is logged as it arrives.
        self._advisor = advisor_factory(log, on_advice=self._on_advice) if advisor_factory else None
        self._last_plan: tuple[str, str] | None = None
        # Haiku squad leads for the raid squad and the garrison ({} = rules only).
        self._leads = squad_lead_factory(log, on_directive=self._matchlog.squad) if squad_lead_factory else {}

    # -- the hot path ---------------------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        raw = getattr(state, "raw", None)
        if not isinstance(raw, dict) or not isinstance(raw.get("observation"), dict):
            return WAIT  # nothing to look at yet

        view = read(raw, getattr(context, "game_config", None))
        if not view.ok:
            return WAIT

        # Remember enemies through the fog: the brain weighs the enemy army we
        # remember, and the army can target buildings we saw earlier.
        self._models_switch(context)
        self._audit_placements(view)
        self._memory.update(view)
        view = self._memory.apply(view)
        self._failed_cells = {c: t for c, t in self._failed_cells.items()
                              if view.game_seconds - t < C.FAILED_SPOT_SECONDS}
        if self._failed_cells:
            view = dataclasses.replace(view, failed_cells=frozenset(self._failed_cells))

        # Track the threat window (sets/expires defensively).
        self._commander.track_threat(view)

        self._intel.update(view)
        self._patrol.update(view, taken={*self._raid.unit_ids, *self._garrison.unit_ids})
        threat = self._patrol.alarm(view) or self._intel.rushing(view)
        # The opening is the standard build order's: the advisor is consulted
        # once it stands, or sooner if a rush is read or the base is attacked.
        self._opening_done = self._opening_done or buildplan.opening_done(view)
        advice = (self._poll_advisor(view)
                  if self._opening_done or threat or self._commander.cheese_since is not None else None)
        hunting = self._hunter.hunting(view)
        self._break_stall(view)
        # testrush (TEST ONLY): against ourselves in a test match, one seat rushes.
        rushing = self._rusher.active(context)
        forced = self._rusher.mode(view, self._commander.cheese_since) if rushing else None
        # The army's one decision (commander.py).
        self._commander.decide(
            view, model_pref=_POSTURE.get(getattr(advice, "posture", None)), scout_id=self._patrol.unit_id,
            squad_ids=frozenset({*self._patrol.unit_ids, *self._raid.unit_ids, *self._garrison.unit_ids}),
            hunting=hunting, alarm=threat, distraction=self._raid.distracting(view),
            escorting=self._expansion.escorting, rushing=rushing, forced=forced,
        )
        if self._commander.changed:
            # While the raid squad holds their army, straight down the centre.
            self._flank_via = (None if self._mode is not Mode.ALL_IN or self._raid.distracting(view)
                               else self._pick_route(view, advice))
        self._update_flank(view)
        if not rushing:   # testrush: the rusher never expands
            self._expansion.update(view, self._mode, advice, ore_running_out=self._harvest.home_depleted(view))
        if self._expansion.scout_target is not None and self._patrol.visit is None:
            self._patrol.visit = self._expansion.scout_target
        self._game_seconds = view.game_seconds

        self._raid.update(view, self._mode, self._patrol.unit_id, advice,
                          taken=set(self._patrol.unit_ids))
        self._garrison.update(view, self._mode, self._expansion,
                              {*self._patrol.unit_ids, *self._raid.unit_ids})
        self._matchlog.observe(raw, view, self._mode, context, extra={
            "scout": self._patrol.unit_id, "scouts": dict(self._patrol.roles),
            "enemy_opening": self._intel.opening(), "threat": threat,
            "confirming": self._confirm_at is not None,
            "raid_squad": len(self._raid.unit_ids),

            # Confirms the live game_config carries the map picture.
            "map_ore_cells": len(view.ore_cells), "enemy_ore_field": enemy_ore_field(view),
            "raid_target": self._raid.target(view),
            "flank_via": self._flank_via,
            "expansion": self._expansion.state, "expansion_site": self._expansion.site,
            "expansions_done": self._expansion.sites, "garrison": len(self._garrison.unit_ids),
            # Each expansion's ore, whether a refinery of ours serves it, and
            # how often the engine moved an expansion refinery off its spot.
            "expansion_ores": [d["ore"] for d in self._expansion.done],
            "expansions_served": [self._expansion.served(view, d["ore"]) for d in self._expansion.done],
            "expand_push": self._expansion._push(view.game_seconds),
            "harvesters_by_field": {f"{f[0]},{f[1]}": n for f, n in self._harvest.last_counts.items()},
            "depleted_fields": sorted(f"{f[0]},{f[1]}" for f in self._harvest.depleted()),
            "home_depleted": self._harvest.home_depleted(view),
            "early_push_warning": scout.early_push_warning(view),
            "hunting": hunting,
            "test_rusher": rushing,   # testrush
            "models": "on" if (self._advisor is not None or self._leads) else "off",
            "active_base": self._active_base(view),
            "harvester_acts": dict(Counter(h.act for h in view.units_of(C.HARVESTER) if h.act)),
            "unstuck": self._unstick.unstuck,
            "enemy_expansions": _enemy_expansions(view),
            "advice": getattr(advice, "posture", None), "advisor_calls": getattr(self._advisor, "calls", 0),
        })
        squads = {*self._patrol.unit_ids, *self._raid.unit_ids, *self._garrison.unit_ids}
        micro_orders, micro_ids = micro.plan(view, self._mode, frozenset(squads), scout.choke_cell(view))
        hunt = self._hunter.targets(view, C.HUNT_GROUPS) if self._mode is Mode.ALL_IN else []
        sell = buildplan.sell_for_income(view)
        if sell is not None and view.game_seconds - self._sold_at >= C.SELL_FOR_INCOME_SECONDS:
            self._sold_at = view.game_seconds
        else:
            sell = None
        planned = ([sell] if sell else []) + executor.plan(
            view, self._mode,
            scout_id=self._patrol.unit_id,
            scout_orders=(self._patrol.orders(view, self._hunter.targets(view, len(self._patrol.roles))
                                              if hunting else None) + self._patrol.production(view)),
            raid_ids=[*self._raid.unit_ids, *self._garrison.unit_ids, *micro_ids,
                      *self._patrol.unit_ids],
            raid_orders=(self._raid_orders(view)
                         + self._garrison_orders(view) + micro_orders),
            advice=self._army_advice(advice, hunting, view),
            flank_via=self._flank_via, expansion=self._expansion,
            hunt=hunt[0] if hunt else None, hunt_points=hunt,
            harvest_orders=self._harvest.orders(view),
            depleted=frozenset(self._harvest.depleted()),
            home=self._active_base(view),
            clearing=self._clearing,
            rusher=rushing,   # testrush
            threat=threat,
            probe=self._probe_point(view),
        ) + self._unstick.orders(view) + executor.stance_orders(
            view, self._mode, set(self._garrison.unit_ids))
        now = self._clock()
        batch_orders = [
            o for o in planned
            if self._blocked.get(_order_key(o), 0.0) <= now
            and not self._army_repeat(o, now) and not self._production_pending(o, now)
        ]

        move = orders.batch(batch_orders)
        if move is None:
            return WAIT  # nothing to do right now

        self._last_sent = move["orders"]
        self._note_placements(view, self._last_sent)
        for order in self._last_sent:
            if order.get("cmd") in _ARMY_CMDS:
                self._army_sent[_army_key(order)] = (now, frozenset(order.get("units") or ()))
            elif order.get("cmd") in _PRODUCTION_CMDS:
                self._production_sent[_production_key(order)] = now
        return WithReasoning(move, self._reason(view))

    def on_action_result(self, result: Any, context: DecisionContext) -> None:
        """Learn from each play_action answer. Orders the server refused (the
        whole batch on INVALID_ACTION, or single rows marked ``rejected``) are
        held back for ``REFUSAL_BLOCK_SECONDS``, then allowed again — a refusal
        never blocks an order for the rest of the match."""
        if not isinstance(result, dict):
            return
        until = self._clock() + C.REFUSAL_BLOCK_SECONDS
        sent = self._last_sent
        if str(result.get("error") or "").upper() == "INVALID_ACTION":
            refused = [(o, "batch_refused", result.get("detail")) for o in sent]
        else:
            refused = [
                (sent[row["index"]], row.get("reason"), row.get("detail"))
                for row in result.get("results") or []
                if isinstance(row, dict) and row.get("status") == "rejected"
                and isinstance(row.get("index"), int) and 0 <= row["index"] < len(sent)
            ]
        for order, reason, detail in refused:
            self._blocked[_order_key(order)] = until
            self._army_sent.pop(_army_key(order), None)
            if order.get("cmd") in _PRODUCTION_CMDS:
                self._production_sent.pop(_production_key(order), None)
            self._matchlog.refused(order, reason, detail, self._game_seconds)

    def on_match_end(self, final_state: Any, context: DecisionContext) -> None:
        """Record how the match ended (the runner calls this once)."""
        self._matchlog.end(final_state, context)
        if self._advisor is not None:
            self._advisor.close()
        for lead in self._leads.values():
            lead.close()

    def _probe_point(self, view):
        """While an attack call waits for confirmation: a point a little way
        from the massing point toward the target, where the army edges to see."""
        if self._confirm_at is None:
            return None
        stage = scout.choke_cell(view)
        target = executor._attack_target(view)
        if stage is None or target is None:
            return None
        f = C.PROBE_FRACTION
        return (round(stage[0] + (target[0] - stage[0]) * f), round(stage[1] + (target[1] - stage[1]) * f))

    def _main_fight(self, view):
        """Where the main army is fighting (ALL_IN, enemies near it), else None."""
        if self._mode is not Mode.ALL_IN:
            return None
        squads = {*self._patrol.unit_ids, *self._raid.unit_ids, *self._garrison.unit_ids}
        army = [u.cell for u in view.combat_units if u.cell is not None and u.id not in squads]
        if not army:
            return None
        centre = (sum(c[0] for c in army) / len(army), sum(c[1] for c in army) / len(army))
        r2 = C.MAIN_FIGHT_RADIUS_CELLS ** 2
        near = [a.cell for a in view.enemy_units if a.cell is not None and not a.wreck
                and a.type not in C.HARVESTER and dist2(a.cell, centre) <= r2]
        return min(near, key=lambda c: dist2(c, centre)) if near else None

    def _raid_orders(self, view) -> list[dict]:
        """The raid squad's orders. The fight in front of it comes first and
        is the rules' call, not the lead's: defenders before harvesters, pull
        back when outmatched, join the main army's fight as a second prong.
        Otherwise its Haiku lead's call when fresh and usable, else the rules."""
        lead = self._leads.get("raid")
        ids = list(self._raid.unit_ids)
        fight = self._main_fight(view)
        if ids and (self._raid.situation(view)[0] in ("fight", "outmatched") or fight is not None):
            if lead is not None:
                lead.poll(view, [], "")
            return self._raid.orders(view, fight)
        if lead is not None and ids:
            d = lead.poll(view, ids, "Hunt enemy harvesters and weakly defended expansions; avoid defenses.",
                          {"rules_target": self._raid.target(view)})
            o = squadlead.squad_orders(view, ids, d, scout.choke_cell(view)) if d is not None else None
            if o is not None:
                return o
        elif lead is not None:
            lead.poll(view, [], "")
        return self._raid.orders(view)

    def _garrison_orders(self, view) -> list[dict]:
        """The garrison's orders: its Haiku lead's call when fresh, usable and
        within ``GARRISON_LEASH_CELLS`` of its post, else the garrison rules."""
        lead = self._leads.get("garrison")
        ids = list(self._garrison.unit_ids)
        post = self._garrison.post(view, self._expansion)
        if lead is not None and ids and post is not None:
            d = lead.poll(view, ids, f"Hold our expansion; post at {list(post)}. Engage threats near it; don't chase far.",
                          {"post": list(post)})
            leashed = d is not None and (d.cell is None or dist2(d.cell, post) <= C.GARRISON_LEASH_CELLS ** 2)
            o = squadlead.squad_orders(view, ids, d, post) if leashed else None
            if o is not None:
                return o
        elif lead is not None:
            lead.poll(view, [], "")
        return self._garrison.orders(view, self._expansion)

    def _active_base(self, view):
        """The expansion that takes over building once home is mined out (or None)."""
        return self._expansion.active_base(view, self._harvest.home_depleted(view), self._harvest.depleted())

    def _pick_route(self, view, advice) -> tuple[int, int] | None:
        """The flank waypoint for a new attack (None: straight at the target).
        The advisor's route when it gave one, else the side with the fewest
        known enemies, rotating on ties."""
        target = getattr(advice, "army_target", None) or executor._attack_target(view)
        route = getattr(advice, "attack_route", "auto")
        if view.game_seconds < C.FLANK_AFTER_SECONDS:
            return None   # early: straight down the middle, keeping map control
        if route in ("left", "right"):
            return executor.flank_waypoint(view, target, -1 if route == "left" else 1)
        if route == "center":
            return None
        return executor.choose_flank(view, target, self._attacks)[1]

    def _update_flank(self, view) -> None:
        """End the flank leg once the army reaches the waypoint, or after
        ``FLANK_MAX_SECONDS``: from then on it attacks the target directly."""
        if self._flank_via is None:
            return
        if view.game_seconds - self._mode_since > C.FLANK_MAX_SECONDS:
            self._flank_via = None
            return
        army = [u.cell for u in view.combat_units if u.cell is not None and u.id not in self._patrol.unit_ids]
        if army:
            cx = sum(c[0] for c in army) / len(army)
            cy = sum(c[1] for c in army) / len(army)
            if (cx - self._flank_via[0]) ** 2 + (cy - self._flank_via[1]) ** 2 <= C.FLANK_ARRIVED_CELLS ** 2:
                self._flank_via = None

    def _on_advice(self, advice, latency_s: float) -> None:
        """Each answer goes to the match log; the console gets a line only
        when the posture or plan changes, so it can be followed live without
        a line every few seconds. Local only: never in reasoning_summary."""
        self._matchlog.advice(advice, latency_s)
        key = (advice.posture, advice.plan)
        if key != self._last_plan:
            self._last_plan = key
            self._log(
                f"[redalert] advisor t={int(advice.made_at)}s ({latency_s:.1f}s) {advice.posture}: "
                f"{advice.plan or '(no plan)'} | why: {advice.reasoning or '-'}"
            )

    def _poll_advisor(self, view):
        """The advisor's fresh advice, or ``None``. Never blocks, never raises."""
        if self._advisor is None:
            return None
        try:
            raid_state = self._raid.situation(view)[0]
            return self._advisor.poll(view, {
                "mode": str(self._mode),
                "mode_for_seconds": round(self._game_seconds - self._mode_since) if self._game_seconds else 0,
                "confirming_attack": self._commander.confirming,
                "massing_point": list(scout.choke_cell(view) or []) or None,
                "our_unit_value": round(brain.our_unit_value(view)),
                "enemy_unit_value_known": round(brain.enemy_unit_value(view)),
                "enemy_turrets_known": len(executor.enemy_defenses(view)),
                "raid_squad": {"ids": list(self._raid.unit_ids), "situation": raid_state,
                               "drawing_their_army": self._raid.distracting(view)},
                "scouts": dict(self._patrol.roles),
                "intel": {"enemy_opening": self._intel.opening(),
                          "threat_alarm": self._patrol.alarm(view) or self._intel.rushing(view),
                          "their_base_seen_at": self._patrol.base_seen_at},
            })
        except Exception as exc:  # noqa: BLE001 - the advisor must never stall the game
            self._log(f"[redalert] advisor poll failed, rules only: {type(exc).__name__}: {exc}")
            return None

    # The commander's state, under the names the rest of the agent (and older
    # tests) use.
    _mode = property(lambda self: self._commander.mode,
                     lambda self, v: setattr(self._commander, "mode", v))
    _mode_since = property(lambda self: self._commander.mode_since,
                           lambda self, v: setattr(self._commander, "mode_since", v))
    _confirm_at = property(lambda self: self._commander.confirm_at,
                           lambda self, v: setattr(self._commander, "confirm_at", v))
    _cheese_since = property(lambda self: self._commander.cheese_since,
                             lambda self, v: setattr(self._commander, "cheese_since", v))
    _retreated_at = property(lambda self: self._commander.retreated_at,
                             lambda self, v: setattr(self._commander, "retreated_at", v))
    _retreat_army = property(lambda self: self._commander.retreat_army,
                             lambda self, v: setattr(self._commander, "retreat_army", v))
    _attacks = property(lambda self: self._commander.attacks)

    def _break_stall(self, view) -> None:
        """An attack that kills nothing for ``ATTACK_STALL_SECONDS`` with no
        enemy in sight is parked on a stale target: forget the remembered
        buildings around the army and let the hunter lead for a while. (Match
        c2fec761: from 10:00 a 50-unit army sat at the razed enemy main, held
        there by the advisor's target and buildings remembered 5-10 cells off,
        while the enemy's last building stood 15 cells away.)"""
        kills, since = self._kills_seen
        if view.kills_cost != kills:
            self._kills_seen = (view.kills_cost, view.game_seconds)
            return
        if self._mode is not Mode.ALL_IN or view.enemy_units or view.enemy_buildings:
            self._kills_seen = (kills, view.game_seconds) if self._mode is not Mode.ALL_IN else self._kills_seen
            return
        if view.game_seconds - since < C.ATTACK_STALL_SECONDS:
            return
        army = [u.cell for u in view.combat_units if u.cell is not None]
        if army:
            centre = (sum(c[0] for c in army) / len(army), sum(c[1] for c in army) / len(army))
            self._memory.forget_near(centre, C.STALL_FORGET_CELLS)
        self._ignore_advice_until = view.game_seconds + C.STALL_IGNORE_ADVICE_SECONDS
        self._kills_seen = (kills, view.game_seconds)

    def _army_advice(self, advice, hunting: bool, view):
        """The advice the army follows: without its army target while we
        finish a razed enemy (the rules walk onto each remembered building,
        which clears stale ones) or after a stall."""
        if advice is None or getattr(advice, "army_target", None) is None:
            return advice
        if hunting or view.game_seconds < self._ignore_advice_until:
            import dataclasses
            return dataclasses.replace(advice, army_target=None)
        return advice

    def _models_switch(self, context) -> None:
        """A/B testing in self-play: ``REDALERT_MODELS_OFF_SEAT=<seat>`` plays
        that seat rules-only (no advisor, no squad leads) while the other
        keeps them. Test matches only (never a tournament game); checked once
        per match."""
        if self._models_checked:
            return
        self._models_checked = True
        seat = os.environ.get("REDALERT_MODELS_OFF_SEAT", "").strip()
        if (not seat.isdigit() or getattr(context, "tournament_id", None) is not None
                or getattr(context, "seat_position", None) != int(seat)):
            return
        if self._advisor is not None:
            self._advisor.close()
            self._advisor = None
        for lead in self._leads.values():
            lead.close()
        self._leads = {}
        self._log(f"[redalert] seat {seat}: models off for this test match (A/B: rules only)")

    def _note_placements(self, view, sent: list[dict]) -> None:
        """Remember where each building was asked for (placement audit)."""
        for o in sent:
            if o.get("cmd") != "place":
                continue
            item = str(o.get("item") or "").lower()
            p = self._placing.get(item)
            if p is None or view.game_seconds - p["t"] > 15.0:
                known = {b.id for b in view.buildings if b.type == item}
                self._placing[item] = {"asked": o.get("at"), "t": view.game_seconds, "known": known}
            else:
                p["asked"] = o.get("at")

    def _audit_placements(self, view) -> None:
        """Log where each placed building came up against where it was asked
        for: the placement rule the platform doesn't document (refineries at
        expansions keep landing at home while power plants beside them don't)."""
        for item, p in list(self._placing.items()):
            new = [b for b in view.buildings if b.type == item and b.id not in p["known"] and b.cell is not None]
            if new:
                landed = new[0].cell
                self._matchlog.placement({"t": round(view.game_seconds), "item": item,
                                          "asked": p["asked"], "landed": list(landed)})
                if p["asked"] and tuple(p["asked"]) != tuple(landed):
                    # Moved although legal by our rule (match 89356b51: the same
                    # expansion refinery spot three times): ask elsewhere for a while.
                    for c in buildplan._cells(item, tuple(p["asked"])):
                        self._failed_cells[c] = view.game_seconds
                del self._placing[item]
            elif view.game_seconds - p["t"] > 15.0:
                del self._placing[item]

    def _production_pending(self, order: dict, now: float) -> bool:
        """A build/train of an item we sent moments ago: the engine applies
        orders 0.2-0.4 s later, so the view may not show it queued yet and a
        resend would queue it twice (skill/redalert §9, "Don't overlap")."""
        if order.get("cmd") not in _PRODUCTION_CMDS:
            return False
        sent_at = self._production_sent.get(_production_key(order))
        return sent_at is not None and now - sent_at < C.PENDING_PRODUCTION_SECONDS

    def _army_repeat(self, order: dict, now: float) -> bool:
        """An army order with the same command and destination, sent recently,
        to no new units: skip it so units aren't re-ordered 5x a second."""
        if order.get("cmd") not in _ARMY_CMDS:
            return False
        last = self._army_sent.get(_army_key(order))
        if last is None:
            return False
        sent_at, units = last
        return now - sent_at < C.ARMY_REISSUE_SECONDS and set(order.get("units") or ()) <= units

    def _reason(self, view) -> str:
        return f"{self._mode} | army {int(view.army_value)} vs {int(view.enemy_army_value)} | t={int(view.game_seconds)}s"


_POSTURE = {"attack": Mode.ALL_IN, "hold": Mode.BUILD}
_ARMY_CMDS = frozenset({"attack_move", "attack", "move"})
_PRODUCTION_CMDS = frozenset({"build", "train", "cancel", "place", "repair"})


def _production_key(order: dict) -> str:
    """The item (build/train/cancel/place), or the building id (repair)."""
    return str(order.get("item") or order.get("building")).lower()


def _army_key(order: dict) -> str:
    """Command + destination on a coarse grid + target: chasing an enemy that
    shifts a cell or two is the same order, not a fresh one every view."""
    to = order.get("to")
    coarse = [v // _ARMY_GRID for v in to] if isinstance(to, list) else to
    return json.dumps([order.get("cmd"), coarse, order.get("target")], default=str)


_ARMY_GRID = 4


def _order_key(order: dict) -> str:
    body = {k: v for k, v in order.items() if k != "queued"}
    if isinstance(body.get("units"), list):
        body["units"] = sorted(body["units"], key=str)
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
