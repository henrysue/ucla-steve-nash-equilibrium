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
from . import sparring  # TEST ONLY (self-play sparring archetypes; never in a tournament)
from . import intel, opponents, postgame, scouting
from .commander import Commander
from .expand import Expansion
from .expand import enemy_expansions as _enemy_expansions
from .guard import Garrison
from .raid import RaidCorps
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
        self._raw_log = log
        self._seat: int | None = None   # our seat (from your_seat "Multi1"): every console line says which seat it is
        self._log = self._tagged
        self._clock = clock
        self._commander = Commander()   # the army's one decision each view (commander.py)
        self._failed_cells: dict = {}   # cell -> game time the engine moved a building off it
        self._enemy_seen: list = []     # (game s, enemy army seen, our kills then): brain.enemy_estimate
        # The batch play_action answers next; refused order key -> blocked
        # until (clock); army (cmd, destination) -> (last sent, unit ids).
        self._last_sent: list[dict] = []
        self._sent_batches: list[list[dict]] = []    # the last few batches sent (lost orders: last_orders)
        self._lost_batch = None                       # the last_orders batch already read for lost orders
        self._outbox: list[tuple[float, dict]] = []   # orders that didn't fit the last batch
        self._damage = DamageScan()                    # under_attack / unit_lost: where we are hit
        self._blocked: dict[str, float] = {}
        self._army_sent: dict[str, tuple[float, frozenset]] = {}
        self._unit_dest: dict[int, tuple[float, str, tuple]] = {}   # unit -> (time, cmd, destination) last sent
        # build/train item -> when we last sent it (until production shows it).
        self._production_sent: dict[str, float] = {}
        self._patrol = scouting.ScoutCorps()
        self._intel = intel.EnemyIntel()
        self._memory = EnemyMemory()
        self._raid = RaidCorps()
        self._expansion = Expansion()
        self._garrison = Garrison()
        self._unstick = Unstick()
        self._hunter = scout.Hunter()
        self._placing: dict = {}   # item -> {"asked", "t", "known"}: where we asked to place it
        self._sold_at = float("-inf")
        self._kills_seen = (0.0, 0.0)               # (kills_cost, game time it last changed)
        self._ignore_advice_until = float("-inf")   # advisor's army target ignored until (stall breaker)
        self._sparring = sparring.Sparring()       # sparring: TEST ONLY
        self._models_checked = False                # REDALERT_MODELS_OFF_SEAT applied yet
        self._clearing: dict = {}   # building item -> game time its spot was first found occupied
        self._harvest = HarvestManager()
        # Per-match review log (off unless a log directory is configured).
        self._matchlog = MatchLog(self._log, directory=log_dir)
        self._mode_changes: list[dict] = []           # our army's recent mode changes (the advisor's memory)
        self._attack_open: dict | None = None          # the attack under way: when, who called it, trades then
        self._attack_results: list[dict] = []          # how our recent attacks went (the advisor's memory)
        self._opponents: tuple = ()                    # their display names (DecisionContext.opponents)
        self._match_info: dict = {}                    # map, their faction (the opponent profile)
        self._trades = (0.0, 0.0)                      # kills, deaths at the last view
        self._game_seconds: float | None = None
        # The current attack's waypoints: round the chokes, through a flank point (_pick_route).
        self._flank_via: list = []
        # The model advisor (None: rules only). Built after the match log so
        # each piece of advice is logged as it arrives.
        self._advisor = advisor_factory(self._log, on_advice=self._on_advice) if advisor_factory else None
        self._last_plan: tuple[str, str] | None = None
        # Haiku squad leads for the raid squad and the garrison ({} = rules only).
        self._leads = squad_lead_factory(self._log, on_directive=self._matchlog.squad) if squad_lead_factory else {}

    # -- the hot path ---------------------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        raw = getattr(state, "raw", None)
        if not isinstance(raw, dict) or not isinstance(raw.get("observation"), dict):
            return WAIT  # nothing to look at yet

        if self._seat is None:
            self._seat = _seat_number(raw.get("your_seat"), getattr(context, "seat_position", None))
            self._start_profile(raw, context)
        view = read(raw, getattr(context, "game_config", None))
        if not view.ok:
            return WAIT
        self._forget_lost_orders(raw)
        attacked = self._damage.update(view)

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
        seen_at = self._patrol.base_seen_at
        view = dataclasses.replace(view, enemy_army_peak=self._enemy_peak(view),
                                   enemy_seen_recent=max((r[1] for r in self._enemy_seen), default=0.0),
                                   ore_running_dry=self._harvest.running_dry(view),
                                   enemy_base_scouted=(seen_at is not None
                                                       and view.game_seconds - seen_at <= C.SCOUT_FRESH_SECONDS))

        # Track the threat window (sets/expires defensively).
        self._commander.track_threat(view)

        self._intel.update(view)
        view = dataclasses.replace(view, enemy_turtling=self._intel.turtling(view))
        self._patrol.update(view, taken={*self._raid.unit_ids, *self._garrison.unit_ids})
        if self._patrol.alarm(view):
            self._intel.scouts_lost(view)
        threat = self._patrol.alarm(view) or self._intel.rushing(view)
        # What the scouts saw for the first time: the log and the console.
        self._log_sightings(view)
        for raid in self._raid.drain():                 # raids that ended last view: the post-game analysis
            self._matchlog.note("raid_result", raid)
        # sparring (TEST ONLY): against ourselves in a test match, one seat
        # plays a fixed archetype (rusher / turtler / macro), without the advisor.
        spar = self._sparring.resolve(context, raw.get("your_seat"))
        if spar is not None:
            advice = spar.advice(view)
        else:
            # From the first second: it reads what the scouts see and predicts
            # their posture while the opening runs (it used to wait for the
            # build order to stand, ~3 minutes, so the scouts' look at their
            # base went unread).
            advice = self._poll_advisor(view)
        hunting = self._hunter.hunting(view)
        self._break_stall(view)
        rushing = spar is not None and spar.skip_confirm
        cheese = self._commander.cheese_since
        forced = (lambda policy: spar.mode(view, cheese, policy)) if spar is not None else None
        # Part of their army held by our raid squads: an attack goes away from it.
        drawing = self._raid.drawing(view)
        avoid = self._commander.attack_avoid if self._mode is Mode.ALL_IN else (drawing[2] if drawing else None)
        view = dataclasses.replace(view, attack_avoid=avoid)
        # The army's one decision (commander.py).
        self._commander.decide(
            view, model_pref=_POSTURE.get(getattr(advice, "posture", None)), scout_id=self._patrol.unit_id,
            squad_ids=frozenset({*self._patrol.unit_ids, *self._raid.unit_ids, *self._garrison.unit_ids}),
            hunting=hunting, alarm=threat, distraction=drawing,
            escorting=self._expansion.escorting, rushing=rushing, forced=forced,
        )
        if self._mode is Mode.ALL_IN and view.attack_avoid != self._commander.attack_avoid:
            view = dataclasses.replace(view, attack_avoid=self._commander.attack_avoid)
        # A blind attack call: a scout looks where their army masses first.
        self._patrol.look = scout.enemy_staging(view) if self._commander.wants_look else None
        self._trades = (view.kills_cost, view.deaths_cost)
        if self._commander.changed:
            self._mode_changes = (self._mode_changes + [
                {"t": round(view.game_seconds), "to": str(self._mode)}])[-C.ADVISOR_MEMORY_READS - 2:]
            self._track_attack(view, advice)
            self._flank_via = self._pick_route(view, advice) if self._mode is Mode.ALL_IN else []
        self._update_flank(view)
        if spar is None or spar.expands:   # sparring: the rusher never expands
            # Expand again once the worked fields are forecast to run dry (not
            # only once harvesters idle on an empty one).
            self._expansion.update(view, self._mode, advice, ore_running_out=view.ore_running_dry)
        if self._expansion.scout_target is not None and self._patrol.visit is None:
            self._patrol.visit = self._expansion.scout_target
        self._game_seconds = view.game_seconds

        if spar is None or spar.raids:     # sparring: the archetypes don't raid
            self._raid.update(view, self._mode, self._patrol.unit_id, advice,
                              taken={*self._patrol.unit_ids, *self._garrison.unit_ids})
        self._garrison.update(view, self._mode, self._expansion,
                              {*self._patrol.unit_ids, *self._raid.unit_ids})
        self._matchlog.observe(raw, view, self._mode, context, extra={
            "scout": self._patrol.unit_id, "scouts": dict(self._patrol.roles),
            "enemy_opening": self._intel.opening(), "threat": threat,
            "confirming": self._confirm_at is not None,
            "raid_squad": len(self._raid.unit_ids),
            "positions": self._positions(view),

            # Confirms the live game_config carries the map picture.
            "map_ore_cells": len(view.ore_cells), "enemy_ore_field": enemy_ore_field(view),
            "raid_target": self._raid.target(view),
            "raid_squads": self._raid.active, "raid_targets": self._raid.targets(view),
            "attacked_spots": [list(c) for c in attacked[:6]], "outbox": len(self._outbox),
            "flank_via": self._flank_via,
            "expansion": self._expansion.state, "expansion_site": self._expansion.site,
            "expansions_done": self._expansion.sites, "garrison": len(self._garrison.unit_ids),
            "ore_runway": (round(self._harvest.runway_seconds) if self._harvest.runway_seconds is not None else None),
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
            "sparring": spar.name if spar is not None else None,   # sparring: TEST ONLY
            "models": "on" if (self._advisor is not None or self._leads) else "off",
            "active_base": self._active_base(view),
            "harvester_acts": dict(Counter(h.act for h in view.units_of(C.HARVESTER) if h.act)),
            "unstuck": self._unstick.unstuck,
            "enemy_expansions": _enemy_expansions(view),
            "advice": getattr(advice, "posture", None), "enemy_style": getattr(advice, "enemy_style", None),
            "balance": getattr(advice, "balance", None),
            "advisor_calls": getattr(self._advisor, "calls", 0),
        })
        squads = {*self._patrol.unit_ids, *self._raid.unit_ids, *self._garrison.unit_ids}
        micro_orders, micro_ids = micro.plan(view, self._mode, frozenset(squads), scout.choke_cell(view))
        # Their main base razed: as many search cells as fanned-out groups.
        k = max(C.HUNT_GROUPS, len(view.combat_units) // C.FAN_GROUP_UNITS) if hunting else C.HUNT_GROUPS
        hunt = self._hunter.targets(view, k) if self._mode is Mode.ALL_IN else []
        sell = buildplan.sell_for_income(view)
        if sell is not None and view.game_seconds - self._sold_at >= C.SELL_FOR_INCOME_SECONDS:
            self._sold_at = view.game_seconds
            self._damage.sold.add(sell.get("building"))   # it will vanish: not a loss to hit back at
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
            sparring=spar,   # sparring: TEST ONLY
            threat=threat,
            probe=self._probe_point(view),
            warning=self._intel.suspect(view),
            fan=hunting,
            attacked=attacked,
        ) + self._unstick.orders(view, set(self._garrison.unit_ids)) + executor.stance_orders(
            view, self._mode, set(self._garrison.unit_ids), getattr(advice, "army_stance", None))
        now = self._clock()
        batch_orders = [
            o for o in self._steady(planned, view, now)
            if self._blocked.get(_order_key(o), 0.0) <= now
            and not self._army_repeat(o, now) and not self._production_pending(o, now)
        ]

        # What didn't fit the last batch goes first now (unless a newer order
        # covers the same units or is the same order): the tail of the plan —
        # the army's orders — no longer falls off every batch.
        keys = {_order_key(o) for o in batch_orders}
        busy = {u for o in batch_orders for u in (o.get("units") or ())}
        carried = [o for at, o in self._outbox if now - at <= C.OUTBOX_MAX_AGE_SECONDS
                   and _order_key(o) not in keys and not set(o.get("units") or ()) & busy]
        take, rest = orders.split(carried + batch_orders)
        self._outbox = [(now, o) for o in rest]
        if not take:
            return WAIT  # nothing to do right now
        move = {"type": "orders", "orders": take}

        self._last_sent = move["orders"]
        self._sent_batches = (self._sent_batches + [self._last_sent])[-4:]
        self._note_placements(view, self._last_sent)
        for order in self._last_sent:
            if order.get("cmd") in _ARMY_CMDS:
                self._army_sent[_army_key(order)] = (now, frozenset(order.get("units") or ()))
                if order.get("cmd") in ("attack_move", "move") and not order.get("queued") \
                        and isinstance(order.get("to"), list):
                    for uid in order.get("units") or ():
                        self._unit_dest[uid] = (now, order["cmd"], tuple(order["to"]))
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
        """Record how the match ended (the runner calls this once), and what we
        learned about the opponent for next time (opponents.py)."""
        self._matchlog.end(final_state, context)
        if self._opponents:
            try:
                record = self._profile_record(final_state)
                record["analysis"] = self._analyze_opponent()
                opponents.save(self._opponents, record)
            except Exception as exc:  # noqa: BLE001 - the profile must never break the end of a match
                self._log(f"[redalert] opponent profile not saved: {type(exc).__name__}: {exc}")
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

    def _forget_lost_orders(self, raw: dict) -> None:
        """An order the server reports ``lost`` never reached its units:
        forget that we sent it to them, so the plan sends it again next view
        (skill/redalert: "resend lost orders"; the 20 s no-repeat memory would
        have held it back). Only those units, once per batch: a ``rejected``
        order is on_action_result's (held back briefly), and wiping every
        unit's memory on each re-ordered the whole army — they are common in
        fights (focus fire on a target that just died)."""
        last = raw.get("last_orders")
        if not isinstance(last, dict) or last.get("pending"):
            return
        batch = last.get("batch")
        if batch is not None and batch == self._lost_batch:
            return
        self._lost_batch = batch
        rows = [r for r in last.get("results") or [] if isinstance(r, dict) and r.get("status") == "lost"
                and r.get("cmd") in ("move", "attack_move", "attack", "guard") and isinstance(r.get("index"), int)]
        if not rows:
            return
        # The batch these rows answer: the newest one sent whose orders match them.
        sent = next((b for b in reversed(self._sent_batches)
                     if all(0 <= r["index"] < len(b) and b[r["index"]].get("cmd") == r["cmd"] for r in rows)), None)
        if sent is None:
            return
        for r in rows:
            order = sent[r["index"]]
            for uid in order.get("units") or ():
                self._unit_dest.pop(uid, None)
                self._garrison._guarding.pop(uid, None)
            self._army_sent.pop(_army_key(order), None)

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

    def _pick_route(self, view, advice) -> list:
        """The waypoints of a new attack ([]: straight at the target): round
        the chokes on the road between the bases (terrain.choke_zone) when
        the way round is at most ``DETOUR_MAX_RATIO`` as long — never into a
        concave at a choke (82bdca3e lost 4600 in 50 s that way) — and
        through a flank point first when flanking (from 10:00, or decisively
        ahead; not while our raid holds their army): the advisor's side, else
        the side with the fewest known enemies, rotating on ties."""
        from . import terrain

        target = getattr(advice, "army_target", None) or executor._attack_target(view)
        army = [u.cell for u in view.combat_units if u.cell is not None and u.id not in self._patrol.unit_ids]
        if target is None or not (army or view.base_center):
            return []
        via = None
        if (not self._raid.distracting(view) and (view.game_seconds >= C.FLANK_AFTER_SECONDS
                                                  or brain._dominant(view) or brain.beaten(view))):
            route = getattr(advice, "attack_route", "auto")
            if route in ("left", "right"):
                via = executor.flank_waypoint(view, target, -1 if route == "left" else 1)
            elif route != "center":
                via = executor.choose_flank(view, target, self._attacks)[1]
        here = ((round(sum(c[0] for c in army) / len(army)), round(sum(c[1] for c in army) / len(army)))
                if army else tuple(view.base_center))
        avoid = terrain.choke_zone(view)
        points: list = []
        for goal in ([tuple(via)] if via else []) + [tuple(target)]:
            path = terrain.route_around(view, here, goal, avoid, C.DETOUR_MAX_RATIO)
            points += terrain.legs(path, C.ARMY_LEG_CELLS) if path else [goal]
            here = goal
        return points[:-1]                         # the target is the attack's own last leg

    def _update_flank(self, view) -> None:
        """Drop each waypoint of the attack's route as the army reaches it;
        all of them after ``FLANK_MAX_SECONDS`` (straight at the target)."""
        if not self._flank_via:
            return
        if view.game_seconds - self._mode_since > C.FLANK_MAX_SECONDS:
            self._flank_via = []
            return
        army = [u.cell for u in view.combat_units if u.cell is not None and u.id not in self._patrol.unit_ids]
        if army:
            cx = sum(c[0] for c in army) / len(army)
            cy = sum(c[1] for c in army) / len(army)
            nxt = self._flank_via[0]
            if (cx - nxt[0]) ** 2 + (cy - nxt[1]) ** 2 <= C.FLANK_ARRIVED_CELLS ** 2:
                self._flank_via = self._flank_via[1:]

    def _on_advice(self, advice, latency_s: float) -> None:
        """Each answer goes to the match log; the console gets a line only
        when the posture or plan changes, so it can be followed live without
        a line every few seconds. Local only: never in reasoning_summary."""
        self._matchlog.advice(advice, latency_s)
        key = (advice.posture, advice.enemy_style, advice.balance, advice.plan, advice.mistakes)
        if key != self._last_plan:
            self._last_plan = key
            mistakes = "; ".join(f"{m['kind']}: {m['note']}" for m in advice.mistakes)
            self._log(
                f"[redalert] advisor t={int(advice.made_at)}s ({latency_s:.1f}s) {advice.posture}, enemy {advice.enemy_style} "
                f"({advice.confidence:.0%} sure), "
                f"{advice.balance}: {advice.plan or '(no plan)'} | why: {advice.reasoning or '-'}"
                + (f" | mistakes: {mistakes}" if mistakes else "")
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
                "mode_changes": list(self._mode_changes),
                "attack_results": list(self._attack_results),
                "counters": {"kills": round(view.kills_cost), "deaths": round(view.deaths_cost),
                             "units_lost": self._damage.units_lost, "buildings_lost": self._damage.buildings_lost,
                             "scouts_lost": self._patrol.scouts_lost, "sightings": self._intel.sighted},
                "confirming_attack": self._commander.confirming,
                "massing_point": list(scout.choke_cell(view) or []) or None,
                "ore_runway_seconds": (round(self._harvest.runway_seconds)
                                       if self._harvest.runway_seconds is not None else None),
                "our_unit_value": round(brain.our_unit_value(view)),
                "enemy_unit_value_known": round(brain.enemy_unit_value(view)),
                "army_odds": round(min(brain.army_odds(view), 99.0), 2),   # by strength (1.0 = even)
                "enemy_turrets_known": len(executor.enemy_defenses(view)),
                "raid_squad": {"ids": list(self._raid.unit_ids), "situation": raid_state,
                               "drawing_their_army": self._raid.distracting(view)},
                "scouts": dict(self._patrol.roles),
                "scouting": {**self._intel.report(view),
                             "scouts_alive": dict(self._patrol.roles),
                             "scouts_lost": [{"t": round(t), "cell": list(c)} for t, c, _ in self._patrol._lost]},
                "intel": {"enemy_opening": self._intel.opening(),
                          "enemy_turtling": view.enemy_turtling,
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

    def _start_profile(self, raw: dict, context) -> None:
        """Our past games against this opponent go to the advisor's memory."""
        self._opponents = tuple(n for n in (getattr(context, "opponents", None) or ()) if isinstance(n, str))
        obs = raw.get("observation") if isinstance(raw.get("observation"), dict) else {}
        self._match_info = {"map": (raw.get("map") or {}).get("id") if isinstance(raw.get("map"), dict) else None,
                            "our_faction": obs.get("faction"), "their_faction": obs.get("enemy_faction")}
        if self._advisor is not None and self._opponents:
            history = opponents.load(self._opponents)
            self._advisor.opponent_history = [{k: v for k, v in r.items() if k != "analysis"} for r in history]
            self._advisor.addendum = postgame.addendum(history)
            if history:
                self._log(f"[redalert] {len(history)} past games against this opponent go to the advisor"
                          + (" (with our analysis of them)" if self._advisor.addendum else ""))

    def _analyze_opponent(self) -> dict | None:
        """Claude reads this match's log (postgame.py): their playstyle,
        opening, what worked for each side, how to beat them. None when
        logging or the analyst is off, or it fails."""
        path = self._matchlog.path
        provider = postgame.provider_from_env(self._log) if path is not None else None
        if provider is None:
            return None
        try:
            analysis = postgame.analyze(path, provider, opponents.load(self._opponents), postgame.timeout_from_env())
        except Exception as exc:  # noqa: BLE001 - the analysis is optional
            self._log(f"[redalert] post-game analysis failed: {type(exc).__name__}: {exc}")
            return None
        if analysis:
            self._log(f"[redalert] post-game analysis: {analysis['summary']} | to beat them: {analysis['how_to_beat']}")
        return analysis

    def _profile_record(self, final_state) -> dict:
        """One game against this opponent: what they did and how it went."""
        raw = getattr(final_state, "raw", None)
        raw = raw if isinstance(raw, dict) else {}
        result = getattr(final_state, "final_result", None) or raw.get("result")
        adv = self._advisor
        return {
            "date": time.strftime("%Y-%m-%d %H:%M"), **self._match_info,
            "result": result.get("your_return") if isinstance(result, dict) else None,
            "their_opening": self._intel.opening(),
            "their_first_push_at": self._intel.first_push_at,
            "first_seen": {k: v for k, v in self._intel.first_seen.items() if v <= 300},
            "our_first_read": adv.first_read if adv is not None else None,
            "our_final_read": adv.memory()["in_effect"].get("enemy_style") if adv is not None else None,
            "kills": round(self._trades[0]), "deaths": round(self._trades[1]),
            "attacks": list(self._attack_results),
        }

    def _track_attack(self, view, advice) -> None:
        """When an attack starts, who called it and the trades then; when it
        ends, how it went (the advisor's memory: attack_results)."""
        t = view.game_seconds
        if self._mode is Mode.ALL_IN:
            self._attack_open = {"t": round(t), "kills": view.kills_cost, "deaths": view.deaths_cost,
                                 "army": round(view.army_value),
                                 "called_by": "advisor" if getattr(advice, "posture", None) == "attack" else "rules"}
        elif self._attack_open is not None:
            a, self._attack_open = self._attack_open, None
            result = {"t": a["t"], "called_by": a["called_by"], "seconds": round(t - a["t"]), "army": a["army"],
                      "killed": round(view.kills_cost - a["kills"]), "lost": round(view.deaths_cost - a["deaths"]),
                      "ended": str(self._mode)}
            self._attack_results = (self._attack_results + [result])[-3:]
            self._matchlog.note("attack_result", result)

    def _log_sightings(self, view) -> None:
        """This view's first sightings into the match log, and one console
        line naming what is new: each building, and each unit type the first
        time it shows up."""
        fresh = self._intel.fresh
        if not fresh:
            return
        self._matchlog.scouted(view.game_seconds, fresh)
        new_types = {it["type"] for it in fresh if self._intel.first_seen.get(it["type"]) == it["t"]}
        shown = [it for it in fresh if it["kind"] == "building" or it["type"] in new_types]
        if shown:
            counts = Counter((it["kind"], it["type"]) for it in shown)
            where = shown[0]["cell"]
            what = ", ".join(f"{t}" + (f" x{n}" if n > 1 else "") for (_, t), n in counts.items())
            self._log(f"[redalert] scouts t={int(view.game_seconds)}s saw: {what} (near {where})")

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

    def _enemy_peak(self, view) -> float:
        """The most enemy army seen in the last ``ENEMY_PEAK_SECONDS``, less
        ``KILL_ARMY_SHARE`` of what we have killed since each sighting (the
        rest were buildings and trucks: f581756e killed 1800 while their army
        fell 900). No guess at what they built since: decisions read
        sightings and trades, and a scout looks before a blind attack."""
        t = view.game_seconds
        self._enemy_seen.append((t, view.enemy_army_value, view.kills_cost))
        self._enemy_seen = [r for r in self._enemy_seen if t - r[0] <= C.ENEMY_PEAK_SECONDS]
        return max(0.0, max(seen - C.KILL_ARMY_SHARE * (view.kills_cost - kills) for _, seen, kills in self._enemy_seen))

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
                    if item == C.REFINERY:
                        # An expansion's refinery came up elsewhere: the next
                        # one at once, at the nearest valid spot there.
                        self._expansion.refinery_misplaced(p["asked"])
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

    def _steady(self, planned: list[dict], view, now: float) -> list[dict]:
        """Drop units from a move or attack-move when they were sent within
        ``ORDER_JITTER_CELLS`` of the same spot in the last
        ``UNIT_REISSUE_SECONDS``: the attack points move with the army every
        view, and each small re-order made units turn and swap groups (match
        906dbb32: the big push split and went back and forth). A unit standing
        idle away from its spot (stuck) is sent again. A queued leg keeps only
        the units its first leg kept."""
        units = {u.id: u for u in view.units if u.id is not None}
        out, kept_last = [], None
        r2 = C.ORDER_JITTER_CELLS ** 2
        for o in planned:
            cmd, to = o.get("cmd"), o.get("to")
            if cmd not in ("attack_move", "move") or not isinstance(to, list):
                out.append(o)
                kept_last = None
                continue
            if o.get("queued"):
                ids = [i for i in o.get("units") or () if kept_last is None or i in kept_last]
            else:
                ids = []
                for i in o.get("units") or ():
                    last = self._unit_dest.get(i)
                    u = units.get(i)
                    stuck = u is not None and u.idle and u.cell is not None and dist2(u.cell, tuple(to)) > 9
                    if (last is not None and last[1] == cmd and now - last[0] < C.UNIT_REISSUE_SECONDS
                            and dist2(last[2], tuple(to)) <= r2 and not stuck):
                        continue
                    ids.append(i)
                kept_last = set(ids)
            if ids:
                out.append({**o, "units": ids})
        return out

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

    def _positions(self, view) -> dict:
        """Where things stand, for the match log: the army's centre, the
        massing point, each factory's rally point, our fighters (type, cell,
        role), harvesters and the enemy units in sight. Logs used to hold only
        totals, so "the army went behind the base" needed the replay."""
        roles = {i: "raid" for i in self._raid.unit_ids}
        roles.update({i: "scout" for i in self._patrol.unit_ids})
        roles.update({i: "outpost" for i in self._garrison.unit_ids})
        army = [u.cell for u in view.combat_units if u.cell is not None and u.id not in roles]
        centre = ([round(sum(c[0] for c in army) / len(army)), round(sum(c[1] for c in army) / len(army))]
                  if army else None)
        return {
            "army_centre": centre,
            "massing": list(scout.choke_cell(view) or []) or None,
            "rally": [[b.type, *b.rally] for b in view.buildings
                      if b.rally is not None and (b.type in C.BARRACKS or b.type == C.WAR_FACTORY)],
            "units": [[u.type, u.cell[0], u.cell[1], roles.get(u.id, "army")]
                      for u in view.combat_units if u.cell is not None][:80],
            "harvesters": [list(h.cell) for h in view.units_of(C.HARVESTER) if h.cell is not None][:16],
            "enemies": [[a.type, a.cell[0], a.cell[1]] for a in view.enemy_units
                        if a.cell is not None and not a.wreck][:40],
        }

    def _tagged(self, line: str) -> None:
        """Console lines say which seat they come from ("[redalert seat 1] ..."):
        both seats print to the same terminals in self-play. The platform
        gives no colour, only the seat."""
        if self._seat is not None and line.startswith("[redalert]"):
            line = f"[redalert seat {self._seat}]" + line[len("[redalert]"):]
        self._raw_log(line)

    def _reason(self, view) -> str:
        seat = f"seat {self._seat} | " if self._seat is not None else ""
        return f"{seat}{self._mode} | army {int(view.army_value)} vs {int(view.enemy_army_value)} | t={int(view.game_seconds)}s"


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


def _seat_number(your_seat, seat_position) -> int | None:
    """0 / 1 from the state's ``your_seat`` ("Multi0"), else the context's seat."""
    digits = "".join(ch for ch in str(your_seat or "") if ch.isdigit())
    if digits:
        return int(digits)
    return seat_position if isinstance(seat_position, int) else None


class DamageScan:
    """Where we are being hit: ``under_attack`` events (the attacked actor's
    cell), ``unit_lost`` / ``building_lost`` events, and anything of ours that
    vanished since the last view (events scroll out in a busy fight —
    skill/redalert: compare with the previous view). Spots from the last
    ``RETALIATE_SECONDS``."""

    def __init__(self) -> None:
        self._seq = -1
        self._last: dict[int, tuple[tuple[int, int], str]] = {}
        self.recent: list[tuple[float, tuple[int, int], str]] = []
        self.sold: set = set()                        # buildings we sold: they vanish, nobody hit them
        self.units_lost = 0                           # ours that vanished (the advisor's since_last_ask)
        self.buildings_lost = 0

    def update(self, view) -> list[tuple[int, int]]:
        t = view.game_seconds
        cur = {a.id: (a.cell, a.type) for a in list(view.units) + list(view.buildings)
               if a.id is not None and a.cell is not None and not a.wreck}
        for ev in view.events or []:
            seq = ev.get("seq")
            if not isinstance(seq, int) or seq <= self._seq:
                continue
            self._seq = seq
            kind = str(ev.get("kind") or "")
            if kind not in ("under_attack", "unit_lost", "building_lost"):
                continue
            aid = ev.get("id", ev.get("actor_id"))
            cell = (cur.get(aid) or self._last.get(aid) or (None, None))[0]
            if cell is None and isinstance(ev.get("x"), int) and isinstance(ev.get("y"), int):
                cell = (ev["x"], ev["y"])
            if cell is not None:
                self.recent.append((t, cell, kind))
        for aid, (cell, kind) in self._last.items():
            if aid not in cur and kind != C.MCV and aid not in self.sold:   # an MCV that deployed became a yard
                self.recent.append((t, cell, "vanished"))
                if kind in C.ALL_BUILDING_TYPES:
                    self.buildings_lost += 1
                else:
                    self.units_lost += 1
        self._last = cur
        self.recent = [r for r in self.recent if t - r[0] <= C.RETALIATE_SECONDS]
        return [c for _, c, _ in self.recent]
