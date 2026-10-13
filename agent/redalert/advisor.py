"""The model advisor: a professional player's eye on the game, off the hot path.

The rules play every view, about 5 times a second, and carry the strategy in
PLAN.md (decisions in commander.py / brain.py). Every
``ADVISOR_INTERVAL_SECONDS`` this module asks a model for *advice*: a few
choices the rules then carry out, within their limits.

Its read of the game:

- ``enemy_style``: what the enemy is playing — ``aggressive`` (we hold,
  turrets up, and win on macro), ``macro`` (a bigger raid party that flanks
  to their economy), ``standard`` (our plan) or ``unknown``;
- ``balance``: ``more_army`` / ``more_economy`` / ``balanced`` — where the
  money goes, inside the rules' own gates (never over-index on either);
- ``mistakes``: up to 3 problems it sees (idle units, a
  piling bank, stuck units), for our log and console, and to inform the next agent decision.

Its calls, each inside the game plan and its limits:

- ``posture``: ``attack`` (at even odds, never blind or into turrets we can't break), ``hold``, or ``default``;
- ``attack_route``: ``center`` / ``left`` / ``right`` / ``auto`` (from 10:00, or sooner when decisively ahead);
- ``expand``: ``now`` / ``wait`` / ``auto`` (``auto`` follows the balance);
- ``army_target``: the cell the main army attacks (used only with a known
  enemy near it, and not while finishing a razed enemy);
- ``raid_target_id`` / ``raid_target_cell``: the harvester or cell the raid
  squad goes for once nothing defends it;
- ``defense_at``: which side of the base the next turret faces;
- ``plan`` / ``reasoning``: its read and why, for our log and console.

Where the army masses, how it is formed and how it attacks are the rules'
calls. Every field may be null: the rules decide.

Real time rules everything here:

- The call runs on a background thread. :meth:`Advisor.poll` never waits: it
  starts a call when one is due and none is in flight, and returns the newest
  advice that is still fresh (``ADVICE_TTL_SECONDS`` of game time).
- Advice is checked before use: cells must be inside the map, a raid target
  must be a visible enemy harvester. The safety rails (defense, the early
  hold, retreat, finishing, protect-the-lead) come first and cannot be
  overridden.
- A failing or slow model only means the rules play alone. Errors back off
  (5 s, doubling to 60 s), and calls are capped per match.

Configuration: ``REDALERT_ADVISOR=off`` disables it; ``REDALERT_MODEL`` (else
``VGC_MODEL``) picks the model; ``REDALERT_ADVISOR_SECONDS`` sets the interval.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
import dataclasses
from dataclasses import dataclass
from typing import Any

from . import config as C
from .expand import enemy_expansion_sites as _enemy_expansion_sites
from .expand import enemy_expansions as _enemy_expansions
from .observe import View, enemy_ore_field, enemy_spawn_guess

SYSTEM_PROMPT = """\
You are a professional Command & Conquer: Red Alert (OpenRA) player in a 1v1 tournament match, \
coaching a fast rules engine that plays every moment. Make strong, generally good moves: the calls \
a top player makes on any map against any opponent. Play our game plan well; don't gamble, don't \
invent a new plan. Your best value is reading the opponent, keeping army and economy in balance, \
and catching mistakes.

THE MATCH: at most 20 minutes. Destroying every enemy building wins outright; otherwise the higher \
kills_cost - deaths_cost wins, then higher assets_value. Trade efficiently and finish a beaten enemy.

OUR GAME PLAN (the engine does all of this by itself):
- Opening: power, barracks, refinery, refinery, power, war factory (the pros' double refinery). Then, \
unless outgunned, a third refinery and the service depot; more refineries over time, expansion to \
the natural field, a second war factory after the depot, up to 3 barracks; about half the army's \
value in main battle tanks. Units spend what the economy doesn't need.
- Information: two scouts circle behind the enemy base early, then pickets watch the approach and \
one re-checks their base every 2.5 minutes. One refinery plus an early war factory or 2nd barracks, \
or an early army, reads as a rush (intel.enemy_opening).
- Every fight is judged by strength (unit types, hit points, damage per armor), not price: \
riflemen barely scratch tanks. enemy.army_value_estimate is what we saw of their units lately or, \
if more, what they could have built since (0.6 of what we spend on our army), less half our kills \
since; turrets not included.
- The army masses in a concave at the main choke and holds it; it never walks out to an expansion \
(an outpost, a turret and nearby units hold those). A push seen coming: it gets ready at the main \
base's turret line. In a fight at the main base it attacks if it wins clearly, else holds at the \
turret line. Idle units nearby answer small raids and any spot we are hit.
- It attacks at 1.3x (1.15x after 7:00, 1.05x with a 15000+ army; their turrets at the target count), \
or at once after clearly winning a fight (1500+ net, 1.5x). Knowing little of their army, a scout \
looks first. It retreats (breaks off to the turret line) when losing 1.3x what it kills, and does \
not flank before 10:00 unless decisively ahead. Raid squads go only for enemy expansions our \
scouts found, never into a turret's reach with infantry: they walk round it to a spot it doesn't \
cover, or wait for tanks or siege units to take along (enemy.expansions_known: turrets and \
raid_spot, None = out of reach for now); when they hold part of their army, the main army is judged \
against the rest and attacks away from the raid.

LESSONS FROM OUR GAMES (call on them):
- Holding is not free. Against a turtle or a macro player the side that waits gets out-built: \
c1c37a10 held 13 minutes at 1.4-1.9x odds (their real army was a third of ours at 9:14) while a \
turtle built 28800 behind pillboxes and then crushed us. A "hold" only delays the engine's attack \
by 90 s; use it for a real reason (a push coming, a look pending), not to "shore up the economy".
- Pillboxes and turrets eat infantry raids. c1c37a10 sent four raids into pillboxed expansions and \
lost 7300 for 1950. A pillbox rarely covers every side: aim raid_target_cell at the side of the \
expansion away from its turrets (harvesters on the far edge of the field, a refinery out of reach). \
If every side is covered, leave it until tanks or V2s can go, or for the main army.
- Turrets you haven't seen are still there: a turtle's are on the front of its base, where our \
scouts don't walk. Few units seen plus a quiet enemy is a turtle until a look at their front says \
otherwise — expect siege units to be needed, and don't call them "passive macro".
- The opponent's history is a hypothesis: c1c37a10's advisor kept "passive macro, thin army" from \
the last game for 13 minutes while this game's turtle out-built us.
- An army parked on their side of the middle is their concave, not a push. e2bc988e read a turtle \
as "aggressive" from it (and from history), held all game and put turrets up at home while the \
turtle out-expanded us 10 harvesters to 7. When situation.intel.enemy_passive is true they are not \
coming: say so, expand ("now"), keep "balanced"/"more_economy", and let the engine mass tanks \
(it goes to 70% tanks and three war factories against a passive enemy) for a decisive push.
- Raid only their remote expansions (a yard or refinery 25+ cells from their main base); the \
engine won't raid their main base's edge.

EARLY GAME (you are consulted from the first second): situation.scouting is what our scouts \
have seen — buildings_known, first_seen (enemy type -> game second), recent_sightings, the rules' \
own read and our scouts alive/lost — and base_report: what our scouts saw in their base in the \
first 4 minutes (riflemen and every unit type counted by unit, the buildings present by type), \
with the second scout walking round the back of their base into its centre 30 s after the first \
(reached_their_base_centre_at). Many riflemen and one refinery there is a rush coming. Predict their posture from it as soon as our scouts reach \
their base (~1:00-1:40), and say in plan when and from where you expect their first push. Tells:
  rush: one refinery by 1:40, a second barracks or an early war factory, 5+ riflemen in front of \
  their base before 2:00, our scouts killed on the way in;
  macro: two or three refineries, an early MCV or second yard, few units;
  turtle: pillboxes/turrets before 3:00, units kept home;
  standard: the double-refinery build into a war factory (~2:40).
An "aggressive" read puts up a warning turret on the road in (one before our war factory); keep \
"unknown" until the scouts have actually seen their base.

MEMORY (memory) — build on it, don't start over each time:
- first_read: your first read of this enemy, from the early scouting, with its evidence.
- read_changes: your last changes of read, oldest first — what changed, and how many answers \
(answers_between) and seconds the read before held; latest: your newest answer and how many answers \
it has held; in_effect: what the engine acts on now. Keep your read unless new evidence contradicts \
it; when you change it, put that evidence in evidence and reasoning. Flip-flopping (changes a few \
answers apart) marches the army out and back: change only on something new.
- since_last_ask: what happened since you were last asked — our kills and losses, units, buildings \
and scouts lost, new sightings, our army's mode changes. Read the change, not just the snapshot.
- opponent_history (and AGAINST THIS OPPONENT, after the plan): our past games against this \
opponent. A hypothesis, not a fact: players change. Use it to know what to look for, confirm it with \
this game's scouting, and drop it the moment a sighting contradicts it (tanks early against a \
"passive" read, no push when a rush was expected). Until confirmed it is worth confidence 0.5 at most.
- situation.attack_results: how our recent attacks went (who called them, seconds, killed, lost, how \
they ended) — calibrate your posture on them. situation.mode_changes: our army's recent moves.
Track your first-push prediction: if it came, say how it went; if it didn't, say what changed.

CONFIDENCE: confidence (0-1) is how sure you are of enemy_style and your posture; evidence lists \
what backs it (sightings with times, e.g. "2 barracks at 1:35", "12 riflemen at their exit 2:10"). \
An "aggressive" read puts up turrets only at 0.6+; at 0.8+ with evidence your change takes effect at \
once instead of waiting for a second answer.

YOUR READ:
- enemy_style: what the enemy is playing, from their buildings, army and moves:
  "aggressive": early or repeated pushes, 2+ barracks or an early war factory on one refinery, units \
  at our base — attacks on us, not their army standing on their side of the middle. Never while \
  situation.intel.enemy_passive is true (no attack on us for 4 minutes). The engine puts up to 3 turrets on the approach and wins on macro; once their pushes \
  are spent and their economy is behind ours, punish them (a rusher on one refinery can't rebuild).
  "macro": many refineries or harvesters, fast expansions, few units, no pressure. The engine sends \
  a bigger raid party round the flank to their harvesters and expansions.
  "turtle": many turrets (intel.enemy_turtling, enemy_turrets_known 4+), few units seen, no pushes. \
  Not "macro": raids die on the turrets. The engine builds the dome and siege units (they outrange \
  turrets), out-expands it, and attacks only with the siege or an overwhelming army.
  "standard": a normal build and army: the engine plays its plan. "unknown": not enough seen yet.
  Change it only on new evidence.
- balance: where our money should go, from our bank (cash + ore), harvesters, refineries, \
production, situation.ore_runway_seconds and army against enemy.army_value_estimate. "balanced" \
is the default and the usual answer. "more_army" only when we clearly trail (our army under \
about 0.7x their estimate) or are under attack (no extra refineries, dome or expansion for now). \
"more_economy" when the ore we mine runs out within 3 minutes (situation.ore_runway_seconds under \
180: the next expansion and its refinery come first), or we are safe and our income trails theirs. \
Never over-index: a big bank with a small army and a big army on 4 harvesters are both mistakes, \
and so is "more_army" all game.

YOUR CALLS (null or "default"/"auto" = the engine decides; each works inside the plan's limits):
- posture: your vote on the engine's decisions, counted only NEAR each bar (situation.decisions \
lists every bar, our number and your band): "attack" fires an action up to 15% below its bar \
(starting an attack, striking an expansion, fighting on instead of retreating), "hold" stops one up \
to 15% above it; beyond the band the numbers decide either way. So call it when our number sits in \
your band and you see what the numbers miss. "attack" never goes under 1.0x odds, never on a \
blind estimate (a scout looks first) and never into turrets the army can't break. Not on a stale \
estimate either: when situation.enemy_army_last_seen_seconds_ago is over 60, what we know of their \
army is old and they have rebuilt (a rusher trains riflemen nonstop) — hold until a look; "hold" keeps \
massing (ignored above its band, when we clearly outnumber them or are behind at the end) and \
only DELAYS an attack the engine calls: after 90 s of holding against its call the engine attacks anyway. Holding is not \
free: a turtle or macro player out-builds a waiting army. Use them for what the \
numbers miss (their army is out of position, a counter-push is coming). Keep a posture 30+ seconds: \
flip-flopping marches the army out and back.
- attack_route: "center", "left" or "right" for the next attack, from 10:00 (sooner only when \
decisively ahead: engaged early, a flank leaves our base open). Flank a defended middle; go straight \
when the raid squad is drawing their army away.
- expand: "now" to expand earlier, "wait" if the site or map is contested; "auto" follows your balance.
- army_target: [x, y] the main army attacks, used only if a known enemy building or unit is within \
10 cells of it. Good targets: power plants, then production, then refineries; an undefended expansion.
- raid_target_id / raid_target_cell: the raid squad's harvester (a visible harvester's id) or a \
cell to sweep, at an enemy expansion (18+ cells from their main base; their main field is never \
raided). It kills any defenders first.
- defense_at: [x, y] the side of our base the next turret should face (the enemy's approach).
- army_stance: the waiting (massing) army's stance — "defend" (fire at what comes in range, hold \
the line: the default), "attack_anything" (go after any enemy in sight: when they poke at the \
concave from outside its range), "return_fire" (only shoot back), "hold_fire"; "auto" = the engine. \
Attacking and defending armies always engage freely.
- guard: "on" (default): idle units of the massing army go at small groups that come within 12 cells \
of them, then back to their slots; "off" keeps the concave still (a bait you don't want chased).
- plan: your read of the game and plan for the next minute or two, in 1-2 short sentences.
- reasoning: the evidence for it, in 1-2 short sentences.

Use only cells inside map.bounds and ids you are shown. Be decisive and brief.\
"""

_CELL = {"type": ["array", "null"], "items": {"type": "integer"}}
STEADY_FIELDS = ("posture", "enemy_style", "balance", "expand", "attack_route")
STYLES = ("aggressive", "macro", "turtle", "standard", "unknown")
BALANCES = ("more_army", "more_economy", "balanced")
MISTAKES = ("idle_units", "exposed_harvesters", "bank_piling", "stuck_units", "losing_fight", "other")
ADVICE_SCHEMA = {
    "type": "object",
    "properties": {
        "enemy_style": {"type": "string", "enum": list(STYLES)},
        "balance": {"type": "string", "enum": list(BALANCES)},
        # No maxItems: Claude's structured outputs reject it (HTTP 400, every
        # call failed in match 82908a24); parse_advice keeps the first 3.
        "mistakes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["kind", "note"],
            "properties": {"kind": {"type": "string", "enum": list(MISTAKES)}, "note": {"type": "string"}}}},
        "posture": {"type": "string", "enum": ["attack", "hold", "default"]},
        "attack_route": {"type": "string", "enum": ["auto", "center", "left", "right"]},
        "expand": {"type": "string", "enum": ["auto", "now", "wait"]},
        "army_target": _CELL,
        "raid_target_id": {"type": ["integer", "null"]},
        "raid_target_cell": _CELL,
        "defense_at": _CELL,
        "army_stance": {"type": "string", "enum": ["auto", "defend", "attack_anything", "return_fire", "hold_fire"]},
        "guard": {"type": "string", "enum": ["on", "off"]},
        "confidence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "plan": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["enemy_style", "balance", "mistakes", "posture", "attack_route", "expand", "army_target", "raid_target_id", "raid_target_cell",
                 "defense_at", "army_stance", "guard", "confidence", "evidence", "plan", "reasoning"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Advice:
    made_at: float                       # game seconds of the view it was asked about
    enemy_style: str = "unknown"         # its read of the enemy: aggressive / macro / standard / unknown
    balance: str = "balanced"            # where the money goes: more_army / more_economy / balanced
    mistakes: tuple = ()                 # ({"kind", "note"}, ...): for the log and console
    posture: str = "default"
    attack_route: str = "auto"
    expand: str = "auto"
    army_target: tuple[int, int] | None = None
    raid_target_id: int | None = None
    raid_target_cell: tuple[int, int] | None = None
    defense_at: tuple[int, int] | None = None
    army_stance: str = "auto"            # the waiting army's stance (executor.stance_orders)
    guard: str = "on"                    # the massing army's guard response
    confidence: float = 0.5              # how sure it is of its read (0-1)
    evidence: tuple = ()                 # what it saw that backs the read (short notes)
    plan: str = ""
    reasoning: str = ""

    def as_log(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def parse_advice(answer: Any, view: View, made_at: float) -> Advice | None:
    """A validated :class:`Advice`, dropping any field that is malformed or
    off the map. ``None`` when the answer is not an object at all."""
    if not isinstance(answer, dict):
        return None

    def cell(value):
        if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in value)):
            return None
        if view.bounds is not None:
            x, y, w, h = view.bounds
            if not (x <= value[0] < x + w and y <= value[1] < y + h):
                return None
        return value[0], value[1]

    style = answer.get("enemy_style") if answer.get("enemy_style") in STYLES else "unknown"
    balance = answer.get("balance") if answer.get("balance") in BALANCES else "balanced"
    mistakes = tuple(
        {"kind": m["kind"], "note": " ".join(str(m.get("note") or "").split())[:160]}
        for m in (answer.get("mistakes") or [])[:3]
        if isinstance(m, dict) and m.get("kind") in MISTAKES)
    posture = answer.get("posture") if answer.get("posture") in ("attack", "hold", "default") else "default"
    expand = answer.get("expand") if answer.get("expand") in ("now", "wait") else "auto"
    if expand == "auto":
        expand = {"more_economy": "now", "more_army": "wait"}.get(balance, "auto")
    raid_id = answer.get("raid_target_id")
    return Advice(
        made_at=made_at,
        enemy_style=style,
        balance=balance,
        mistakes=mistakes,
        posture=posture,
        attack_route=answer.get("attack_route") if answer.get("attack_route") in ("center", "left", "right") else "auto",
        expand=expand,
        army_target=cell(answer.get("army_target")),
        raid_target_id=raid_id if isinstance(raid_id, int) and not isinstance(raid_id, bool) else None,
        raid_target_cell=cell(answer.get("raid_target_cell")),
        defense_at=cell(answer.get("defense_at")),
        army_stance=answer.get("army_stance") if answer.get("army_stance") in C.ARMY_STANCES else "auto",
        guard="off" if answer.get("guard") == "off" else "on",
        # No evidence from this game, no more than half sure: a read carried
        # over from past games alone must not raise turrets or act early
        # (9ff91807: "passive macro" from history held while they fielded tanks).
        confidence=(_confidence(answer.get("confidence")) if any(isinstance(e, str) and e.strip()
                                                                  for e in (answer.get("evidence") or []))
                    else min(_confidence(answer.get("confidence")), C.ADVISOR_NO_EVIDENCE_CONFIDENCE)),
        evidence=tuple(" ".join(e.split())[:120] for e in (answer.get("evidence") or [])[:4]
                       if isinstance(e, str) and e.strip()),
        plan=" ".join(str(answer.get("plan") or "").split())[:300],
        reasoning=" ".join(str(answer.get("reasoning") or "").split())[:300],
    )


def _confidence(value) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    return 0.5


def build_prompt(view: View, situation: dict) -> dict:
    """A compact, model-readable summary of the view (a few KB at most)."""

    def groups(actors, cap):
        by_type: dict[str, list] = {}
        for a in actors:
            if a.cell is not None and not a.wreck:
                by_type.setdefault(a.type, []).append(list(a.cell))
        return {t: {"count": len(cells), "cells": cells[:cap]} for t, cells in by_type.items()}

    return {
        "time": {"game_seconds": round(view.game_seconds), "remaining_seconds": view.remaining_seconds},
        "you": {
            "cash": round(view.cash), "ore": round(view.ore), "power": [round(view.power_provided), round(view.power_drained)],
            "army_value": round(view.army_value), "kills_cost": round(view.kills_cost),
            "bank": round(view.cash + view.ore),
            "harvesters": len(view.units_of(C.HARVESTER)),
            "in_production": [str(p.get("item")) for p in view.production if not p.get("ready")][:10],
            "deaths_cost": round(view.deaths_cost), "assets_value": round(view.assets_value),
            "buildings": [{"type": b.type, "cell": list(b.cell)} for b in view.buildings if b.cell is not None and not b.wreck][:30],
            "units": groups(view.units, 6),
            "base_center": list(view.base_center) if view.base_center else None,
            **situation,
        },
        "enemy": {
            "army_value_seen": round(view.enemy_army_value),
            "army_value_estimate": round(_estimate(view)),   # fog included: what the rules attack against
            "visible_units": [{"id": a.id, "type": a.type, "cell": list(a.cell)}
                              for a in view.enemy_units if a.cell is not None and not a.wreck][:20],
            "remembered_units": groups(view.known_enemy_units, 4),
            "known_buildings": [{"type": a.type, "cell": list(a.cell)}
                                for a in view.known_enemy_buildings if a.cell is not None][:25],
            "spawn_guess": list(enemy_spawn_guess(view) or []) or None,
            "ore_field_guess": list(enemy_ore_field(view) or []) or None,
            "expansions_known": _expansions_known(view),
            "expansion_sites_suspected": [list(c) for c in _enemy_expansion_sites(view)[:3]],
        },
        "map": {"bounds": list(view.bounds) if view.bounds else None,
                "spawns": [list(s) for s in view.spawns]},
    }


class Advisor:
    def __init__(
        self,
        provider,
        *,
        log: Callable[[str], None],
        clock: Callable[[], float] = time.monotonic,
        interval: float | None = None,
        on_advice: Callable[[Advice, float], None] | None = None,
    ) -> None:
        self._provider = provider
        self._log = log
        self._clock = clock
        self._interval = interval if interval is not None else C.ADVISOR_INTERVAL_SECONDS
        self._on_advice = on_advice
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ra-advisor")
        self._future: Future | None = None
        self._asked_at = 0.0
        self._asked_game_seconds = 0.0
        self._last_call = float("-inf")
        self._backoff_until = 0.0
        self._errors_in_row = 0
        self._last_raw: dict = {}
        self._steady_values: dict = {k: getattr(Advice(made_at=0.0), k) for k in STEADY_FIELDS}
        self._advice: Advice | None = None
        self.calls = 0
        # Its memory, sent with every call (memory()): the first read once the
        # scouts had seen their base, the changes of its read, what happened
        # since it was last asked, and our past games against this opponent.
        self.first_read: dict | None = None
        self._changes: list[dict] = []           # each change of its read (STEADY_FIELDS), oldest first
        self._latest: dict | None = None         # its newest answer, briefly
        self._answers = 0
        self._changed_at: tuple[int, float] | None = None   # (answer number, game s) of the last change
        self._last_ask: dict | None = None       # {"t", "counters"} when it was last asked
        self.opponent_history: list[dict] = []   # opponents.load: our past games against them
        self.addendum = ""                       # postgame.addendum: how to beat this opponent, after the plan

    def poll(self, view: View, situation: dict) -> Advice | None:
        """Collect a finished answer, start a call if one is due, and return
        the newest still-fresh advice. Never blocks."""
        now = self._clock()
        if self._future is not None and self._future.done():
            self._collect(view, now)
        if (self._future is None and self.calls < C.ADVISOR_MAX_CALLS
                and now - self._last_call >= self._interval and now >= self._backoff_until):
            self._start(view, situation, now)
        advice = self._advice
        if advice is not None and view.game_seconds - advice.made_at <= C.ADVICE_TTL_SECONDS:
            return advice
        return None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- internals -------------------------------------------------------------

    def _start(self, view: View, situation: dict, now: float) -> None:
        prompt = {**build_prompt(view, situation), "memory": {**self.memory(), "since_last_ask": self._since(view, situation)}}
        system = SYSTEM_PROMPT + ("\n\n" + self.addendum if self.addendum else "")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(prompt, separators=(",", ":"), default=str)},
        ]
        self._last_call = now
        self._asked_at = now
        self._asked_game_seconds = view.game_seconds
        self.calls += 1
        self._future = self._pool.submit(
            self._provider.complete_structured, messages, "red_alert_advice", ADVICE_SCHEMA,
            timeout=C.ADVISOR_TIMEOUT_SECONDS, effort="none",
        )

    def memory(self) -> dict:
        """What it said before, so its read stays consistent and changes on
        evidence:

        - ``first_read``: its first read of the enemy (once our scouts had
          seen them), with its evidence;
        - ``read_changes``: its last ``ADVISOR_MEMORY_READS`` changes of read
          (posture, style, balance, expansion, route) — what changed, and how
          many answers and seconds the read before held;
        - ``latest``: its newest answer and how many answers it has held;
        - ``in_effect``: the read the engine acts on (a change takes effect
          once two answers agree, or at once when it is sure and says why);
        - ``opponent_history``: our past games against this opponent."""
        return {
            "first_read": self.first_read,
            "read_changes": self._changes[-C.ADVISOR_MEMORY_READS:],
            "changes_so_far": len(self._changes),
            "answers_so_far": self._answers,
            "latest": self._latest,
            "in_effect": dict(self._steady_values),
            "opponent_history": self.opponent_history,
        }

    def _remember(self, advice: "Advice") -> None:
        read = {k: getattr(advice, k) for k in STEADY_FIELDS}
        t = round(advice.made_at)
        self._answers += 1
        before = self._latest["read"] if self._latest else None
        if read != before:
            entry = {"t": t, "read": read, "plan": (advice.plan or "")[:200],
                     "confidence": advice.confidence, "evidence": list(advice.evidence)}
            if before is None:
                entry["changed"] = "first answer"
            else:
                entry["changed"] = {k: [before[k], v] for k, v in read.items() if before[k] != v}
                n, since = self._changed_at
                entry["answers_between"] = self._answers - n         # how long the read before held
                entry["seconds_between"] = t - round(since)
            self._changes = (self._changes + [entry])[-2 * C.ADVISOR_MEMORY_READS:]
            self._changed_at = (self._answers, advice.made_at)
        n, _ = self._changed_at
        self._latest = {"t": t, "read": read, "plan": (advice.plan or "")[:200], "confidence": advice.confidence,
                        "answers_unchanged": self._answers - n + 1}   # answers giving this read
        if self.first_read is None and advice.enemy_style != "unknown":
            self.first_read = {"t": t, "read": read, "plan": (advice.plan or "")[:240],
                               "reasoning": (advice.reasoning or "")[:240], "evidence": list(advice.evidence),
                               "confidence": advice.confidence}

    def _since(self, view: View, situation: dict) -> dict | None:
        """What happened since it was last asked: seconds, our kills and
        losses, units, buildings and scouts lost, the first sightings and our
        army's mode changes in between (``situation.counters`` and the
        scouting report). None the first time."""
        counters = situation.get("counters") or {}
        t = view.game_seconds
        last, self._last_ask = self._last_ask, {"t": t, "counters": dict(counters)}
        if last is None:
            return None
        sightings = (situation.get("scouting") or {}).get("recent_sightings") or []
        return {
            "seconds": round(t - last["t"]),
            **{k: round(v - last["counters"].get(k, 0)) for k, v in counters.items()
               if isinstance(v, (int, float))},
            "new_sightings": [s for s in sightings if s.get("t", 0) > last["t"]][:10],
            "mode_changes": [m for m in situation.get("mode_changes") or [] if m.get("t", 0) > last["t"]],
        }

    def _steady(self, advice: "Advice") -> "Advice":
        """Its posture, read, balance, expand and route take effect only when
        two answers in a row agree (about 12 s): one odd answer doesn't swing
        the plan — unless it is sure (``ADVISOR_FAST_CONFIDENCE``) and gives
        its evidence. Targets and notes pass straight through."""
        raw = {k: getattr(advice, k) for k in STEADY_FIELDS}
        sure = advice.confidence >= C.ADVISOR_FAST_CONFIDENCE and bool(advice.evidence)
        for k, v in raw.items():
            if self._last_raw.get(k) == v or sure:   # sure, and says why: at once
                self._steady_values[k] = v
        self._last_raw = raw
        return dataclasses.replace(advice, **self._steady_values)

    def _collect(self, view: View, now: float) -> None:
        future, self._future = self._future, None
        try:
            answer = future.result()
        except Exception as exc:  # noqa: BLE001 - a model problem must never stall the game
            self._errors_in_row += 1
            pause = min(60.0, 5.0 * 2 ** (self._errors_in_row - 1))
            self._backoff_until = now + pause
            self._log(f"[redalert] advisor call failed ({type(exc).__name__}: {exc}); rules only for {pause:g} s")
            return
        self._errors_in_row = 0
        advice = parse_advice(answer, view, self._asked_game_seconds)
        if advice is None:
            return
        self._remember(advice)
        advice = self._steady(advice)
        self._advice = advice
        if self._on_advice is not None:
            self._on_advice(advice, now - self._asked_at)


def advisor_from_env(log: Callable[[str], None], on_advice=None) -> Advisor | None:
    """An :class:`Advisor` when a model is configured and not switched off,
    else ``None`` (the rules play alone)."""
    if os.environ.get("REDALERT_ADVISOR", "").strip().lower() in ("off", "0", "false", "no"):
        return None
    try:
        from agent.vgc.providers import DEFAULT_ANTHROPIC_MODEL, AnthropicProvider, anthropic_credentials, provider_from_env
    except Exception as exc:  # noqa: BLE001
        log(f"[redalert] advisor unavailable ({type(exc).__name__}); rules only")
        return None
    model = os.environ.get("REDALERT_MODEL") or os.environ.get("VGC_MODEL") or DEFAULT_ANTHROPIC_MODEL
    provider = None
    try:
        provider = AnthropicProvider(model=model) if anthropic_credentials() else provider_from_env(log)
    except Exception as exc:  # noqa: BLE001
        log(f"[redalert] advisor model unusable ({exc}); rules only")
    if provider is None:
        return None
    interval = None
    raw = os.environ.get("REDALERT_ADVISOR_SECONDS")
    if raw:
        try:
            interval = max(2.0, float(raw))
        except ValueError:
            pass
    log(f"[redalert] advisor on: {provider!r}, every {interval or C.ADVISOR_INTERVAL_SECONDS:g} s")
    return Advisor(provider, log=log, interval=interval, on_advice=on_advice)


def _expansions_known(view) -> list[dict]:
    """Each known enemy expansion: its cell, the turrets known by it, and
    where our raid would go now (raid.raid_spot: round the turrets for
    infantry, or with tanks) — None: out of reach until tanks or siege."""
    from .raid import raid_spot, would_form

    out = []
    for c in _enemy_expansions(view):
        r2 = (C.EXPAND_CLUSTER_CELLS + C.RAID_TURRET_CLEAR_CELLS) ** 2
        turrets = [a.type for a in list(view.enemy_buildings) + list(view.known_enemy_buildings)
                   if a.type in C.TURRETS and a.cell is not None and not a.wreck
                   and (a.cell[0] - c[0]) ** 2 + (a.cell[1] - c[1]) ** 2 <= r2]
        spot = raid_spot(view, would_form(view), view.base_center, c)
        out.append({"cell": list(c), "turrets": turrets, "raid_spot": list(spot) if spot else None})
    return out


def _estimate(view) -> float:
    from .brain import enemy_estimate

    return enemy_estimate(view)
