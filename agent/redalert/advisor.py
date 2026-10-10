"""The model advisor: a professional player's eye on the game, off the hot path.

The rules (executor, brain) play every view, about 5 times a second, and stay
simple. Every ``ADVISOR_INTERVAL_SECONDS`` this module asks a model for
*advice*: a few tactical choices the rules then carry out.

- ``posture``: ``attack`` (go now), ``hold`` (keep massing), or ``default``;
- ``army_target``: the cell the main army attacks;
- ``raid_target_id`` / ``raid_target_cell``: the harvester the raid squad
  focuses, or where it sweeps;
- ``stage_at`` / ``stage_facing`` / ``formation``: where the army waits
  while massing, which way it faces, and ``concave`` (an arc whose ends curve
  toward the enemy, so every unit can fire into the gap) or ``clump``;
- ``defense_at``: where the next base defense (pillbox, turret, Tesla) goes;
- ``plan`` / ``reasoning``: its overall plan and why it chose these directives
  now, for our own log and console (never sent to the server).

Every field may be null, meaning "let the rules decide".

Real time rules everything here:

- The call runs on a background thread. :meth:`Advisor.poll` never waits: it
  starts a call when one is due and none is in flight, and returns the newest
  advice that is still fresh (``ADVICE_TTL_SECONDS`` of game time).
- Advice is checked before use: cells must be inside the map, a raid target
  must be a visible enemy harvester. The safety rails (rush and base defense,
  retreat, protect-the-lead) are applied before any advice and cannot be
  overridden; the model can hold back an attack but never force one while we
  are behind (brain.py).
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
from dataclasses import dataclass
from typing import Any

from . import config as C
from .expand import enemy_expansion_sites as _enemy_expansion_sites
from .expand import enemy_expansions as _enemy_expansions
from .observe import View, enemy_ore_field, enemy_spawn_guess

SYSTEM_PROMPT = """\
You are a professional Command & Conquer: Red Alert (OpenRA) player in a 1v1 tournament match. \
Make the calls a smart, strategic top player would.

The game is real time. A fast rules engine plays every moment for you: build order, economy, \
production, repairs, rush and base defense, and retreating from losing fights. You are the \
commander above it. Every few seconds you see a summary and give a few directives; the engine \
carries them out until your next answer. Leave any field null to let the engine decide.

The match lasts at most 20 minutes. If no one is destroyed, the higher kills_cost - deaths_cost \
wins, then higher assets_value. Trade efficiently: don't feed units into defenses, and keep \
your army alive.

Your directives:
- posture: "attack" to send the main army now, "hold" to keep massing, "default" for the engine's call. Commit to a posture for at least 30 seconds (you are shown the current mode and how long it has lasted): flip-flopping marches the army out and back for nothing. Passivity loses a 20-minute game: when your army is clearly bigger than what you know of the enemy, attack.
- attack_route: "center", "left" or "right" for the next attack ("auto" lets the engine pick \
the side with fewer known enemies). Flank around defended or contested middles instead of \
always pushing straight through the center.
- expand: "now" to start the expansion (a second construction yard on a second ore field, \
normally around 4:00-4:30, needing 4+ harvesters, 2 refineries, a war factory and a real army), \
"wait" to hold it (e.g. the site or the map is contested), "auto" for the engine's timing.
- army_target: [x, y] the main army attack-moves to when attacking. Strong targets: power plants \
first (low power slows everything), then production buildings, then refineries.
- raid_target_id / raid_target_cell: the small raid squad hunts enemy harvesters. Give a visible \
harvester's id to focus it, or a cell to sweep (an ore field the enemy harvests, their refinery). \
A known enemy expansion (enemy.expansions_known) is a prime raid target: fresh, far from their \
army, and lightly defended.
- stage_at / stage_facing / formation: where the army waits while massing (a choke on the enemy's \
approach is best), the cell it faces (where the enemy will come from), and "concave" (an arc \
whose ends curve toward the enemy, so every unit can fire into the gap) or "clump".
- defense_at: [x, y] where the next base defense (pillbox, turret, flame turret, Tesla coil) goes. Put it on the perimeter of the base facing the enemy's approach, covering buildings and the path in, not in the middle of the base; spread defenses so they cover different ground.
- plan: your overall plan for the next minute or two, in 1-2 short sentences.
- reasoning: why these directives now, from what you see, in 1-2 short sentences.

Use only cells inside map.bounds and ids you are shown. Be decisive and brief.\
"""

_CELL = {"type": ["array", "null"], "items": {"type": "integer"}}
ADVICE_SCHEMA = {
    "type": "object",
    "properties": {
        "posture": {"type": "string", "enum": ["attack", "hold", "default"]},
        "attack_route": {"type": "string", "enum": ["auto", "center", "left", "right"]},
        "expand": {"type": "string", "enum": ["auto", "now", "wait"]},
        "army_target": _CELL,
        "raid_target_id": {"type": ["integer", "null"]},
        "raid_target_cell": _CELL,
        "stage_at": _CELL,
        "stage_facing": _CELL,
        "formation": {"type": "string", "enum": ["concave", "clump"]},
        "defense_at": _CELL,
        "plan": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["posture", "attack_route", "expand", "army_target", "raid_target_id", "raid_target_cell",
                 "stage_at", "stage_facing", "formation", "defense_at", "plan", "reasoning"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Advice:
    made_at: float                       # game seconds of the view it was asked about
    posture: str = "default"
    attack_route: str = "auto"
    expand: str = "auto"
    army_target: tuple[int, int] | None = None
    raid_target_id: int | None = None
    raid_target_cell: tuple[int, int] | None = None
    stage_at: tuple[int, int] | None = None
    stage_facing: tuple[int, int] | None = None
    formation: str = "clump"
    defense_at: tuple[int, int] | None = None
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

    posture = answer.get("posture") if answer.get("posture") in ("attack", "hold", "default") else "default"
    raid_id = answer.get("raid_target_id")
    return Advice(
        made_at=made_at,
        posture=posture,
        attack_route=answer.get("attack_route") if answer.get("attack_route") in ("center", "left", "right") else "auto",
        expand=answer.get("expand") if answer.get("expand") in ("now", "wait") else "auto",
        army_target=cell(answer.get("army_target")),
        raid_target_id=raid_id if isinstance(raid_id, int) and not isinstance(raid_id, bool) else None,
        raid_target_cell=cell(answer.get("raid_target_cell")),
        stage_at=cell(answer.get("stage_at")),
        stage_facing=cell(answer.get("stage_facing")),
        formation="concave" if answer.get("formation") == "concave" else "clump",
        defense_at=cell(answer.get("defense_at")),
        plan=" ".join(str(answer.get("plan") or "").split())[:300],
        reasoning=" ".join(str(answer.get("reasoning") or "").split())[:300],
    )


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
            "deaths_cost": round(view.deaths_cost), "assets_value": round(view.assets_value),
            "buildings": [{"type": b.type, "cell": list(b.cell)} for b in view.buildings if b.cell is not None and not b.wreck][:30],
            "units": groups(view.units, 6),
            "base_center": list(view.base_center) if view.base_center else None,
            **situation,
        },
        "enemy": {
            "army_value_estimate": round(view.enemy_army_value),
            "visible_units": [{"id": a.id, "type": a.type, "cell": list(a.cell)}
                              for a in view.enemy_units if a.cell is not None and not a.wreck][:20],
            "remembered_units": groups(view.known_enemy_units, 4),
            "known_buildings": [{"type": a.type, "cell": list(a.cell)}
                                for a in view.known_enemy_buildings if a.cell is not None][:25],
            "spawn_guess": list(enemy_spawn_guess(view) or []) or None,
            "ore_field_guess": list(enemy_ore_field(view) or []) or None,
            "expansions_known": [list(c) for c in _enemy_expansions(view)],
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
        self._advice: Advice | None = None
        self.calls = 0

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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(build_prompt(view, situation), separators=(",", ":"), default=str)},
        ]
        self._last_call = now
        self._asked_at = now
        self._asked_game_seconds = view.game_seconds
        self.calls += 1
        self._future = self._pool.submit(
            self._provider.complete_structured, messages, "red_alert_advice", ADVICE_SCHEMA,
            timeout=C.ADVISOR_TIMEOUT_SECONDS, effort="none",
        )

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
