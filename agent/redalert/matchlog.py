"""A per-match record of how a Red Alert game went, for reviewing strategy.

When ``REDALERT_LOG_DIR`` (or, failing that, ``VGC_LOG_DIR``) is set, each
match writes ``<stamp>-red-alert-<session>-seat<n>.jsonl`` (append-only, so a
crash still leaves everything up to that point) and, at the end, a short
``.md`` summary next to it:

- ``start``: factions, map, match length.
- ``sample`` every ``SAMPLE_SECONDS`` of game time: mode, cash, ore and
  storage, harvesters, power, army value vs the enemy value (seen or remembered), enemy buildings known, the
  tiebreak figures (kills_cost, deaths_cost, assets_value), unit and building
  counts, how much of the map is explored.
- ``mode``: every posture change, with the army figures that drove it.
- ``event``: our buildings lost (from the observation's events).
- ``refused``: orders the server refused, with the reason, from the immediate
  answer and from the engine's later ``recent_order_problems`` (first few per
  command+reason, then only counted).
- ``end``: the server's result and the final military figures.

Logging is best effort and off the decision path: any I/O problem disables
the log for the match and never reaches the game.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

SAMPLE_SECONDS = 15.0
REFUSALS_LOGGED_PER_REASON = 3
_EVENT_KINDS = {"building_lost"}


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


class MatchLog:
    def __init__(self, log, directory: str | None = None) -> None:
        self._log = log
        raw = directory if directory is not None else (
            os.environ.get("REDALERT_LOG_DIR") or os.environ.get("VGC_LOG_DIR")
        )
        self._directory = Path(raw).expanduser() if raw else None
        self._path: Path | None = None
        self._disabled = self._directory is None
        self._next_sample = 0.0
        self._last_mode: str | None = None
        self._last_event_seq = -1
        self._seen_problems: set = set()
        self._refusals: Counter = Counter()
        self._samples: list[dict] = []
        self._modes: list[dict] = []
        self._buildings_lost: list[dict] = []
        self._ended = False
        self._advice_count = 0
        self._squad_counts: dict[str, int] = {}
        self._latencies: list[float] = []
        self._plans: list[dict] = []

    @property
    def enabled(self) -> bool:
        return not self._disabled

    # -- recording -------------------------------------------------------------

    def observe(self, raw: dict, view, mode, context, extra: dict | None = None) -> None:
        """Called every view; writes only at sample times and on changes."""
        if self._disabled:
            return
        if self._path is None:
            self._open(raw, context, getattr(view, "map_legend", None), getattr(view, "map_picture", None))
            if self._disabled:
                return
        t = view.game_seconds
        mode_s = str(mode)
        if mode_s != self._last_mode:
            row = {"t": round(t), "from": self._last_mode, "to": mode_s,
                   "army": round(view.army_value), "enemy_seen": round(view.enemy_army_value)}
            self._modes.append(row)
            self._write("mode", row)
            self._last_mode = mode_s
        self._events(raw, t)
        self._problems(raw, t)
        if t >= self._next_sample:
            self._next_sample = t + SAMPLE_SECONDS
            row = {**self._sample(raw, view, mode_s), **(extra or {})}
            self._samples.append(row)
            self._write("sample", row)

    def placement(self, data: dict) -> None:
        """A building we placed: the asked top-left cell and where it stands."""
        self._write("placement", data)

    def refused(self, order: dict, reason: Any, detail: Any, t: float | None) -> None:
        if self._disabled or self._path is None:
            return
        key = (str(order.get("cmd")), str(reason or "rejected"))
        self._refusals[key] += 1
        if self._refusals[key] <= REFUSALS_LOGGED_PER_REASON:
            self._write("refused", {"t": round(t) if t is not None else None, "order": order,
                                    "reason": key[1], "detail": str(detail or "")[:200]})

    def advice(self, advice, latency_s: float) -> None:
        """Each piece of model advice as it arrives, with how long it took."""
        if self._disabled or self._path is None:
            return
        self._advice_count += 1
        self._latencies.append(latency_s)
        self._write("advice", {"latency_s": round(latency_s, 2), **advice.as_log()})
        # The summary keeps each change of plan (not every repeat).
        key = (advice.posture, advice.plan)
        if not self._plans or self._plans[-1]["key"] != key:
            self._plans.append({"key": key, "t": round(advice.made_at), "posture": advice.posture,
                                "plan": advice.plan, "reasoning": advice.reasoning})

    def squad(self, name: str, directive, latency_s: float) -> None:
        """Each Haiku squad-lead directive as it arrives."""
        if self._disabled or self._path is None:
            return
        self._squad_counts[name] = self._squad_counts.get(name, 0) + 1
        self._write("squad_lead", {"squad": name, "latency_s": round(latency_s, 2), **directive.__dict__})

    def end(self, final_state, context) -> None:
        """Write the result and the markdown summary (once)."""
        if self._disabled or self._ended:
            return
        self._ended = True
        raw = _dict(getattr(final_state, "raw", None))
        if self._path is None:
            self._open(raw, context)
            if self._disabled:
                return
        result = getattr(final_state, "final_result", None) or raw.get("result")
        military = _dict(_dict(raw.get("observation")).get("military"))
        self._write("end", {"result": result, "military": military,
                            "refusals": {f"{c}:{r}": n for (c, r), n in self._refusals.items()}})
        try:
            self._path.with_suffix(".md").write_text(self._summary(result, military), encoding="utf-8")
        except OSError as exc:
            self._fail(exc)

    # -- internals -------------------------------------------------------------

    def _open(self, raw: dict, context, legend=None, picture=None) -> None:
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            session = str(getattr(context, "session_id", None) or "session")[:8]
            seat = getattr(context, "seat_position", None)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self._path = self._directory / f"{stamp}-red-alert-{session}-seat{seat}.jsonl"
        except OSError as exc:
            self._fail(exc)
            return
        obs = _dict(raw.get("observation"))
        self._write("start", {
            "session": getattr(context, "session_id", None),
            "seat": raw.get("your_seat"),
            "faction": obs.get("faction"),
            "enemy_faction": obs.get("enemy_faction"),
            "map": _dict(raw.get("map")).get("id"),
            "max_minutes": _dict(raw.get("time")).get("max_minutes"),
            "match_rules": raw.get("match_rules"),
            "map_legend": legend,     # how the map picture marks terrain (cliffs, water, ...)
            "map_picture": picture,   # {"origin", "rows"}: the terrain, for analysing placement later
        })

    def _sample(self, raw: dict, view, mode: str) -> dict:
        obs = _dict(raw.get("observation"))
        types = Counter(b.type for b in view.buildings if not b.wreck)
        return {
            "t": round(view.game_seconds),
            "mode": mode,
            "cash": round(view.cash),
            "ore": round(view.ore),
            "storage": round(view.resource_capacity),
            "harvesters": sum(1 for u in view.units if u.type == "harv" and not u.wreck),
            "power": [round(view.power_provided), round(view.power_drained)],
            "army": round(view.army_value),
            "enemy_seen": round(view.enemy_army_value),
            "enemy_buildings_known": len(view.known_enemy_buildings),
            "combat_units": len(view.combat_units),
            "kills_cost": round(view.kills_cost),
            "deaths_cost": round(view.deaths_cost),
            "assets": round(view.assets_value),
            "buildings": dict(types),
            # Where each building actually stands: ``place at`` is only a hint.
            "placed": [[b.type, *b.cell] for b in view.buildings if b.cell is not None and not b.wreck][:30],
            "queued": [str(p.get("item")) for p in view.production if not p.get("ready")],
            "explored_pct": obs.get("explored_percent"),
        }

    def _events(self, raw: dict, t: float) -> None:
        for event in _dict(raw.get("observation")).get("events") or []:
            if not isinstance(event, dict):
                continue
            seq = event.get("seq")
            if not isinstance(seq, int) or seq <= self._last_event_seq:
                continue
            self._last_event_seq = seq
            if event.get("kind") in _EVENT_KINDS:
                row = {"t": round(t), **{k: event.get(k) for k in ("kind", "type", "id")}}
                self._buildings_lost.append(row)
                self._write("event", row)

    def _problems(self, raw: dict, t: float) -> None:
        """The engine's later verdicts (``recent_order_problems``): rejected,
        partial and lost orders that the immediate answer reported as pending."""
        for row in raw.get("recent_order_problems") or []:
            if not isinstance(row, dict):
                continue
            key = (row.get("batch"), row.get("index"))
            if key in self._seen_problems:
                continue
            self._seen_problems.add(key)
            reason = f"{row.get('status')}:{row.get('reason')}"
            self.refused({"cmd": row.get("cmd")}, reason, row.get("detail"), t)

    def _write(self, kind: str, data: dict) -> None:
        if self._disabled or self._path is None:
            return
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"kind": kind, **data}, default=str) + "\n")
        except OSError as exc:
            self._fail(exc)

    def _fail(self, exc: Exception) -> None:
        self._disabled = True
        self._log(f"[redalert] match log disabled: {type(exc).__name__}: {exc}")

    def _summary(self, result: Any, military: dict) -> str:
        r = _dict(result)
        lines = [
            f"# Red Alert match {self._path.stem}",
            "",
            f"- Outcome: your_return={r.get('your_return')} reason={r.get('termination_reason') or r.get('reason')}",
            f"- Final: kills_cost={military.get('kills_cost')} deaths_cost={military.get('deaths_cost')} "
            f"assets_value={military.get('assets_value')} army_value={military.get('army_value')}",
            "",
            "## Timeline",
            "",
            "| t (s) | mode | cash | harv | army | enemy seen | kills | deaths | assets | explored |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for s in self._samples:
            lines.append(
                f"| {s['t']} | {s['mode']} | {s['cash']} | {s['harvesters']} | {s['army']} | "
                f"{s['enemy_seen']} | {s['kills_cost']} | {s['deaths_cost']} | {s['assets']} | {s['explored_pct']} |"
            )
        lines += ["", "## Mode changes", ""]
        lines += [f"- {m['t']}s: {m['from']} -> {m['to']} (army {m['army']} vs seen {m['enemy_seen']})" for m in self._modes]
        lines += ["", "## Buildings lost", ""]
        lines += [f"- {b['t']}s: {b.get('type')}" for b in self._buildings_lost] or ["- none"]
        lines += ["", "## Model plan", ""]
        if self._advice_count:
            avg = sum(self._latencies) / len(self._latencies)
            lines.append(f"{self._advice_count} answers, {avg:.1f} s average; each change of plan below "
                         "(every answer is in the JSONL, kind \"advice\").")
            lines.append("")
            lines += [f"- **{p['t']}s, {p['posture']}**: {p['plan'] or '(no plan)'} *Why:* {p['reasoning'] or '-'}"
                      for p in self._plans]
        else:
            lines.append("- no advisor this match (rules only)")
        lines += ["", "## Squad leads", ""]
        lines += [f"- {name}: {n} directives (JSONL kind \"squad_lead\")" for name, n in self._squad_counts.items()] or ["- none"]
        lines += ["", "## Refused orders", ""]
        lines += [f"- {c} ({r}): {n}" for (c, r), n in self._refusals.most_common()] or ["- none"]
        return "\n".join(lines) + "\n"
