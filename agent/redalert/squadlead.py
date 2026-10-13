"""Haiku "squad leads": a fast model directing one small squad each.

The big picture stays with the advisor (advisor.py) and the rules. A squad
lead looks after one small detachment with its own job — the harvester raid
squad, the expansion garrison — and every ``SQUAD_LEAD_INTERVAL_SECONDS``
picks what it does next: attack one enemy, attack-move to a cell, hold, or
pull back. A fast, cheap model (Haiku) is enough for a handful of units and
a narrow question; a few calls per squad, never one per unit.

Like the advisor, it never blocks a view: the call runs on a background
thread and the newest still-fresh answer (``SQUAD_LEAD_TTL_SECONDS``) is used.
Answers are checked (only enemy ids we can see, only cells on the map); a
missing, stale or failed answer means the squad's own rules play. Safety
rails win: squads are released to the army in defense and retreat modes by
their own logic (raid.py, guard.py).

Configuration: ``REDALERT_SQUAD_LEADS=off`` disables them;
``REDALERT_SQUAD_MODEL`` picks the model (default ``claude-haiku-5-5``).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from . import config as C
from . import orders
from .observe import View, dist2

DEFAULT_SQUAD_MODEL = "claude-haiku-5-5"

SYSTEM_PROMPT = """\
You are a professional Command & Conquer: Red Alert (OpenRA) player commanding one small squad \
in a real-time 1v1 match. Make the call a sharp, experienced squad leader would. You are shown \
the squad (its units, health and cells), its mission, and the enemies near it. Choose one action:

- "attack": focus fire on one enemy, by its id (only ids you are shown).
- "move": attack-move to a cell [x, y] (inside map.bounds); the squad fights what it meets.
- "hold": stay and fight where it is.
- "retreat": pull back toward home, fighting what it meets (when the squad would just die: outnumbered, badly hurt).

Every time you issue an order your units will recalculate their pathfinding. Issue an order and leave them to do it: keep the same action and target from one call to the next unless the situation has really changed.

Kill the high-value target the squad can actually win against; avoid defended spots; keep the \
squad alive — a dead squad scores nothing. One short line of reasoning. Be decisive.\
"""

SQUAD_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["attack", "move", "hold", "retreat"]},
        "target_id": {"type": ["integer", "null"]},
        "cell": {"type": ["array", "null"], "items": {"type": "integer"}},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "target_id", "cell", "reasoning"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Directive:
    made_at: float
    action: str
    target_id: int | None = None
    cell: tuple[int, int] | None = None
    reasoning: str = ""


def parse_directive(answer, view: View, made_at: float) -> Directive | None:
    if not isinstance(answer, dict) or answer.get("action") not in ("attack", "move", "hold", "retreat"):
        return None
    cell = answer.get("cell")
    ok_cell = None
    if isinstance(cell, list) and len(cell) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in cell):
        x, y = cell
        if view.bounds is None or (view.bounds[0] <= x < view.bounds[0] + view.bounds[2]
                                   and view.bounds[1] <= y < view.bounds[1] + view.bounds[3]):
            ok_cell = (x, y)
    tid = answer.get("target_id")
    return Directive(
        made_at=made_at,
        action=answer["action"],
        target_id=tid if isinstance(tid, int) and not isinstance(tid, bool) else None,
        cell=ok_cell,
        reasoning=" ".join(str(answer.get("reasoning") or "").split())[:200],
    )


def squad_orders(view: View, ids: list[int], d: Directive, home: tuple[int, int] | None) -> list[dict] | None:
    """Orders for the squad from a directive, or None when it is not usable
    now (the squad's own rules then play)."""
    if not ids:
        return None
    if d.action == "attack":
        hit = next((a for a in list(view.enemy_units) + list(view.enemy_buildings)
                    if d.target_id is not None and a.id == d.target_id and a.cell is not None), None)
        if hit is None:
            return None
        # By id only once near (an attack on one unit walks past the rest
        # without firing back); from afar, attack-move to it.
        cells = [u.cell for u in view.units if u.id in set(ids) and u.cell is not None]
        if cells:
            cx, cy = sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells)
            if (hit.cell[0] - cx) ** 2 + (hit.cell[1] - cy) ** 2 > C.ATTACK_ID_CELLS ** 2:
                return [orders.attack_move(ids, list(hit.cell))]
        return [orders.attack(ids, d.target_id)]
    if d.action == "move":
        return [orders.attack_move(ids, list(d.cell))] if d.cell is not None else None
    if d.action == "hold":
        return []
    if d.action == "retreat" and home is not None:
        return [orders.attack_move(ids, list(home))]   # never walk away under fire without shooting back
    return None


def build_prompt(view: View, ids: list[int], mission: str, extra: dict) -> dict:
    squad = [u for u in view.units if u.id in set(ids) and u.cell is not None]
    if squad:
        cx = sum(u.cell[0] for u in squad) / len(squad)
        cy = sum(u.cell[1] for u in squad) / len(squad)
    else:
        cx, cy = view.base_center or (0, 0)
    r2 = C.SQUAD_LEAD_SIGHT_CELLS ** 2
    near = [a for a in list(view.enemy_units) + list(view.enemy_buildings)
            if a.cell is not None and not a.wreck and dist2(a.cell, (cx, cy)) <= r2]
    return {
        "time": round(view.game_seconds),
        "mission": mission,
        "squad": [{"id": u.id, "type": u.type, "hp": round(u.hp), "cell": list(u.cell)} for u in squad],
        "enemies_near": [{"id": a.id, "type": a.type, "hp": round(a.hp), "cell": list(a.cell)} for a in near][:20],
        "map": {"bounds": list(view.bounds) if view.bounds else None},
        **extra,
    }


class SquadLead:
    def __init__(self, name: str, provider, *, log: Callable[[str], None], clock: Callable[[], float] = time.monotonic,
                 interval: float | None = None, on_directive: Callable | None = None) -> None:
        self.name = name
        self._provider = provider
        self._log = log
        self._clock = clock
        self._interval = interval if interval is not None else C.SQUAD_LEAD_INTERVAL_SECONDS
        self._on_directive = on_directive
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ra-{name}")
        self._future: Future | None = None
        self._asked_at = 0.0
        self._asked_game = 0.0
        self._last_call = float("-inf")
        self._backoff_until = 0.0
        self._errors = 0
        self._directive: Directive | None = None
        self.calls = 0

    def poll(self, view: View, ids: list[int], mission: str, extra: dict | None = None) -> Directive | None:
        """Collect a finished answer, start a call when due (only while the
        squad exists), return the freshest usable directive. Never blocks."""
        now = self._clock()
        if self._future is not None and self._future.done():
            self._collect(view, now)
        if not ids:
            self._directive = None
            return None
        if (self._future is None and self.calls < C.SQUAD_LEAD_MAX_CALLS
                and now - self._last_call >= self._interval and now >= self._backoff_until):
            self._start(view, ids, mission, extra or {}, now)
        d = self._directive
        if d is not None and view.game_seconds - d.made_at <= C.SQUAD_LEAD_TTL_SECONDS:
            return d
        return None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _start(self, view, ids, mission, extra, now) -> None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(build_prompt(view, ids, mission, extra), separators=(",", ":"), default=str)},
        ]
        self._last_call = self._asked_at = now
        self._asked_game = view.game_seconds
        self.calls += 1
        self._future = self._pool.submit(
            self._provider.complete_structured, messages, f"red_alert_{self.name}", SQUAD_SCHEMA,
            timeout=C.SQUAD_LEAD_TIMEOUT_SECONDS, effort="none",
        )

    def _collect(self, view, now) -> None:
        future, self._future = self._future, None
        try:
            answer = future.result()
        except Exception as exc:  # noqa: BLE001 - a model problem must never stall the game
            self._errors += 1
            pause = min(60.0, 5.0 * 2 ** (self._errors - 1))
            self._backoff_until = now + pause
            self._log(f"[redalert] {self.name} lead failed ({type(exc).__name__}); rules only for {pause:g} s")
            return
        self._errors = 0
        d = parse_directive(answer, view, self._asked_game)
        if d is not None:
            self._directive = d
            if self._on_directive is not None:
                self._on_directive(self.name, d, now - self._asked_at)


def squad_leads_from_env(log: Callable[[str], None], on_directive=None) -> dict[str, SquadLead]:
    """{"raid": ..., "garrison": ...} when enabled and a model is available, else {}."""
    # Off unless asked for (REDALERT_SQUAD_LEADS=on): match 5e840776, ~180 calls a
    # seat, 80% repeated the rules' own move 2 s late and the rest overrode the
    # raid's pull-back and the outpost's "back at any notice".
    if os.environ.get("REDALERT_SQUAD_LEADS", "").strip().lower() not in ("on", "1", "true", "yes"):
        return {}
    try:
        from agent.vgc.providers import AnthropicProvider, anthropic_credentials
    except Exception:  # noqa: BLE001
        return {}
    if not anthropic_credentials():
        return {}
    model = os.environ.get("REDALERT_SQUAD_MODEL") or DEFAULT_SQUAD_MODEL
    try:
        provider = AnthropicProvider(model=model)
    except Exception as exc:  # noqa: BLE001
        log(f"[redalert] squad leads unavailable ({exc}); rules only")
        return {}
    log(f"[redalert] squad leads on: {model}, every {C.SQUAD_LEAD_INTERVAL_SECONDS:g} s per squad")
    return {name: SquadLead(name, provider, log=log, on_directive=on_directive) for name in ("raid", "garrison")}
