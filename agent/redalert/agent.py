"""The Red Alert agent: wires the executor and brain into one match player.

``create_agent()`` builds one :class:`RedAlertAgent` per match. The agent is
real time, so ``choose_action`` must return fast and never block on a model:

- The **executor** runs every view and returns a legal batch immediately.
- The **brain** (mode selection) is deterministic and cheap, so it also runs
  every view — but the *optional* model consultation inside it is throttled and
  strictly off the critical path (a stale suggestion from a previous view is
  used; a failing/absent model just means the deterministic policy decides).
- ``on_action_result`` feeds refusals back so a batch refused as a whole is not
  repeated verbatim.

With no model configured the agent plays a complete, aggressive, legal match
from the deterministic policy alone — it never stalls or forfeits.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from altruagent import WAIT, DecisionContext, GameState, WithReasoning

from . import brain, executor, orders
from . import config as C
from .modes import Mode
from .observe import read


def create_agent():
    return RedAlertAgent()


class RedAlertAgent:
    def __init__(
        self,
        *,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
        model_pref_fn: Callable[[Any], Mode | None] | None = None,
    ) -> None:
        self._log = log
        self._clock = clock
        # Optional hook: (view) -> Mode suggestion. None => pure deterministic
        # policy. Kept injectable so a model-backed brain can be added without
        # touching the hot path, and so tests run without a provider.
        self._model_pref_fn = model_pref_fn
        self._cheese_since: float | None = None
        self._mode: Mode = Mode.OPEN
        self._model_pref: Mode | None = None
        self._last_model_call = 0.0
        self._model_calls = 0
        self._last_batch_keys: list[str] = []
        self._refused_last = False

    # -- the hot path ---------------------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        raw = getattr(state, "raw", None)
        if not isinstance(raw, dict) or not isinstance(raw.get("observation"), dict):
            return WAIT  # nothing to look at yet

        view = read(raw)
        if not view.ok:
            return WAIT

        # Track the cheese window (sets/expires defensively; strategy hole #3).
        self._cheese_since = brain.update_cheese_since(view, self._cheese_since)

        # Optional, throttled model suggestion — used on a LATER view when
        # ready, never blocking this one.
        self._maybe_refresh_model_pref(view)

        self._mode = brain.choose_mode(
            view, cheese_since=self._cheese_since, model_pref=self._model_pref
        )

        batch_orders = executor.plan(view, self._mode)
        if self._refused_last:
            # Don't resend an identical batch that was just refused as a whole.
            batch_orders = [o for o in batch_orders if _order_key(o) not in self._last_batch_keys]

        move = orders.batch(batch_orders)
        if move is None:
            return WAIT  # nothing to do right now

        self._last_batch_keys = [_order_key(o) for o in move["orders"]]
        self._refused_last = False
        return WithReasoning(move, self._reason(view))

    def on_action_result(self, result: Any, context: DecisionContext) -> None:
        """Learn from each play_action answer. A whole-batch refusal
        (INVALID_ACTION) sets a flag so the next view does not resend it."""
        if isinstance(result, dict) and str(result.get("error") or "").upper() == "INVALID_ACTION":
            self._refused_last = True

    # -- the brain's optional model hook -------------------------------------

    def _maybe_refresh_model_pref(self, view) -> None:
        if self._model_pref_fn is None:
            return
        if self._model_calls >= C.MAX_BRAIN_CALLS:
            return
        now = self._clock()
        if now - self._last_model_call < C.BRAIN_MIN_INTERVAL_SECONDS:
            return
        self._last_model_call = now
        self._model_calls += 1
        try:
            self._model_pref = self._model_pref_fn(view)
        except Exception as exc:  # noqa: BLE001 - a model problem must never stall
            self._log(f"[redalert] model suggestion failed, using policy: {type(exc).__name__}: {exc}")
            self._model_pref = None

    def _reason(self, view) -> str:
        return f"{self._mode} | army {int(view.army_value)} vs {int(view.enemy_army_value)} | t={int(view.game_seconds)}s"


def _order_key(order: dict) -> str:
    import json
    body = {k: v for k, v in order.items() if k != "queued"}
    if isinstance(body.get("units"), list):
        body["units"] = sorted(body["units"], key=str)
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
