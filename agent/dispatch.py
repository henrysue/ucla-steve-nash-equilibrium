"""A game-dispatching agent: route each match to the right specialist.

``--agent`` picks ONE agent factory for the whole process, but this tournament
spans more than one game. ``create_agent()`` here returns a thin wrapper that,
on the first ``choose_action`` of a match, looks at ``context.game_type`` and
builds the specialist for that game — the Red Alert agent for ``red_alert``,
the VGC Draft strategist for everything else — then delegates every call to it
for the rest of the match.

Why lazy (not eager in ``create_agent``): the runtime calls ``create_agent()``
once per match with no arguments; the game type only arrives later, on the
decision context. So selection has to happen on the first decision and be
cached for the match.

Robustness: the wrapper never raises on its own. If the game type is missing on
the first call it defaults to the VGC strategist (whose existing behaviour is to
play Pokémon and to say plainly when it is handed a game it does not play), so
the dispatcher can never be the thing that forfeits a match. ``on_action_result``
is forwarded only to a specialist that defines it (the Red Alert agent does; the
VGC strategist does not).

Run it with:

    python -m agent --tournament --agent agent.dispatch
    python -m agent --match --agent agent.dispatch
"""

from __future__ import annotations

from typing import Any

# Game types that go to the Red Alert specialist. Everything else is handled by
# the VGC strategist (which itself refuses games it does not play).
_RED_ALERT = frozenset({"red_alert"})


def create_agent() -> GameDispatcher:
    return GameDispatcher()


class GameDispatcher:
    """One match's agent. Picks the specialist on first use, then delegates."""

    def __init__(self) -> None:
        self._delegate: Any = None
        self._game_type: str | None = None

    def choose_action(self, state: Any, context: Any):
        return self._resolve(context).choose_action(state, context)

    def on_action_result(self, result: Any, context: Any) -> None:
        delegate = self._resolve(context)
        handler = getattr(delegate, "on_action_result", None)
        if callable(handler):
            handler(result, context)

    def on_match_end(self, final_state: Any, context: Any) -> None:
        handler = getattr(self._resolve(context), "on_match_end", None)
        if callable(handler):
            handler(final_state, context)

    # -- selection ------------------------------------------------------------

    def _resolve(self, context: Any) -> Any:
        """The specialist for this match, built once and cached.

        Keyed on the first non-empty ``game_type`` we see. Defaults to the VGC
        strategist when the game type is absent, so a missing field never
        leaves us without an agent.
        """
        if self._delegate is not None:
            return self._delegate
        game_type = str(getattr(context, "game_type", "") or "")
        self._game_type = game_type
        self._delegate = _build(game_type)
        return self._delegate


def _build(game_type: str) -> Any:
    """Construct the specialist for ``game_type``. Imports are local so a
    specialist's own dependencies are only loaded for the games that use it."""
    if game_type in _RED_ALERT:
        from agent.redalert.agent import create_agent as make_redalert
        return make_redalert()
    from agent.vgc import create_strategist
    return create_strategist()
