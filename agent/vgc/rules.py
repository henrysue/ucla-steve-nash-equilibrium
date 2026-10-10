"""The one rules layer for a battle turn: what is withheld, what is rejected.

Everything that makes a move pointless or a blunder lives behind this one
entry point, built fresh each turn:

- from the move tracker (moverules.py, per-match history): Fake Out after
  its user has moved, a third Protect in a row, Blood Moon twice in a row,
  and the opening-turn Fake Out lead;
- from the move list (movelist.py, static move data): attacks every target
  is immune to, status moves every target ignores, plain attacks another of
  the slot's moves beats on every target;
- row checks (movelist.row_problem): spread moves that hit our own partner,
  attacks into immune or resisting targets, chipping a Stamina user, hostile
  moves at our own partner.

``blocked`` options are hidden from the model and never picked by the best
line; ``check`` turns an answer row into a payload or raises InvalidAnswer.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import actions as act
from . import movelist
from .moverules import MoveTracker, SlotRules


@dataclass
class TurnRules:
    slots: list
    board: movelist.Board | None
    per_slot: list[SlotRules]

    def check(self, answer: dict) -> dict:
        """A legal payload that breaks no hard rule, or InvalidAnswer."""
        payload = act.build_doubles(self.slots, answer, self.per_slot)
        problem = movelist.row_problem(self.board, payload) if self.board is not None else None
        if problem:
            raise act.InvalidAnswer(problem)
        return payload

    def legal(self, answer: dict) -> bool:
        try:
            self.check(answer)
            return True
        except act.InvalidAnswer:
            return False


def build(slots, observation: dict, tracker: MoveTracker, table, log) -> TurnRules:
    """This turn's rules. Never raises: a failing layer is logged and skipped."""
    try:
        per_slot = tracker.begin(slots, observation)
    except Exception as exc:  # noqa: BLE001 - advice only; never cost the move
        log(f"[vgc] move tracker unavailable this turn [{type(exc).__name__}]: {exc}")
        per_slot = [SlotRules(None) for _ in slots]
    board = None
    try:
        board = movelist.Board(slots, observation, table)
        per_slot = movelist.merge(per_slot, movelist.option_blocks(board), slots)
    except Exception as exc:  # noqa: BLE001
        log(f"[vgc] move list checks unavailable this turn [{type(exc).__name__}]: {exc}")
    return TurnRules(slots, board, per_slot)
