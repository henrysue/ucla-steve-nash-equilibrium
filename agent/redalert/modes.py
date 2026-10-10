"""The army's posture: the brain picks one each view, the executor plays it."""

from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    OPEN = "open"                   # no construction yard or barracks yet
    BUILD = "build"                 # mass at the staging point (default)
    ALL_IN = "all_in"               # attack
    RETREAT = "retreat"             # losing a fight: pull back to the staging point
    DEFEND_RUSH = "defend_rush"     # attacked in the opening: rifle spam, fight at home
    DEFEND_BASE = "defend_base"     # attacked later: fight at home, production as normal
    PROTECT_LEAD = "protect_lead"   # ahead at the very end: stay home

    def __str__(self) -> str:  # so it logs/serializes as the plain value
        return self.value


AGGRESSIVE = frozenset({Mode.ALL_IN})

DEFENSIVE = frozenset({Mode.DEFEND_RUSH, Mode.DEFEND_BASE, Mode.PROTECT_LEAD})
