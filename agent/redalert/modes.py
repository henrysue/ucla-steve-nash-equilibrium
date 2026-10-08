"""The high-level Modes the brain sets and the executor obeys.

A tiny, explicit vocabulary is the seam between the two layers: the brain
reasons about *which* mode fits the game state; the executor just plays the
mode fast. Keeping the set small is what makes the agent both quick and
predictable (and easy to test).
"""

from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    OPEN = "open"                   # run the build order, start economy, scout
    BUILD = "build"                 # macro the deathball (default mid-game)
    DEFEND_RUSH = "defend_rush"     # cheese detected: cheapest units, rally home
    ALL_IN = "all_in"               # commit the deathball at the enemy base
    HARASS = "harass"               # pick at enemy economy with a detachment
    PROTECT_LEAD = "protect_lead"   # clock low and ahead: preserve assets

    def __str__(self) -> str:  # so it logs/serializes as the plain value
        return self.value


# Modes in which the executor should actively push units at the enemy.
AGGRESSIVE = frozenset({Mode.ALL_IN, Mode.HARASS})

# Modes in which the executor should keep army at home / rally defensively.
DEFENSIVE = frozenset({Mode.DEFEND_RUSH, Mode.PROTECT_LEAD})
