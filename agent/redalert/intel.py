"""What our scouts have learned about the enemy: their opening.

Pros open with two refineries (1:00 and 1:38) before the war factory. An
opponent with one refinery but a war factory or a second barracks by
``RUSH_READ_SECONDS`` has spent the second refinery's money on an army:
an early rush is coming. Two refineries is the economic opening. The read
needs a look at their base (``INTEL_MIN_BUILDINGS`` buildings known); a rush
read holds until ``EARLY_PUSH_WINDOW_SECONDS`` unless a second refinery
turns up.
"""

from __future__ import annotations

from collections import Counter

from . import config as C
from .observe import View


class EnemyIntel:
    def __init__(self) -> None:
        self._rush = False
        self._economy = False

    @staticmethod
    def buildings(view: View) -> Counter:
        """Enemy building types we see or remember (by id, no wrecks)."""
        seen = {a.id: a.type for a in list(view.known_enemy_buildings) + list(view.enemy_buildings)
                if a.id is not None and not a.wreck}
        return Counter(seen.values())

    def update(self, view: View) -> None:
        c = self.buildings(view)
        if sum(c.values()) < C.INTEL_MIN_BUILDINGS:
            return
        procs = c.get(C.REFINERY, 0)
        rax = sum(c.get(b, 0) for b in C.BARRACKS)
        if procs >= 2:
            self._economy, self._rush = True, False
        elif (view.game_seconds >= C.RUSH_READ_SECONDS and procs <= 1 and not self._economy
              and (c.get(C.WAR_FACTORY, 0) or rax >= 2)):
            self._rush = True
        elif (procs <= 1 and not self._economy and view.game_seconds < C.RUSH_HOLD_SECONDS
              and view.enemy_army_value >= C.RUSH_ARMY_SEEN_VALUE):
            # Their army tells it too: one refinery and ~10 infantry already
            # (match 6b457914: our scouts missed the second barracks at 1:48
            # but saw the army; the read came at 4:38, after the rush).
            self._rush = True

    def suspect(self, view: View) -> bool:
        """Their base seen after ``RUSH_READ_SECONDS`` with one refinery (pros
        have two by 1:38), no read either way, before ``RUSH_HOLD_SECONDS``:
        maybe a rush. Mild on purpose (a second refinery can hide in the fog):
        one warning turret on the road in, no hold. Match 6b457914 saw five
        buildings and one refinery at 1:48; the rush hit at 2:26."""
        if self._rush or self._economy or not (C.RUSH_READ_SECONDS <= view.game_seconds < C.RUSH_HOLD_SECONDS):
            return False
        c = self.buildings(view)
        return sum(c.values()) >= C.INTEL_MIN_BUILDINGS and c.get(C.REFINERY, 0) <= 1

    def opening(self) -> str | None:
        return "rush" if self._rush else "economy" if self._economy else None

    def rushing(self, view: View) -> bool:
        """An early rush read, until ``RUSH_HOLD_SECONDS``: a rush arrives by
        then or it wasn't one (it used to hold the army home until 6:00)."""
        return self._rush and view.game_seconds < C.RUSH_HOLD_SECONDS
