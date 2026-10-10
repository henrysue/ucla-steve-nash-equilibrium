"""Memory of enemy actors seen through the fog of war.

The observation only lists enemies we can see *right now*. Without memory,
an enemy army that steps out of sight counts as zero (so the brain thinks it
can attack anything), and a base we scouted is forgotten the moment the scout
moves on. This keeps, per enemy id: type, last cell, last time seen, and
whether it is a building.

Forgetting:

- A **wreck** is dropped at once.
- A **unit** is dropped ``ENEMY_UNIT_MEMORY_SECONDS`` after we last saw it
  (it has moved, or died out of sight).
- Either is dropped when one of our actors stands within
  ``MEMORY_CLEAR_RADIUS_CELLS`` of its last cell and it is not visible: we are
  looking at that spot and it is not there (moved, or destroyed).
- Buildings otherwise stay remembered for the whole match — at the cell we
  last saw them, refreshed each time we see them again (same id).
- A construction yard that packs up into an MCV gets a new id: an enemy MCV
  seen beside a remembered yard means the yard has moved, so the yard is
  dropped and the MCV followed; a new yard seen beside a remembered MCV means
  it deployed there, so the MCV is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import config as C
from .observe import Actor, View, dist2


@dataclass
class Memory:
    actor: Actor
    last_seen: float
    building: bool


class EnemyMemory:
    def __init__(self) -> None:
        self._known: dict[int, Memory] = {}

    def update(self, view: View) -> None:
        now = view.game_seconds
        visible: set[int] = set()
        for group, building in ((view.enemy_units, False), (view.enemy_buildings, True)):
            for actor in group:
                if actor.id is None:
                    continue
                if actor.wreck:
                    self._known.pop(actor.id, None)
                    continue
                visible.add(actor.id)
                self._known[actor.id] = Memory(actor, now, building)   # refreshed: the cell we see now
        self._follow_transforms(view, visible)

        lookers = [a.cell for a in view.units + view.buildings if a.cell is not None and not a.wreck]
        r2 = C.MEMORY_CLEAR_RADIUS_CELLS ** 2
        for actor_id, mem in list(self._known.items()):
            if actor_id in visible:
                continue
            expired = not mem.building and now - mem.last_seen > C.ENEMY_UNIT_MEMORY_SECONDS
            looked = mem.actor.cell is not None and any(dist2(c, mem.actor.cell) <= r2 for c in lookers)
            if expired or looked:
                del self._known[actor_id]

    def _follow_transforms(self, view: View, visible: set) -> None:
        """A yard that packed up (MCV seen beside it) or an MCV that deployed
        (a yard seen beside it) left a stale memory under its old id."""
        r2 = C.TRANSFORM_RADIUS_CELLS ** 2
        mcvs = [a.cell for a in view.enemy_units if a.type == C.MCV and a.cell is not None and not a.wreck]
        yards = [a.cell for a in view.enemy_buildings if a.type == C.CONSTRUCTION_YARD and a.cell is not None and not a.wreck]
        for actor_id, mem in list(self._known.items()):
            if actor_id in visible or mem.actor.cell is None:
                continue
            if mem.building and mem.actor.type == C.CONSTRUCTION_YARD and any(dist2(c, mem.actor.cell) <= r2 for c in mcvs):
                del self._known[actor_id]      # it packed up and drives off as that MCV
            elif not mem.building and mem.actor.type == C.MCV and any(dist2(c, mem.actor.cell) <= r2 for c in yards):
                del self._known[actor_id]      # it deployed into that yard

    def forget_near(self, cell, radius: float) -> int:
        """Drop remembered buildings within ``radius`` of ``cell`` (the army
        stalled there: they are gone). Returns how many."""
        r2 = radius * radius
        gone = [i for i, m in self._known.items()
                if m.building and m.actor.cell is not None and dist2(m.actor.cell, cell) <= r2]
        for i in gone:
            del self._known[i]
        return len(gone)

    # -- what the rest of the agent reads ----------------------------------------

    @property
    def units(self) -> list[Actor]:
        return [m.actor for m in self._known.values() if not m.building]

    @property
    def buildings(self) -> list[Actor]:
        return [m.actor for m in self._known.values() if m.building]

    def army_value(self, costs: dict[str, float]) -> float:
        """The value of every enemy combat unit we remember (visible or not),
        plus the enemy defenses we know of: an attack into pillboxes is a
        fight against them too (match 01930376: seat 1 rated 7700 vs 1100 and
        attacked into two pillboxes for 2 minutes, losing 8000 to kill 2200)."""
        skip = set(C.HARVESTER) | {C.MCV}
        units = sum(costs.get(a.type, C.DEFAULT_UNIT_COST) for a in self.units if a.type not in skip)
        defenses = sum(costs.get(a.type, C.DEFAULT_UNIT_COST) for a in self.buildings if a.type in C.DEFENSES)
        return units + defenses

    def apply(self, view: View) -> View:
        """``view`` with the enemy army value raised to what we remember and
        the remembered enemy buildings attached (for targeting)."""
        remembered = self.army_value(view.unit_costs)
        return replace(
            view,
            enemy_army_value=max(view.enemy_army_value, remembered),
            known_enemy_buildings=self.buildings,
            known_enemy_units=self.units,
        )
