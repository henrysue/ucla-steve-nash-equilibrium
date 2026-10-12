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
        self._push_at: float | None = None   # their fighters last seen pushing near our base
        self._seen_ids: set = set()
        self.first_seen: dict[str, float] = {}   # enemy type -> game s it was first seen
        self.sightings: list[dict] = []          # first sightings of each enemy actor, newest last
        self.fresh: list[dict] = []              # this view's first sightings
        self.first_push_at: int | None = None    # when their fighters first pushed near our base
        self._main_pushes: list[float] = []      # when their fighters pushed at our main base
        self._our_buildings: set | None = None
        self._lost_at: float | None = None       # when a building of ours last went

    @staticmethod
    def buildings(view: View) -> Counter:
        """Enemy building types we see or remember (by id, no wrecks)."""
        seen = {a.id: a.type for a in list(view.known_enemy_buildings) + list(view.enemy_buildings)
                if a.id is not None and not a.wreck}
        return Counter(seen.values())

    def update(self, view: View) -> None:
        self._note_sightings(view)
        self._track_pushes(view)
        ids = {b.id for b in view.buildings if b.id is not None and not b.wreck}
        if self._our_buildings is not None and self._our_buildings - ids:
            self._lost_at = view.game_seconds
        self._our_buildings = ids
        c = self.buildings(view)
        if sum(c.values()) < C.INTEL_MIN_BUILDINGS:
            self._army_read(view)
            return
        procs = c.get(C.REFINERY, 0)
        rax = sum(c.get(b, 0) for b in C.BARRACKS)
        if procs >= 2 and not (self._rush and view.game_seconds < C.RUSH_HOLD_SECONDS):
            self._economy, self._rush = True, False
        elif (view.game_seconds >= C.RUSH_READ_SECONDS and procs <= 1 and not self._economy
              and (c.get(C.WAR_FACTORY, 0) or rax >= 2)):
            self._rush = True
        self._army_read(view)

    def _army_read(self, view: View) -> None:
        """Their army tells a rush whatever their refineries say: ~10
        infantry seen before ``RUSH_HOLD_SECONDS``, or ~6 before
        ``RUSH_EARLY_ARMY_SECONDS`` (6b457914: the read came at 4:38, after
        the rush; 3bd6e450: two refineries hid a rifle wave — seen at 2:04,
        it hit at 2:50, the read came at 4:07)."""
        t = view.game_seconds
        seen = view.enemy_army_value
        if (t < C.RUSH_HOLD_SECONDS and seen >= C.RUSH_ARMY_SEEN_VALUE) or \
                (t < C.RUSH_EARLY_ARMY_SECONDS and seen >= C.RUSH_EARLY_ARMY_VALUE):
            self._rush = True

    def scouts_lost(self, view: View) -> None:
        """Our early scouts killed (the scout alarm) before ``RUSH_HOLD_SECONDS``:
        something is out there — read it as a rush and hold it (the alarm
        itself lasts a minute: 3bd6e450 dropped it at 3:05, 15 s before the wave)."""
        if view.game_seconds < C.RUSH_HOLD_SECONDS:
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

    def _note_sightings(self, view: View) -> None:
        """Every enemy building and unit the first time we see it: when, what,
        where (the scouting report the advisor reads, and the match log)."""
        t = view.game_seconds
        new = []
        for group, kind in ((view.enemy_buildings, "building"), (view.enemy_units, "unit")):
            for a in group:
                if a.id is None or a.wreck or a.cell is None or a.id in self._seen_ids:
                    continue
                self._seen_ids.add(a.id)
                self.first_seen.setdefault(a.type, round(t))
                new.append({"t": round(t), "kind": kind, "type": a.type, "cell": list(a.cell)})
        self.fresh = new
        self.sightings = (self.sightings + new)[-200:]

    def recent_mix(self, view: View, seconds: float) -> tuple:
        """The enemy unit types first seen in the last ``seconds`` (each unit
        once; no harvesters or MCVs): what their army is made of, over more
        than the units in sight right now."""
        t = view.game_seconds
        skip = set(C.HARVESTER) | {C.MCV}
        return tuple(s["type"] for s in self.sightings
                     if s["kind"] == "unit" and s["type"] not in skip and t - s["t"] <= seconds)

    @property
    def sighted(self) -> int:
        """Enemy actors seen so far."""
        return len(self._seen_ids)

    def report(self, view: View) -> dict:
        """What our scouts have seen of them, for the advisor: the buildings
        we know by type, when each enemy type was first seen, the latest first
        sightings, and the rules' own reads."""
        return {
            "buildings_known": dict(self.buildings(view)),
            "first_seen": dict(self.first_seen),
            "recent_sightings": self.sightings[-15:],
            "rules_read": {"opening": self.opening(), "rush_suspected": self.suspect(view),
                           "turtling": self.turtling(view), "passive": view.enemy_passive},
        }

    def _track_pushes(self, view: View) -> None:
        from .scout import our_yards

        yards = list(our_yards(view)) or ([view.base_center] if view.base_center else [])
        r2 = C.PUSH_NEAR_BASE_CELLS ** 2
        near = [a for a in view.enemy_units if a.cell is not None and not a.wreck and a.type not in C.HARVESTER
                and any((a.cell[0] - y[0]) ** 2 + (a.cell[1] - y[1]) ** 2 <= r2 for y in yards)]
        cost = lambda k: view.unit_costs.get(k) or C.KNOWN_COSTS.get(k, C.DEFAULT_UNIT_COST)  # noqa: E731
        if sum(cost(a.type) for a in near) >= C.PUSH_SEEN_VALUE:
            self._push_at = view.game_seconds
            if self.first_push_at is None:
                self.first_push_at = round(view.game_seconds)
        # At our buildings (not our turrets, not the concave out front): an
        # attack on us, for the passive read.
        ours = [b.cell for b in view.buildings if b.cell is not None and not b.wreck and b.type not in C.TURRETS]
        b2 = C.PASSIVE_PUSH_CELLS ** 2
        at_ours = [a for a in view.enemy_units if a.cell is not None and not a.wreck and a.type not in C.HARVESTER
                   and any((a.cell[0] - c[0]) ** 2 + (a.cell[1] - c[1]) ** 2 <= b2 for c in ours)]
        if sum(cost(a.type) for a in at_ours) >= C.PUSH_SEEN_VALUE:
            self._main_pushes = (self._main_pushes + [view.game_seconds])[-50:]

    def passive(self, view: View, retreated_at: float | None = None) -> bool:
        """They don't come out: past ``PASSIVE_READ_SECONDS``, no rush read,
        no building of ours lost and no push at our buildings (within
        ``PASSIVE_PUSH_CELLS``) for ``PASSIVE_QUIET_SECONDS`` — their units
        chasing our retreat (within ``PASSIVE_CHASE_SECONDS`` of
        ``retreated_at``) don't count, nor do pokes at the concave. Answer: expand early and often, more
        tanks (e2bc988e: a turtle took 10 harvesters to our 7 while we
        skirmished in the middle and held for a push that never came)."""
        t = view.game_seconds
        if t < C.PASSIVE_READ_SECONDS or self._rush:
            return False
        if self._lost_at is not None and t - self._lost_at < C.PASSIVE_QUIET_SECONDS:
            return False
        chase = (lambda p: retreated_at is not None and retreated_at <= p <= retreated_at + C.PASSIVE_CHASE_SECONDS)
        return not any(t - p < C.PASSIVE_QUIET_SECONDS and not chase(p) for p in self._main_pushes)

    def turtling(self, view: View) -> bool:
        """They turtle: ``TURTLE_READ_DEFENSES`` ground turrets known, past
        ``TURTLE_READ_SECONDS``, no push near our base for
        ``TURTLE_QUIET_SECONDS`` and no rush read. Answer: the dome, siege
        units (they outrange turrets), more expansions, and no attack before
        the siege is there (match efefdca8 attacked a turtle's ring twice)."""
        if view.game_seconds < C.TURTLE_READ_SECONDS or self._rush:
            return False
        if self._push_at is not None and view.game_seconds - self._push_at < C.TURTLE_QUIET_SECONDS:
            return False
        ids = {(a.id if a.id is not None else a.cell) for a in list(view.known_enemy_buildings) + list(view.enemy_buildings)
               if a.type in C.TURRETS and not a.wreck}
        return len(ids) >= C.TURTLE_READ_DEFENSES

    def opening(self) -> str | None:
        return "rush" if self._rush else "economy" if self._economy else None

    def rushing(self, view: View) -> bool:
        """An early rush read, until ``RUSH_HOLD_SECONDS``: a rush arrives by
        then or it wasn't one (it used to hold the army home until 6:00)."""
        return self._rush and view.game_seconds < C.RUSH_HOLD_SECONDS
