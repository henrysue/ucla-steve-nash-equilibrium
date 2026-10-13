"""The last-seen map: a mirror of the map with what we last saw in each area.

The map is cut into ``LASTSEEN_SECTOR_CELLS`` squares. Every view, each
square our vision covers (one of our actors in it, or its centre within
``LASTSEEN_VISION_CELLS`` of one) is stamped seen now, and its record is
replaced by the enemies visible in it now (an empty look clears it); a square
a scout covers is also stamped scouted now. Enemy units are kept by id in one
square only — seen elsewhere, they move there — so nothing is counted twice.
Squares we don't see keep their last record and its age.

It informs the advisor only (:meth:`LastSeenMap.summary`): what we last saw
where, and how long ago, so it can weigh a stale estimate. It never changes
the estimate the rules use (agent._believe).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config as C
from .observe import View, dist2, enemy_spawn_guess


@dataclass
class Sector:
    seen_at: float | None = None
    scouted_at: float | None = None
    units: dict = field(default_factory=dict)        # enemy unit id -> type
    buildings: dict = field(default_factory=dict)    # enemy building id (or cell) -> type


class LastSeenMap:
    def __init__(self) -> None:
        self._sectors: dict[tuple[int, int], Sector] = {}
        self._unit_at: dict[int, tuple[int, int]] = {}   # enemy unit id -> the sector holding it
        self.refreshed_at: float | None = None             # the last view folded in
        self.scouted_at: float | None = None               # the last time a scout saw anything

    @staticmethod
    def _key(cell) -> tuple[int, int]:
        s = C.LASTSEEN_SECTOR_CELLS
        return (int(cell[0]) // s, int(cell[1]) // s)

    @staticmethod
    def _centre(key) -> tuple[int, int]:
        s = C.LASTSEEN_SECTOR_CELLS
        return (key[0] * s + s // 2, key[1] * s + s // 2)

    def _covered(self, cells) -> set:
        """The sectors these actors see: their own, and any whose centre is
        within ``LASTSEEN_VISION_CELLS``."""
        r = C.LASTSEEN_VISION_CELLS
        s = C.LASTSEEN_SECTOR_CELLS
        out = set()
        for c in cells:
            out.add(self._key(c))
            for kx in range((c[0] - r) // s, (c[0] + r) // s + 1):
                for ky in range((c[1] - r) // s, (c[1] + r) // s + 1):
                    if kx >= 0 and ky >= 0 and dist2(self._centre((kx, ky)), c) <= r * r:
                        out.add((kx, ky))
        return out

    def update(self, view: View, scout_ids=()) -> None:
        """Fold in this view: stamp what our vision covers and record the
        enemies visible there (``scout_ids``: our scouts, for scouted_at)."""
        t = view.game_seconds
        ours = [a for a in list(view.units) + list(view.buildings) if a.cell is not None and not a.wreck]
        seen = self._covered([a.cell for a in ours])
        scouts = set(scout_ids)
        scouted = self._covered([a.cell for a in ours if a.id in scouts]) if scouts else set()
        foes = [a for a in view.enemy_units if a.cell is not None and not a.wreck and a.id is not None]
        blds = [b for b in view.enemy_buildings if b.cell is not None and not b.wreck]
        seen |= {self._key(a.cell) for a in foes} | {self._key(b.cell) for b in blds}   # what we see, we see
        for key in seen:
            sec = self._sectors.setdefault(key, Sector())
            sec.seen_at = t
            for uid in sec.units:
                if self._unit_at.get(uid) == key:
                    del self._unit_at[uid]
            sec.units, sec.buildings = {}, {}
            if key in scouted:
                sec.scouted_at = t
        for a in foes:
            key = self._key(a.cell)
            old = self._unit_at.get(a.id)
            if old is not None and old != key and old in self._sectors:
                self._sectors[old].units.pop(a.id, None)          # it moved: one place only
            self._sectors[key].units[a.id] = a.type
            self._unit_at[a.id] = key
        for b in blds:
            self._sectors[self._key(b.cell)].buildings[b.id if b.id is not None else tuple(b.cell)] = b.type
        if scouted and any(self._key(a.cell) in scouted for a in foes + blds):
            self.scouted_at = t
        self.refreshed_at = t

    def summary(self, view: View) -> dict:
        """For the advisor: the map's freshness, the share of it seen in the
        last ``LASTSEEN_FRESH_SECONDS``, their base's square, the enemy army
        as last seen (each unit where we last saw it, however long ago) and
        the squares with enemies, most valuable first (at most
        ``LASTSEEN_MAX_SECTORS``), each with its age."""
        t = view.game_seconds
        cost = lambda k: view.unit_costs.get(k) or C.KNOWN_COSTS.get(k, C.DEFAULT_UNIT_COST)  # noqa: E731

        def counts(d: dict) -> dict:
            out: dict = {}
            for kind in d.values():
                out[kind] = out.get(kind, 0) + 1
            return out

        def ago(at):
            return None if at is None else round(t - at)

        def describe(key, sec: Sector) -> dict:
            return {"cell": list(self._centre(key)), "seen_ago": ago(sec.seen_at), "scouted_ago": ago(sec.scouted_at),
                    "units": counts(sec.units), "buildings": counts(sec.buildings)}

        total = 1
        if view.bounds is not None:
            s = C.LASTSEEN_SECTOR_CELLS
            total = max(1, -(-view.bounds[2] // s) * -(-view.bounds[3] // s))
        fresh = sum(1 for sec in self._sectors.values()
                    if sec.seen_at is not None and t - sec.seen_at <= C.LASTSEEN_FRESH_SECONDS)
        army = [(kind, sec.seen_at) for sec in self._sectors.values() for kind in sec.units.values()
                if kind not in C.HARVESTER]
        value = sum(cost(k) for k, _ in army)
        oldest = min((at for _, at in army if at is not None), default=None)
        with_enemies = [(k, s) for k, s in self._sectors.items() if s.units or s.buildings]
        with_enemies.sort(key=lambda ks: -(sum(cost(v) for v in ks[1].units.values())
                                           + sum(cost(v) for v in ks[1].buildings.values())))
        spawn = enemy_spawn_guess(view)
        base_key = self._key(spawn) if spawn is not None else None
        base = self._sectors.get(base_key) if base_key is not None else None
        return {
            "sector_cells": C.LASTSEEN_SECTOR_CELLS,
            "refreshed_at": None if self.refreshed_at is None else round(self.refreshed_at),
            "last_scouted_at": None if self.scouted_at is None else round(self.scouted_at),
            "seen_last_minute_share": round(min(1.0, fresh / total), 2),
            "their_base": describe(base_key, base) if base is not None else None,
            "enemy_army_as_last_seen": {"value": round(value), "units": counts(dict(enumerate(k for k, _ in army))),
                                        "oldest_sighting_ago": ago(oldest)},
            "sectors_with_enemies": [describe(k, s) for k, s in with_enemies[:C.LASTSEEN_MAX_SECTORS]],
        }
