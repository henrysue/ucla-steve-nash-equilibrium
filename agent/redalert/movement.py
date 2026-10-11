"""Keep units moving: spot units stuck on the way and get them going again.

A unit with an order (not idle) that has not moved for
``STUCK_SECONDS`` is jammed — a traffic knot in a choke, wedged on terrain,
or waiting on a path that will never clear. It gets a ``stop``; the next
view's normal logic (staging, attack, harvest, patrol) sees it idle and
orders it afresh, which makes it look for a new path.

Not stuck, though stationary:

- fighting: an enemy unit or building within ``ENGAGED_RADIUS_CELLS`` (it
  is shooting; siege units within ``SIEGE_TARGET_RADIUS_CELLS``: they shell
  turrets from out of their range);
- guarding (the outpost's ``guard`` order, ``skip``): it stands by its ward;
- a harvester beside one of our refineries (unloading);
- an MCV (it may be deploying).
"""

from __future__ import annotations

from . import config as C
from . import orders
from .observe import View, dist2


class Unstick:
    def __init__(self) -> None:
        # unit id -> (cell, game time it was first seen at that cell, not idle)
        self._still: dict[int, tuple[tuple[int, int], float]] = {}
        self.unstuck = 0          # how many stops we have sent (for the log)

    def orders(self, view: View, skip=frozenset()) -> list[dict]:
        t = view.game_seconds
        stuck: list[int] = []
        seen: set[int] = set()
        enemies = [a.cell for a in list(view.enemy_units) + list(view.enemy_buildings)
                   if a.cell is not None and not a.wreck]
        refineries = [b.cell for b in view.buildings if b.type == C.REFINERY and b.cell is not None]
        for u in view.units:
            if (u.id is None or u.cell is None or u.wreck or u.idle or u.type == C.MCV or u.id in skip
                    or "guard" in (u.act or "").lower()):
                continue
            seen.add(u.id)
            reach = C.SIEGE_TARGET_RADIUS_CELLS if u.type in C.SIEGE_UNITS else C.ENGAGED_RADIUS_CELLS
            if any(dist2(u.cell, e) <= reach * reach for e in enemies):
                self._still.pop(u.id, None)          # fighting, not stuck
                continue
            if u.type in C.HARVESTER and any(dist2(u.cell, r) <= 16 for r in refineries):
                self._still.pop(u.id, None)          # unloading
                continue
            last = self._still.get(u.id)
            if last is None or last[0] != u.cell:
                self._still[u.id] = (u.cell, t)
            elif t - last[1] >= C.STUCK_SECONDS:
                stuck.append(u.id)
                self._still.pop(u.id, None)
        for gone in set(self._still) - seen:
            del self._still[gone]                    # idle now, or dead
        if not stuck:
            return []
        self.unstuck += len(stuck)
        return [orders.stop(stuck[:orders.MAX_UNITS_PER_ORDER])]
