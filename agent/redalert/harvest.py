"""Keep harvesters on ore that still exists.

A bare ``harvest`` lets the engine pick ore near the harvester. Once the home
field is mined out it finds none, idles by the empty field, and gets the same
useless order again — while our expansions' refineries sit waiting (match
1e6ba89d). So harvesters are steered between our fields:

- A **field** is ore beside one or more of our refineries (home or expansion).
- A field is **depleted** when a harvester sits idle near it for
  ``FIELD_IDLE_SECONDS``, or when harvesters are around it but none has been
  on its ore for ``FIELD_STALE_SECONDS`` (OpenRA harvesters often keep
  searching instead of going idle). Ore mines regrow, so a depleted field is
  re-checked after ``FIELD_RECHECK_SECONDS``.
- **Idle harvesters** are sent (``harvest`` with ``to``) to the live field with
  the fewest harvesters per refinery, nearest first.
- **Rebalance**: harvesters working a depleted field move to a live one, up
  to ``REBALANCE_PER_VIEW`` at a time.
- **Balance**: fields are kept staffed in proportion to their refineries —
  one harvester at a time moves from the most crowded field to the most
  under-served one, so an expansion's refinery gets trucks even while home
  still has ore. A moved harvester stays put ``HARVESTER_SETTLE_SECONDS``.
"""

from __future__ import annotations

from . import config as C
from . import orders
from .observe import View, dist2


def refinery_fields(view: View) -> dict[tuple[int, int], int]:
    """{ore cell of each field: our refineries serving it}: each refinery's
    field is the ore its trucks can WALK to (not ore across a cliff)."""
    from .terrain import ring, walk_distances

    if not view.ore_cells:
        return {}
    ore_set = set(view.ore_cells)
    out: dict[tuple[int, int], int] = {}
    for b in view.buildings:
        if b.type != C.REFINERY or b.cell is None or b.wreck:
            continue
        starts = [c for c in ring((b.cell[0], b.cell[1], b.cell[0] + 2, b.cell[1] + 2))
                  if c not in view.blocked_cells]   # a cliff beside it is not a way out
        walk = walk_distances(view, starts, C.FIELD_WALK_STEPS)
        near = [c for c in walk if c in ore_set]
        if not near:
            continue
        ore = min(near, key=lambda c: walk[c])
        key = next((k for k in out if dist2(k, ore) <= C.EXPAND_CLUSTER_CELLS ** 2), ore)
        out[key] = out.get(key, 0) + 1
    return out


class HarvestManager:
    def __init__(self) -> None:
        self._depleted: dict[tuple[int, int], float] = {}   # field ore cell -> game time marked
        self._idle_since: dict[int, float] = {}             # harvester id -> game time it went idle
        self._moved_at: dict[int, float] = {}               # harvester id -> game time we last moved it
        self.last_counts: dict[tuple[int, int], int] = {}   # harvesters per field (for the log)
        self._unworked_since: dict[tuple[int, int], float] = {}  # field -> harvesters near, none on its ore
        self._fields: dict[tuple[int, int], int] = {}
        self._t = 0.0

    def depleted(self) -> set[tuple[int, int]]:
        """Fields currently believed mined out (as of the last view)."""
        return {f for f, at in self._depleted.items() if self._t - at < C.FIELD_RECHECK_SECONDS}

    def home_depleted(self, view: View) -> bool:
        """The field nearest our base is mined out."""
        if view.base_center is None or not self._fields:
            return False
        home = min(self._fields, key=lambda f: dist2(f, view.base_center))
        return home in self.depleted()

    def fields(self, view: View) -> dict[tuple[int, int], int]:
        """{ore cell of each field: refineries serving it}, from our refineries."""
        return refinery_fields(view)

    def orders(self, view: View) -> list[dict] | None:
        """Harvest orders for this view, or None when we know no fields (the
        caller then falls back to a bare ``harvest`` for idle harvesters)."""
        t = view.game_seconds
        self._t = t
        fields = self.fields(view)
        self._fields = fields
        if not fields:
            return None
        harvesters = [h for h in view.units_of(C.HARVESTER) if h.id is not None and h.cell is not None]
        near2 = C.FIELD_NEAR_CELLS ** 2

        def field_of(cell):
            best = min(fields, key=lambda f: dist2(f, cell))
            return best if dist2(best, cell) <= near2 else None

        # Mark fields depleted where harvesters stay idle.
        idle_now = set()
        for h in harvesters:
            if not h.idle:
                self._idle_since.pop(h.id, None)
                continue
            idle_now.add(h.id)
            since = self._idle_since.setdefault(h.id, t)
            f = field_of(h.cell)
            if f is not None and t - since >= C.FIELD_IDLE_SECONDS:
                self._depleted[f] = t
        for gone in set(self._idle_since) - {h.id for h in harvesters}:
            del self._idle_since[gone]

        # Mark fields depleted where harvesters hang around but nobody mines.
        work2 = C.FIELD_WORK_CELLS ** 2
        field2 = (C.EXPAND_CLUSTER_CELLS * 1.5) ** 2
        for f in fields:
            ore = [c for c in view.ore_cells if dist2(c, f) <= field2]
            around = [h for h in harvesters if dist2(h.cell, f) <= near2]
            mining = any(not h.idle and any(dist2(h.cell, c) <= work2 for c in ore) for h in around)
            if mining or not around:
                self._unworked_since.pop(f, None)
                continue
            since = self._unworked_since.setdefault(f, t)
            if t - since >= C.FIELD_STALE_SECONDS:
                self._depleted[f] = t
                self._unworked_since.pop(f, None)

        live = [f for f in fields if t - self._depleted.get(f, -1e9) >= C.FIELD_RECHECK_SECONDS]
        if not live:
            idle = [h.id for h in harvesters if h.idle]
            return [orders.harvest(idle)] if idle else []

        load = {f: 0 for f in fields}
        for h in harvesters:
            f = field_of(h.cell)
            if f is not None and h.id not in idle_now:
                load[f] += 1

        def pick(cell):
            f = min(live, key=lambda f: (load[f] / fields[f], dist2(f, cell)))
            load[f] += 1
            return f

        out: dict[tuple[int, int], list[int]] = {}
        for h in harvesters:
            if h.idle:
                out.setdefault(pick(h.cell), []).append(h.id)
        moved = 0
        settled = lambda h: t - self._moved_at.get(h.id, -1e9) >= C.HARVESTER_SETTLE_SECONDS  # noqa: E731
        for h in harvesters:
            if moved >= C.REBALANCE_PER_VIEW or h.idle or not settled(h):
                continue
            f = field_of(h.cell)
            if f is not None and f not in live:
                load[f] -= 1
                out.setdefault(pick(h.cell), []).append(h.id)
                moved += 1
        # Balance: one truck from the most crowded field (per refinery) to the
        # most under-served live one, when the gap is at least one truck.
        if moved < C.REBALANCE_PER_VIEW and len(live) > 1:
            busiest = max(live, key=lambda f: load[f] / fields[f])
            emptiest = min(live, key=lambda f: load[f] / fields[f])
            share = sum(load[f] for f in live) / sum(fields[f] for f in live)
            if load[busiest] - share * fields[busiest] >= 1 and share * fields[emptiest] - load[emptiest] >= 1:
                spare = [h for h in harvesters if not h.idle and settled(h) and field_of(h.cell) == busiest]
                if spare:
                    h = min(spare, key=lambda h: dist2(h.cell, emptiest))
                    load[busiest] -= 1
                    load[emptiest] += 1
                    out.setdefault(emptiest, []).append(h.id)
        for ids in out.values():
            for i in ids:
                self._moved_at[i] = t
        self.last_counts = dict(load)
        return [orders.harvest(ids, to=f) for f, ids in out.items()]
