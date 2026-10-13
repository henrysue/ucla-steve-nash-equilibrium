"""The harvester snipe: a few fast vehicles kill one harvester at their main
field and leave before the defenses catch them.

Against an opponent with no expansion to raid (Yuriko takes none before
~6:40) the raid squads (raid.py) have no target; their harvesters at home are
the tiebreak points within reach (a harvester is 1100). From
``SNIPE_TRAIN_FROM_SECONDS`` we keep ``SNIPE_UNITS`` to ``SNIPE_MAX_UNITS``
Rangers or light tanks (``SNIPE_UNIT_TYPES``, the first offered; none for the
Soviets) out of the main army. A run:

With neither (the Soviets) it takes ``SNIPE_HEAVY_UNITS`` heavy tanks from
the army: slower, but they kill a harvester fast and survive a pillbox.

- **go**: from ``SNIPE_FROM_SECONDS``, while their push is in our half
  (scout.incoming_push: their army is away) or ``SNIPE_EVERY_SECONDS`` after
  the last run, to the edge of their main ore field out of their turrets'
  reach (raid.turret_zone), by a road round their sight lines — the chokes,
  the front of their base and their turrets (:meth:`HarvesterSnipe.route`);
  plain moves, not attack-moves — it slips past, it doesn't fight on the way;
- **hit**: the nearest harvester in sight within ``SNIPE_HIT_CELLS``, by id,
  until it dies;
- **leave**: home, after the kill, when their fighters near it outmatch the
  squad, when it has lost ``SNIPE_LOSS_SHARE`` of its units, after
  ``SNIPE_HIT_SECONDS`` on target or ``SNIPE_TIMEOUT_SECONDS`` in all.

Its units are kept out of the army like the raid squads' (agent: raid_ids).
"""

from __future__ import annotations

from . import config as C
from . import orders
from .observe import View, dist2, enemy_ore_field


class HarvesterSnipe:
    def __init__(self) -> None:
        self.unit_ids: list[int] = []
        self.state = "gather"
        self._since = 0.0                      # when the current state began
        self._launched_at: float | None = None
        self._last_run_end = float("-inf")
        self._launch_size = 0
        self._target: int | None = None        # the harvester being shot
        self._legs: list | None = None
        self.runs = 0
        self.kills = 0

    # -- membership ---------------------------------------------------------------

    def update(self, view: View, taken=frozenset()) -> None:
        """Drop the dead; claim Rangers / light tanks not in another squad
        up to ``SNIPE_MAX_UNITS`` (they are trained for it: production) —
        never the army's own main tank (a light tank is that until the
        depot offers better)."""
        alive = {u.id for u in view.combat_units if u.id is not None and u.cell is not None}
        self.unit_ids = [i for i in self.unit_ids if i in alive]
        if view.game_seconds < C.SNIPE_TRAIN_FROM_SECONDS:
            return
        kinds, cap = self._kinds(view)
        for u in view.combat_units:
            if len(self.unit_ids) >= cap:
                break
            if (u.id is not None and u.type in kinds and u.id not in self.unit_ids
                    and u.id not in taken and self.state == "gather"):
                self.unit_ids.append(u.id)

    @staticmethod
    def _kinds(view: View) -> tuple[list, int]:
        """(the unit types the squad takes, how many): Rangers or light tanks
        that aren't the army's main tank, up to ``SNIPE_MAX_UNITS``; with
        neither (the Soviets) ``SNIPE_HEAVY_UNITS`` heavy tanks."""
        main_tank = C.resolve(C.TANK_PREFERENCE, view.available)
        light = [k for k in C.SNIPE_UNIT_TYPES if k != main_tank and k in view.available]
        if light:
            return light, C.SNIPE_MAX_UNITS
        if main_tank == "3tnk":
            return ["3tnk"], C.SNIPE_HEAVY_UNITS
        return [k for k in C.SNIPE_UNIT_TYPES if k != main_tank], C.SNIPE_MAX_UNITS

    def _wanted(self, view: View) -> int:
        """Units a run needs: ``SNIPE_UNITS``, or ``SNIPE_HEAVY_UNITS`` of heavy tanks."""
        kinds, _ = self._kinds(view)
        return C.SNIPE_HEAVY_UNITS if kinds == ["3tnk"] else C.SNIPE_UNITS

    def production(self, view: View, defending: bool) -> list[dict]:
        """One Ranger / light tank at a time while the squad is short (not
        while defending, nor with a harvester or war factory missing)."""
        kinds, _ = self._kinds(view)
        if kinds == ["3tnk"]:
            return []                          # heavy tanks come from the army's own production
        kind = C.resolve(tuple(kinds), view.available)
        if (kind is None or defending or view.game_seconds < C.SNIPE_TRAIN_FROM_SECONDS
                or not view.has_building(C.WAR_FACTORY) or len(self.unit_ids) >= C.SNIPE_UNITS
                or view.queued_count(kind) or view.cash + view.ore < C.FUNDS_LOW):
            return []
        return [orders.train(kind, count=1)]

    # -- the run --------------------------------------------------------------------

    def _members(self, view: View) -> list:
        ids = set(self.unit_ids)
        return [u for u in view.combat_units if u.id in ids and u.cell is not None]

    def _set(self, state: str, t: float) -> None:
        self.state, self._since, self._legs = state, t, None

    def target_spot(self, view: View):
        """The edge of their main ore field out of their turrets' reach and,
        when the field allows, off the front of their base (the scouts' avoid
        zone): its ore cell nearest the heart there — the flank side."""
        from .raid import turret_zone
        from .scouting import ScoutCorps

        heart = enemy_ore_field(view)
        if heart is None:
            return None
        zone = turret_zone(view)
        r2 = (C.SNIPE_FIELD_CELLS * 1.5) ** 2
        ore = [c for c in view.ore_cells if dist2(c, heart) <= r2 and c not in zone]
        front = ScoutCorps().avoid(view)
        flank = [c for c in ore if c not in front]
        pool = flank or [c for c in ore if dist2(c, heart) <= C.SNIPE_FIELD_CELLS ** 2]
        return min(pool, key=lambda c: dist2(c, heart)) if pool else None

    @staticmethod
    def route(view: View, origin, spot):
        """Round their sight lines: out of the chokes between the bases, the
        front of their base and their turrets' reach (the scouts' avoid zone
        plus raid.turret_zone); with no way round that, round the turrets only."""
        from .raid import raid_route, turret_zone
        from .scouting import ScoutCorps

        flank = raid_route(view, origin, spot, zone=frozenset(turret_zone(view) | ScoutCorps().avoid(view)))
        return flank if flank is not None else raid_route(view, origin, spot)

    def orders(self, view: View, push_on: bool = False) -> list[dict]:
        members = self._members(view)
        t = view.game_seconds
        if not members:
            if self.state != "gather":
                self._end(t)
            return []
        ids = [u.id for u in members][:orders.MAX_UNITS_PER_ORDER]
        centre = (sum(u.cell[0] for u in members) // len(members), sum(u.cell[1] for u in members) // len(members))
        home = view.base_center
        if self.state == "gather":
            due = push_on or t - self._last_run_end >= C.SNIPE_EVERY_SECONDS
            if len(members) >= self._wanted(view) and t >= C.SNIPE_FROM_SECONDS and due:
                spot = self.target_spot(view)
                if spot is not None:
                    route = self.route(view, centre, spot)
                    if route is not None:
                        self._set("go", t)
                        self._legs = list(route) + [spot]
                        self._launched_at, self._launch_size = t, len(members)
                        self.runs += 1
            if self.state == "gather":
                return []
        if self.state in ("go", "hit"):
            if self._should_leave(view, members, t):
                self._set("leave", t)
            else:
                harv = [a for a in view.enemy_units if a.type in C.HARVESTER and a.cell is not None and not a.wreck
                        and a.id is not None and dist2(a.cell, centre) <= C.SNIPE_HIT_CELLS ** 2]
                if self.state == "hit" and self._target is not None and not any(a.id == self._target for a in harv):
                    self.kills += 1                      # gone under our guns: one down, out we go
                    self._set("leave", t)
                elif harv:
                    if self.state != "hit":
                        self._set("hit", t)
                    target = next((a for a in harv if a.id == self._target), None) or min(
                        harv, key=lambda a: dist2(a.cell, centre))
                    self._target = target.id
                    return [orders.attack(ids, target.id)]
                elif self.state == "go":
                    while self._legs and dist2(centre, self._legs[0]) <= 9:
                        self._legs.pop(0)
                    if self._legs:
                        return [orders.move(ids, list(self._legs[0]))]
                    if t - self._since >= C.SNIPE_WAIT_SECONDS:   # at the field, nothing to shoot: home
                        self._set("leave", t)
                    return []
        if self.state == "leave":
            if home is None or dist2(centre, home) <= C.SNIPE_HOME_CELLS ** 2:
                self._end(t)
                return []
            return [orders.move(ids, list(home))]
        return []

    def _should_leave(self, view: View, members, t: float) -> bool:
        from . import strength

        if self._launched_at is not None and t - self._launched_at >= C.SNIPE_TIMEOUT_SECONDS:
            return True
        if self.state == "hit" and t - self._since >= C.SNIPE_HIT_SECONDS:
            return True
        if self._launch_size and len(members) <= (1 - C.SNIPE_LOSS_SHARE) * self._launch_size:
            return True
        r2 = C.SNIPE_THREAT_CELLS ** 2
        foes = [a for a in view.enemy_units if a.cell is not None and not a.wreck and a.type not in C.HARVESTER
                and any(dist2(a.cell, u.cell) <= r2 for u in members)]
        foes += [b for b in view.enemy_buildings if b.type in C.TURRETS and b.cell is not None and not b.wreck
                 and any(dist2(b.cell, u.cell) <= r2 for u in members)]
        return bool(foes) and strength.odds(view, [u.type for u in members], [a.type for a in foes]) < 1.0

    def _end(self, t: float) -> None:
        self._set("gather", t)
        self._last_run_end, self._target, self._launched_at = t, None, None
