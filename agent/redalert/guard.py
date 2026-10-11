"""The expansion garrison, and which units count as "at home".

The garrison: units posted at our newest expansion ("bring support to
capture and defend an expansion"); see :class:`Garrison`. Squads claim only
units at home (:func:`_at_home`), never from the front. (A home guard of two
riflemen at the power plants used to live here; the idle units nearby —
executor.respond — now deal with small raids on the base.)
"""

from __future__ import annotations

from . import config as C
from . import orders
from .modes import DEFENSIVE, Mode
from .observe import View, dist2

_POST_RADIUS_CELLS = 2


def _at_home(view: View, cell) -> bool:
    """Near one of our yards or the massing point: a unit not out fighting.
    Guards and garrisons claim only these, never units at the front (match
    fd507a58: the garrison pulled riflemen out of a winning attack and walked
    them back to the expansion)."""
    from .scout import choke_cell, our_yards

    homes = list(our_yards(view))
    stage = choke_cell(view)
    if stage is not None:
        homes.append(stage)
    r2 = C.SQUAD_CLAIM_RADIUS_CELLS ** 2
    return cell is not None and any(dist2(cell, h) <= r2 for h in homes)


class Garrison:
    """A small outpost at a forward expansion (one nearer the enemy than our
    main base): ``GARRISON_SIZE`` infantry posted on its enemy-facing side.
    Expansions behind or beside the base get none.

    Available at any notice: they are posted only while the army masses in
    calm (BUILD / OPEN). An attack (the rules' or the advisor's), a defense,
    a retreat, or a real fight near the army (executor.army_contact) takes them
    straight back — unless their own expansion is the one attacked, and then
    they fight the attackers there. Match 0ee93a05: four units, tanks first,
    stood at an expansion through every fight with the rusher.
    """

    def __init__(self) -> None:
        self.unit_ids: list[int] = []
        self._guarding: dict[int, int] = {}    # unit -> the actor it was told to guard

    def update(self, view: View, mode: Mode, expansion, taken: set) -> None:
        alive = {u.id for u in view.combat_units}
        self.unit_ids = [i for i in self.unit_ids if i in alive]
        self._prune()
        site = forward_site(view, expansion)
        if site is None:
            self._release()
            return
        threatened = self._threatened(view, site)
        if not threatened and (mode not in (Mode.BUILD, Mode.OPEN) or self._army_fighting(view)):
            self._release()              # needed elsewhere: back with the army at once
            return
        if threatened or len(self.unit_ids) >= C.GARRISON_SIZE:
            return
        free = [u for u in view.combat_units if u.id is not None and u.id not in taken
                and u.id not in self.unit_ids and _at_home(view, u.cell)]
        if len(free) < C.MIN_DEFENDERS + C.GARRISON_SIZE:
            return                       # the army can't spare them yet
        for kind in C.GARRISON_TYPES:
            for u in free:
                if len(self.unit_ids) >= C.GARRISON_SIZE:
                    return
                if u.type == kind and u.id not in self.unit_ids:
                    self.unit_ids.append(u.id)

    def _release(self) -> None:
        self.unit_ids = []
        self._prune()

    def _prune(self) -> None:
        """Forget the guard orders of units no longer in the outpost: one
        claimed again later is told to guard again (it used to be skipped as
        "already guarding" and stood idle wherever it was)."""
        self._guarding = {i: w for i, w in self._guarding.items() if i in self.unit_ids}

    @staticmethod
    def _ward(view: View, expansion) -> int | None:
        """What the outpost guards: the MCV on its way to a forward site, else
        our construction yard at the forward expansion; None without one."""
        if getattr(expansion, "escorting", False) and getattr(expansion, "mcv_id", None) is not None:
            if any(u.id == expansion.mcv_id for u in view.units):
                return expansion.mcv_id
        site = forward_site(view, expansion)
        if site is None:
            return None
        yards = [b for b in view.buildings if b.type == C.CONSTRUCTION_YARD and b.cell is not None and not b.wreck
                 and b.id is not None and dist2(b.cell, site) <= C.GARRISON_LEASH_CELLS ** 2]
        return min(yards, key=lambda b: dist2(b.cell, site)).id if yards else None

    def _army_fighting(self, view: View) -> bool:
        from .executor import army_contact

        ids = set(self.unit_ids)
        return bool(army_contact(view, [u for u in view.combat_units if u.id not in ids]))

    @staticmethod
    def _threatened(view: View, site) -> bool:
        """Enemies near the outpost's expansion."""
        from .scout import enemy_threat_near_base

        r2 = C.GARRISON_LEASH_CELLS ** 2
        return any(a.cell is not None and dist2(a.cell, site) <= r2 for a in enemy_threat_near_base(view))

    def post(self, view: View, expansion) -> tuple[int, int] | None:
        """A few cells out from the forward expansion toward the enemy."""
        from .observe import enemy_spawn_guess

        site = forward_site(view, expansion)
        if site is None:
            return None
        enemy = enemy_spawn_guess(view)
        if enemy is None:
            return site
        dx, dy = enemy[0] - site[0], enemy[1] - site[1]
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        return round(site[0] + dx / length * 3), round(site[1] + dy / length * 3)

    def orders(self, view: View, expansion) -> list[dict]:
        post = self.post(view, expansion)
        if post is None or not self.unit_ids:
            return []
        ids = set(self.unit_ids)
        # A threat at the expansion: attack it (nearest the post first).
        from .scout import enemy_threat_near_base
        leash2 = C.GARRISON_LEASH_CELLS ** 2
        threats = [a.cell for a in enemy_threat_near_base(view) if a.cell is not None and dist2(a.cell, post) <= leash2]
        if threats:
            self._guarding.clear()           # off to fight: guard again after
            return [orders.attack_move(list(self.unit_ids), list(min(threats, key=lambda c: dist2(c, post))))]
        # With the engine's guard order: follow the MCV on its way, then stay
        # with the expansion's construction yard and fight what comes at it.
        ward = self._ward(view, expansion)
        if ward is not None:
            fresh = [i for i in self.unit_ids if self._guarding.get(i) != ward]
            for i in fresh:
                self._guarding[i] = ward
            return [orders.guard(fresh, ward)] if fresh else []
        r2 = 3 ** 2
        away = [u.id for u in view.combat_units if u.id in ids and u.idle and dist2(u.cell, post) > r2]
        return [orders.attack_move(away, list(post))] if away else []


def forward_site(view: View, expansion) -> tuple[int, int] | None:
    """Where the outpost goes: the forward site an MCV is on its way to (the
    outpost escorts it; the army stays at the main choke), else our newest
    expansion that lies nearer the enemy than our main base, else None."""
    from .observe import enemy_spawn_guess

    enemy, base = enemy_spawn_guess(view), view.base_center
    if enemy is None or base is None:
        return None
    if getattr(expansion, "escorting", False) and expansion.site is not None:
        return expansion.site
    ahead = [s for s in expansion.sites if dist2(s, enemy) < dist2(base, enemy)]
    return ahead[-1] if ahead else None
