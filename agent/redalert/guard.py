"""The expansion garrison, and which units count as "at home".

The garrison: units posted at our newest expansion ("bring support to
capture and defend an expansion"); see :class:`Garrison`. Squads claim only
units at home (:func:`_at_home`), never from the front. (A home guard of two
riflemen at the power plants used to live here; home responders —
executor.home_response — now deal with small raids on the base.)
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
    """Units posted at our newest expansion: "bring support to capture and
    defend an expansion". ``GARRISON_SIZE`` fighters (tanks first) are claimed
    once an expansion is deployed and we can spare them, posted on its
    enemy-facing side, and kept out of the army. When the expansion is
    complete (refinery and defense) and the army pushes (ALL_IN), they are
    released to join the push; they are claimed again after it.
    """

    def __init__(self) -> None:
        self.unit_ids: list[int] = []

    def update(self, view: View, mode: Mode, expansion, taken: set) -> None:
        alive = {u.id for u in view.combat_units}
        self.unit_ids = [i for i in self.unit_ids if i in alive]
        if not expansion.sites:
            self.unit_ids = []
            return
        if mode in DEFENSIVE and not self._threatened(view, expansion):
            self.unit_ids = []           # the main base is under attack: go help
            return
        if mode is Mode.ALL_IN and expansion.complete(view):
            self.unit_ids = []           # join the push
            return
        if mode is Mode.ALL_IN or len(self.unit_ids) >= C.GARRISON_SIZE:
            return   # no claims during an attack: the army commits together
        free = [u for u in view.combat_units if u.id is not None and u.id not in taken
                and u.id not in self.unit_ids and _at_home(view, u.cell)]
        if len(free) < C.MIN_DEFENDERS:
            return
        for kind in C.GARRISON_TYPES:
            for u in free:
                if len(self.unit_ids) >= C.GARRISON_SIZE:
                    return
                if u.type == kind and u.id not in self.unit_ids:
                    self.unit_ids.append(u.id)

    @staticmethod
    def _threatened(view: View, expansion) -> bool:
        """Enemies near one of our expansions (then the garrison stays put)."""
        from .scout import enemy_threat_near_base

        r2 = C.GARRISON_LEASH_CELLS ** 2
        return any(a.cell is not None and dist2(a.cell, s) <= r2
                   for a in enemy_threat_near_base(view) for s in expansion.sites)

    def post(self, view: View, expansion) -> tuple[int, int] | None:
        """A few cells out from the newest expansion toward the enemy."""
        from .observe import enemy_spawn_guess

        if not expansion.sites:
            return None
        site, enemy = expansion.sites[-1], enemy_spawn_guess(view)
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
        r2 = 3 ** 2
        away = [u.id for u in view.combat_units if u.id in ids and u.idle and dist2(u.cell, post) > r2]
        return [orders.attack_move(away, list(post))] if away else []
