"""A small infantry squad that hunts enemy harvesters.

Harvesters are the enemy's whole income: each one lost costs them 1100
credits and the ore it would have brought in, and a few cheap infantry can
kill one far from its army. So, alongside the main army, a squad of
``RAID_SQUAD_SIZE`` infantry (rocket soldiers first: harvesters are armored)
goes after them:

1. a visible enemy harvester -> ``attack`` it by id (focus fire);
2. else a known, lightly defended enemy expansion -> ``attack_move`` to it;
2b. else the last cell of a harvester we remember -> ``attack_move`` there;
3. else a known enemy refinery, where harvesters unload -> ``attack_move``;
4. else the enemy's ore field from the map layout -> ``attack_move``.

The squad forms only after ``RAID_START_SECONDS``, only when it has a target,
and only if ``MIN_DEFENDERS`` stay with the army. It is released back to the
army when it has lost half its members, has no target, or the base needs
defending; then it waits ``RAID_COOLDOWN_SECONDS`` before forming again.
The army code leaves the squad's units alone while it exists.
"""

from __future__ import annotations

from . import config as C
from . import orders
from .modes import DEFENSIVE, Mode
from .observe import Actor, View, dist2, enemy_ore_field


class Raid:
    def __init__(self) -> None:
        self.unit_ids: list[int] = []
        self._formed_size = 0
        self._released_at: float | None = None
        self._advice = None

    def update(self, view: View, mode: Mode, scout_id: int | None, advice=None, taken: set = frozenset()) -> None:
        """Drop dead members; release or form the squad."""
        self._advice = advice
        alive = {u.id for u in view.combat_units}
        self.unit_ids = [i for i in self.unit_ids if i in alive]
        if self.unit_ids:
            lost_half = len(self.unit_ids) * 2 <= self._formed_size
            # An attack (ALL_IN) takes the raiders along: commit together.
            if (mode in DEFENSIVE or mode is Mode.RETREAT or mode is Mode.ALL_IN or lost_half
                    or _target(view, self._members(view), self._advice) is None):
                self._release(view)
            return
        if self._may_form(view, mode):
            self._form(view, scout_id, taken)

    def target(self, view: View):
        """The squad's current target (a harvester id or a cell), for the log."""
        members = self._members(view)
        return _target(view, members, self._advice) if members else None

    def orders(self, view: View) -> list[dict]:
        members = self._members(view)
        if not members:
            return []
        target = _target(view, members, self._advice)
        if target is None:
            return []
        ids = [u.id for u in members]
        if isinstance(target, int):
            return [orders.attack(ids, target)]
        return [orders.attack_move(ids, list(target))]

    # -- internals -------------------------------------------------------------

    def _members(self, view: View) -> list[Actor]:
        ids = set(self.unit_ids)
        return [u for u in view.combat_units if u.id in ids]

    def _may_form(self, view: View, mode: Mode) -> bool:
        if view.game_seconds < C.RAID_START_SECONDS or mode in DEFENSIVE or mode is Mode.RETREAT or mode is Mode.ALL_IN:
            return False
        if self._released_at is not None and view.game_seconds - self._released_at < C.RAID_COOLDOWN_SECONDS:
            return False
        return _target(view, [], self._advice) is not None

    def _form(self, view: View, scout_id: int | None, taken: set = frozenset()) -> None:
        fighters = [u for u in view.combat_units if u.id is not None and u.id != scout_id and u.id not in taken]
        if len(fighters) < C.MIN_DEFENDERS + C.RAID_SQUAD_SIZE:
            return  # can't spare them
        squad: list[int] = []
        for kind in C.RAID_UNIT_TYPES:  # rockets first
            squad += [u.id for u in fighters if u.type == kind and u.id not in squad]
        self.unit_ids = squad[:C.RAID_SQUAD_SIZE]
        if len(self.unit_ids) < C.RAID_MIN_SIZE:
            self.unit_ids = []
            return
        self._formed_size = len(self.unit_ids)

    def _release(self, view: View) -> None:
        self.unit_ids = []
        self._formed_size = 0
        self._released_at = view.game_seconds


def _target(view: View, members: list[Actor], advice=None) -> int | tuple[int, int] | None:
    """The advisor's pick when it is usable (a visible enemy harvester's id,
    or a cell), else, nearest the squad (or our base, before it exists): a
    visible harvester's id, a known lightly defended enemy expansion, a
    remembered harvester, a known refinery — else the enemy ore field."""
    from .expand import enemy_expansions, expansion_defenses

    picked = getattr(advice, "raid_target_id", None)
    if picked is not None and any(a.id == picked and a.type in C.HARVESTER for a in view.enemy_units):
        return picked
    if getattr(advice, "raid_target_cell", None) is not None:
        return advice.raid_target_cell
    origin = _centroid(members) or view.base_center

    def nearest(actors):
        actors = [a for a in actors if a.cell is not None]
        return min(actors, key=lambda a: dist2(a.cell, origin)) if actors else None

    visible = nearest(a for a in view.enemy_units if a.type in C.HARVESTER and a.id is not None)
    if visible is not None:
        return visible.id
    # A known enemy expansion, lightly defended: its refinery is where its
    # harvesters unload, and a fresh expansion has the least protection.
    soft = [c for c in enemy_expansions(view) if expansion_defenses(view, c) <= C.RAID_EXPANSION_MAX_DEFENSES]
    if soft:
        return min(soft, key=lambda c: dist2(c, origin))
    remembered = nearest(a for a in view.known_enemy_units if a.type in C.HARVESTER)
    if remembered is not None:
        return remembered.cell
    # A refinery, unless defenses guard it: raids hunt soft targets.
    refinery = nearest(a for a in view.enemy_buildings + view.known_enemy_buildings if a.type == C.REFINERY
                       and a.cell is not None and expansion_defenses(view, a.cell) <= C.RAID_EXPANSION_MAX_DEFENSES)
    if refinery is not None:
        return refinery.cell
    # Nothing seen yet: their ore field from the map, where harvesters work.
    return enemy_ore_field(view)


def _centroid(actors: list[Actor]) -> tuple[int, int] | None:
    cells = [a.cell for a in actors if a.cell is not None]
    if not cells:
        return None
    return (sum(c[0] for c in cells) // len(cells), sum(c[1] for c in cells) // len(cells))
