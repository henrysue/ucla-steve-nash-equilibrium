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

Before any of that, the fight in front of it (:meth:`Raid.situation`):
enemy fighters within ``RAID_THREAT_RADIUS_CELLS`` are defending, so the
squad kills them first (weakest first) and only then the harvesters; if they
are worth more than ``RAID_OUTMATCH_RATIO`` x the squad, it pulls back,
fighting, to our staging point. Either way it has drawn their army: for
``DISTRACTION_SECONDS`` the main army may attack down the centre
(brain.py), and once the main army is in a fight the squad attack-moves into
it — the second prong. (A lone raider used to keep shooting a harvester while
infantry shot it, and die.)

The squad forms only after ``RAID_START_SECONDS``, only when it has a target,
and only if ``MIN_DEFENDERS`` stay with the army. It is released back to the
army when it has lost half its members, has no target, or the base needs
defending; then it waits ``RAID_COOLDOWN_SECONDS`` before forming again.
The army code leaves the squad's units alone while it exists.

:class:`RaidCorps` runs the squads. With healthy macro (``HEALTHY_HARVESTERS``
harvesters, not outgunned) more squads form the longer the game goes: a
second from ``RAID_EXTRA_START_SECONDS``, one more every
``RAID_EXTRA_EVERY_SECONDS`` up to ``RAID_MAX_SQUADS``, each a little bigger,
re-forming sooner, and never taking past ``RAID_MAX_ARMY_SHARE`` of the
fighters. Each squad picks a target ``RAID_SPREAD_CELLS`` from the others',
and small enemy groups in the field (their scouts and pickets) are targets
too. Unhealthy again: the extra squads rejoin the army.
"""

from __future__ import annotations

from . import config as C
from . import orders
from .modes import DEFENSIVE, Mode
from .observe import Actor, View, dist2, enemy_ore_field, enemy_spawn_guess


class Raid:
    def __init__(self) -> None:
        self.unit_ids: list[int] = []
        self._formed_size = 0
        self._released_at: float | None = None
        self._advice = None
        self._drawing_at: float | None = None   # last time enemy fighters were on the squad
        self._via: tuple[int, int] | None = None  # flank waypoint (macro enemy), until reached
        self._visited: dict[tuple[int, int], float] = {}   # remote field -> when the squad was last there
        self._via_done = False
        self.avoid: list[tuple[int, int]] = []    # the other squads' targets (RaidCorps)
        self.cooldown = C.RAID_COOLDOWN_SECONDS
        self.size_bonus = 0
        self.limit: int | None = None             # most units this squad may take (extra squads)

    def update(self, view: View, mode: Mode, scout_id: int | None, advice=None, taken: set = frozenset()) -> None:
        """Drop dead members; release or form the squad."""
        self._advice = advice
        alive = {u.id for u in view.combat_units}
        self.unit_ids = [i for i in self.unit_ids if i in alive]
        if self.unit_ids:
            lost_half = len(self.unit_ids) * 2 <= self._formed_size
            # In an attack the squad stays: it is the second prong (orders()).
            if (mode in DEFENSIVE or mode is Mode.RETREAT or lost_half
                    or _target(view, self._members(view), self._advice, avoid=self.avoid) is None):
                self._release(view)
            return
        if self._may_form(view, mode):
            self._form(view, scout_id, taken)

    def target(self, view: View):
        """The squad's current target (a harvester id or a cell), for the log."""
        members = self._members(view)
        return _target(view, members, self._advice, self._visited, self.avoid) if members else None

    def situation(self, view: View) -> tuple[str, list[Actor], float]:
        """("none" | "clear" | "fight" | "outmatched", the enemy fighters on
        the squad, their value)."""
        members = self._members(view)
        if not members:
            return "none", [], 0.0
        centre = _centroid(members)
        r2 = C.RAID_THREAT_RADIUS_CELLS ** 2
        threats = [a for a in view.enemy_units if a.cell is not None and not a.wreck and a.id is not None
                   and a.type not in C.HARVESTER and dist2(a.cell, centre) <= r2]
        if not threats:
            return "clear", [], 0.0
        cost = lambda a: view.unit_costs.get(a.type, C.DEFAULT_UNIT_COST)  # noqa: E731
        theirs, ours = sum(cost(a) for a in threats), sum(cost(u) for u in members)
        return ("outmatched" if theirs > C.RAID_OUTMATCH_RATIO * ours else "fight"), threats, theirs

    def distracting(self, view: View) -> bool:
        """Enemy fighters have been on the squad recently: their army is busy."""
        return self._drawing_at is not None and view.game_seconds - self._drawing_at <= C.DISTRACTION_SECONDS

    def orders(self, view: View, main_fight: tuple[int, int] | None = None) -> list[dict]:
        members = self._members(view)
        if not members:
            return []
        ids = [u.id for u in members]
        state, threats, value = self.situation(view)
        if state != "clear" and value >= C.DISTRACTION_MIN_VALUE:
            self._drawing_at = view.game_seconds
        if state == "outmatched":
            from .scout import choke_cell
            back = choke_cell(view) or view.base_center
            return [orders.attack_move(ids, list(back))] if back else []
        if state == "fight":
            weakest = min(threats, key=lambda a: (a.hp if a.hp else 100, dist2(a.cell, _centroid(members))))
            return [orders.attack(ids, weakest.id)]
        if main_fight is not None:
            return [orders.attack_move(ids, list(main_fight))]   # the second prong
        centre = _centroid(members)
        for s, _ in list(self._visited.items()) + [(c, 0) for c in _remote(view)]:
            if centre is not None and dist2(centre, s) <= 36:
                self._visited[s] = view.game_seconds
        target = _target(view, members, self._advice, self._visited, self.avoid)
        if target is None:
            return []
        if isinstance(target, int):
            return [orders.attack(ids, target)]
        via = self._flank(view, members, target)
        if via is not None:
            return [orders.attack_move(ids, list(via)), orders.attack_move(ids, list(target), queued=True)]
        return [orders.attack_move(ids, list(target))]

    def _flank(self, view: View, members: list[Actor], target) -> tuple[int, int] | None:
        """Against a macro-heavy enemy (the advisor's read), round the side
        with the fewest known enemies to a far target, then straight in."""
        if getattr(self._advice, "enemy_style", None) != "macro" or self._via_done:
            return None
        centre = _centroid(members)
        if self._via is None:
            if centre is None or dist2(centre, target) < C.RAID_FLANK_MIN_CELLS ** 2:
                return None
            from .executor import flank_waypoint
            r2 = C.FLANK_DANGER_RADIUS_CELLS ** 2
            enemies = [a.cell for a in list(view.enemy_units) + list(view.known_enemy_units)
                       + list(view.known_enemy_buildings) if a.cell is not None and a.type not in C.HARVESTER]
            sides = [w for w in (flank_waypoint(view, target, s) for s in (1, -1)) if w is not None]
            self._via = min(sides, key=lambda w: sum(1 for c in enemies if dist2(c, w) <= r2), default=None)
        if self._via is not None and centre is not None and dist2(centre, self._via) <= C.FLANK_ARRIVED_CELLS ** 2:
            self._via_done = True
            return None
        return self._via

    # -- internals -------------------------------------------------------------

    def _members(self, view: View) -> list[Actor]:
        ids = set(self.unit_ids)
        return [u for u in view.combat_units if u.id in ids]

    def _may_form(self, view: View, mode: Mode) -> bool:
        if view.game_seconds < C.RAID_START_SECONDS or mode in DEFENSIVE or mode is Mode.RETREAT or mode is Mode.ALL_IN:
            return False
        if self._released_at is not None and view.game_seconds - self._released_at < self.cooldown:
            return False
        return _target(view, [], self._advice, self._visited, self.avoid) is not None

    def _form(self, view: View, scout_id: int | None, taken: set = frozenset()) -> None:
        fighters = [u for u in view.combat_units if u.id is not None and u.id != scout_id and u.id not in taken]
        if len(fighters) < C.MIN_DEFENDERS + C.RAID_SQUAD_SIZE:
            return  # can't spare them
        size = C.RAID_SQUAD_SIZE_MACRO if getattr(self._advice, "enemy_style", None) == "macro" else C.RAID_SQUAD_SIZE
        # The squad grows with the army: a big army idling at home annoys them more.
        size = max(size, min(C.RAID_SQUAD_MAX, len(fighters) // C.RAID_ARMY_SHARE))
        if len(fighters) < C.MIN_DEFENDERS + size:
            size = C.RAID_SQUAD_SIZE      # can't spare the bigger party
        size = min(size + self.size_bonus, len(fighters) - C.MIN_DEFENDERS)
        if self.limit is not None:
            size = min(size, self.limit)
        squad: list[int] = []
        for kind in C.RAID_UNIT_TYPES:  # rockets first
            squad += [u.id for u in fighters if u.type == kind and u.id not in squad]
        self.unit_ids = squad[:size]
        self._via, self._via_done = None, False
        if len(self.unit_ids) < C.RAID_MIN_SIZE:
            self.unit_ids = []
            return
        self._formed_size = len(self.unit_ids)

    def _release(self, view: View) -> None:
        self.unit_ids = []
        self._formed_size = 0
        self._released_at = view.game_seconds


def healthy(view: View) -> bool:
    """Macro is healthy: ``HEALTHY_HARVESTERS`` harvesters and not outgunned."""
    from .brain import outgunned

    return len(view.units_of(C.HARVESTER)) >= C.HEALTHY_HARVESTERS and not outgunned(view)


def raid_steps(view: View) -> int:
    """How far into a healthy game we are: 0 before ``RAID_EXTRA_START_SECONDS``
    or when unhealthy, then 1 and one more every ``RAID_EXTRA_EVERY_SECONDS``."""
    t = view.game_seconds
    if t < C.RAID_EXTRA_START_SECONDS or not healthy(view):
        return 0
    return 1 + int((t - C.RAID_EXTRA_START_SECONDS) // C.RAID_EXTRA_EVERY_SECONDS)


def squads_allowed(view: View) -> int:
    return min(C.RAID_MAX_SQUADS, 1 + raid_steps(view))


class RaidCorps:
    """The raid squads (see the module docstring). The first squad is the
    original one: it takes the advisor's raid target and always joins the
    main army's fight; the extra squads join only from nearby."""

    def __init__(self) -> None:
        self.squads: list[Raid] = [Raid()]

    @property
    def unit_ids(self) -> list[int]:
        return [i for s in self.squads for i in s.unit_ids]

    @unit_ids.setter
    def unit_ids(self, ids) -> None:
        self.squads[0].unit_ids = list(ids)

    @property
    def active(self) -> int:
        return sum(1 for s in self.squads if s.unit_ids)

    def update(self, view: View, mode: Mode, scout_id: int | None, advice=None, taken: set = frozenset()) -> None:
        steps = raid_steps(view)
        allowed = min(C.RAID_MAX_SQUADS, 1 + steps)
        while len(self.squads) < allowed:
            self.squads.append(Raid())
        fighters = [u for u in view.combat_units if u.id is not None and u.id != scout_id and u.id not in taken]
        busy = set(taken)
        claimed: list[tuple[int, int]] = []
        for i, s in enumerate(self.squads):
            if i >= allowed:
                if s.unit_ids:
                    s._release(view)              # unhealthy again: back to the army
                continue
            s.avoid = list(claimed)
            s.cooldown = C.RAID_LATE_COOLDOWN_SECONDS if steps else C.RAID_COOLDOWN_SECONDS
            s.size_bonus = max(0, steps - 1)
            raiders = len(busy - set(taken))
            s.limit = (None if i == 0 else
                       max(0, int(C.RAID_MAX_ARMY_SHARE * len(fighters)) - raiders))
            s.update(view, mode, scout_id, advice if i == 0 else None, taken=busy)
            busy |= set(s.unit_ids)
            cell = _cell_of(view, s.target(view))
            if cell is not None:
                claimed.append(cell)

    def target(self, view: View):
        return self.squads[0].target(view)

    def targets(self, view: View) -> list:
        return [s.target(view) for s in self.squads if s.unit_ids]

    def situation(self, view: View) -> tuple[str, list[Actor], float]:
        rank = {"none": 0, "clear": 1, "fight": 2, "outmatched": 3}
        state, threats, value = "none", [], 0.0
        for s in self.squads:
            st, th, v = s.situation(view)
            if rank[st] > rank[state]:
                state = st
            threats += th
            value += v
        return state, threats, value

    def distracting(self, view: View) -> bool:
        return any(s.distracting(view) for s in self.squads)

    def orders(self, view: View, main_fight: tuple[int, int] | None = None) -> list[dict]:
        out: list[dict] = []
        j2 = C.RAID_JOIN_FIGHT_CELLS ** 2
        for i, s in enumerate(self.squads):
            fight = main_fight
            if i > 0 and fight is not None:
                centre = _centroid(s._members(view))
                if centre is None or dist2(centre, fight) > j2:
                    fight = None                  # far off on its own errand: keep at it
            out += s.orders(view, fight)
        return out


def _cell_of(view: View, target) -> tuple[int, int] | None:
    if isinstance(target, int):
        hit = next((a for a in view.enemy_units if a.id == target and a.cell is not None), None)
        return hit.cell if hit is not None else None
    return tuple(target) if target is not None else None


def _picket(view: View, origin, free) -> Actor | None:
    """The nearest small enemy group out in the field (their scouts and
    pickets, at least ``RAID_PICKET_SIDE`` of the way from their base): worth
    ``RAID_PICKET_MAX_VALUE`` or less with its friends close by."""
    base, enemy = view.base_center, enemy_spawn_guess(view)
    if base is None or enemy is None:
        return None
    side2 = dist2(base, enemy) * C.RAID_PICKET_SIDE ** 2
    r2 = C.RAID_PICKET_RADIUS_CELLS ** 2
    cost = lambda a: view.unit_costs.get(a.type, C.DEFAULT_UNIT_COST)  # noqa: E731
    field = [a for a in view.enemy_units if a.id is not None and a.cell is not None and not a.wreck
             and a.type not in C.HARVESTER and a.type != C.MCV and dist2(a.cell, enemy) >= side2]
    small = [a for a in field if free(a.cell)
             and sum(cost(b) for b in field if dist2(b.cell, a.cell) <= r2) <= C.RAID_PICKET_MAX_VALUE]
    return min(small, key=lambda a: dist2(a.cell, origin)) if small else None


def _remote(view: View) -> list[tuple[int, int]]:
    from .scouting import remote_sites
    return remote_sites(view)


def _target(view: View, members: list[Actor], advice=None, visited: dict | None = None,
            avoid=()) -> int | tuple[int, int] | None:
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
    s2 = C.RAID_SPREAD_CELLS ** 2

    def free(cell) -> bool:
        """Not another squad's target."""
        return all(dist2(cell, a) > s2 for a in avoid)

    def nearest(actors):
        actors = [a for a in actors if a.cell is not None and free(a.cell)]
        return min(actors, key=lambda a: dist2(a.cell, origin)) if actors else None

    visible = nearest(a for a in view.enemy_units if a.type in C.HARVESTER and a.id is not None)
    if visible is not None:
        return visible.id
    # Their scouts and pickets out in the field: cheap kills that blind them.
    picket = _picket(view, origin, free)
    if picket is not None:
        return picket.id
    # A known enemy expansion, lightly defended: its refinery is where its
    # harvesters unload, and a fresh expansion has the least protection.
    soft = [c for c in enemy_expansions(view)
            if expansion_defenses(view, c) <= C.RAID_EXPANSION_MAX_DEFENSES and free(c)]
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
    # Nothing seen yet: a remote field they may have expanded to (not their
    # defended main field: f333a770's squads went there and died), nearest
    # first and not one we checked lately; else their main field.
    t = view.game_seconds
    due = [s for s in _remote(view)
           if t - (visited or {}).get(s, -1e9) >= C.REMOTE_SITE_RECHECK_SECONDS and free(s)]
    if due:
        return min(due, key=lambda s: dist2(s, origin))
    main = enemy_ore_field(view)
    return main if main is None or free(main) else None


def _centroid(actors: list[Actor]) -> tuple[int, int] | None:
    cells = [a.cell for a in actors if a.cell is not None]
    if not cells:
        return None
    return (sum(c[0] for c in cells) // len(cells), sum(c[1] for c in cells) // len(cells))
