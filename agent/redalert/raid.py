"""A small infantry squad that hunts enemy harvesters.

Harvesters are the enemy's whole income: each one lost costs them 1100
credits and the ore it would have brought in, and a few cheap infantry can
kill one far from its army. So, alongside the main army, a squad of
``RAID_SQUAD_SIZE`` infantry (rocket soldiers first: harvesters are armored)
goes after them — only at an enemy expansion our scouts have found
(``ENEMY_EXPANSION_MIN_CELLS`` or more from their main base), never their
main field, where their army and turrets wait (match 52a77fb1). No
expansion found: no raid. At an expansion, nearest first:

1. the advisor's pick there, or a visible enemy harvester -> ``attack`` it by id;
2. else a known, lightly defended enemy expansion -> ``attack_move`` to it;
2b. else the last cell of a harvester we remember there -> ``attack_move``;
3. else a known enemy refinery there, where harvesters unload -> ``attack_move``.

Before any of that, the fight in front of it (:meth:`Raid.situation`):
enemy fighters within ``RAID_THREAT_RADIUS_CELLS`` are defending, so the
squad kills them first (weakest first) and only then the harvesters; if they
beat the squad (strength.odds, the one fight judgment), it breaks off to our
massing point. Either way it has drawn part of their army: for
``DISTRACTION_SECONDS`` the main army is judged against the rest and may
attack away from the raid (brain.py), and once the main army is in a fight
the squad attack-moves into it — the second prong. (A lone raider used to keep shooting a harvester while
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
from .observe import Actor, View, dist2, enemy_spawn_guess


class Raid:
    def __init__(self) -> None:
        self.unit_ids: list[int] = []
        self._formed_size = 0
        self._released_at: float | None = None
        self._advice = None
        self._drawing: tuple | None = None       # (when, enemy ids, their value, where) fighters were last on the squad
        self._via: tuple[int, int] | None = None  # flank waypoint (macro enemy), until reached
        self._via_done = False
        self.avoid: list[tuple[int, int]] = []    # the other squads' targets (RaidCorps)
        self.cooldown = C.RAID_COOLDOWN_SECONDS
        self.size_bonus = 0
        self.limit: int | None = None             # most units this squad may take (extra squads)
        self._start: tuple[float, float] | None = None   # (kills, deaths) when formed
        self.results: list[float] = []            # net trade (kills - losses) of each finished raid
        self._aim: tuple[int, int] | None = None  # the cell it was last sent at
        self.failed: list[tuple[tuple[int, int], float]] = []   # (target, when) of raids that lost
        self.finished: list[dict] = []            # each raid's outcome, until RaidCorps.drain takes it
        self._formed_at = 0.0
        self._armor = False                       # the target needs tanks or siege (raid_spot)
        self._no_target_since: float | None = None   # the formed squad has had no target since
        self._legs: tuple | None = None           # (target, waypoints round the turrets still to walk)

    def update(self, view: View, mode: Mode, scout_id: int | None, advice=None, taken: set = frozenset(),
               may_form: bool = True) -> None:
        """Drop dead members; release or form the squad (``may_form``: not
        while raids are backing off)."""
        self._advice = advice
        alive = {u.id for u in view.combat_units}
        self.unit_ids = [i for i in self.unit_ids if i in alive]
        if self.unit_ids:
            lost_half = len(self.unit_ids) * 2 <= self._formed_size
            # Outmatched: released (the target marked failed), not walked back
            # and forth between the choke and the target (the audit).
            if self.situation(view)[0] == "outmatched":
                if self._aim is not None:
                    self.failed.append((self._aim, view.game_seconds))
                self._release(view)
                return
            # No target for a moment (it flickers as their units come and go
            # out of sight) isn't the end: RAID_NO_TARGET_GRACE_SECONDS of it is
            # (5a4bc83f formed and dropped five squads in 25 s, walking them
            # out and back).
            if _target(view, self._members(view), self._advice, avoid=self.avoid) is None:
                self._no_target_since = self._no_target_since or view.game_seconds
            else:
                self._no_target_since = None
            gone = (self._no_target_since is not None
                    and view.game_seconds - self._no_target_since >= C.RAID_NO_TARGET_GRACE_SECONDS)
            # In an attack the squad stays: it is the second prong (orders()).
            if mode in DEFENSIVE or mode is Mode.RETREAT or lost_half or gone:
                self._release(view)
            return
        if may_form and self._may_form(view, mode):
            self._form(view, scout_id, taken)

    def target(self, view: View):
        """The squad's current target (a harvester id or a cell), for the log."""
        members = self._members(view)
        return _target(view, members, self._advice, self.avoid) if members else None

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
        from . import strength

        theirs = strength.value(view, [a.type for a in threats])
        odds = strength.odds(view, [u.type for u in members], [a.type for a in threats])
        return ("outmatched" if odds < C.RAID_OUTMATCH_ODDS else "fight"), threats, theirs

    def drawing(self, view: View) -> tuple | None:
        """(enemy ids, their value, where) of the fighters on the squad in the
        last ``DISTRACTION_SECONDS``, else None."""
        d = self._drawing
        if d is None or view.game_seconds - d[0] > C.DISTRACTION_SECONDS:
            return None
        return d[1], d[2], d[3]

    def distracting(self, view: View) -> bool:
        """Enemy fighters have been on the squad recently: part of their army is busy."""
        return self.drawing(view) is not None

    def orders(self, view: View, main_fight: tuple[int, int] | None = None) -> list[dict]:
        members = self._members(view)
        if not members:
            return []
        ids = [u.id for u in members]
        state, threats, value = self.situation(view)
        if state != "clear" and value >= C.DISTRACTION_MIN_VALUE:
            self._drawing = (view.game_seconds, frozenset(a.id for a in threats), value, _centroid(threats))
        if state == "outmatched":
            # Break off with a plain move (an attack-move stops to shoot every
            # chaser and never gets away), back to the army.
            from .scout import choke_cell
            back = choke_cell(view) or view.base_center
            return [orders.move(ids, list(back))] if back else []
        if state == "fight":
            weakest = min(threats, key=lambda a: (a.hp if a.hp else 100, dist2(a.cell, _centroid(members))))
            return [orders.attack(ids, weakest.id)]
        if main_fight is not None:
            return [orders.attack_move(ids, list(main_fight))]   # the second prong
        target = _target(view, members, self._advice, self.avoid)
        if target is None:
            return []
        self._aim = _cell_of(view, target) or self._aim
        if isinstance(target, int):
            # A harvester by id only once near: an attack on one unit walks
            # past everything else without firing back — from afar the squad
            # attack-moves along its route to it, fighting what it meets.
            cell, centre = _cell_of(view, target), _centroid(members)
            if cell is None or centre is None or dist2(cell, centre) <= C.ATTACK_ID_CELLS ** 2:
                return [orders.attack(ids, target)]
            target = cell
        via = self._flank(view, members, target)
        route = self._round_turrets(view, members, target, via)
        if route:
            return [orders.attack_move(ids, list(route[0]))] + [
                orders.attack_move(ids, list(p), queued=True) for p in route[1:] + [target]]
        return [orders.attack_move(ids, list(target))]

    def _round_turrets(self, view: View, members: list[Actor], target, via=None) -> list:
        """The squad's waypoints to ``target``, kept until walked: round the
        reach of the turrets we know (raid_route) for infantry, the road
        itself for a squad with the tanks to go straight — always in legs, so
        the squad keeps to one road and no terrain splits it."""
        from .terrain import legs, walk_path

        centre = _centroid(members)
        if centre is None:
            return []
        key = (tuple(target), tuple(via) if via is not None else None)
        if self._legs is None or self._legs[0] != key:
            armored = _armored([u.type for u in members]) >= C.RAID_ARMOR_UNITS

            def road(a, b):
                if armored:
                    return legs(walk_path(view, a, tuple(b)), C.ARMY_LEG_CELLS)[:-1]
                return raid_route(view, a, b) or []

            # Through the flank point first when there is one (macro read).
            path = (road(centre, via) + [tuple(via)] + road(tuple(via), target)) if via is not None else road(centre, target)
            self._legs = (key, list(path))
        points = self._legs[1]
        while points and dist2(centre, points[0]) <= 9:
            points.pop(0)
        return points

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
        # Infantry first; tanks or siege only where nothing else gets at it.
        self._armor = False
        if _target(view, [], self._advice, self.avoid) is not None:
            return True
        armored = would_form(view, armor=True)
        if _armored(armored) >= C.RAID_ARMOR_UNITS and _target(view, [], self._advice, self.avoid, armored) is not None:
            self._armor = True
            return True
        return False

    def _form(self, view: View, scout_id: int | None, taken: set = frozenset()) -> None:
        fighters = [u for u in view.combat_units if u.id is not None and u.id != scout_id and u.id not in taken]
        if len(fighters) < C.MIN_DEFENDERS + C.RAID_SQUAD_SIZE:
            return  # can't spare them
        size = C.RAID_SQUAD_SIZE_MACRO if getattr(self._advice, "enemy_style", None) == "macro" else C.RAID_SQUAD_SIZE
        # The squad grows with the army: a big army idling at home annoys them more.
        size = max(size, min(C.RAID_SQUAD_MAX, len(fighters) // C.RAID_ARMY_SHARE))
        # Late, a bigger, more decisive party: a quarter of the fighters
        # (4e9828e1's raids were 8-12 of a 35k army; still within
        # RAID_MAX_ARMY_SHARE for all squads together).
        if view.game_seconds >= C.RAID_LATE_SECONDS:
            size = max(size, min(C.RAID_SQUAD_MAX_LATE, int(C.RAID_LATE_SHARE * len(fighters))))
        # Mid-game, an expansion our scouts found goes in force: the army
        # doesn't just mass at home while their economy grows.
        from .expand import enemy_expansions

        if view.game_seconds >= C.RAID_EXPANSION_SECONDS and enemy_expansions(view):
            size = max(size, C.RAID_EXPANSION_SQUAD)
        if len(fighters) < C.MIN_DEFENDERS + size:
            size = C.RAID_SQUAD_SIZE      # can't spare the bigger party
        size = min(size + self.size_bonus, len(fighters) - C.MIN_DEFENDERS)
        if self.limit is not None:
            size = min(size, self.limit)
        squad: list[int] = []
        if self._armor:                 # only tanks or siege get at the target: they lead
            squad += [u.id for u in fighters if u.type in C.TANK_PREFERENCE or u.type in C.SIEGE_UNITS][:C.RAID_ARMOR_UNITS]
        for kind in C.RAID_UNIT_TYPES:  # rockets first
            squad += [u.id for u in fighters if u.type == kind and u.id not in squad]
        self.unit_ids = squad[:size]
        self._via, self._via_done = None, False
        if len(self.unit_ids) < C.RAID_MIN_SIZE:
            self.unit_ids = []
            return
        self._formed_size = len(self.unit_ids)
        # Its target from the start: outmatched at once, it still marks the
        # spot failed (4e9828e1: three squads died in 4-10 s at the same
        # expansion, each released before it had aimed anywhere).
        self._aim = _cell_of(view, _target(view, self._members(view), self._advice, self.avoid))
        self._legs = None                          # a fresh route (not a released squad's leftover)
        self._start = (view.kills_cost, view.deaths_cost)
        self._formed_at = view.game_seconds

    def _release(self, view: View) -> None:
        if self._start is not None:
            killed, lost = view.kills_cost - self._start[0], view.deaths_cost - self._start[1]
            net = killed - lost
            self.results.append(net)
            self.finished.append({"t": round(self._formed_at), "seconds": round(view.game_seconds - self._formed_at),
                                  "size": self._formed_size, "survivors": len(self.unit_ids),
                                  "target": list(self._aim) if self._aim else None,
                                  "killed": round(killed), "lost": round(lost)})
            if net < 0 and self._aim is not None:
                self.failed.append((self._aim, view.game_seconds))   # it lost there: leave it alone a while
            self._start = None
        self._aim = None
        self._no_target_since = None
        self.unit_ids = []
        self._formed_size = 0
        self._released_at = view.game_seconds


def healthy(view: View) -> bool:
    """Macro is healthy: ``HEALTHY_HARVESTERS`` harvesters with ore ahead of
    them, an army worth ``RAID_EXTRA_MIN_ARMY_VALUE``, and not outgunned
    (match ee2ac21b raided on with a 9-unit army and mined-out fields)."""
    from .brain import our_unit_value, outgunned

    return (len(view.units_of(C.HARVESTER)) >= C.HEALTHY_HARVESTERS and not view.ore_running_dry
            and our_unit_value(view) >= C.RAID_EXTRA_MIN_ARMY_VALUE and not outgunned(view))


def raid_steps(view: View, is_healthy: bool | None = None) -> int:
    """How far into a healthy game we are: 0 before ``RAID_EXTRA_START_SECONDS``
    or when unhealthy, then 1 and one more every ``RAID_EXTRA_EVERY_SECONDS``."""
    t = view.game_seconds
    if t < C.RAID_EXTRA_START_SECONDS or not (healthy(view) if is_healthy is None else is_healthy):
        return 0
    return 1 + int((t - C.RAID_EXTRA_START_SECONDS) // C.RAID_EXTRA_EVERY_SECONDS)


def squads_allowed(view: View, is_healthy: bool | None = None) -> int:
    return min(C.RAID_MAX_SQUADS, 1 + raid_steps(view, is_healthy))


class RaidCorps:
    """The raid squads (see the module docstring). The first squad is the
    original one: it takes the advisor's raid target and always joins the
    main army's fight; the extra squads join only from nearby."""

    def __init__(self) -> None:
        self.squads: list[Raid] = [Raid()]
        self._backoff_until = -1e9

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
        t = view.game_seconds
        # Raids trading at a loss: no new squad for a while (they bleed the
        # army; c1c37a10 lost 7300 for 1950 in four raids, one after another).
        recent = [r for s in self.squads for r in s.results][-3:]
        if len(recent) >= 2 and sum(recent) <= -C.RAID_LOSING_NET:
            self._backoff_until = t + C.RAID_BACKOFF_SECONDS
            for s in self.squads:
                s.results.clear()
        # Healthy held RAID_HEALTHY_HOLD_SECONDS: extra squads released and
        # re-formed as the reading flickered (the audit).
        if healthy(view):
            self._healthy_until = t + C.RAID_HEALTHY_HOLD_SECONDS
        from .brain import outgunned

        well = t <= getattr(self, "_healthy_until", float("-inf")) and not outgunned(view)   # outgunned: at once
        steps = 0 if t < self._backoff_until else raid_steps(view, well)
        allowed = 1 if t < self._backoff_until else squads_allowed(view, well)
        while len(self.squads) < allowed:
            self.squads.append(Raid())
        fighters = [u for u in view.combat_units if u.id is not None and u.id != scout_id and u.id not in taken]
        # A unit is in one squad only (ee2ac21b: squads re-forming took each
        # other's units, 28 "raiders" were 13 units pulled two ways).
        seen: set = set(taken)
        for s in self.squads:
            s.unit_ids = [i for i in s.unit_ids if i not in seen]
            seen |= set(s.unit_ids)
        # Where a raid lost more than it killed (efefdca8: a squad a minute fed
        # to a turtle's natural): no squad goes there for a while.
        blocked = [c for s in self.squads for c, at in s.failed if t - at < C.RAID_FAILED_TARGET_SECONDS]
        claimed: list[tuple[int, int]] = list(blocked)
        for i, s in enumerate(self.squads):
            if i >= allowed:
                if s.unit_ids:
                    s._release(view)              # unhealthy again: back to the army
                continue
            others = {u for j, o in enumerate(self.squads) if j != i for u in o.unit_ids}
            s.avoid = list(claimed)
            s.cooldown = C.RAID_LATE_COOLDOWN_SECONDS if steps else C.RAID_COOLDOWN_SECONDS
            s.size_bonus = max(0, steps - 1)
            # Every squad (the first too: "in force" took 8 of 14) within
            # RAID_MAX_ARMY_SHARE of the fighters.
            s.limit = max(0, int(C.RAID_MAX_ARMY_SHARE * len(fighters)) - len(others))
            s.update(view, mode, scout_id, advice if i == 0 else None, taken=set(taken) | others,
                     may_form=t >= self._backoff_until)
            cell = _cell_of(view, s.target(view))
            if cell is not None:
                claimed.append(cell)

    def target(self, view: View):
        return self.squads[0].target(view)

    def drain(self) -> list[dict]:
        """The outcomes of the raids that ended since the last call."""
        out = [r for s in self.squads for r in s.finished]
        for s in self.squads:
            s.finished = []
        return out

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

    def drawing(self, view: View) -> tuple | None:
        """(enemy ids, their value, where the biggest share fights) the squads
        have drawn lately, else None (brain: the distraction)."""
        ds = [d for d in (s.drawing(view) for s in self.squads) if d is not None]
        if not ds:
            return None
        ids = frozenset(i for d in ds for i in d[0])
        return ids, sum(d[1] for d in ds), max(ds, key=lambda d: d[1])[2]

    def distracting(self, view: View) -> bool:
        return self.drawing(view) is not None

    def orders(self, view: View, main_fight: tuple[int, int] | None = None) -> list[dict]:
        out: list[dict] = []
        j2 = C.RAID_JOIN_FIGHT_CELLS ** 2
        for i, s in enumerate(self.squads):
            fight = main_fight
            if fight is not None:                 # (every squad: the first flip-flopped across the map)
                centre = _centroid(s._members(view))
                if centre is None or dist2(centre, fight) > j2:
                    fight = None                  # far off on its own errand: keep at it
            out += s.orders(view, fight)
        return out


def _cell_of(view: View, target) -> tuple[int, int] | None:
    if isinstance(target, int):
        actors = list(view.enemy_units) + list(view.enemy_buildings) + list(view.known_enemy_buildings)
        hit = next((a for a in actors if a.id == target and a.cell is not None), None)
        return hit.cell if hit is not None else None
    return tuple(target) if target is not None else None


_CLEAR_ORDER = {C.REFINERY: 0, C.CONSTRUCTION_YARD: 1, C.WAR_FACTORY: 2, "tent": 2, "barr": 2}


def _clear_target(view: View, squad, origin, remote, avoid) -> int | None:
    """At an expansion (``RAID_CLEAR_CELLS``): the next building of theirs
    there, by id — the refinery first, then the yard, production, the rest;
    not their turrets, nor anything in a turret's reach, for infantry
    (4e9828e1's raiders killed the refinery and stood idle beside the
    rest). None when not at one, or it is clear."""
    if origin is None:
        return None
    c2 = C.RAID_CLEAR_CELLS ** 2
    here = [r for r in remote if dist2(r, origin) <= c2]
    if not here:
        return None
    armored = _armored(squad) >= C.RAID_ARMOR_UNITS
    zone = frozenset() if armored else turret_zone(view)
    by2 = (C.EXPAND_CLUSTER_CELLS + 4) ** 2
    s2 = C.RAID_SPREAD_CELLS ** 2
    seen: dict = {}
    for b in list(view.enemy_buildings) + list(view.known_enemy_buildings):
        if (b.id is None or b.cell is None or b.wreck or (b.type in C.TURRETS and not armored)
                or b.cell in zone or not any(dist2(b.cell, r) <= by2 for r in here)
                or any(dist2(b.cell, a) <= s2 for a in avoid)):
            continue
        seen[b.id] = b
    if not seen:
        return None
    return min(seen.values(), key=lambda b: (_CLEAR_ORDER.get(b.type, 3), dist2(b.cell, origin))).id


def _target(view: View, members: list[Actor], advice=None, avoid=(), squad=None) -> int | tuple[int, int] | None:
    """Only at their expansions (``ENEMY_EXPANSION_MIN_CELLS`` or more from
    their main base) that our scouts have found: never their main field,
    where their army and turrets are (match 52a77fb1 sent squad after squad
    to it, to die). Nearest the squad (or our base, before it exists): the
    advisor's pick there, a visible harvester there, an expansion, a
    remembered harvester there, a refinery there — each only where the
    squad can get at it (:func:`raid_spot`: round the turrets, or with
    tanks). ``None`` (no raid) until a scout finds an expansion.
    ``squad``: the unit types to judge with (default: the members, else the
    squad that would form now)."""
    from .expand import enemy_expansions

    main = enemy_spawn_guess(view)
    far2 = C.ENEMY_EXPANSION_MIN_CELLS ** 2
    s2 = C.RAID_SPREAD_CELLS ** 2
    squad = squad or [u.type for u in members] or would_form(view)
    origin = _centroid(members) or view.base_center
    remote = remote_expansions(view)
    by2 = (C.EXPAND_CLUSTER_CELLS + 4) ** 2

    def spot(cell):
        """Where the squad goes for ``cell`` (it, or a spot by it out of the
        turrets' reach), or None: not at a remote expansion, another
        squad's, or out of reach."""
        if cell is None or (main is not None and dist2(cell, main) < far2) \
                or not any(dist2(cell, r) <= by2 for r in remote) \
                or any(dist2(cell, a) <= s2 for a in avoid):
            return None
        return raid_spot(view, squad, origin, cell)

    picked = getattr(advice, "raid_target_id", None)
    hit = next((a for a in view.enemy_units if a.id == picked and a.type in C.HARVESTER), None) if picked else None
    if hit is not None and spot(hit.cell) == hit.cell:
        return picked
    advised = spot(getattr(advice, "raid_target_cell", None))
    if advised is not None:
        return advised

    def nearest(cells, area):
        found = [(s, c) for s, c in ((spot(c), c) for c in cells if dist2(c, area) <= by2) if s is not None]
        return min(found, key=lambda f: dist2(f[1], origin)) if found and origin else (found[0] if found else None)

    def harvester_at(area):
        here = [a for a in view.enemy_units if a.type in C.HARVESTER and a.id is not None and a.cell is not None
                and dist2(a.cell, area) <= by2 and spot(a.cell) == a.cell]
        return min(here, key=lambda a: dist2(a.cell, origin) if origin else 0).id if here else None

    # The squad already at an expansion finishes it first: its harvesters,
    # then its buildings one by one (not off to another mid-job).
    if members and origin is not None:
        c2 = C.RAID_CLEAR_CELLS ** 2
        for area in (r for r in remote if dist2(r, origin) <= c2):
            hit_id = harvester_at(area)
            if hit_id is not None:
                return hit_id
        building = _clear_target(view, squad, origin, remote, avoid)
        if building is not None:
            return building
    # Else the expansion farthest from their start first — the least
    # defended, the slowest for their army to reach (20095e3e's raids went for
    # the natural by their main while the scouts had seen one 11 cells
    # farther out): a harvester there in sight, else the expansion, a
    # harvester remembered there, its refinery; the next farthest only when
    # nothing there can be reached.
    areas = sorted(remote, key=lambda r: -dist2(r, main) if main is not None else 0)
    known_harvesters = [a.cell for a in view.known_enemy_units if a.type in C.HARVESTER]
    refineries = [a.cell for a in view.enemy_buildings + view.known_enemy_buildings if a.type == C.REFINERY]
    expansions = enemy_expansions(view)
    for area in areas:
        hit_id = harvester_at(area)
        if hit_id is not None:
            return hit_id
        for cells in (expansions, known_harvesters, refineries, [area]):
            best = nearest(cells, area)
            if best is not None:
                return best[0]
    return None


def remote_expansions(view: View) -> list[tuple[int, int]]:
    """Their remote expansions: any building of theirs we know
    ``RAID_REMOTE_MIN_CELLS`` or more from their main base — a yard, a
    refinery, or the turret they put up there first (d1615ac6 saw an
    expansion's turret at 12:12 and waited for its refinery until 13:19).
    Raids go only there (e2bc988e raided a refinery 18 cells out, their
    main base's edge, and lost)."""
    main = enemy_spawn_guess(view)
    if main is None:
        return []
    far2 = C.RAID_REMOTE_MIN_CELLS ** 2
    return [a.cell for a in list(view.enemy_buildings) + list(view.known_enemy_buildings)
            if a.cell is not None and not a.wreck
            and dist2(a.cell, main) >= far2]


def would_form(view: View, armor: bool = False) -> list[str]:
    """The squad a raid would take now (rockets first; ``armor``: with
    ``RAID_ARMOR_UNITS`` tanks or siege units in front), for judging a
    target before it forms."""
    size = C.RAID_EXPANSION_SQUAD if view.game_seconds >= C.RAID_EXPANSION_SECONDS else C.RAID_SQUAD_SIZE
    kinds = sorted((u.type for u in view.combat_units if u.type in C.RAID_UNIT_TYPES), key=C.RAID_UNIT_TYPES.index)
    heavy = [u.type for u in view.combat_units if u.type in C.TANK_PREFERENCE or u.type in C.SIEGE_UNITS]
    return (heavy[:C.RAID_ARMOR_UNITS] if armor else []) + kinds[:size]


def _armored(kinds) -> int:
    return sum(1 for k in kinds if k in C.TANK_PREFERENCE or k in C.SIEGE_UNITS)


def turret_zone(view: View) -> frozenset:
    """Cells within ``RAID_TURRET_CLEAR_CELLS`` of an enemy turret we know."""
    r = C.RAID_TURRET_CLEAR_CELLS
    zone = set()
    for a in list(view.enemy_buildings) + list(view.known_enemy_buildings):
        if a.type in C.TURRETS and a.cell is not None and not a.wreck:
            zone.update((a.cell[0] + i, a.cell[1] + j) for i in range(-r, r + 1) for j in range(-r, r + 1)
                        if i * i + j * j <= r * r)
    return frozenset(zone)


def raid_spot(view: View, squad: list[str], origin, cell) -> tuple[int, int] | None:
    """Where a raid on ``cell`` goes, or None (not now). With no turret of
    theirs known by it: ``cell``. By turrets (c1c37a10: four raids into
    pillboxed expansions killed 1950 and lost 7300):

    - infantry only: ``cell`` if it is out of the turrets' reach, else the
      ore by it that is, either way only when a road there keeps out of
      their reach (:func:`raid_route`) — a pillbox rarely covers every side;
    - with ``RAID_ARMOR_UNITS`` tanks or siege units: straight at ``cell``,
      the turrets counted in the odds;
    - otherwise None: wait for the tanks or the siege.

    Either way the squad must clearly beat (``RAID_TARGET_ODDS``) the enemy
    fighters we know there."""
    from . import strength

    r2 = (C.EXPAND_CLUSTER_CELLS + C.RAID_TURRET_CLEAR_CELLS) ** 2
    turrets = [a.type for a in list(view.enemy_buildings) + list(view.known_enemy_buildings)
               if a.type in C.TURRETS and a.cell is not None and not a.wreck and dist2(a.cell, cell) <= r2]
    near2 = C.EXPAND_CLUSTER_CELLS ** 2
    units = [a.type for a in list(view.enemy_units) + list(view.known_enemy_units)
             if a.type not in C.HARVESTER and a.type != C.MCV and a.cell is not None and not a.wreck
             and dist2(a.cell, cell) <= near2]
    if not squad:
        return None
    if not turrets:
        return tuple(cell) if not units or strength.odds(view, squad, units) >= C.RAID_TARGET_ODDS else None
    if _armored(squad) >= C.RAID_ARMOR_UNITS:
        return tuple(cell) if strength.odds(view, squad, turrets + units) >= C.RAID_TARGET_ODDS else None
    if units and strength.odds(view, squad, units) < C.RAID_TARGET_ODDS:
        return None
    zone = turret_zone(view)
    ore = sorted((c for c in view.ore_cells if dist2(c, cell) <= near2 and c not in zone),
                 key=lambda c: dist2(c, cell))
    for c in ([tuple(cell)] if tuple(cell) not in zone else []) + ore[:6]:
        if origin is None or raid_route(view, origin, c, zone) is not None:
            return c
    return None


def raid_route(view: View, origin, goal, zone: frozenset | None = None) -> list | None:
    """Waypoints from ``origin`` to ``goal`` out of the turrets' reach
    (:func:`turret_zone`): the straight road's legs when it is clear, the
    legs of the way round when there is one (at most
    ``RAID_ROUTE_MAX_RATIO`` as long), None when there is none."""
    from .terrain import legs, route_around, walk_path

    zone = turret_zone(view) if zone is None else zone
    if tuple(goal) in zone:
        return None
    direct = walk_path(view, tuple(origin), tuple(goal))
    if not direct:
        return None
    if not any(c in zone for c in direct[3:]):
        return legs(direct, C.ARMY_LEG_CELLS)[:-1]   # the road itself, in legs (one road for the whole squad)
    path = route_around(view, origin, goal, zone, C.RAID_ROUTE_MAX_RATIO)
    if path is None or any(c in zone for c in path[3:]):
        return None
    return legs(path, C.ARMY_LEG_CELLS)


def _centroid(actors: list[Actor]) -> tuple[int, int] | None:
    cells = [a.cell for a in actors if a.cell is not None]
    if not cells:
        return None
    return (sum(c[0] for c in cells) // len(cells), sum(c[1] for c in cells) // len(cells))
