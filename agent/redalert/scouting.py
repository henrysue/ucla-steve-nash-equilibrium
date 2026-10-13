"""The scout corps: information over survival.

Early (before ``EARLY_SCOUT_END_SECONDS``) two scouts take the left and the
right path round the enemy's main base — out along a flank, then round its
side, past the back and onto the other side — to see every building (their
opening: intel.py). The left runs an outer ring (``CIRCUIT_OUTER_CELLS``
from their spawn), the right an inner one through the base
(``CIRCUIT_INNER_CELLS``), ``CIRCUIT_STAGGER_SECONDS`` behind: a scout sees
about 5 cells, and one ring of 11 missed a second barracks at 1:48 (match
6b457914). They don't stop to fight on the way: a
scout dying back there after seeing the base was worth it. Once round, each
pulls back to a picket post and is kept for later checks.

The roamer (``p0``, and ``x0`` too from ``EXPANSION_ROAM_SECONDS``) walks
round the remote ore fields where an enemy expansion can hide, seeing each
from beyond its four extreme ore cells (:func:`field_points`): an expansion
found there is remembered (memory.py) and flagged for the raid squads
(expand.enemy_expansions, raid.py).

From ``MID_PICKET_SECONDS`` up to ``MID_PICKETS`` scouts hold picket posts
spread left and right across the approach (``PICKET_ALONG`` of the way to
the enemy, ``PICKET_SPREAD`` sideways): advance warning of a push. Pickets
back off from enemy fighters. Every ``BASE_CHECK_SECONDS`` one of them runs
the circuit again (their buildings, economy, army) and returns to its post.

Scouts are dogs where we can train them (Soviet kennel: fast and cheap),
else riflemen, else a scout vehicle; claimed from the army, or trained.
"""

from __future__ import annotations

import math

from . import config as C
from . import orders
from .observe import View, dist2, enemy_spawn_guess


class ScoutCorps:
    def __init__(self) -> None:
        self.roles: dict[str, int] = {}              # role -> unit id
        self._route: dict[str, list] = {}            # role -> remaining circuit waypoints
        self._circling: set[str] = set()             # roles out on a circuit
        self._last_check = float("-inf")
        self._last_cell: dict[int, tuple[int, int]] = {}
        self._lost: list[tuple[float, tuple[int, int], bool]] = []   # (time, cell, was circling)
        self.visit: tuple[int, int] | None = None    # a one-off look (our expansion site)
        self.look: tuple[int, int] | None = None     # where their army masses, before a blind attack
        self._peek_at: dict[str, float] = {}          # role -> when its last look at their exit ended
        self._peeking: set = set()                    # roles out on a look right now
        self._army_check: set = set()                 # roles that, round their base, now look at their army
        self._look_leg: dict[int, int] = {}           # scout id -> its look's stop (0 their exit, 1 their base)
        self._look_saw: set = set()                   # scout ids whose look has seen something (a stop or their army)
        self.unreachable: set = set()                 # places a scout can't reach without walking a choke corridor
        self.base_seen_at: float | None = None
        self.army_looked_at: float | None = None      # when a scout last looked where their army masses (look)
        self._sent: dict[int, tuple[int, int]] = {}  # unit id -> waypoint last ordered
        self._start_at: dict[str, float] = {}        # role -> when its circuit may set out (the stagger)
        self._site_seen: dict[tuple[int, int], float] = {}   # remote field watch point -> when we last saw it
        self._roam_to: dict[str, tuple[int, int]] = {}       # roamer role -> the watch point it is walking to
        self._checks = 0                             # mid-game base checks run (they alternate rings)
        self.scouts_lost = 0                         # scouts killed this match
        self._legs: dict = {}                        # unit id -> (destination, its remaining legs)
        self._avoid: tuple = (None, frozenset())     # (view, where scouts don't walk) for this view

    # -- what the rest of the agent reads ----------------------------------------

    @property
    def unit_ids(self) -> list[int]:
        return list(self.roles.values())

    @property
    def unit_id(self) -> int | None:
        """One scout (for code that keeps a single scout out of the army)."""
        return next(iter(self.roles.values()), None)

    def request_army_look(self) -> bool:
        """A look at their army now (e.g. they just expanded): the first
        free scout — not on a circuit or already looking — goes
        (:meth:`_army_look`). False when none is free."""
        for role in ("p1", "left", "right", "p0", "x0"):
            if role in self.roles and role not in self._circling and role not in self._army_check:
                self._army_check.add(role)
                return True
        return False

    def backing_off(self, view: View) -> bool:
        """``SCOUT_BACKOFF_LOSSES`` scouts lost within
        ``SCOUT_BACKOFF_WINDOW_SECONDS``: no looks or peeks until
        ``SCOUT_BACKOFF_SECONDS`` after the last — they are feeding scouts to
        whatever kills them (5a4bc83f lost 58)."""
        t = view.game_seconds
        recent = [lt for lt, _, _ in self._lost if t - lt <= C.SCOUT_BACKOFF_WINDOW_SECONDS]
        return len(recent) >= C.SCOUT_BACKOFF_LOSSES and t - max(recent) <= C.SCOUT_BACKOFF_SECONDS

    def alarm(self, view: View) -> bool:
        """``SCOUT_ALARM_LOSSES`` scouts killed past the middle early, within
        ``SCOUT_ALARM_SECONDS``: something is out there — an early-push alarm.
        One dead scout is just a dead scout (it held the army home for 30 s)."""
        base, enemy = view.base_center, enemy_spawn_guess(view)
        if base is None or enemy is None or view.game_seconds >= C.EARLY_PUSH_WINDOW_SECONDS:
            return False
        recent = [c for t, c, _ in self._lost
                  if view.game_seconds - t <= C.SCOUT_ALARM_SECONDS and dist2(c, enemy) < dist2(c, base)]
        return len(recent) >= C.SCOUT_ALARM_LOSSES

    # -- each view -----------------------------------------------------------------

    def wanted(self, view: View) -> list[str]:
        t = view.game_seconds
        if t < C.EARLY_SCOUT_END_SECONDS:
            return ["left", "right"]
        if t < C.MID_PICKET_SECONDS:
            return ["left", "right"]                 # back on their posts after the circuit
        roamers = ["x0"] if t >= C.EXPANSION_ROAM_SECONDS else []
        return ["left", "right"] + [f"p{i}" for i in range(max(0, C.MID_PICKETS - 2))] + roamers

    def update(self, view: View, taken: set = frozenset()) -> None:
        """Drop dead scouts (remembering where they died), fill empty roles."""
        alive = {u.id: u for u in view.units if u.id is not None and not u.wreck}
        for role, uid in list(self.roles.items()):
            u = alive.get(uid)
            if u is None:
                cell = self._last_cell.get(uid)
                if cell is not None:
                    self._lost.append((view.game_seconds, cell, role in self._circling))
                    self.scouts_lost += 1
                self._lost = self._lost[-6:]
                del self.roles[role]
                self._circling.discard(role)
                self._route.pop(role, None)
            elif u.cell is not None:
                self._last_cell[uid] = u.cell
        for role in self.wanted(view):
            if role in self.roles:
                continue
            busy = taken | set(self.roles.values())
            if role not in ("left", "right"):
                # Pickets come from spare units only: never strip the army
                # below MIN_DEFENDERS (they are trained instead).
                free = [u for u in view.combat_units if u.id is not None and u.id not in busy]
                if len(free) <= C.MIN_DEFENDERS and not any(u.type in C.SCOUT_UNIT_TYPES[:1] for u in free):
                    break
            uid = self._claim(view, busy)
            if uid is None:
                break
            self.roles[role] = uid
            if role in ("left", "right") and view.game_seconds < C.EARLY_SCOUT_END_SECONDS and self.base_seen_at is None:
                self._start_circuit(view, role)
        # Mid-game base check: one picket runs the circuit again.
        if (view.game_seconds >= C.MID_PICKET_SECONDS and view.game_seconds - self._last_check >= C.BASE_CHECK_SECONDS
                and not self._circling):
            role = next((r for r in ("left", "right") if r in self.roles), None)
            if role is not None:
                self._start_circuit(view, role, inner=self._checks % 2 == 1)   # outer, inner, outer, ...
                self._checks += 1
                self._last_check = view.game_seconds

    def orders(self, view: View, search: list | None = None) -> list[dict]:
        """``search``: the hunter's unchecked search points while we hunt the
        enemy's last buildings — each scout not on a circuit takes one.
        Every trip is walked in legs round the chokes and the front of their
        base (:meth:`_go`)."""
        out: list[dict] = []
        units = {u.id: u for u in view.units if u.id is not None}
        visit_done = look_done = False
        if search:
            out += self._sweep(view, units, list(search))
        for role, uid in self.roles.items():
            u = units.get(uid)
            if u is None or u.cell is None:
                continue
            if role in self._circling:
                out += self._circuit_orders(view, role, u)
                continue
            if role in self._army_check:
                moves = self._army_look(view, role, u)
                if moves is not None:
                    out += moves
                    continue
            if search:
                continue                             # sweeping (above)
            if self.look is not None and not look_done:
                look_done = True
                moves = self._look(view, u)
                if moves is None:
                    self.look = None                 # seen: a fresh look at them
                else:
                    out += moves
                    continue
            if role in ("p0", "x0") and view.game_seconds >= C.MID_PICKET_SECONDS:
                site = self._next_remote_site(view, u, role)
                if site is not None:                 # the roamer: remote fields they may have expanded to
                    out += self._go(view, u, site)[0]
                    continue
            if self.visit is not None and not visit_done:
                visit_done = True
                if dist2(u.cell, self.visit) <= 36:
                    self.visit = None
                else:
                    out += self._go(view, u, self.visit)[0]
                    continue
            post = self.post(view, role)
            if post is None:
                continue
            # Out on a peek it turns back as soon as it sees them (it has
            # seen what it went for); at its post, when they come close.
            r2 = (C.SCOUT_SEEN_CELLS if role in self._peeking else C.SCOUT_FLEE_CELLS) ** 2
            threatened = any(a.cell is not None and not a.wreck and a.type not in C.HARVESTER
                             and dist2(a.cell, u.cell) <= r2 for a in view.enemy_units)
            if threatened and view.base_center is not None:
                # A picket backs off: it is there to see the push, not stop it.
                self._end_peek(role, view)
                back = _step_toward(u.cell, view.base_center, C.SCOUT_FLEE_CELLS)
                self._sent[u.id] = back
                self._legs.pop(u.id, None)
                out.append(orders.move([uid], list(back)))
                continue
            # The lane scouts don't stand idle at their posts: every
            # SCOUT_PEEK_SECONDS each takes a look at the enemy's exit from its
            # own side (their army and its moves), then goes back (3bd6e450:
            # two scouts stood at the same corners for minutes).
            peek = self.peek_point(view, role) if role in ("left", "right") else None
            if peek is not None and self.backing_off(view):
                self._end_peek(role, view)
                peek = None
            if peek is not None and role not in self._peeking and \
                    view.game_seconds - self._peek_at.get(role, self._peek_offset(role)) >= C.SCOUT_PEEK_SECONDS:
                self._peeking.add(role)
            if role in self._peeking and peek is not None:
                moves, arrived = self._go(view, u, peek)
                if arrived:
                    self._end_peek(role, view)       # seen (or as near as it gets): back to the post
                else:
                    out += moves
                    continue
            if dist2(u.cell, post) > 9:
                out += self._go(view, u, post)[0]
        return out

    def production(self, view: View) -> list[dict]:
        """Train a scout when a role is empty and nothing can be claimed."""
        missing = len(self.wanted(view)) - len(self.roles)
        if missing <= 0:
            return []
        kind = C.resolve(C.SCOUT_UNIT_TYPES, view.available)
        if kind is None or view.queued_count(kind) >= 1:
            return []
        return [orders.train(kind, count=1)]

    # -- moving: the long way round, in legs ----------------------------------------

    def avoid(self, view: View) -> frozenset:
        """Where scouts don't walk: the chokes on the road between the bases
        (terrain.choke_zone), the front of their base (within
        ``SCOUT_FRONT_CELLS`` of their spawn, ``SCOUT_FRONT_DEGREES`` either
        side of their way out toward us) and ``SCOUT_TURRET_CELLS`` round
        every enemy turret we know (82bdca3e: the left scout's road ran
        through Trapped's gap, the right one cut across the front of their
        base, past where their defenses stand)."""
        from .terrain import choke_zone, walk_path

        if self._avoid[0] is view:
            return self._avoid[1]
        zone = set(choke_zone(view))
        enemy, base = enemy_spawn_guess(view), view.base_center
        if enemy is not None and base is not None:
            path = walk_path(view, tuple(enemy), tuple(base))
            exit_ = path[min(len(path) - 1, C.APPROACH_RING_CELLS)] if path else tuple(base)
            ex, ey = exit_[0] - enemy[0], exit_[1] - enemy[1]
            el = (ex * ex + ey * ey) ** 0.5 or 1.0
            cos_max = math.cos(math.radians(C.SCOUT_FRONT_DEGREES))
            r = C.SCOUT_FRONT_CELLS
            for i in range(-r, r + 1):
                for j in range(-r, r + 1):
                    d = (i * i + j * j) ** 0.5
                    if 0 < d <= r and (i * ex + j * ey) / (d * el) >= cos_max:
                        zone.add((enemy[0] + i, enemy[1] + j))
        t = C.SCOUT_TURRET_CELLS
        for a in list(view.enemy_buildings) + list(view.known_enemy_buildings):
            if a.type in C.TURRETS and a.cell is not None and not a.wreck:
                zone.update((a.cell[0] + i, a.cell[1] + j) for i in range(-t, t + 1) for j in range(-t, t + 1)
                            if i * i + j * j <= t * t)
        out = frozenset(zone)
        self._avoid = (view, out)
        return out

    def _go(self, view: View, u, dest) -> tuple[list[dict], bool]:
        """(orders, arrived) to take scout ``u`` to ``dest`` round the places
        it mustn't walk (:meth:`avoid`), in legs of ``SCOUT_LEG_CELLS`` so its
        own pathing can't cut back through them, each an attack-move: the
        next leg when it reaches one (or stands idle short of it).
        ``arrived``: on the last leg and there, or idle as near as it gets."""
        from .terrain import legs, route_around, walk_path

        dest = tuple(dest)
        plan = self._legs.get(u.id)
        if plan is None or plan[0] != dest or (u.idle and plan[1] and dist2(u.cell, plan[1][0]) > (2 * C.SCOUT_LEG_CELLS) ** 2):
            path = route_around(view, u.cell, dest, self.avoid(view), C.SCOUT_MAX_DETOUR)
            if path is None and self._through_choke(view, u.cell, dest):
                # No way round, and the straight road runs down a choke
                # corridor into their concave (ae03adbd: scouts walked
                # Sidestep's gap to their army's spot and died, again and
                # again): the nearest spot beside the choke reached the long
                # way round instead, else not reachable at all.
                path = self._beside_choke(view, u.cell, dest)
                if path is None:
                    self.unreachable.add(dest)
                    self._legs.pop(u.id, None)
                    return [], True
            if path is None:                         # the straight road is clear: still walked in legs,
                path = walk_path(view, tuple(u.cell), dest) or None   # so it is the road we checked
            plan = (dest, legs(path, C.SCOUT_LEG_CELLS) if path else [dest])
            self._legs[u.id] = plan
        points = plan[1]
        dest = points[-1] if points else dest          # beside the choke, when the place itself is in it
        while len(points) > 1 and (dist2(u.cell, points[0]) <= 9
                                   or (u.idle and self._sent.get(u.id) == points[0])):
            points.pop(0)
        nxt = points[0]
        last = len(points) == 1
        if last and (dist2(u.cell, dest) <= 9 or (u.idle and self._sent.get(u.id) == dest)):
            return [], True
        if u.idle or self._sent.get(u.id) != nxt:
            self._sent[u.id] = nxt
            # attack-move: a scout meeting something on the way shoots back
            # instead of being shot in the back (only backing off is a move)
            return [orders.attack_move([u.id], list(nxt))], False
        return [], False

    def _beside_choke(self, view: View, start, dest) -> list | None:
        """A path to the nearest cell (within ``SCOUT_PEEK_BACK_CELLS`` + 4)
        to ``dest`` outside the choke zone that a scout reaches without
        walking the corridor, or None."""
        from .terrain import choke_zone, route_around, walk_path

        zone, avoid = choke_zone(view), self.avoid(view)
        r = C.SCOUT_PEEK_BACK_CELLS + 4
        near = sorted(((dest[0] + i, dest[1] + j) for i in range(-r, r + 1) for j in range(-r, r + 1)
                       if i * i + j * j <= r * r), key=lambda c: dist2(c, dest))
        tried = 0
        for c in near:
            if c in zone or c in view.blocked_cells:
                continue
            tried += 1
            if tried > 12:
                break
            path = route_around(view, start, c, avoid, C.SCOUT_MAX_DETOUR)
            if path is None and not self._through_choke(view, start, c):
                path = walk_path(view, tuple(start), c) or None
            if path is not None:
                return path
        return None

    @staticmethod
    def _through_choke(view: View, start, dest) -> bool:
        """The straight road from ``start`` to ``dest`` crosses the choke
        zone (terrain.choke_zone) farther than ``SCOUT_CHOKE_ENTRY_CELLS``
        from ``dest`` (and 3 from ``start``)."""
        from .terrain import choke_zone, walk_path

        zone = choke_zone(view)
        if not zone:
            return False
        e2, s2 = C.SCOUT_CHOKE_ENTRY_CELLS ** 2, 9
        return any(c in zone and dist2(c, dest) > e2 and dist2(c, start) > s2
                   for c in walk_path(view, tuple(start), tuple(dest)))

    # -- places ------------------------------------------------------------------

    def _peek_offset(self, role: str) -> float:
        """The left and right scouts look half a period apart."""
        return -C.SCOUT_PEEK_SECONDS / 2 if role == "right" else 0.0

    def _end_peek(self, role: str, view: View) -> None:
        if role in self._peeking:
            self._peeking.discard(role)
            self._peek_at[role] = view.game_seconds

    def peek_point(self, view: View, role: str) -> tuple[int, int] | None:
        """Where a scout looks at the enemy's exit: ``SCOUT_PEEK_BACK_CELLS``
        short of their massing point toward us, ``SCOUT_PEEK_SIDE_CELLS`` to
        the scout's own side — off the centre lane their army walks."""
        from .scout import _open_cell, enemy_staging

        base, enemy = view.base_center, enemy_spawn_guess(view)
        staging = enemy_staging(view)
        if base is None or enemy is None or staging is None:
            return None
        dx, dy = base[0] - enemy[0], base[1] - enemy[1]
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / length, dy / length
        side = -1.0 if role == "left" else 1.0
        px, py = uy * side, -ux * side
        cell = (staging[0] + ux * C.SCOUT_PEEK_BACK_CELLS + px * C.SCOUT_PEEK_SIDE_CELLS,
                staging[1] + uy * C.SCOUT_PEEK_BACK_CELLS + py * C.SCOUT_PEEK_SIDE_CELLS)
        return self._outside(view, _reachable(view, _open_cell(view, cell)))

    def post(self, view: View, role: str) -> tuple[int, int] | None:
        """Picket posts across the approach: the two early scouts on the outer
        posts; the extra pickets watch the enemy's exit from the flanks, never
        the centre lane (3bd6e450: a picket stood in the middle of the
        approach, where the armies meet)."""
        from .scout import _open_cell

        base, enemy = view.base_center, enemy_spawn_guess(view)
        if base is None or enemy is None:
            return None
        if role.startswith("p") and role != "p0":
            # an exit watcher: alternate sides, on the flank of their massing point
            return self.peek_point(view, "left" if int(role[1:] or 0) % 2 else "right")
        slots = {"left": -1.0, "right": 1.0}
        extra = max(0, C.MID_PICKETS - 2)
        for i in range(extra):
            slots[f"p{i}"] = -1.0 + 2.0 * (i + 1) / (extra + 1)
        k = slots.get("p0" if role == "x0" else role, 0.0)   # the second roamer waits where the first does
        dx, dy = enemy[0] - base[0], enemy[1] - base[1]
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        px, py = -dy / length, dx / length
        spread = C.PICKET_SPREAD * length
        ax, ay = base[0] + dx * C.PICKET_ALONG, base[1] + dy * C.PICKET_ALONG
        return self._outside(view, _reachable(view, _open_cell(view, (ax + px * spread * k, ay + py * spread * k))))

    def _outside(self, view: View, cell) -> tuple[int, int]:
        """``cell``, or the nearest open cell out of the places scouts don't
        stand (:meth:`avoid`), leaning toward our base: a post or a look at
        their exit from inside the choke or the front of their base is in
        their concave's fire (82bdca3e: the flank picket stood in Trapped's
        gap and died five times in a minute)."""
        zone = self.avoid(view)
        if cell not in zone:
            return cell
        base = view.base_center or cell
        r = C.SCOUT_FRONT_CELLS
        near = [(cell[0] + i, cell[1] + j) for i in range(-r, r + 1) for j in range(-r, r + 1)]
        ok = [c for c in near if c not in zone and c not in view.blocked_cells
              and (view.bounds is None or (view.bounds[0] <= c[0] < view.bounds[0] + view.bounds[2]
                                           and view.bounds[1] <= c[1] < view.bounds[1] + view.bounds[3]))]
        return min(ok, key=lambda c: dist2(c, cell) + 0.25 * dist2(c, base)) if ok else cell

    def circuit(self, view: View, side: int, radius: float = C.CIRCUIT_OUTER_CELLS) -> list[tuple[int, int]]:
        """Waypoints out along one flank, round that side of the enemy base at
        ``radius`` from their spawn — from its side (``CIRCUIT_DEGREES`` start
        at 90: never across the front) past the back onto the other side."""
        from .scout import _along, _open_cell

        base, enemy = view.base_center, enemy_spawn_guess(view)
        if base is None or enemy is None:
            return []
        dx, dy = base[0] - enemy[0], base[1] - enemy[1]
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / length, dy / length            # from their base toward ours
        px, py = -uy * side, ux * side
        mid = _along(base, enemy, 0.5)
        bow = C.SCOUT_FLANK_FRACTION * length
        points = [_open_cell(view, (mid[0] + px * bow, mid[1] + py * bow))]
        for deg in C.CIRCUIT_DEGREES:
            a = math.radians(deg)
            cx = ux * math.cos(a) + px * math.sin(a)
            cy = uy * math.cos(a) + py * math.sin(a)
            points.append(_open_cell(view, (enemy[0] + cx * radius, enemy[1] + cy * radius)))
        return [_reachable(view, q) for q in points]

    def _next_remote_site(self, view: View, u, role: str = "p0") -> tuple[int, int] | None:
        """The nearest watch point round a remote ore field (:func:`field_points`)
        that none of our units has seen for ``REMOTE_SITE_RECHECK_SECONDS``
        and the other roamer isn't walking to (match f333a770: neither seat
        found any of the other's 2-3 expansions; d2d8dc27: the scouts passed
        the middle of a field and missed the yard at its far end)."""
        t = view.game_seconds
        points = [p for s in remote_sites(view) for p in field_points(view, s)]
        ours = [a.cell for a in view.units if a.cell is not None and not a.wreck]
        r2 = C.FIELD_WATCH_SEEN_CELLS ** 2
        for p in points:
            if any(dist2(c, p) <= r2 for c in ours):
                self._site_seen[p] = t
        taken = {q for r, q in self._roam_to.items() if r != role and r in self.roles}
        due = [p for p in points if t - self._site_seen.get(p, -1e9) >= C.REMOTE_SITE_RECHECK_SECONDS
               and p not in taken]
        nxt = min(due, key=lambda p: dist2(p, u.cell)) if due else None
        if nxt is None:
            self._roam_to.pop(role, None)
        else:
            self._roam_to[role] = nxt
        return nxt

    # -- internals ---------------------------------------------------------------

    def _sweep(self, view: View, units: dict, points: list) -> list[dict]:
        """Each free scout to the nearest search point no other scout took."""
        out = []
        free = [units[uid] for role, uid in self.roles.items()
                if role not in self._circling and uid in units and units[uid].cell is not None]
        for u in sorted(free, key=lambda u: u.id):
            if not points:
                break
            p = min(points, key=lambda p: dist2(p, u.cell))
            points.remove(p)
            out += self._go(view, u, p)[0]
        return out

    def _start_circuit(self, view: View, role: str, inner: bool | None = None) -> None:
        side = 1 if role in ("left", "p0") else -1
        if inner is None:
            inner = role == "right"
        route = self.circuit(view, side, C.CIRCUIT_INNER_CELLS if inner else C.CIRCUIT_OUTER_CELLS)
        if route:
            self._route[role] = route
            self._circling.add(role)
            start = view.game_seconds
            if role == "right" and "left" in self._start_at:   # staggered behind the left
                start = max(start, self._start_at["left"] + C.CIRCUIT_STAGGER_SECONDS)
            self._start_at[role] = start

    def _circuit_orders(self, view: View, role: str, u) -> list[dict]:
        if view.game_seconds < self._start_at.get(role, 0.0):
            return []                                # the stagger: not yet
        route = self._route.get(role) or []
        while route and (dist2(u.cell, route[0]) <= C.CIRCUIT_ARRIVED_CELLS ** 2):
            route.pop(0)
        while route:
            moves, arrived = self._go(view, u, route[0])
            if not arrived:
                return moves
            route.pop(0)                             # there (or as near as it gets): the next waypoint
        self._circling.discard(role)                 # round the back: info gathered...
        self.base_seen_at = view.game_seconds
        self._army_check.add(role)                   # ...and, still alive, a look at their army
        moves = self._army_look(view, role, u)
        if moves is not None:
            return moves
        post = self.post(view, role)
        return self._go(view, u, post)[0] if post else []

    def _army_look(self, view: View, role: str, u) -> list[dict] | None:
        """A scout back from round their base looks at their army
        (:meth:`_look`), then goes back to its post. None when done."""
        moves = self._look(view, u)
        if moves is None:
            self._army_check.discard(role)
        return moves

    def _look(self, view: View, u) -> list[dict] | None:
        """A look at their army: first where it masses (scout.enemy_staging:
        the mouth of the choke on their side), then, if nobody is there, on
        into the front of their base — a turtle's army waits at home
        (d2d8dc27: the scout found the open-ground spot empty, the look
        counted, and the attack met 3250 on an estimate of 250). It counts
        (``army_looked_at``: the attack's blind check) once the scout sees
        their fighters, or stands in front of their base and sees none.
        Walked in legs round the chokes (:meth:`_go`); a stop it can't reach
        is passed. None when done."""
        from .scout import enemy_staging

        if any(a.cell is not None and not a.wreck and a.type not in C.HARVESTER
               and dist2(a.cell, u.cell) <= C.SCOUT_SEEN_CELLS ** 2 for a in view.enemy_units):
            # Their army seen: the look counts, and the scout turns back (its
            # post's flee rule) — walking on through their army fed a scout
            # every ~30 s (5a4bc83f). An empty spot goes on to their base front.
            self._look_saw.add(u.id)
            return self._looked(view, u)
        if self.backing_off(view):
            self._look_leg.pop(u.id, None)
            return None                              # too many scouts lost lately: no look (and no credit)
        stops = [s for s in (enemy_staging(view), self.base_front(view)) if s is not None]
        if not stops:
            self._look_leg.pop(u.id, None)       # nowhere to look: no credit
            return None
        leg = self._look_leg.get(u.id, 0)
        while leg < len(stops):
            if dist2(u.cell, stops[leg]) > 36:
                moves, arrived = self._go(view, u, stops[leg])
                if not arrived:
                    self._look_leg[u.id] = leg
                    return moves
            if leg == len(stops) - 1 and stops[leg] not in self.unreachable:
                self._look_saw.add(u.id)             # stood in front of their base: nobody there
            leg += 1                                 # nobody: the next stop (an empty exit doesn't count)
        return self._looked(view, u)

    def _looked(self, view: View, u) -> None:
        """Done: it counts (``army_looked_at``) only if the scout stood at a
        stop or saw their army — not when every stop was down a choke."""
        self._look_leg.pop(u.id, None)
        if u.id in self._look_saw:
            self.base_seen_at = self.army_looked_at = view.game_seconds
        self._look_saw.discard(u.id)
        return None

    def base_front(self, view: View) -> tuple[int, int] | None:
        """The front of their main base: ``LOOK_BASE_FRONT_CELLS`` from their
        spawn toward us, where a turtle's army stands."""
        from .scout import _forward

        enemy, base = enemy_spawn_guess(view), view.base_center
        if enemy is None or base is None:
            return None
        return _reachable(view, _forward(view, tuple(enemy), tuple(base), C.LOOK_BASE_FRONT_CELLS, tuple(enemy)))

    @staticmethod
    def _claim(view: View, taken: set) -> int | None:
        """A dog, else a rifleman, else a scout vehicle — never a tank."""
        for kind in C.SCOUT_UNIT_TYPES + C.SCOUT_VEHICLES:
            for u in view.units_of(kind):
                if u.id is not None and u.id not in taken and not u.wreck:
                    return u.id
        return None


def _reachable(view: View, cell) -> tuple[int, int]:
    """``cell``, or the nearest cell (within ``SCOUT_SNAP_CELLS``) our units
    can walk to from the base: an open cell walled in by cliffs is no post."""
    from .terrain import walk_distances

    if view.base_center is None or not view.blocked_cells:
        return cell
    walk = walk_distances(view, [view.base_center], C.SCOUT_WALK_STEPS)
    if cell in walk:
        return cell
    r = C.SCOUT_SNAP_CELLS
    near = [(cell[0] + i, cell[1] + j) for i in range(-r, r + 1) for j in range(-r, r + 1)
            if (cell[0] + i, cell[1] + j) in walk]
    return min(near, key=lambda c: dist2(c, cell)) if near else cell


def _step_toward(cell, target, cells: float) -> tuple[int, int]:
    dx, dy = target[0] - cell[0], target[1] - cell[1]
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    s = min(cells, length)
    return round(cell[0] + dx / length * s), round(cell[1] + dy / length * s)


_FIELD_POINTS: dict = {}


def field_points(view: View, ore) -> list[tuple[int, int]]:
    """Where a scout stands to see all round the ore field at ``ore``: beyond
    its northmost, southmost, westmost and eastmost ore cells,
    ``FIELD_WATCH_OUT_CELLS`` further out from the field's centre, on ground
    we can reach. An enemy yard unfolds beside the ore, which on a big field
    is far from its centre (d2d8dc27: scouts passed (20,30) by the field at
    (25,23) and never saw the yard at (29,11), 12 cells off)."""
    from .expand import _ore_components
    from .scout import _open_cell

    key = (id(view.ore_cells), tuple(ore))
    hit = _FIELD_POINTS.get(key)
    if hit is not None and hit[0] is view.ore_cells:
        return hit[1]
    comp = next((c for c in _ore_components(view) if tuple(ore) in c), None) or [tuple(ore)]
    cx, cy = sum(c[0] for c in comp) / len(comp), sum(c[1] for c in comp) / len(comp)
    out: list[tuple[int, int]] = []
    # Ties go to the cell nearest the field's middle (not a corner, which
    # would look the same way as its neighbour).
    for e in (min(comp, key=lambda c: (c[1], abs(c[0] - cx))), max(comp, key=lambda c: (c[1], -abs(c[0] - cx))),
              min(comp, key=lambda c: (c[0], abs(c[1] - cy))), max(comp, key=lambda c: (c[0], -abs(c[1] - cy)))):
        dx, dy = e[0] - cx, e[1] - cy
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        p = _reachable(view, _open_cell(view, (e[0] + dx / d * C.FIELD_WATCH_OUT_CELLS,
                                               e[1] + dy / d * C.FIELD_WATCH_OUT_CELLS)))
        if all(dist2(p, q) > C.FIELD_WATCH_SEEN_CELLS ** 2 for q in out):
            out.append(p)
    if len(_FIELD_POINTS) > 64:
        _FIELD_POINTS.clear()
    _FIELD_POINTS[key] = (view.ore_cells, out)
    return out


def remote_sites(view: View) -> list[tuple[int, int]]:
    """Ore fields far from both main bases (``ENEMY_EXPANSION_MIN_CELLS``) and
    from our own yards: where an enemy expansion can hide."""
    from .expand import _ore_fields

    base, enemy = view.base_center, enemy_spawn_guess(view)
    if base is None or enemy is None:
        return []
    far2 = C.ENEMY_EXPANSION_MIN_CELLS ** 2
    yards = [b.cell for b in view.buildings if b.type == C.CONSTRUCTION_YARD and b.cell is not None and not b.wreck]
    return [ore for _, centre, ore in _ore_fields(view)
            if dist2(centre, enemy) >= far2 and dist2(centre, base) >= far2
            and all(dist2(centre, y) > 100 for y in yards)]
