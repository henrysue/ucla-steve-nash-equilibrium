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
        self.base_seen_at: float | None = None
        self._sent: dict[int, tuple[int, int]] = {}  # unit id -> waypoint last ordered
        self._start_at: dict[str, float] = {}        # role -> when its circuit may set out (the stagger)
        self._site_seen: dict[tuple[int, int], float] = {}   # remote ore field -> when a scout last saw it
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
        return ["left", "right"] + [f"p{i}" for i in range(max(0, C.MID_PICKETS - 2))]

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
            if search:
                continue                             # sweeping (above)
            if self.look is not None and not look_done:
                look_done = True
                if dist2(u.cell, self.look) <= 36:
                    self.look = None                 # seen: a fresh look at them
                    self.base_seen_at = view.game_seconds
                else:
                    out += self._go(view, u, self.look)[0]
                    continue
            if role == "p0" and view.game_seconds >= C.MID_PICKET_SECONDS:
                site = self._next_remote_site(view, u)
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
            r2 = C.SCOUT_FLEE_CELLS ** 2
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
        from .terrain import legs, route_around

        dest = tuple(dest)
        plan = self._legs.get(u.id)
        if plan is None or plan[0] != dest or (u.idle and plan[1] and dist2(u.cell, plan[1][0]) > (2 * C.SCOUT_LEG_CELLS) ** 2):
            path = route_around(view, u.cell, dest, self.avoid(view), C.SCOUT_MAX_DETOUR)
            plan = (dest, legs(path, C.SCOUT_LEG_CELLS) if path else [dest])
            self._legs[u.id] = plan
        points = plan[1]
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
        k = slots.get(role, 0.0)
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

    def _next_remote_site(self, view: View, u) -> tuple[int, int] | None:
        """The nearest remote ore field not seen for ``REMOTE_SITE_RECHECK_SECONDS``
        (match f333a770: neither seat found any of the other's 2-3 expansions)."""
        t = view.game_seconds
        for s in remote_sites(view):
            if dist2(u.cell, s) <= 36:
                self._site_seen[s] = t
        due = [s for s in remote_sites(view) if t - self._site_seen.get(s, -1e9) >= C.REMOTE_SITE_RECHECK_SECONDS]
        return min(due, key=lambda s: dist2(s, u.cell)) if due else None

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
        self._circling.discard(role)                 # round the back: info gathered, back to its post
        self.base_seen_at = view.game_seconds
        post = self.post(view, role)
        return self._go(view, u, post)[0] if post else []

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
