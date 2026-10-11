"""Expansion: a second construction yard on a second ore field, around 4:00.

Benchmarks the user gave for OpenRA competitive play (cited in PLAN.md):
expand around 4:00-4:30, only on a stable economy (double refinery and war
factory, 4-5 harvesters), with a real army to protect the expansion line,
after scouting the site, and with a mobile force guarding the haul.

The plan, one step at a time (``state``):

1. ``idle`` -> ``scouting``: once the gates pass (time, harvesters,
   refineries, war factory, army value, not defending), pick the site — the
   richest ore field on our side of the map, away from our main base — and
   send the scout to look at it.
2. ``scouting`` -> ``training``: a unit of ours has seen the site within
   ``EXPAND_SCOUT_VALID_SECONDS`` and no known enemy is near it; train an MCV
   (needs a war factory and a service depot). Unsafe or unseen for too long:
   back to ``idle`` with a cooldown.
3. ``training`` -> ``moving``: the MCV is out; it drives to the site while the
   army stages there as its escort (agent.py holds attacks meanwhile).
4. ``moving`` -> ``deployed``: at the site it deploys into a construction
   yard; if there is no room, the spot shifts toward home and it tries again.
   The build plan then puts a refinery beside the new ore.

A destroyed MCV sends the plan back to ``idle`` with a cooldown; after
``EXPAND_MAX_FAILURES`` the agent stops trying. Up to ``EXPAND_MAX_EXPANSIONS``
expansions: the next one ``EXPAND_AGAIN_AFTER_SECONDS`` after the last, or
sooner once harvesters sit idle (the ore is running out — match f9f06e34
stalled around 10:00 on a depleted economy). Each deployed expansion gets a
refinery (asked for once, capped by ``MAX_REFINERIES``), a defense on its
enemy-facing edge, and a garrison (guard.Garrison) that joins the main army's
pushes once the expansion is complete.
"""

from __future__ import annotations

from . import config as C
from . import orders
from .brain import our_unit_value, outgunned
from .modes import DEFENSIVE, Mode
from .observe import View, dist2, enemy_spawn_guess


class Expansion:
    def __init__(self) -> None:
        self.state = "idle"
        self.site: tuple[int, int] | None = None
        self.ore: tuple[int, int] | None = None     # the field's ore cell nearest the site
        self.mcv_id: int | None = None
        # Expansions we have deployed: {"site", "ore", "asked_at"} (asked_at:
        # when its refinery was last requested, so it is asked for once, not
        # every view — match f9f06e34 built 16 refineries).
        self.done: list[dict] = []
        self._since = 0.0
        self._seen_at: float | None = None
        self._cooldown_until = 0.0
        self._failures = 0
        self._last_deploy = float("-inf")
        self._deploy_tries = 0
        self._spots: list[tuple[int, int]] = []     # other deploy spots around the site
        self._idle_since: float | None = None       # harvesters idling: the ore is running out
        self._strapped_since: float | None = None   # money under EXPAND_STRAPPED_FUNDS since
        self._even = False                          # trades about even or better (last view)
        self._last_done_at: float | None = None
        self._timed_out = False

    # -- what other parts read -----------------------------------------------

    @property
    def scout_target(self) -> tuple[int, int] | None:
        """Where the scout should look (while scouting the site)."""
        return self.site if self.state == "scouting" else None

    @property
    def escorting(self) -> bool:
        """The MCV is on the road to a forward site (nearer the enemy than our
        base): the army escorts it instead of attacking. A site behind our
        base is safe: no escort (match f333a770 walked 50-60 units behind
        both bases for a minute to guard an MCV nobody threatened)."""
        return self.state == "moving" and getattr(self, "_site_forward", True)

    @property
    def stage_override(self) -> tuple[int, int] | None:
        return self.site if self.escorting else None

    def track_site(self, view: View) -> None:
        """Whether the site lies forward (nearer the enemy than our base)."""
        enemy = enemy_spawn_guess(view)
        self._site_forward = (self.site is not None and enemy is not None and view.base_center is not None
                              and dist2(self.site, enemy) < dist2(view.base_center, enemy))

    @property
    def sites(self) -> list[tuple[int, int]]:
        return [d["site"] for d in self.done]

    def refinery_ore(self, view: View) -> tuple[int, int] | None:
        """The ore of a deployed expansion that still has no refinery beside
        it — unless we are at the refinery cap or asked for it recently."""
        if _count(view, C.REFINERY) >= C.MAX_REFINERIES or outgunned(view):
            return None
        for d in self.done:
            if not self._wants_refinery(view, d):
                continue
            if d["asked_at"] is not None and view.game_seconds - d["asked_at"] < C.EXPAND_REFINERY_RETRY_SECONDS:
                continue
            return d["ore"]
        return None

    def active_base(self, view: View, home_depleted: bool, depleted: set) -> tuple[int, int] | None:
        """Where the base grows once the main base's ore is mined out: the
        newest expansion that still has its construction yard and live ore.
        None while home still has ore (the main base keeps growing)."""
        if not home_depleted:
            return None
        field2 = (C.EXPAND_CLUSTER_CELLS * 1.5) ** 2
        for d in reversed(self.done):
            yard = any(b.type == C.CONSTRUCTION_YARD and b.cell is not None and not b.wreck
                       and dist2(b.cell, d["site"]) <= 16 for b in view.buildings)
            dry = any(dist2(d["ore"], f) <= field2 for f in depleted)
            if yard and not dry:
                return d["site"]
        return None

    def refinery_job(self, view: View) -> dict | None:
        """For PLACEMENT: the first expansion with no refinery serving its ore
        yet ({"site", "ore", ...}), ignoring the request throttle."""
        return next((d for d in self.done if self._wants_refinery(view, d)), None)

    def _wants_refinery(self, view: View, d: dict) -> bool:
        return not self._served(view, d["ore"])

    def served(self, view: View, ore) -> bool:
        """Whether a refinery of ours serves this ore (for the match log)."""
        return self._served(view, ore)

    def needs_defense(self, view: View) -> tuple[int, int] | None:
        """For PLACEMENT: an expansion short of defenses, ignoring the throttle."""
        return next((d["site"] for d in self.done
                     if self._defenses(view, d["site"]) < C.EXPANSION_DEFENSES), None)

    def defense_site(self, view: View) -> tuple[int, int] | None:
        """A deployed expansion still short of ``EXPANSION_DEFENSES`` defenses,
        unless its turret was asked for recently (once per retry window, not
        every view — match 1e6ba89d built 11 turrets at the main base this way)."""
        for d in self.done:
            if self._defenses(view, d["site"]) >= C.EXPANSION_DEFENSES:
                continue
            if d.get("defense_asked_at") is not None and \
                    view.game_seconds - d["defense_asked_at"] < C.EXPAND_REFINERY_RETRY_SECONDS:
                continue
            return d["site"]
        return None

    def complete(self, view: View) -> bool:
        """Every expansion has its refinery and its defense."""
        return bool(self.done) and all(
            self._served(view, d["ore"]) and self._defenses(view, d["site"]) >= C.EXPANSION_DEFENSES
            for d in self.done
        )

    # -- the plan --------------------------------------------------------------

    def update(self, view: View, mode: Mode, advice=None, ore_running_out: bool = False) -> None:
        t = view.game_seconds
        self.track_site(view)
        self._ore_running_out = ore_running_out
        self._track_depletion(view)
        self._track_money(view)
        self._track_refinery_requests(view)
        self._track_defense_requests(view)
        if self.state == "idle":
            if (len(self.done) < C.EXPAND_MAX_EXPANSIONS and self._failures < C.EXPAND_MAX_FAILURES
                    and t >= self._cooldown_until and self._again(t) and self._gates(view, mode, advice)):
                site = choose_site(view, exclude=self.sites)
                if site is not None:
                    self.site, self.ore = site
                    self._go("scouting", t)
            return
        if self.state == "scouting":
            if not self._safe(view):
                return self._abort(t)
            if self._watched(view):
                self._seen_at = t
            fresh = self._seen_at is not None and t - self._seen_at <= C.EXPAND_SCOUT_VALID_SECONDS
            # The MCV is built while the scout looks (pros queue it right after
            # the service depot, ~3:50); it only leaves once the site is seen.
            mcvs = [u for u in view.units_of(C.MCV) if u.id is not None]
            if fresh and mcvs:
                self.mcv_id = mcvs[0].id
                self._spots = _spots_around(self.site, view)
                self._yards_before = self._yard_ids(view)
                self._go("moving", t)
            elif fresh and C.MCV in view.available and self._gates(view, mode, advice):
                self._go("training", t)
            elif t - self._since > C.EXPAND_SCOUT_TIMEOUT_SECONDS:
                self._abort(t)
            return
        if self.state == "training":
            mcvs = [u for u in view.units_of(C.MCV) if u.id is not None]
            if mcvs:
                self.mcv_id = mcvs[0].id
                self._spots = _spots_around(self.site, view)
                self._yards_before = self._yard_ids(view)
                self._go("moving", t)
            elif t - self._since > C.EXPAND_TRAIN_TIMEOUT_SECONDS:
                self._abort(t)
            return
        if self.state == "moving":
            mcv = next((u for u in view.units if u.id == self.mcv_id), None)
            if mcv is not None and mcv.cell is not None and t - self._since > C.EXPAND_MOVE_TIMEOUT_SECONDS:
                if not self._timed_out:
                    # Taking too long: deploy where it stands (and around it).
                    self._timed_out = True
                    self.site = roomy_spot(mcv.cell, view) or mcv.cell
                    self._spots = _spots_around(self.site, view)
                    self._deploy_tries = 0
                elif t - self._since > C.EXPAND_MOVE_TIMEOUT_SECONDS + C.EXPAND_GIVE_UP_SECONDS:
                    self._failures += 1
                    return self._abort(t)    # give up: the escort ends, the army is free
            if mcv is None:
                new_yard = self._new_yard(view)
                if new_yard is not None:
                    # Wherever it actually unfolded (match 0561274e: a yard 1-2
                    # cells off the planned spot counted as a lost MCV, so the
                    # expansion never got its refinery or harvesters).
                    self.site = new_yard
                    self.done.append({"site": self.site, "ore": self._ore_for(view, new_yard), "asked_at": None})
                    self._last_done_at = t
                    self.state, self._since = "idle", t
                    self.site = self.ore = self.mcv_id = None
                else:
                    self._failures += 1      # the MCV died on the road
                    self._abort(t)

    def orders(self, view: View) -> list[dict]:
        out: list[dict] = []   # harvesters: harvest.HarvestManager, the one owner of their orders
        if self.state in ("scouting", "training") and C.MCV in view.available and not view.units_of(C.MCV):
            # Checked again when it trains, not just when the expansion began
            # (08560d9b: the gates passed at 2:30, the MCV trained at 4:22 with 3 fighters).
            if outgunned(view):
                # Outgunned while preparing: the expansion is called off (MCV
                # cancelled, the money back, the cooldown); a new one must pass
                # every gate again. Kept alive, it queued and cancelled its MCV
                # each time the balance tipped (b90e492f: 2.5 minutes of it).
                self._abort(view.game_seconds)
                return out + ([orders.cancel(C.MCV)] if view.queued_count(C.MCV) else [])
            if (not view.queued_count(C.MCV) and len(view.combat_units) >= C.EXPAND_MIN_FIGHTERS
                    and our_unit_value(view) >= C.EXPAND_MCV_ARMY_VALUE):
                return out + [orders.train(C.MCV, count=1)]
        return out + self._mcv_orders(view)

    def _mcv_orders(self, view: View) -> list[dict]:
        if self.state != "moving" or self.site is None:
            return []
        mcv = next((u for u in view.units if u.id == self.mcv_id), None)
        if mcv is None or mcv.cell is None:
            return []
        if dist2(mcv.cell, self.site) > C.EXPAND_DEPLOY_RADIUS_CELLS ** 2:
            stuck = (mcv.idle and view.game_seconds - self._since >= C.EXPAND_STUCK_AFTER_SECONDS
                     and dist2(mcv.cell, self.site) <= C.EXPAND_STUCK_RADIUS_CELLS ** 2)
            if not stuck:
                return [orders.move([mcv.id], list(self.site))]
            # Stopped short (the exact cell may be unreachable): unfold at the
            # nearest roomy spot around where it stands.
            self.site = roomy_spot(mcv.cell, view) or mcv.cell
            self._spots = _spots_around(self.site, view)
            if dist2(mcv.cell, self.site) > C.EXPAND_DEPLOY_RADIUS_CELLS ** 2:
                return [orders.move([mcv.id], list(self.site))]
        if view.game_seconds - self._last_deploy < C.EXPAND_DEPLOY_RETRY_SECONDS:
            return []
        self._last_deploy = view.game_seconds
        self._deploy_tries += 1
        if self._deploy_tries > 2 and self._spots:
            # No room to unfold here: try the next spot around the site (not a
            # march toward home — match f9f06e34 walked ~25 cells that way).
            self.site = self._spots.pop(0)
            self._deploy_tries = 0
            return [orders.move([mcv.id], list(self.site))]
        return [orders.deploy(mcv.id)]

    # -- internals ---------------------------------------------------------------

    def _again(self, t: float) -> bool:
        """The first expansion any time; later ones ``EXPAND_AGAIN_AFTER_SECONDS``
        after the last, or sooner once harvesters idle (the ore is running
        out), or ``EXPAND_AGAIN_STRAPPED_SECONDS`` after it when we are cash
        strapped while trading even or better (:meth:`_push`)."""
        if self._last_done_at is None:
            return True
        depleted = self._idle_since is not None and t - self._idle_since >= C.EXPAND_DEPLETED_IDLE_SECONDS
        home_dry = getattr(self, "_ore_running_out", False)   # harvest.HarvestManager says home is mined out
        since = t - self._last_done_at
        return (depleted or home_dry or since >= C.EXPAND_AGAIN_AFTER_SECONDS
                or (self._push(t) and since >= C.EXPAND_AGAIN_STRAPPED_SECONDS))

    def _push(self, t: float) -> bool:
        """Cash strapped (money under ``EXPAND_STRAPPED_FUNDS`` for
        ``EXPAND_STRAPPED_SECONDS``) while trades are about even or better:
        income is what we lack, and we can afford to take more ground.
        (Matches a1329ceb, b7fcbd4e: zero cash from minute 3 on, and a second
        expansion late or never.)"""
        return (self._even and self._strapped_since is not None
                and t - self._strapped_since >= C.EXPAND_STRAPPED_SECONDS)

    def _track_money(self, view: View) -> None:
        if view.cash + view.ore < C.EXPAND_STRAPPED_FUNDS:
            self._strapped_since = self._strapped_since if self._strapped_since is not None else view.game_seconds
        else:
            self._strapped_since = None
        self._even = view.kills_cost >= C.EXPAND_EVEN_TRADES * view.deaths_cost

    def _track_depletion(self, view: View) -> None:
        idle = sum(1 for h in view.units_of(C.HARVESTER) if h.idle)
        if idle >= C.EXPAND_DEPLETED_IDLE_HARVESTERS:
            self._idle_since = self._idle_since if self._idle_since is not None else view.game_seconds
        else:
            self._idle_since = None

    def _track_defense_requests(self, view: View) -> None:
        """Note when an expansion's turret has been put in production."""
        if not any(view.queued_count(d) for d in C.DEFENSES):
            return
        site = self.defense_site(view)
        if site is not None:
            next(d for d in self.done if d["site"] == site)["defense_asked_at"] = view.game_seconds

    def _track_refinery_requests(self, view: View) -> None:
        """Note when an expansion's refinery has been put in production."""
        in_production = view.queued_count(C.REFINERY) or any(
            p.get("ready") and str(p.get("item") or "").lower() == C.REFINERY for p in view.production)
        if not in_production:
            return
        for d in self.done:
            if self._wants_refinery(view, d) and (
                    d["asked_at"] is None or view.game_seconds - d["asked_at"] >= C.EXPAND_REFINERY_RETRY_SECONDS):
                d["asked_at"] = view.game_seconds
                return

    def _served(self, view: View, ore) -> bool:
        """A refinery of ours whose trucks can walk to this ore."""
        from .terrain import field_cells, steps_from_rect, walk_distances

        if ore is None:
            return False
        r2 = C.EXPAND_REFINERY_RADIUS_CELLS ** 2
        near = [b for b in view.buildings if b.type == C.REFINERY and b.cell is not None
                and not b.wreck and dist2(b.cell, ore) <= r2]
        if not near:
            return False
        field = field_cells(view, ore, C.EXPAND_CLUSTER_CELLS * 1.5) or [ore]
        walk = walk_distances(view, field, C.FIELD_WALK_STEPS)
        for b in near:
            steps = steps_from_rect(walk, (b.cell[0], b.cell[1], b.cell[0] + 2, b.cell[1] + 2))
            if steps is not None and steps <= C.FIELD_WALK_STEPS:   # 0 = right beside the ore
                return True
        return False

    def _defenses(self, view: View, site) -> int:
        r2 = C.EXPAND_REFINERY_RADIUS_CELLS ** 2
        return sum(1 for b in view.buildings if b.type in C.DEFENSES and b.cell is not None
                   and not b.wreck and dist2(b.cell, site) <= r2)

    def _gates(self, view: View, mode: Mode, advice) -> bool:
        wish = getattr(advice, "expand", "auto")
        # Running out of ore overrides a "wait" (match c4caa45e: the advisor's
        # "more_army" read meant "wait" in 151 of 159 answers, and the ore ran
        # out with a second site free).
        # An even game (kills - losses within EXPAND_EVEN_TRADE_BAND) keeps taking
        # ground: the first three fields run dry around 9:00 (ef66403c: both
        # seats on one expansion until 9:30 at even trades).
        even = abs(view.kills_cost - view.deaths_cost) <= C.EXPAND_EVEN_TRADE_BAND
        if wish == "wait" and not self._push(view.game_seconds) and not getattr(self, "_ore_running_out", False)                 and not even:
            return False   # broke and trading even: income beats the advisor's caution
        earliest = C.EXPAND_EARLIEST_SECONDS if wish == "now" else C.EXPAND_AFTER_SECONDS
        return (
            view.game_seconds >= earliest
            and len(view.units_of(C.HARVESTER)) >= C.EXPAND_MIN_HARVESTERS
            and _count(view, C.REFINERY) >= C.EXPAND_MIN_REFINERIES
            and view.has_building(C.WAR_FACTORY)
            and view.army_value >= C.EXPAND_MIN_ARMY_VALUE
            and not outgunned(view)
            and mode not in DEFENSIVE and mode is not Mode.RETREAT
            and (view.remaining_seconds is None or view.game_seconds <= C.EXPAND_LAST_START_SECONDS)
        )

    def _safe(self, view: View) -> bool:
        r2 = C.EXPAND_DANGER_RADIUS_CELLS ** 2
        threats = [a for a in list(view.enemy_units) + list(view.known_enemy_units) if a.type not in C.HARVESTER]
        threats += list(view.enemy_buildings) + list(view.known_enemy_buildings)
        return not any(a.cell is not None and dist2(a.cell, self.site) <= r2 for a in threats)

    def _watched(self, view: View) -> bool:
        r2 = C.EXPAND_SEEN_RADIUS_CELLS ** 2
        return any(u.cell is not None and dist2(u.cell, self.site) <= r2 for u in view.units)

    @staticmethod
    def _yard_ids(view: View) -> set:
        return {b.id for b in view.buildings if b.type == C.CONSTRUCTION_YARD and b.id is not None}

    def _new_yard(self, view: View) -> tuple[int, int] | None:
        """The centre of a construction yard that wasn't there when the MCV
        set out (nearest the site if several), else None."""
        before = getattr(self, "_yards_before", None)
        if before is None:   # no snapshot of the yards from when it set out: nearby yard only
            return self.site if self.site is not None and self._deployed(view) else None
        new = [b.cell for b in view.buildings if b.type == C.CONSTRUCTION_YARD and b.cell is not None
               and not b.wreck and b.id not in before]
        if not new:
            return None
        cell = min(new, key=lambda c: dist2(c, self.site or c))
        return (cell[0] + 1, cell[1] + 1)

    def _ore_for(self, view: View, site) -> tuple[int, int] | None:
        """The planned field's ore if the yard can walk to it, else the ore
        nearest the yard by walking."""
        from .terrain import walk_distances

        walk = walk_distances(view, [site], C.FIELD_WALK_STEPS)
        if self.ore is not None and self.ore in walk:
            return self.ore
        reachable = [c for c in view.ore_cells if c in walk]
        return min(reachable, key=lambda c: walk[c]) if reachable else self.ore

    def _deployed(self, view: View) -> bool:
        r2 = C.EXPAND_DEPLOY_RADIUS_CELLS ** 2 * 4
        return any(b.type == C.CONSTRUCTION_YARD and b.cell is not None and dist2(b.cell, self.site) <= r2
                   for b in view.buildings)

    def _go(self, state: str, t: float) -> None:
        self.state, self._since = state, t
        self._timed_out = False
        if state == "scouting":
            self._seen_at = None

    def _abort(self, t: float) -> None:
        self.state, self._since = "idle", t
        self.site = self.ore = self.mcv_id = None
        self._cooldown_until = t + C.EXPAND_COOLDOWN_SECONDS


def _count(view: View, kind: str) -> int:
    return sum(1 for b in view.buildings if b.type == kind and not b.wreck)


def has_room(cell, view: View) -> bool:
    """Whether an MCV at ``cell`` can unfold: the yard (3x4 with its bib, top-
    left at ``cell`` - (1, 1): the MCV's Transforms offset) is inside the map,
    passable, ore-free, and clear of our buildings."""
    from .buildplan import _clashes

    x, y = cell
    if view.bounds is not None:
        bx, by, bw, bh = view.bounds
        if not (bx + 1 <= x - 1 and x + 1 <= bx + bw - 2 and by + 1 <= y - 1 and y + 2 <= by + bh - 2):
            return False
    taken = view.blocked_cells | frozenset(view.ore_cells)
    if any((x + i, y + j) in taken for i in (-1, 0, 1) for j in (-1, 0, 1, 2)):
        return False
    return not _clashes(C.CONSTRUCTION_YARD, (x - 1, y - 1), view)


def roomy_spot(cell, view: View) -> tuple[int, int] | None:
    """The nearest cell to ``cell`` (within ``EXPAND_ROOM_SEARCH_CELLS``) with
    room to unfold an MCV."""
    r = C.EXPAND_ROOM_SEARCH_CELLS
    near = sorted(((cell[0] + i, cell[1] + j) for i in range(-r, r + 1) for j in range(-r, r + 1)),
                  key=lambda c: dist2(c, cell))
    return next((c for c in near if has_room(c, view)), None)


def _spots_around(site, view: View) -> list[tuple[int, int]]:
    """Other deploy spots in rings around ``site`` with room to unfold,
    nearest first — those that can walk back to ``site`` (same side of any
    cliff) before the rest."""
    from .terrain import walk_distances

    walk = walk_distances(view, [site], 12)
    out = []
    for r in (3, 5, 7):
        for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r)):
            spot = (site[0] + dx, site[1] + dy)
            if spot not in out and has_room(spot, view):
                out.append(spot)
    return sorted(out, key=lambda c: c not in walk)


_FIELDS_CACHE: dict = {}


def _ore_components(view: View) -> list[list[tuple[int, int]]]:
    """The map's ore fields: ore cells joined when at most one cell apart
    (a grid used to split one field in two, or merge two)."""
    key = id(view.ore_cells)
    hit = _FIELDS_CACHE.get(key)
    if hit is not None and hit[0] is view.ore_cells:
        return hit[1]
    left = set(view.ore_cells)
    comps = []
    while left:
        seed = left.pop()
        comp, stack = [seed], [seed]
        while stack:
            x, y = stack.pop()
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    n = (x + dx, y + dy)
                    if n in left:
                        left.remove(n)
                        comp.append(n)
                        stack.append(n)
        comps.append(comp)
    _FIELDS_CACHE.clear()
    _FIELDS_CACHE[key] = (view.ore_cells, comps)
    return comps


def _ore_fields(view: View) -> list[tuple[int, tuple[float, float], tuple[int, int]]]:
    """The map's ore fields: (cell count, centre, the ore cell nearest the centre)."""
    fields = []
    for cells in _ore_components(view):
        centre = (sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells))
        fields.append((len(cells), centre, min(cells, key=lambda c: dist2(c, centre))))
    return fields


def enemy_expansion_sites(view: View) -> list[tuple[int, int]]:
    """Ore fields the enemy may expand to, richest first: at least
    ``ENEMY_EXPANSION_MIN_CELLS`` from their main base and not on our side of
    the map (our own expansion logic, mirrored)."""
    base, enemy = view.base_center, enemy_spawn_guess(view)
    if base is None or enemy is None:
        return []
    far2 = C.ENEMY_EXPANSION_MIN_CELLS ** 2
    sites = [(n, ore) for n, centre, ore in _ore_fields(view)
             if dist2(centre, enemy) >= far2 and dist2(centre, enemy) < dist2(centre, base)]
    return [ore for _, ore in sorted(sites, key=lambda s: -s[0])]


def enemy_expansions(view: View) -> list[tuple[int, int]]:
    """Known enemy buildings far from their main base, i.e. expansions: the
    cell of a construction yard or refinery there when we know one (where
    harvesters unload), else of any building. One cell per expansion."""
    enemy = enemy_spawn_guess(view)
    if enemy is None:
        return []
    far2 = C.ENEMY_EXPANSION_MIN_CELLS ** 2
    out: list[tuple[int, int]] = []
    seen = list(view.enemy_buildings) + list(view.known_enemy_buildings)
    for prefer in (True, False):
        for b in seen:
            if b.cell is None or b.wreck or dist2(b.cell, enemy) < far2:
                continue
            if prefer != (b.type in (C.CONSTRUCTION_YARD, C.REFINERY)):
                continue
            if all(dist2(b.cell, c) > C.EXPAND_CLUSTER_CELLS ** 2 for c in out):
                out.append(b.cell)
    return out


def expansion_defenses(view: View, cell) -> int:
    """Known enemy defenses near an enemy expansion."""
    r2 = C.EXPAND_CLUSTER_CELLS ** 2
    return sum(1 for b in list(view.enemy_buildings) + list(view.known_enemy_buildings)
               if b.type in C.DEFENSES and b.cell is not None and dist2(b.cell, cell) <= r2)


def choose_site(view: View, exclude=()) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """(deploy cell, ore cell) for the next expansion: the NATURAL first —
    the ore field nearest our base by walking that no refinery of ours mines
    yet, on our side of the map, at least ``EXPAND_MIN_DISTANCE_CELLS`` out
    and not a straggler (``EXPAND_MIN_FIELD_SHARE`` of the biggest field).
    The nearest is the safest and quickest to take and to defend; the
    richest used to win, and both seats of match b7fcbd4e walked past their
    natural (~16 cells behind the base) to a field 28 cells away. The deploy
    cell walks to the ore (not across a cliff), nearest our base."""
    from .terrain import steps_from_rect, walk_distances

    base, enemy = view.base_center, enemy_spawn_guess(view)
    if base is None or enemy is None or not view.ore_cells:
        return None
    comps = _ore_components(view)
    biggest = max(len(c) for c in comps)
    floor = max(C.EXPAND_MIN_FIELD_CELLS, C.EXPAND_MIN_FIELD_SHARE * biggest)
    refineries = [b for b in view.buildings if b.type == C.REFINERY and b.cell is not None and not b.wreck]
    from_base = walk_distances(view, [base], C.EXPAND_SEARCH_WALK_STEPS)
    far2 = C.EXPAND_MIN_DISTANCE_CELLS ** 2
    taken2 = (C.EXPAND_SITE_WALK_STEPS + 2) ** 2
    from .scout import our_yards
    yards = our_yards(view)
    reach2 = C.REFINERY_ORE_SEARCH_CELLS ** 2
    best = None
    for cells in comps:
        if len(cells) < floor:
            continue
        centre = (sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells))
        if dist2(centre, base) < far2 or dist2(centre, base) >= dist2(centre, enemy):
            continue   # the home field, or their side of the map
        if any(min(dist2(c, y) for c in cells) <= reach2 for y in yards):
            continue   # our base reaches it: refineries mine it, an MCV there is wasted (654f1b2e)
        if any(min(dist2(c, e) for c in cells) <= taken2 for e in exclude):
            continue   # one of our expansions already
        near = walk_distances(view, cells, C.REFINERY_ORE_WALK_STEPS + 2)
        if any((s := steps_from_rect(near, (r.cell[0], r.cell[1], r.cell[0] + 2, r.cell[1] + 2))) is not None
               and s <= C.REFINERY_ORE_WALK_STEPS + 2 for r in refineries):
            continue   # a refinery of ours mines it already
        steps = min((from_base[c] for c in cells if c in from_base), default=None)
        rank = (steps if steps is not None else 1000 + dist2(centre, base) ** 0.5, -len(cells))
        if best is None or rank < best[0]:
            best = (rank, cells, min(cells, key=lambda c: dist2(c, centre)))
    if best is None:
        return None
    _, cells, ore = best
    spot = walkable_site(view, ore, base, field=cells)
    if spot is None:   # no roomy spot that walks to this ore: the old way
        spot = _toward(ore, base, C.EXPAND_DEPLOY_OFFSET_CELLS, view)
        spot = roomy_spot(spot, view) or spot
    return spot, ore


def walkable_site(view: View, ore, base, field=None) -> tuple[int, int] | None:
    """A deploy spot with room to unfold that WALKS to the field's ore: as
    close to the field as leaves room for its refinery (``EXPAND_SITE_MIN_STEPS``
    or more walking steps, at most ``EXPAND_SITE_WALK_STEPS``; not across a
    cliff), the side nearer our base breaking ties. Nearest our base used to
    win, and expansion yards unfolded beside the main base, the field at the
    edge of their reach (match 14a78c43: both seats' naturals). ``field``: the
    field's ore cells (default: ore around ``ore``)."""
    from .terrain import field_cells, walk_distances

    field = field or field_cells(view, ore, C.EXPAND_CLUSTER_CELLS * 1.5) or [ore]
    walk = walk_distances(view, field, C.EXPAND_SITE_WALK_STEPS)
    spots = [c for c, d in walk.items() if C.EXPAND_SITE_MIN_STEPS <= d <= C.EXPAND_SITE_WALK_STEPS and has_room(c, view)]
    if not spots:
        return None
    return min(spots, key=lambda c: (walk[c], dist2(c, base)))


def _toward(cell, target, cells: float, view: View) -> tuple[int, int]:
    """``cells`` from ``cell`` toward ``target``, on a passable, ore-free cell
    inside the map."""
    if target is None:
        return cell
    dx, dy = target[0] - cell[0], target[1] - cell[1]
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    x, y = round(cell[0] + dx / length * cells), round(cell[1] + dy / length * cells)
    taken = view.blocked_cells | frozenset(view.ore_cells)
    if (x, y) in taken:
        free = [(x + i, y + j) for i in range(-4, 5) for j in range(-4, 5) if (x + i, y + j) not in taken]
        if free:
            x, y = min(free, key=lambda c: dist2(c, (x, y)))
    if view.bounds is not None:
        bx, by, bw, bh = view.bounds
        x, y = min(max(x, bx + 2), bx + bw - 3), min(max(y, by + 2), by + bh - 3)
    return x, y
