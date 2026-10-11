"""Typed, defensive reads of the Red Alert observation.

Everything the executor and brain need to know about the game state, pulled
from ``state.raw`` into small typed views. Field names match the platform's
observation, documented in https://api.altruagent-game.com/skill/redalert
section 4 (and used by ``examples/llm/redalert.py``): ``observation`` holds
``economy``, ``military``, ``units``, ``buildings``, ``production``,
``available_production``, ``visible_enemies``, ``visible_enemy_buildings``,
``events``, ``base_center``; ``time`` holds the clock; ``map`` holds geometry.

Every read is defensive: a missing or malformed field yields a safe empty
default, never an exception. Unit/building ids are re-read every view because
they change and die (a deployed MCV becomes a construction yard with a new id).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from . import config as C


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value) -> list[dict]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _num(value, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _cell(value) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[0], value[1]
    elif isinstance(value, dict):
        x, y = value.get("x"), value.get("y")
    else:
        return None
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        return int(x), int(y)
    return None


@dataclass(frozen=True)
class Actor:
    id: int | None
    type: str
    cell: tuple[int, int] | None
    hp: float
    idle: bool = False
    wreck: bool = False
    repairing: bool = False
    rally: tuple[int, int] | None = None   # production buildings: where new units go
    stance: str | None = None              # units: hold_fire / return_fire / defend / attack_anything
    act: str | None = None                 # units: current activity, as the server reports it


def _stance(value) -> str | None:
    """Normalized stance: "AttackAnything", "attack anything" and
    "attack_anything" all become "attack_anything" (the order's spelling)."""
    import re

    if not isinstance(value, str) or not value.strip():
        return None
    snake = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", value.strip())
    return snake.lower().replace(" ", "_").replace("-", "_")


def _actor(raw: dict) -> Actor:
    return Actor(
        id=raw.get("id") if isinstance(raw.get("id"), int) and not isinstance(raw.get("id"), bool) else None,
        type=str(raw.get("type") or "").lower(),
        cell=_cell(raw),
        hp=_num(raw.get("hp")),
        idle=bool(raw.get("idle")),
        wreck=bool(raw.get("wreck")),
        repairing=bool(raw.get("repairing")),
        rally=_cell(raw.get("rally")),
        stance=_stance(raw.get("stance")),
        act=str(raw["act"]) if isinstance(raw.get("act"), str) and raw.get("act") else None,
    )


def _top_left(b: Actor) -> Actor:
    """Our building at its footprint's top-left cell. The observation reports
    a building's centre cell — (w // 2, h // 2) in from the corner: every
    placement in match fd507a58's audit came up at the asked top-left + (1, 1)
    — while placement takes the top-left. Reading the centre as the corner
    put every "one free cell" spot right of or below a building two cells
    out, past OpenRA's adjacency limit, and the engine moved it home."""
    if b.cell is None:
        return b
    dx, dy = centre_offset(b.type)
    return replace(b, cell=(b.cell[0] - dx, b.cell[1] - dy))


def centre_offset(kind: str) -> tuple[int, int]:
    """Where the server reports a building, from its top-left cell."""
    w, h = C.FOOTPRINTS.get(kind, C.DEFAULT_FOOTPRINT)
    return C.CENTRE_OFFSETS.get(kind, (w // 2, h // 2))


@dataclass(frozen=True)
class View:
    """One decision's complete, defensive snapshot of the match."""

    ok: bool                       # false when there is no observation yet
    tick: int
    game_seconds: float
    remaining_seconds: float | None
    cash: float
    power_provided: float
    power_drained: float
    units: list[Actor]
    buildings: list[Actor]
    enemy_units: list[Actor]
    enemy_buildings: list[Actor]
    available: set[str]            # available_production ids, lowercased
    production: list[dict]         # queue entries (item/queue/ready/progress)
    events: list[dict]
    base_center: tuple[int, int] | None
    spawns: list[tuple[int, int]]
    bounds: tuple[int, int, int, int] | None
    army_value: float
    enemy_army_value: float
    kills_cost: float
    deaths_cost: float
    assets_value: float
    # The server's own hints (raw["legal_actions"]): a ready-to-send attack
    # order and the enemy base cell. Either may be absent.
    attack_now: dict | None = None
    enemy_base: tuple[int, int] | None = None
    ore: float = 0.0                 # stored ore, and how much we can store
    resource_capacity: float = 0.0
    unit_costs: dict = field(default_factory=dict)   # rules.units[<id>].cost
    # Enemy buildings we saw earlier and still believe stand (memory.py).
    known_enemy_buildings: list = field(default_factory=list)
    known_enemy_units: list = field(default_factory=list)  # seen recently, maybe not now
    # Ore and gem cells from the map picture (game_config rules.maps.<id>.layout,
    # as at match start). Empty when the config has no layout.
    ore_cells: tuple = ()
    blocked_cells: frozenset = frozenset()   # impassable map cells (water, rock, cliffs, trees, mines)
    # Cells a building can't stand on although units walk there (rough
    # ground, beach, road/bridge), plus the impassable ones.
    unbuildable_cells: frozenset = frozenset()
    map_legend: dict = field(default_factory=dict)   # the map picture's legend, as given (for the log)
    map_picture: dict = field(default_factory=dict)  # {"origin", "rows"} as given (for the log)
    # Cells of footprints the engine recently moved a building away from
    # (agent.py's placement audit): no candidate spot may cover them.
    failed_cells: frozenset = frozenset()
    # The most enemy army we saw in the last ENEMY_PEAK_SECONDS, less what we
    # killed since (agent.py): the fog hides most of it most of the time.
    enemy_army_peak: float = 0.0
    # The fields our harvesters work have less than EXPAND_LEAD_SECONDS of ore
    # left (harvest.HarvestManager's forecast): expand, no more harvesters.
    ore_running_dry: bool = False

    # -- derived, cheap helpers ----------------------------------------------

    def has_building(self, type_ids) -> bool:
        ids = (type_ids,) if isinstance(type_ids, str) else tuple(type_ids)
        return any(b.type in ids and not b.wreck for b in self.buildings)

    def units_of(self, type_ids) -> list[Actor]:
        ids = (type_ids,) if isinstance(type_ids, str) else tuple(type_ids)
        return [u for u in self.units if u.type in ids and not u.wreck]

    @property
    def power_margin(self) -> float:
        return self.power_provided - self.power_drained

    @property
    def combat_units(self) -> list[Actor]:
        """Our units that can fight: no harvesters, MCV, engineers, medics or
        other non-combat types, and no wrecks (``*husk``)."""
        skip = set(C.HARVESTER) | {C.MCV} | set(C.NON_COMBAT)
        return [u for u in self.units if u.type not in skip and not u.wreck and not u.type.endswith("husk")]

    def queued_count(self, item: str) -> int:
        item = item.lower()
        return sum(
            1 for p in self.production
            if str(p.get("item") or "").lower() == item and not p.get("ready")
        )

    def ready_in_queue(self, item: str) -> bool:
        item = item.lower()
        return any(
            str(p.get("item") or "").lower() == item and p.get("ready")
            for p in self.production
        )


def read(raw: dict, game_config: dict | None = None) -> View:
    """Build a :class:`View` from ``state.raw`` (plus ``context.game_config``,
    which holds the map's spawns and unit costs). Never raises."""
    raw = _dict(raw)
    obs = _dict(raw.get("observation"))
    if not obs:
        return _empty()
    econ = _dict(obs.get("economy"))
    mil = _dict(obs.get("military"))
    time_info = _dict(raw.get("time"))
    rules = _dict(_dict(game_config).get("rules"))
    state_map = _dict(raw.get("map"))
    # The state's map entry has id/bounds; spawns live in get_game_config's
    # rules.maps[<id>] (as in examples/llm/redalert.py). Merge both.
    map_info = {**state_map, **_dict(_dict(rules.get("maps")).get(state_map.get("id") or ""))}

    spawns = [c for c in (_cell(s) for s in map_info.get("spawns") or []) if c is not None]
    bounds = map_info.get("bounds")
    bounds_t = tuple(bounds) if isinstance(bounds, list) and len(bounds) == 4 and all(
        isinstance(n, int) and not isinstance(n, bool) for n in bounds
    ) else None

    units = [_actor(u) for u in _list_of_dicts(obs.get("units"))]
    enemy_units = [_actor(e) for e in _list_of_dicts(obs.get("visible_enemies"))]
    costs = _unit_costs(rules)
    ore = _ore_cells(state_map.get("id"), map_info.get("layout"))
    blocked = _blocked_cells(state_map.get("id"), map_info.get("layout"))
    unbuildable = _unbuildable_cells(state_map.get("id"), map_info.get("layout")) | blocked
    legal = _dict(raw.get("legal_actions"))
    attack_now = legal.get("attack_now")

    return View(
        ok=True,
        tick=int(_num(obs.get("tick"))),
        game_seconds=_num(time_info.get("game_seconds")),
        remaining_seconds=(
            _num(time_info.get("remaining_seconds"))
            if time_info.get("remaining_seconds") is not None else None
        ),
        cash=_num(econ.get("cash", econ.get("credits"))),
        power_provided=_num(econ.get("power_provided")),
        power_drained=_num(econ.get("power_drained")),
        units=units,
        buildings=[_top_left(_actor(b)) for b in _list_of_dicts(obs.get("buildings"))],
        enemy_units=enemy_units,
        enemy_buildings=[_actor(e) for e in _list_of_dicts(obs.get("visible_enemy_buildings"))],
        available={str(x).lower() for x in obs.get("available_production") or [] if isinstance(x, str)},
        production=_list_of_dicts(obs.get("production")),
        events=_list_of_dicts(obs.get("events")),
        base_center=_cell(obs.get("base_center")),
        spawns=spawns,
        bounds=bounds_t,
        # Priced from the actors themselves, combat units only: the server's
        # army_value counts harvesters (match e2f73d65: "13150" was 10
        # harvesters and 7 fighters, so the rules saw an army that wasn't there).
        army_value=_mil_value(mil, ("army_value", "value"), units, costs),
        enemy_army_value=_mil_value(mil, ("visible_enemy_value", "enemy_value"), enemy_units, costs),
        kills_cost=_num(mil.get("kills_cost")),
        deaths_cost=_num(mil.get("deaths_cost")),
        assets_value=_num(econ.get("assets_value", mil.get("assets_value"))),
        attack_now=attack_now if isinstance(attack_now, dict) and attack_now else None,
        enemy_base=_cell(legal.get("enemy_base")),
        ore=_num(econ.get("ore")),
        resource_capacity=_num(econ.get("resource_capacity")),
        unit_costs=costs,
        ore_cells=ore,
        blocked_cells=blocked,
        unbuildable_cells=unbuildable,
        map_legend=_legend_dict(_dict(map_info.get("layout")).get("legend")),
        map_picture={k: _dict(map_info.get("layout")).get(k) for k in ("origin", "rows")},
    )


_ORE_CACHE: dict[object, tuple] = {}
_ORE_CHARS = frozenset("$*M")  # ore, gems, ore mines (skill/redalert section 4 legend)


def _ore_cells(map_id, layout) -> tuple:
    """Ore/gem cells from the map picture: ``rows[i][j]`` is the cell
    ``[origin[0] + j, origin[1] + i]``. Parsed once per map; ``()`` when the
    layout is missing or malformed."""
    if map_id is not None and map_id in _ORE_CACHE:
        return _ORE_CACHE[map_id]
    cells: tuple = ()
    layout = _dict(layout)
    rows, origin = layout.get("rows"), layout.get("origin")
    if isinstance(rows, list) and _cell(origin) is not None:
        ox, oy = _cell(origin)
        cells = tuple(
            (ox + j, oy + i)
            for i, row in enumerate(rows) if isinstance(row, str)
            for j, ch in enumerate(row) if ch in _ORE_CHARS
        )
    if map_id is not None and cells:
        _ORE_CACHE[map_id] = cells  # only a real parse: the config may arrive late
    return cells


_BLOCKED_CACHE: dict[object, frozenset] = {}
_BLOCKED_CHARS = frozenset("~^TM")   # water, rock, trees, ore mine (when there is no legend)
# Legend descriptions: walkable words win first (a bridge over water is a
# bridge), then anything impassable; unknown terrain stays walkable.
_WALKABLE_WORDS = ("bridge", "road", "ford", "clear", "ore", "gem", "spawn", "beach", "rough",
                   "grass", "dirt", "sand", "snow", "ice", "ground", "field")
_BLOCKED_WORDS = ("cliff", "rock", "water", "tree", "wood", "forest", "river", "lake", "sea", "ocean",
                  "wall", "impassable", "mine", "mountain", "boulder", "building", "structure")


def _legend_dict(legend) -> dict:
    """The legend as {char: description}, whatever shape it came in."""
    out: dict = {}
    if isinstance(legend, dict):
        for k, v in legend.items():
            desc = v if isinstance(v, str) else " ".join(str(x) for x in _dict(v).values())
            out[str(k)[:1]] = desc
    elif isinstance(legend, list):
        for entry in legend:
            if isinstance(entry, str) and ":" in entry:
                ch, _, desc = entry.partition(":")
                out[ch.strip()[:1] or ch[:1]] = desc.strip()
            elif isinstance(entry, dict):
                ch = entry.get("char") or entry.get("symbol") or entry.get("key")
                if isinstance(ch, str) and ch:
                    out[ch[:1]] = " ".join(str(v) for k, v in entry.items() if k not in ("char", "symbol", "key"))
    return out


def _legend_blocked_chars(legend: dict) -> frozenset | None:
    """Impassable characters per the legend (None without a usable legend).
    An ore mine ("M") is always impassable."""
    if not legend:
        return None
    blocked = {"M"}
    for ch, desc in legend.items():
        d = str(desc).lower()
        if "mine" in d and ("ore" in d or "gem" in d):
            blocked.add(ch)
        elif any(w in d for w in ("bridge", "road", "ford")):
            continue
        elif any(w in d for w in _BLOCKED_WORDS):
            blocked.add(ch)
    return frozenset(blocked)


_UNBUILDABLE_CACHE: dict[object, frozenset] = {}
# OpenRA's buildings stand only on clear ground and road; units also walk
# rough ground and beaches, so a footprint there is quietly moved next to the
# main yard (match b7fcbd4e: both of an expansion's refineries went home).
# Bridges share the road character, so roads count as unbuildable too.
_UNBUILDABLE_WORDS = ("rough", "beach", "shore", "bridge", "ore", "gem", "mine")


def _legend_unbuildable_chars(legend: dict) -> frozenset:
    """Walkable-but-unbuildable characters per the legend (empty without one);
    unknown terrain counts as buildable."""
    return frozenset(ch for ch, desc in legend.items()
                     if any(w in str(desc).lower() for w in _UNBUILDABLE_WORDS))


def _unbuildable_cells(map_id, layout) -> frozenset:
    """Cells units walk but buildings can't stand on, from the map picture
    (by its legend; nothing without one), parsed once per map."""
    if map_id is not None and map_id in _UNBUILDABLE_CACHE:
        return _UNBUILDABLE_CACHE[map_id]
    layout = _dict(layout)
    rows, origin = layout.get("rows"), layout.get("origin")
    chars = _legend_unbuildable_chars(_legend_dict(layout.get("legend")))
    cells: frozenset = frozenset()
    if chars and isinstance(rows, list) and _cell(origin) is not None:
        ox, oy = _cell(origin)
        cells = frozenset(
            (ox + j, oy + i)
            for i, row in enumerate(rows) if isinstance(row, str)
            for j, ch in enumerate(row) if ch in chars
        )
    if map_id is not None and cells:
        _UNBUILDABLE_CACHE[map_id] = cells
    return cells


def _blocked_cells(map_id, layout) -> frozenset:
    """Impassable cells from the map picture (characters judged by its
    legend when it has one), parsed once per map."""
    if map_id is not None and map_id in _BLOCKED_CACHE:
        return _BLOCKED_CACHE[map_id]
    layout = _dict(layout)
    rows, origin = layout.get("rows"), layout.get("origin")
    chars = _legend_blocked_chars(_legend_dict(layout.get("legend"))) or _BLOCKED_CHARS
    cells: frozenset = frozenset()
    if isinstance(rows, list) and _cell(origin) is not None:
        ox, oy = _cell(origin)
        cells = frozenset(
            (ox + j, oy + i)
            for i, row in enumerate(rows) if isinstance(row, str)
            for j, ch in enumerate(row) if ch in chars
        )
    if map_id is not None and cells:
        _BLOCKED_CACHE[map_id] = cells
    return cells


def _unit_costs(rules: dict) -> dict[str, float]:
    """``rules.units[<id>].cost`` (and ``rules.buildings``, for pricing enemy
    defenses) from the game config, lowercased ids."""
    out = {}
    for section in ("buildings", "units"):
        for item, spec in _dict(rules.get(section)).items():
            cost = _dict(spec).get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                out[str(item).lower()] = float(cost)
    return out


def _mil_value(mil: dict, keys: tuple[str, ...], actors: list[Actor], costs: dict[str, float]) -> float:
    """Combat value only: the server's figure less the harvesters and MCVs
    it counts, else the summed cost of the combat actors we can see."""
    cost = lambda a: costs.get(a.type, C.KNOWN_COSTS.get(a.type, C.DEFAULT_UNIT_COST))  # noqa: E731
    economy = set(C.HARVESTER) | {C.MCV}
    for key in keys:
        if key in mil:
            trucks = sum(cost(a) for a in actors if a.type in economy and not a.wreck)
            return max(0.0, _num(mil.get(key)) - trucks)
    return sum(cost(a) for a in actors if a.type not in economy and not a.wreck)


def _empty() -> View:
    return View(
        ok=False, tick=0, game_seconds=0.0, remaining_seconds=None, cash=0.0,
        power_provided=0.0, power_drained=0.0, units=[], buildings=[],
        enemy_units=[], enemy_buildings=[], available=set(), production=[],
        events=[], base_center=None, spawns=[], bounds=None,
        army_value=0.0, enemy_army_value=0.0, kills_cost=0.0, deaths_cost=0.0,
        assets_value=0.0,
    )


def enemy_ore_field(view: View) -> tuple[int, int] | None:
    """The ore cell at the heart of the enemy's nearest ore field: the ore
    within ``ENEMY_ORE_RADIUS_CELLS`` of the enemy spawn, averaged, then
    snapped to a real ore cell. ``None`` without a layout or spawn."""
    enemy = enemy_spawn_guess(view)
    if enemy is None or not view.ore_cells:
        return None
    r2 = C.ENEMY_ORE_RADIUS_CELLS ** 2
    near = [c for c in view.ore_cells if dist2(c, enemy) <= r2]
    if not near:
        return None
    cx = sum(c[0] for c in near) / len(near)
    cy = sum(c[1] for c in near) / len(near)
    return min(near, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)


def enemy_spawn_guess(view: View) -> tuple[int, int] | None:
    """The enemy base: the server's ``enemy_base`` when given, else the spawn
    farthest from our base — our attack target when we see no enemy."""
    if view.enemy_base is not None:
        return view.enemy_base
    if not view.spawns or view.base_center is None:
        return None
    bx, by = view.base_center
    return max(view.spawns, key=lambda s: (s[0] - bx) ** 2 + (s[1] - by) ** 2)


def dist2(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    """Squared cell distance; ``inf`` when either point is unknown."""
    if a is None or b is None:
        return float("inf")
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
