"""Typed, defensive reads of the Red Alert observation.

Everything the executor and brain need to know about the game state, pulled
from ``state.raw`` into small typed views. Field names match the platform's
observation (as used by ``examples/llm/redalert.py``): ``observation`` holds
``economy``, ``military``, ``units``, ``buildings``, ``production``,
``available_production``, ``visible_enemies``, ``visible_enemy_buildings``,
``events``, ``base_center``; ``time`` holds the clock; ``map`` holds geometry.

Every read is defensive: a missing or malformed field yields a safe empty
default, never an exception. Unit/building ids are re-read every view because
they change and die (a deployed MCV becomes a construction yard with a new id).
"""

from __future__ import annotations

from dataclasses import dataclass

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


def _actor(raw: dict) -> Actor:
    return Actor(
        id=raw.get("id") if isinstance(raw.get("id"), int) and not isinstance(raw.get("id"), bool) else None,
        type=str(raw.get("type") or "").lower(),
        cell=_cell(raw),
        hp=_num(raw.get("hp")),
        idle=bool(raw.get("idle")),
        wreck=bool(raw.get("wreck")),
    )


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
        """Our units that are not harvesters or the MCV (i.e. can fight)."""
        skip = set(C.HARVESTER) | {C.MCV}
        return [u for u in self.units if u.type not in skip and not u.wreck]

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


def read(raw: dict) -> View:
    """Build a :class:`View` from ``state.raw``. Never raises."""
    raw = _dict(raw)
    obs = _dict(raw.get("observation"))
    if not obs:
        return _empty()
    econ = _dict(obs.get("economy"))
    mil = _dict(obs.get("military"))
    time_info = _dict(raw.get("time"))
    map_info = {**_dict(raw.get("map"))}

    spawns = [c for c in (_cell(s) for s in map_info.get("spawns") or []) if c is not None]
    bounds = map_info.get("bounds")
    bounds_t = tuple(bounds) if isinstance(bounds, list) and len(bounds) == 4 and all(
        isinstance(n, int) and not isinstance(n, bool) for n in bounds
    ) else None

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
        units=[_actor(u) for u in _list_of_dicts(obs.get("units"))],
        buildings=[_actor(b) for b in _list_of_dicts(obs.get("buildings"))],
        enemy_units=[_actor(e) for e in _list_of_dicts(obs.get("visible_enemies"))],
        enemy_buildings=[_actor(e) for e in _list_of_dicts(obs.get("visible_enemy_buildings"))],
        available={str(x).lower() for x in obs.get("available_production") or [] if isinstance(x, str)},
        production=_list_of_dicts(obs.get("production")),
        events=_list_of_dicts(obs.get("events")),
        base_center=_cell(obs.get("base_center")),
        spawns=spawns,
        bounds=bounds_t,
        army_value=_num(mil.get("army_value", mil.get("value"))),
        enemy_army_value=_num(mil.get("visible_enemy_value", mil.get("enemy_value"))),
        kills_cost=_num(mil.get("kills_cost")),
        deaths_cost=_num(mil.get("deaths_cost")),
        assets_value=_num(econ.get("assets_value", mil.get("assets_value"))),
    )


def _empty() -> View:
    return View(
        ok=False, tick=0, game_seconds=0.0, remaining_seconds=None, cash=0.0,
        power_provided=0.0, power_drained=0.0, units=[], buildings=[],
        enemy_units=[], enemy_buildings=[], available=set(), production=[],
        events=[], base_center=None, spawns=[], bounds=None,
        army_value=0.0, enemy_army_value=0.0, kills_cost=0.0, deaths_cost=0.0,
        assets_value=0.0,
    )


def enemy_spawn_guess(view: View) -> tuple[int, int] | None:
    """The spawn farthest from our base — our attack target when we see no enemy."""
    if not view.spawns or view.base_center is None:
        return None
    bx, by = view.base_center
    return max(view.spawns, key=lambda s: (s[0] - bx) ** 2 + (s[1] - by) ** 2)


def dist2(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    """Squared cell distance; ``inf`` when either point is unknown."""
    if a is None or b is None:
        return float("inf")
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
