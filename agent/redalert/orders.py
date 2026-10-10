"""Order builders and shape validation.

Thin, typed constructors for the order commands the executor uses, plus a cheap
shape check so a malformed order never leaves this module (the server validates
again). The command/field vocabulary and limits follow the platform's guide,
https://api.altruagent-game.com/skill/redalert section 5 (as does
``examples/llm/redalert.py``): ``build``/``place``/``train`` for production,
``deploy`` for the MCV, ``attack_move``/``attack``/``move``/``stop`` for units,
``harvest`` for ore trucks, ``set_stance``/``rally`` for posture.

Everything returns plain dicts; a batch is ``{"type": "orders", "orders": [...]}``.
The executor assembles batches and caps them at the server's limit.
"""

from __future__ import annotations

MAX_ORDERS_PER_BATCH = 20
MAX_UNITS_PER_ORDER = 50
# Each order expands to engine commands (one per unit, or one per trained
# item); a batch over this is refused whole (skill/redalert section 5).
MAX_ENGINE_COMMANDS = 200

# cmd -> (required fields, optional fields) — a subset of the full schema: only
# what this agent emits. The server rejects unknown fields, so we send none.
_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "build": (frozenset({"item"}), frozenset()),
    "place": (frozenset({"item"}), frozenset({"at"})),
    "train": (frozenset({"item"}), frozenset({"count"})),
    "cancel": (frozenset({"item"}), frozenset()),
    "repair": (frozenset({"building"}), frozenset({"on"})),
    "sell": (frozenset({"building"}), frozenset()),
    "deploy": (frozenset({"units"}), frozenset()),
    "move": (frozenset({"units", "to"}), frozenset({"queued"})),
    "attack_move": (frozenset({"units", "to"}), frozenset({"queued"})),
    "attack": (frozenset({"units"}), frozenset({"target", "to", "queued"})),
    "stop": (frozenset({"units"}), frozenset()),
    "harvest": (frozenset({"units"}), frozenset({"to", "queued"})),
    "set_stance": (frozenset({"units", "stance"}), frozenset()),
    "rally": (frozenset({"building", "to"}), frozenset()),
}


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _cell(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_int(v) for v in value)


def order_ok(order) -> bool:
    """Cheap shape check: known cmd, required fields present, no stray fields,
    field values well formed. The server does the authoritative check."""
    if not isinstance(order, dict):
        return False
    cmd = order.get("cmd")
    if cmd not in _FIELDS:
        return False
    required, optional = _FIELDS[cmd]
    keys = set(order) - {"cmd"}
    if not required <= keys or keys - required - optional:
        return False
    if cmd == "attack" and ("target" in order) == ("to" in order):
        return False  # exactly one of target / to
    return all(_field_ok(key, order[key]) for key in keys)


def _field_ok(key: str, value) -> bool:
    """Shape check for one order field value."""
    if key == "units":
        return (
            isinstance(value, list) and 1 <= len(value) <= MAX_UNITS_PER_ORDER
            and all(_int(u) and u >= 0 for u in value) and len(set(value)) == len(value)
        )
    if key in ("to", "at"):
        return _cell(value)
    if key in ("target", "building"):
        return _int(value) and value >= 1
    if key == "item":
        return isinstance(value, str) and bool(value.strip())
    if key == "count":
        return _int(value) and value >= 1
    if key == "queued":
        return isinstance(value, bool)
    if key == "stance":
        return isinstance(value, str) and bool(value.strip())
    return True  # unknown field: the server decides


# -- builders ----------------------------------------------------------------


def build(item: str) -> dict:
    return {"cmd": "build", "item": item}


def place(item: str, at=None) -> dict:
    order = {"cmd": "place", "item": item}
    if at is not None:
        order["at"] = list(at)
    return order


def train(item: str, count: int = 1) -> dict:
    return {"cmd": "train", "item": item, "count": max(1, int(count))}


def cancel(item: str) -> dict:
    """Cancel one queued item of that type (the money comes back)."""
    return {"cmd": "cancel", "item": item}


def repair(building_id: int) -> dict:
    return {"cmd": "repair", "building": building_id}


def sell(building_id: int) -> dict:
    return {"cmd": "sell", "building": building_id}


def deploy(unit_id: int) -> dict:
    return {"cmd": "deploy", "units": [unit_id]}


def attack_move(unit_ids: list[int], to, queued: bool = False) -> dict:
    order = {"cmd": "attack_move", "units": list(unit_ids), "to": list(to)}
    if queued:
        order["queued"] = True  # after the unit's current order, not instead of it
    return order


def move(unit_ids: list[int], to) -> dict:
    return {"cmd": "move", "units": list(unit_ids), "to": list(to)}


def attack(unit_ids: list[int], target: int) -> dict:
    return {"cmd": "attack", "units": list(unit_ids), "target": target}


def harvest(unit_ids: list[int], to=None) -> dict:
    order = {"cmd": "harvest", "units": list(unit_ids)}
    if to is not None:
        order["to"] = list(to)     # this ore field, not wherever the engine picks
    return order


def rally(building_id: int, to) -> dict:
    return {"cmd": "rally", "building": building_id, "to": list(to)}


def stop(unit_ids: list[int]) -> dict:
    return {"cmd": "stop", "units": list(unit_ids)}


def set_stance(unit_ids: list[int], stance: str) -> dict:
    return {"cmd": "set_stance", "units": list(unit_ids), "stance": stance}


def batch(orders: list[dict]) -> dict:
    """Validate, de-dupe-cap, and wrap orders into a sendable move.

    Drops malformed orders (defensive) and caps at the server's batch limit
    and engine-command budget, keeping orders in priority order. Returns
    ``None`` when nothing valid remains, so the caller can WAIT.
    """
    valid, commands = [], 0
    for order in orders:
        if not order_ok(order) or len(valid) >= MAX_ORDERS_PER_BATCH:
            continue
        cost = engine_commands(order)
        if commands + cost > MAX_ENGINE_COMMANDS:
            continue  # a later, smaller order may still fit
        valid.append(order)
        commands += cost
    if not valid:
        return None
    return {"type": "orders", "orders": valid}


def engine_commands(order: dict) -> int:
    """How many engine commands an order expands to (skill/redalert §5)."""
    if isinstance(order.get("units"), list):
        return len(order["units"])
    if order.get("cmd") == "train":
        return int(order.get("count", 1))
    return 1
