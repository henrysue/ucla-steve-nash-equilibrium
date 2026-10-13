"""Rule-based unit micro: instant, every view, no model call.

A model answers in seconds; a fight turns in fractions of one. So the micro
that has to react at once is rules:

- **Focus fire.** Our units with enemy fighters in reach (``FOCUS_RADIUS_CELLS``,
  about weapon range) shoot the weakest of them, and keep that target while
  it lives and stays in reach. Buildings are left to the attack-move. (It
  used to re-pick every view over a 6-cell radius, buildings included:
  units switched targets and walked out of formation into turret range.)
- No pulling damaged vehicles out of attacks: lone tanks walking home
  through the choke died anyway and left the fight short (match 5e840776).

Units under micro this view are taken out of the army's general orders, so
the two never fight over the same unit. Squads with their own job (scout,
raid squad, home guard, garrison) are left to it.
"""

from __future__ import annotations

from . import config as C
from . import orders
from .modes import Mode
from .observe import Actor, View, dist2


def plan(view: View, mode: Mode, skip: frozenset, stage: tuple[int, int] | None) -> tuple[list[dict], set[int]]:
    """(orders, ids of the units they control)."""
    if mode is Mode.RETREAT or mode is Mode.PROTECT_LEAD:
        return [], set()      # retreating / sitting on a lead: no new fights
    fighters = [u for u in view.combat_units if u.id is not None and u.cell is not None and u.id not in skip]
    out: list[dict] = []
    controlled: set[int] = set()

    # Siege units (V2 / artillery) shell visible enemy defenses first: they
    # outrange them, which is the whole point of building them.
    defenses = [a for a in view.enemy_buildings
                if a.type in C.TURRETS and a.id is not None and a.cell is not None and not a.wreck]
    s2 = C.SIEGE_TARGET_RADIUS_CELLS ** 2
    for u in fighters:
        if u.type in C.SIEGE_UNITS and u.id not in controlled:
            near = [d for d in defenses if dist2(u.cell, d.cell) <= s2]
            if near:
                target = min(near, key=lambda d: dist2(u.cell, d.cell))
                out.append(orders.attack([u.id], target.id))
                controlled.add(u.id)

    # Focus fire: group units by the weakest enemy fighter in their reach;
    # a unit keeps its target while that target lives and stays in reach.
    enemies = {a.id: a for a in view.enemy_units if a.id is not None and a.cell is not None and not a.wreck
               and a.type not in C.HARVESTER}
    if not enemies:
        _FOCUS.clear()
        return out, controlled
    # In an attack, out to weapon range: units hitting a building turn onto
    # their fighters shooting at them (a4a78f76 at 5:40 lost 1500 for 300
    # firing at war factories while the defenders fired at them).
    r2 = (C.FOCUS_RADIUS_ATTACK_CELLS if mode is Mode.ALL_IN else C.FOCUS_RADIUS_CELLS) ** 2
    by_target: dict[int, list[int]] = {}
    for u in fighters:
        if u.id in controlled:
            continue
        kept = enemies.get(_FOCUS.get(u.id))
        if kept is not None and dist2(u.cell, kept.cell) <= r2:
            target = kept
        else:
            near = [e for e in enemies.values() if dist2(u.cell, e.cell) <= r2]
            if not near:
                _FOCUS.pop(u.id, None)
                continue
            target = min(near, key=_focus_key)
        _FOCUS[u.id] = target.id
        by_target.setdefault(target.id, []).append(u.id)
    for target_id, ids in by_target.items():
        out.append(orders.attack(ids[:orders.MAX_UNITS_PER_ORDER], target_id))
        controlled.update(ids)
    return out, controlled


_FOCUS: dict[int, int] = {}   # our unit id -> the enemy it is focusing


def _focus_key(enemy: Actor):
    """Weakest first; combat units before buildings; harvesters last (raids
    handle those); health as the tie-breaker."""
    building = enemy.type in C.ALL_BUILDING_TYPES
    harvester = enemy.type in C.HARVESTER
    return (harvester, building, enemy.hp if enemy.hp > 0 else 100.0)
