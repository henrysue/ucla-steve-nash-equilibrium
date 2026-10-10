"""Rule-based unit micro: instant, every view, no model call.

A model answers in seconds; a fight turns in fractions of one. So the micro
that has to react at once is rules:

- **Focus fire.** Our units with enemies close by all shoot the weakest enemy
  within ``FOCUS_RADIUS_CELLS`` of them (lowest health first; units before
  buildings; harvesters last), so enemies die faster and stop shooting back
  sooner, instead of every unit spreading damage over whatever is nearest.
- **Pull back the badly damaged.** During an attack, a vehicle below
  ``PULL_BACK_HP`` withdraws to the staging point instead of dying for
  nothing: its value stays ours for the tiebreak. (The platform has no order
  to send a unit into a repair bay, so this saves units rather than heals them.)

Units under micro this view are taken out of the army's general orders, so
the two never fight over the same unit. Squads with their own job (scout,
raid squad, home guard, garrison) are left to it.
"""

from __future__ import annotations

from . import config as C
from . import orders
from .modes import DEFENSIVE, Mode
from .observe import Actor, View, dist2


def plan(view: View, mode: Mode, skip: frozenset, stage: tuple[int, int] | None) -> tuple[list[dict], set[int]]:
    """(orders, ids of the units they control)."""
    if mode is Mode.RETREAT or mode is Mode.PROTECT_LEAD:
        return [], set()      # retreating / sitting on a lead: no new fights
    fighters = [u for u in view.combat_units if u.id is not None and u.cell is not None and u.id not in skip]
    out: list[dict] = []
    controlled: set[int] = set()

    # Pull back badly damaged vehicles from an attack (not from base defense),
    # attack-moving: shot at on the way, they shoot back.
    if mode not in DEFENSIVE and stage is not None:
        hurt = [u.id for u in fighters if u.type in C.VEHICLE_TYPES and 0 < u.hp < C.PULL_BACK_HP
                and dist2(u.cell, stage) > C.REGROUP_RADIUS_CELLS ** 2]
        if hurt:
            out.append(orders.attack_move(hurt[:orders.MAX_UNITS_PER_ORDER], list(stage)))
            controlled.update(hurt)

    # Siege units (V2 / artillery) shell visible enemy defenses first: they
    # outrange them, which is the whole point of building them.
    defenses = [a for a in view.enemy_buildings
                if a.type in C.ENEMY_DEFENSE_TYPES and a.id is not None and a.cell is not None and not a.wreck]
    s2 = C.SIEGE_TARGET_RADIUS_CELLS ** 2
    for u in fighters:
        if u.type in C.SIEGE_UNITS and u.id not in controlled:
            near = [d for d in defenses if dist2(u.cell, d.cell) <= s2]
            if near:
                target = min(near, key=lambda d: dist2(u.cell, d.cell))
                out.append(orders.attack([u.id], target.id))
                controlled.add(u.id)

    # Focus fire: group units by the weakest enemy in their reach.
    enemies = [a for a in view.enemy_units if a.id is not None and a.cell is not None and not a.wreck]
    enemies += [a for a in view.enemy_buildings if a.id is not None and a.cell is not None and not a.wreck]
    if not enemies:
        return out, controlled
    r2 = C.FOCUS_RADIUS_CELLS ** 2
    by_target: dict[int, list[int]] = {}
    for u in fighters:
        if u.id in controlled:
            continue
        near = [e for e in enemies if dist2(u.cell, e.cell) <= r2]
        if not near:
            continue
        target = min(near, key=_focus_key)
        by_target.setdefault(target.id, []).append(u.id)
    for target_id, ids in by_target.items():
        out.append(orders.attack(ids[:orders.MAX_UNITS_PER_ORDER], target_id))
        controlled.update(ids)
    return out, controlled


def _focus_key(enemy: Actor):
    """Weakest first; combat units before buildings; harvesters last (raids
    handle those); health as the tie-breaker."""
    building = enemy.type in C.ALL_BUILDING_TYPES
    harvester = enemy.type in C.HARVESTER
    return (harvester, building, enemy.hp if enemy.hp > 0 else 100.0)
