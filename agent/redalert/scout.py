"""Scouting and cheese detection.

Most RTS bots fold to early cheese (MCV rush, base-move, unit rush), so an
early read is high value — but a FALSE cheese alarm is worse than a missed one:
over-reacting to a lone scout trashes our own tempo, the very mistake we want to
punish in the opponent. So detection here is deliberately conservative (a
*threshold* of enemy combat value near our base, not one unit) and the reaction
auto-expires if the threat does not materialize (handled by the brain/executor
via ``CHEESE_EXPIRE_SECONDS``).
"""

from __future__ import annotations

from . import config as C
from .observe import View, dist2

# Enemy unit types that are NOT a threat worth panicking over (a lone harvester
# or a single scout wandering by is normal, not cheese).
_NON_THREAT = set(C.HARVESTER)


def enemy_threat_near_base(view: View) -> list:
    """Enemy combat actors within ``CHEESE_NEAR_BASE_CELLS`` of our base.

    Counts units and a forward enemy MCV (a base-move/MCV-rush tell). Harvesters
    are ignored. Returns the offending actors so the caller can size the threat.
    """
    if view.base_center is None:
        return []
    radius2 = C.CHEESE_NEAR_BASE_CELLS ** 2
    near = []
    for actor in view.enemy_units:
        if actor.type in _NON_THREAT:
            continue
        if dist2(actor.cell, view.base_center) <= radius2:
            near.append(actor)
    # A visible enemy MCV / construction yard close to us is itself cheese.
    for actor in view.enemy_buildings:
        if actor.type in (C.MCV, C.CONSTRUCTION_YARD) and dist2(actor.cell, view.base_center) <= radius2:
            near.append(actor)
    return near


def early_under_attack(view: View) -> bool:
    """An ``under_attack`` event before we have a real army is a cheese tell."""
    if len(view.combat_units) >= 4:
        return False
    for event in view.events:
        kind = str(event.get("type") or event.get("kind") or "").lower()
        if "under_attack" in kind or "attacked" in kind:
            return True
    return False


def detect_cheese(view: View) -> dict | None:
    """A conservative cheese verdict, or ``None`` when the board looks normal.

    Fires only when enough enemy combat value is near our base, or a forward
    enemy MCV is seen, or we are taking damage before we have an army. The
    thresholds are intentionally high to avoid tempo-wrecking false alarms.
    """
    near = enemy_threat_near_base(view)
    forward_mcv = any(a.type in (C.MCV, C.CONSTRUCTION_YARD) for a in near)
    if forward_mcv or len(near) >= C.CHEESE_MIN_ENEMY_UNITS:
        return {
            "kind": "forward_mcv" if forward_mcv else "unit_rush",
            "near_base": len(near),
            "detected_at": view.game_seconds,
        }
    if early_under_attack(view):
        return {"kind": "early_damage", "near_base": len(near), "detected_at": view.game_seconds}
    return None


def choke_cell(view: View) -> tuple[int, int] | None:
    """A choke/approach cell to park the infantry tripwire on.

    Simple and robust: the midpoint between our base and the likely enemy
    spawn, which sits on the natural approach. Falls back to a cell a short way
    toward the map center if spawns are unknown.
    """
    from .observe import enemy_spawn_guess

    base = view.base_center
    if base is None:
        return None
    enemy = enemy_spawn_guess(view)
    if enemy is not None:
        return ((base[0] + enemy[0]) // 2, (base[1] + enemy[1]) // 2)
    if view.bounds is not None:
        x, y, w, h = view.bounds
        center = (x + w // 2, y + h // 2)
        return ((base[0] + center[0]) // 2, (base[1] + center[1]) // 2)
    return None
