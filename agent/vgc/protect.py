"""How likely each opposing active is to Protect this turn.

Damage sent into a Protect is a wasted action, and so is a Fake Out. The
estimate is deterministic and every factor is logged with it:

- carries Protect: revealed moves win; else the table's common moves
  (the Protect family listed, or a common set without it), else unknown;
- first turn on the field: leads and fresh switch-ins Protect more, to
  scout and to dodge Fake Out;
- threatened: one of our moves this turn hits it super-effectively;
- low HP: below half, Protect buys a turn of terrain/partner support;
- protected last turn: a repeat succeeds only 1/3 of the time, so few try.

Numbers are priors, clamped to 0.02-0.85: certainty either way is never
justified by what we can see.
"""

from __future__ import annotations

import re

from .typechart import effectiveness

PROTECT_MOVES = frozenset({
    "protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "burningbulwark", "silktrap", "obstruct",
})
CARRIES, LACKS, UNKNOWN = 0.30, 0.03, 0.20
FIRST_TURN = 1.4
THREATENED = 0.15
LOW_HP, LOW_HP_BONUS = 0.5, 0.10
REPEAT = 0.3
FLOOR, CEILING = 0.02, 0.85
LIKELY = 0.5  # at or above this, treat the Pokemon as probably protecting


def _id(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _moves_listed(entry: dict) -> list[str] | None:
    for key in ("revealed_moves", "known_moves", "moves"):
        value = entry.get(key)
        if isinstance(value, list):
            return [_id(m.get("id") or m.get("name") if isinstance(m, dict) else m) for m in value]
    return None


def _last_move(entry: dict) -> str:
    return _id(entry.get("last_move") or entry.get("last_move_used") or "")


def _hp(entry: dict) -> float | None:
    fraction = entry.get("hp_fraction")
    if isinstance(fraction, (int, float)) and not isinstance(fraction, bool):
        return float(fraction)
    current, maximum = entry.get("current_hp"), entry.get("max_hp")
    if isinstance(current, (int, float)) and isinstance(maximum, (int, float)) and maximum:
        return current / maximum
    return None


def _our_attack_types(observation: dict) -> list[str]:
    types = []
    for moves in observation.get("available_moves") or []:
        for move in moves if isinstance(moves, list) else []:
            if isinstance(move, dict) and (move.get("base_power") or 0) > 0 and isinstance(move.get("type"), str):
                types.append(move["type"].upper())
    return types


def odds(
    observation: dict,
    *,
    first_turn: dict[str, bool],
    common_moves,
    card_set=None,
) -> dict[str, dict]:
    """species id -> {"likelihood": p, "why": [...]} for each opposing active.

    ``first_turn`` maps species id -> on its first turn on the field;
    ``common_moves`` is species -> list of move names (the reference
    table); ``card_set`` is species -> its exact draft card or ``None``, and
    a card's four moves are certain either way.
    """
    attack_types = _our_attack_types(observation)
    found = {}
    for entry in observation.get("opponent_active_pokemon") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("species") or entry.get("name")
        key = _id(name)
        hp = _hp(entry)
        if not key or (hp is not None and hp <= 0):
            continue
        why = []
        revealed = _moves_listed(entry)
        common = {_id(m) for m in common_moves(name) or []}
        card = card_set(name) if card_set else None
        if card:
            exact = {_id(m) for m in card.get("moves") or []}
            p, carries = (CARRIES, "its draft card runs Protect") if exact & PROTECT_MOVES else (0.0, "its draft card has no Protect")
        elif revealed is not None and len(revealed) >= 4 and not set(revealed) & PROTECT_MOVES:
            p, carries = 0.0, "all four moves revealed, no Protect"
        elif revealed and set(revealed) & PROTECT_MOVES:
            p, carries = CARRIES, "Protect revealed"
        elif common & PROTECT_MOVES:
            p, carries = CARRIES, "commonly runs Protect"
        elif common:
            p, carries = LACKS, "common set has no Protect"
        else:
            p, carries = UNKNOWN, "set unknown"
        why.append(carries)
        if p > 0:
            if first_turn.get(key):
                p *= FIRST_TURN
                why.append("first turn on the field")
            types = entry.get("types") if isinstance(entry.get("types"), list) else []
            if types and any(effectiveness(t, types) > 1.0 for t in attack_types):
                p += THREATENED
                why.append("we threaten it super-effectively")
            if hp is not None and hp < LOW_HP:
                p += LOW_HP_BONUS
                why.append("below half HP")
            if _last_move(entry) in PROTECT_MOVES:
                p *= REPEAT
                why.append("protected last turn")
        # A set known to lack Protect (its card, or all four moves seen) cannot use it.
        p = min(CEILING, max(FLOOR, p)) if p > 0 else 0.0
        found[key] = {"likelihood": round(p, 2), "why": why}
    return found


def both_likely(found: dict[str, dict]) -> bool:
    """Every opposing active is probably protecting this turn."""
    return bool(found) and all(v["likelihood"] >= LIKELY for v in found.values())
