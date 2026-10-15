"""How likely each opposing active is to Protect this turn.

Damage sent into a Protect is a wasted action, and so is a Fake Out. One
simple number per foe, with its reason logged:

- it cannot Protect (its draft card, or all four moves seen, without it): 0;
- it protected last turn: 0 (a repeat works 1 time in 3, so nobody tries);
- first turn on the field: the rate this opponent has shown (from 35%; match
  e1f70879: 6 of 16 first turns across four matches were a Protect, not 70%);
- threatened super-effectively: a little above normal;
- otherwise: low.

Match 4e60a5a2: a style layer stacked on top of these read "always Protects"
from three first-turn Protects, held Torkoal at ~58% every turn while it
attacked, and steered every attack away from it. Keep this one table.
"""

from __future__ import annotations

import re

from .typechart import effectiveness

PROTECT_MOVES = frozenset({
    "protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "burningbulwark", "silktrap", "obstruct",
})
FIRST_TURN = 0.35  # leads and fresh switch-ins, before we have seen this opponent's
THREATENED = 0.35
NORMAL = 0.25
UNKNOWN = 0.20  # set unknown: it may not even carry Protect
LACKS = 0.03  # its usage set has no Protect (a guess, not certain)
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
    protected_last_turn=(),
    first_turn_rate: float = FIRST_TURN,
) -> dict[str, dict]:
    """species id -> {"likelihood": p, "why": [...]} for each opposing active.

    ``first_turn`` maps species id -> on its first turn on the field;
    ``common_moves`` is species -> list of move names (the reference
    table); ``card_set`` is species -> its exact draft card or ``None``, and
    a card's four moves are certain either way. ``protected_last_turn`` holds
    the species ids we saw (or inferred) Protect last turn; ``first_turn_rate``
    how often this opponent has Protected on a first turn out (style.py).
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
        revealed = _moves_listed(entry)
        common = {_id(m) for m in common_moves(name) or []}
        card = card_set(name) if card_set else None
        if card:
            exact = {_id(m) for m in card.get("moves") or []}
            base, carries = (NORMAL, "its draft card runs Protect") if exact & PROTECT_MOVES else (0.0, "its draft card has no Protect")
        elif revealed is not None and len(revealed) >= 4 and not set(revealed) & PROTECT_MOVES:
            base, carries = 0.0, "all four moves revealed, no Protect"
        elif revealed and set(revealed) & PROTECT_MOVES:
            base, carries = NORMAL, "Protect revealed"
        elif common & PROTECT_MOVES:
            base, carries = NORMAL, "commonly runs Protect"
        elif common:
            base, carries = LACKS, "common set has no Protect"
        else:
            base, carries = UNKNOWN, "set unknown"
        why = [carries]
        types = entry.get("types") if isinstance(entry.get("types"), list) else []
        if base <= LACKS:
            p = base
        elif key in protected_last_turn or _last_move(entry) in PROTECT_MOVES:
            p = 0.0
            why.append("protected last turn: it will not try again")
        elif first_turn.get(key):
            p = first_turn_rate
            why.append(f"first turn on the field (this opponent: {first_turn_rate:.0%})")
        elif types and any(effectiveness(t, types) > 1.0 for t in attack_types):
            p = max(base, THREATENED)
            why.append("we threaten it super-effectively")
        else:
            p = base
        found[key] = {"likelihood": round(p, 2), "why": why}
    return found


def both_likely(found: dict[str, dict]) -> bool:
    """Every opposing active is probably protecting this turn."""
    return bool(found) and all(v["likelihood"] >= LIKELY for v in found.values())
