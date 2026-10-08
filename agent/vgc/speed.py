"""Deterministic turn-order prior for a doubles turn.

Turn order is the backbone of doubles: it decides whether a KO lands before
the opponent moves, whether Protect is needed, and whether Trick Room flips
everything. The model should not have to re-derive it from base stats every
turn under a clock, so this module computes a coarse ordering once, in code,
and the strategist hands it to the model as ``reference_speed_order`` and
uses it to sharpen the heuristic fallback.

What this is and is NOT, stated plainly because a WRONG turn-order hint is
worse than none:

- It is a BASE-SPEED prior. Base Speed comes from the format table
  (``meta.base_speed``). The agent cannot see the opponent's EVs, nature,
  held item (Choice Scarf!), or stat stages, so the ordering is "all else
  equal, who is faster" and is explicitly labelled as such.
- Priority brackets beat Speed and are NOT reversed by Trick Room. We know
  our own moves' priority (the observation lists it); we do not know the
  opponent's, so the Speed ordering is reported separately from our known
  priority moves rather than pretending to merge them.
- Field effects are applied ONLY when the live observation actually reveals
  them. Because the platform's exact field key names are not documented
  here, several plausible spellings are scanned; an effect that is present
  under none of them is simply treated as absent, which keeps the base-speed
  prior correct rather than inventing a wrong adjustment.

Nothing here raises: missing data drops a Pokemon from the ordering, and a
total absence of base speeds yields an empty result the caller omits.
"""

from __future__ import annotations

from dataclasses import dataclass

# Field-condition key spellings scanned in the observation. The platform's
# own names are authoritative and not documented in-repo, so we look under
# the common variants and treat "found under none" as "not active" — never
# as a guess. Matched case-insensitively against both the observation root
# and a nested "field"/"field_conditions"/"side_conditions" map if present.
_TRICK_ROOM_KEYS = ("trick_room", "trickroom", "trick_room_turns", "is_trick_room")
_TAILWIND_SELF_KEYS = ("tailwind", "our_tailwind", "ally_tailwind", "tailwind_turns")
_TAILWIND_OPP_KEYS = ("opponent_tailwind", "opp_tailwind", "foe_tailwind", "their_tailwind")
_FIELD_CONTAINERS = ("field", "field_conditions", "conditions", "side_conditions", "sides")

# Paralysis quarters/halves Speed depending on generation; Gen 7+ halves it.
_PARALYSIS = 0.5
_TAILWIND = 2.0


@dataclass(frozen=True)
class SpeedUnit:
    """One active Pokemon's effective base-speed figure for ordering."""

    species: str
    side: str  # "ours" or "theirs"
    base_speed: int
    effective: float  # base speed after field/status multipliers we can see
    note: str  # what, if anything, adjusted it (empty when nothing did)


def _truthy(value: object) -> bool:
    """A field flag counts as active when truthy, or a positive turn count."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on", "active"}
    return False


def _scan_flag(observation: dict, keys: tuple[str, ...]) -> bool:
    """True if any of ``keys`` is present-and-truthy at the observation root
    or inside a nested field/conditions container. Case-insensitive."""
    def check(mapping: object) -> bool:
        if not isinstance(mapping, dict):
            return False
        lowered = {str(k).lower(): v for k, v in mapping.items()}
        return any(_truthy(lowered.get(key)) for key in keys)

    if check(observation):
        return True
    for container in _FIELD_CONTAINERS:
        value = observation.get(container) if isinstance(observation, dict) else None
        if check(value):
            return True
        # one more nesting level (e.g. sides -> {our/their: {...}})
        if isinstance(value, dict):
            for inner in value.values():
                if check(inner):
                    return True
        elif isinstance(value, list):
            for inner in value:
                if check(inner):
                    return True
    return False


def _status_of(entry: dict) -> str:
    for key in ("status", "status_condition", "major_status"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _active_entries(observation: dict, key: str) -> list[dict]:
    value = observation.get(key)
    return [e for e in value if isinstance(e, dict)] if isinstance(value, list) else []


def _species_of(entry: dict) -> str:
    name = entry.get("species") or entry.get("name")
    return name if isinstance(name, str) and name else ""


def turn_order(observation: dict, base_speed) -> dict | None:
    """Build a base-speed turn-order prior from the live observation.

    ``base_speed`` is a callable ``species -> int | None`` (``meta.base_speed``).
    Returns a dict ready to drop into a prompt, or ``None`` when no active
    Pokemon has a known base speed (so the caller shows nothing). The result
    is deliberately coarse and labelled as a base-speed prior.
    """
    if not isinstance(observation, dict):
        return None

    trick_room = _scan_flag(observation, _TRICK_ROOM_KEYS)
    tailwind_ours = _scan_flag(observation, _TAILWIND_SELF_KEYS)
    tailwind_theirs = _scan_flag(observation, _TAILWIND_OPP_KEYS)

    units: list[SpeedUnit] = []
    for key, side, has_tailwind in (
        ("active_pokemon", "ours", tailwind_ours),
        ("opponent_active_pokemon", "theirs", tailwind_theirs),
    ):
        for entry in _active_entries(observation, key):
            species = _species_of(entry)
            if not species:
                continue
            base = base_speed(species)
            if base is None:
                continue
            effective = float(base)
            notes = []
            if has_tailwind:
                effective *= _TAILWIND
                notes.append("Tailwind x2")
            if _status_of(entry) in {"par", "paralysis", "paralyzed"}:
                effective *= _PARALYSIS
                notes.append("paralysis x0.5")
            units.append(SpeedUnit(species, side, base, effective, ", ".join(notes)))

    if not units:
        return None

    # Trick Room reverses the SPEED tier (not priority). Sort fast-first by
    # default, slow-first under Trick Room. Species with equal effective
    # speed are a genuine speed tie (coin flip) and flagged as such.
    units_sorted = sorted(units, key=lambda u: u.effective, reverse=not trick_room)

    order = []
    for rank, unit in enumerate(units_sorted, start=1):
        item = {
            "rank": rank,
            "species": unit.species,
            "side": unit.side,
            "base_speed": unit.base_speed,
        }
        if unit.effective != float(unit.base_speed):
            item["effective_speed"] = round(unit.effective, 1)
        if unit.note:
            item["adjusted_by"] = unit.note
        order.append(item)

    # Flag speed ties (same effective speed, which actually resolves 50/50).
    ties = _speed_ties(units_sorted)

    result: dict = {
        "mode": "Trick Room (slower moves first)" if trick_room else "normal (faster moves first)",
        "order_fast_to_slow": order,
        "basis": (
            "BASE Speed from the format table, adjusted only for field effects visible in the "
            "observation (Trick Room, Tailwind, paralysis). It does NOT account for the "
            "opponent's unknown EVs, nature, held item (e.g. Choice Scarf), or stat stages, and "
            "priority moves ignore it. Treat it as a prior and verify against move order you see."
        ),
    }
    if trick_room:
        result["trick_room_active"] = True
    if tailwind_ours or tailwind_theirs:
        result["tailwind"] = {"ours": tailwind_ours, "theirs": tailwind_theirs}
    if ties:
        result["speed_ties_50_50"] = ties
    return result


def _speed_ties(units_sorted: list[SpeedUnit]) -> list[list[str]]:
    """Groups of 2+ Pokemon with identical effective speed (true 50/50 ties)."""
    groups: list[list[str]] = []
    current: list[SpeedUnit] = []
    for unit in units_sorted:
        if current and unit.effective == current[-1].effective:
            current.append(unit)
        else:
            if len(current) >= 2:
                groups.append([u.species for u in current])
            current = [unit]
    if len(current) >= 2:
        groups.append([u.species for u in current])
    return groups


def faster_than(observation: dict, base_speed, ours: str, theirs: str) -> bool | None:
    """Does our ``ours`` outspeed their ``theirs`` under the visible field?

    Returns ``True``/``False`` on a confident base-speed comparison, or
    ``None`` when either base speed is unknown or they are tied (so the
    caller does not act on a coin flip). Used by the heuristic fallback,
    which has no model to reason about speed otherwise.
    """
    order = turn_order(observation, base_speed)
    if order is None:
        return None
    speeds = {}
    for item in order["order_fast_to_slow"]:
        speeds[item["species"]] = item.get("effective_speed", float(item["base_speed"]))
    a, b = speeds.get(ours), speeds.get(theirs)
    if a is None or b is None or a == b:
        return None
    trick_room = order.get("trick_room_active", False)
    # Under Trick Room the SLOWER Pokemon moves first.
    return (a < b) if trick_room else (a > b)
