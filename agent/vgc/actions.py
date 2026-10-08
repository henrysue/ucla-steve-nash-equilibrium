"""Reading the server's action templates and building legal payloads.

GameAPI offers Team Preview and each doubles turn as a single legal action
whose ``input["action"]`` is a *template* to fill in, so this module owns
three jobs:

- parse and validate the template (never guess a payload);
- describe the options to the model the way Section 3.2 wants them — named
  targets, move power/accuracy, and the type multiplier against each legal
  target, with immune targets flagged as the dominated actions they are;
- turn an answer back into a payload GameAPI will accept, and supply a
  deterministic type-aware choice for when no model answer arrives in time.

The validation mirrors ``examples/llm/pokemon.py``: the model only ever
picks option *indices* the server offered, and ``build_*`` re-checks the
result against the server's own numbers before anything is sent.
"""

from __future__ import annotations

from dataclasses import dataclass

from altruagent import GameState, LegalAction

from .typechart import effectiveness

PASS = {"type": "pass"}

# Move-id families the spec's representative action set needs to recognize
# (3.2: always include a Protect line, a speed-control line, a switch line).
PROTECTION = frozenset({
    "protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "burningbulwark",
    "silktrap", "obstruct", "maxguard", "wideguard", "quickguard", "craftyshield",
})
SPEED_CONTROL = frozenset({
    "tailwind", "trickroom", "icywind", "electroweb", "thunderwave", "bulldoze",
    "glaciate", "stringshot", "afteryou", "quash", "scaryface", "cottonspore",
})
REDIRECTION = frozenset({"followme", "ragepowder", "spotlight"})
SPREAD_MOVES_HITTING_ALL = frozenset({"earthquake", "surf", "bulldoze", "discharge", "lavaplume", "sludgewave", "explosion", "selfdestruct"})


class TemplateError(ValueError):
    """The server offered a template this agent cannot read."""


class InvalidAnswer(ValueError):
    """The model's answer does not fit the server's legal options."""


# -- team preview -------------------------------------------------------------


def parse_lineup(action: LegalAction) -> list[str]:
    """The roster a ``select_lineup`` template offers, de-duplicated."""
    template = _template(action, "select_lineup")
    roster = template.get("roster")
    if not isinstance(roster, list) or not all(isinstance(species, str) and species for species in roster):
        raise TemplateError(f"select_lineup template needs a roster of species names, got {roster!r}")
    unique = list(dict.fromkeys(roster))
    if len(unique) < 4:
        raise TemplateError(f"select_lineup needs at least 4 distinct species, got {unique!r}")
    return unique


def build_lineup(roster: list[str], bring: object, leads: object) -> dict:
    """Validate a bring-4 / lead-2 answer into a ``select_lineup`` payload."""
    if not isinstance(bring, list) or len(bring) != 4 or len(set(bring)) != 4 or not set(bring) <= set(roster):
        raise InvalidAnswer(f"bring must be 4 different species from the roster {roster}, got {bring!r}")
    if not isinstance(leads, list) or len(leads) != 2 or len(set(leads)) != 2 or not set(leads) <= set(bring):
        raise InvalidAnswer(f"leads must be 2 different species from bring {bring}, got {leads!r}")
    return {"type": "select_lineup", "bring": list(bring), "leads": list(leads)}


# -- doubles turns ------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    number: int
    active: object
    force_switch: object
    options: list[dict]


def parse_doubles(action: LegalAction) -> list[Slot]:
    """Both slots of a ``doubles_turn`` template, ordered 0 then 1."""
    template = _template(action, "doubles_turn")
    raw_slots = template.get("slots")
    if not isinstance(raw_slots, list) or len(raw_slots) != 2:
        raise TemplateError(f"doubles_turn template needs exactly 2 slots, got {raw_slots!r}")

    slots: dict[int, Slot] = {}
    for index, raw in enumerate(raw_slots):
        if not isinstance(raw, dict):
            raise TemplateError(f"doubles_turn slot {index} is not an object: {raw!r}")
        number = raw.get("slot", index)
        options = raw.get("options")
        if number not in (0, 1) or number in slots:
            raise TemplateError(f"doubles_turn slots must be numbered 0 and 1, got {raw_slots!r}")
        if not isinstance(options, list) or not options or not all(
            isinstance(option, dict) and option.get("type") in ("move", "switch", "pass") for option in options
        ):
            raise TemplateError(f"doubles_turn slot {number} has unreadable options: {options!r}")
        slots[number] = Slot(number, raw.get("active"), raw.get("force_switch"), options)
    return [slots[0], slots[1]]


def build_doubles(slots: list[Slot], answer: object) -> dict:
    """Validate a ``{"slot_0": {...}, "slot_1": {...}}`` answer into a payload.

    Enforces the two cross-slot rules GameAPI would otherwise reject: the
    slots cannot switch the same Pokémon in, and they cannot both pass while
    a real option exists.
    """
    if not isinstance(answer, dict):
        raise InvalidAnswer(f"a doubles turn answer must be an object with slot_0 and slot_1, got {answer!r}")
    slot_0 = _build_slot(slots[0], answer.get("slot_0"))
    slot_1 = _build_slot(slots[1], answer.get("slot_1"))
    if slot_0["type"] == slot_1["type"] == "switch" and slot_0["species"] == slot_1["species"]:
        raise InvalidAnswer(f"both slots cannot switch in the same Pokémon ({slot_0['species']})")
    if slot_0["type"] == slot_1["type"] == "pass" and not all(
        option["type"] == "pass" for slot in slots for option in slot.options
    ):
        raise InvalidAnswer("both slots cannot pass while a move or switch is available")
    return {"type": "doubles_turn", "slot_0": slot_0, "slot_1": slot_1}


def _build_slot(slot: Slot, answer: object) -> dict:
    if not isinstance(answer, dict):
        raise InvalidAnswer(f"slot_{slot.number} must be an object with 'option' and 'target'")
    index, target = answer.get("option"), answer.get("target")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(slot.options):
        raise InvalidAnswer(
            f"slot_{slot.number}.option must be an index 0..{len(slot.options) - 1}, got {index!r}"
        )
    option = slot.options[index]
    if option["type"] == "pass":
        return dict(PASS)
    if option["type"] == "switch":
        species = option.get("species")
        if not isinstance(species, str) or not species:
            raise InvalidAnswer(f"slot_{slot.number} switch option {index} has no species: {option!r}")
        return {"type": "switch", "species": species}

    move_id = option.get("move_id")
    if not isinstance(move_id, str) or not move_id:
        raise InvalidAnswer(f"slot_{slot.number} move option {index} has no move_id: {option!r}")
    choice = {"type": "move", "move_id": move_id}
    targets = [t for t in option.get("targets") or [] if isinstance(t, int) and not isinstance(t, bool)]
    if targets:
        if isinstance(target, bool) or target not in targets:
            raise InvalidAnswer(
                f"slot_{slot.number}: target {target!r} is not legal for {move_id}; choose one of {targets}"
            )
        choice["target"] = target
    return choice


# -- describing the options to the model --------------------------------------

_SIDE_LABEL = {
    "self": "SELF {species} — this Pokémon itself",
    "ally": "ALLY {species} — your OTHER active (legal, but it hits your own side)",
    "opponent": "OPPONENT {species}",
    "none": "no target needed (self, field or spread move)",
}

TARGET_GUIDE = (
    "Every move lists its legal targets with who they are: SELF is the Pokémon using the move, "
    "ALLY is your other active, OPPONENT is a foe. Answer with the target integer exactly as "
    "listed. Targeting ALLY or SELF is legal but hits your own side, so only do it on purpose."
)


def describe_slots(slots: list[Slot], state: GameState) -> list[dict]:
    """The option space as the model sees it (3.2).

    Each move option carries its own numbers plus, per legal target, the
    type multiplier and an ``immune`` flag. Nothing is removed — GameAPI
    stays authoritative about legality — but a dominated line is labelled so
    the model does not spend a matrix row on it.
    """
    observation = _observation(state)
    foes = _species_types(observation.get("opponent_active_pokemon"))
    allies = _species_types(observation.get("active_pokemon"))
    return [
        {
            "slot": slot.number,
            "active": slot.active,
            "force_switch": slot.force_switch,
            "options": [
                {"option": index, **_describe_option(option, slot, observation, foes, allies)}
                for index, option in enumerate(slot.options)
            ],
        }
        for slot in slots
    ]


def _describe_option(option: dict, slot: Slot, observation: dict, foes: dict, allies: dict) -> dict:
    if option.get("type") != "move":
        return {key: value for key, value in option.items()}

    details = _move_details(option, slot.number, observation)
    shown = {key: value for key, value in option.items() if key not in ("targets", "target_options")}
    # The option's own "type" is the option kind ("move"); the move's
    # element type lives in the observation, so it gets a distinct key.
    move_type = _move_type(option, details)
    if move_type:
        shown["move_type"] = move_type
    for key in ("category", "base_power", "accuracy", "priority", "current_pp"):
        if key in details and key not in shown:
            shown[key] = details[key]

    move_id = str(option.get("move_id") or "").lower()
    if move_id in PROTECTION:
        shown["family"] = "protection"
    elif move_id in SPEED_CONTROL:
        shown["family"] = "speed_control"
    elif move_id in REDIRECTION:
        shown["family"] = "redirection"

    shown["targets"] = [
        _describe_target(entry, move_type, foes, allies) for entry in _target_entries(option)
    ]
    return shown


def _describe_target(entry: dict, move_type: str | None, foes: dict, allies: dict) -> dict:
    side = entry.get("side")
    species = entry.get("species") or "(empty position)"
    described = {
        "target": entry.get("target"),
        "is": _SIDE_LABEL.get(side, "target {target}").format(species=species, target=entry.get("target")),
    }
    lookup = foes if side == "opponent" else allies
    types = lookup.get(species)
    if move_type and types:
        multiplier = effectiveness(move_type, types)
        described["type_multiplier"] = multiplier
        if multiplier == 0.0:
            described["immune"] = True
    return described


def _target_entries(option: dict) -> list[dict]:
    """Normalize ``target_options`` (newer servers) or bare ``targets``."""
    entries = option.get("target_options")
    if isinstance(entries, list) and entries:
        return [entry for entry in entries if isinstance(entry, dict)]
    return [
        {"target": target, "side": "opponent" if target > 0 else "ally" if target < 0 else "none"}
        for target in option.get("targets") or []
        if isinstance(target, int) and not isinstance(target, bool)
    ]


def _move_type(option: dict, details: dict) -> str | None:
    """The move's element type (FIRE, GRASS...), never the option kind.

    A template option's ``type`` is always ``"move"``/``"switch"``/``"pass"``,
    so the element type comes from the observation's ``available_moves``, or
    from an explicit ``move_type`` if a server ever supplies one inline.
    """
    for candidate in (option.get("move_type"), details.get("type")):
        if isinstance(candidate, str) and candidate and candidate.lower() not in ("move", "switch", "pass"):
            return candidate
    return None


def _move_details(option: dict, slot_number: int, observation: dict) -> dict:
    """Merge the template option with the observation's ``available_moves``."""
    available = observation.get("available_moves")
    if not isinstance(available, list) or slot_number >= len(available):
        return {}
    moves = available[slot_number]
    if not isinstance(moves, list):
        return {}
    move_id = option.get("move_id")
    for move in moves:
        if isinstance(move, dict) and move.get("id") == move_id:
            return move
    return {}


def _species_types(actives: object) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for entry in actives if isinstance(actives, list) else []:
        if not isinstance(entry, dict):
            continue
        species = entry.get("species") or entry.get("name")
        types = entry.get("types")
        if isinstance(species, str) and isinstance(types, list):
            found[species] = [t for t in types if isinstance(t, str)]
    return found


# -- deterministic fallback ---------------------------------------------------


def heuristic_doubles(slots: list[Slot], state: GameState) -> dict:
    """A legal, type-aware turn with no model call.

    Used when the clock runs out or the model never answers. Per slot it
    takes the highest-scoring damaging move at the best legal opponent
    (never an immunity, never our own side unless nothing else is legal),
    falling back to a status/protection move, then a switch, then pass.
    """
    observation = _observation(state)
    foes = _species_types(observation.get("opponent_active_pokemon"))
    hp = _hp_fractions(observation.get("opponent_active_pokemon"))

    slot_0 = _heuristic_slot(slots[0], observation, foes, hp, taken=None)
    taken = slot_0.get("species") if slot_0["type"] == "switch" else None
    slot_1 = _heuristic_slot(slots[1], observation, foes, hp, taken=taken)
    return build_doubles(slots, {
        "slot_0": {"option": slot_0.pop("_option"), "target": slot_0.get("target")},
        "slot_1": {"option": slot_1.pop("_option"), "target": slot_1.get("target")},
    })


def _heuristic_slot(slot: Slot, observation: dict, foes: dict, hp: dict, *, taken: str | None) -> dict:
    kinds = {option["type"] for option in slot.options}
    if kinds == {"pass"}:
        return {"type": "pass", "_option": _index_of(slot, lambda o: o["type"] == "pass")}

    best: tuple[float, int, int | None] | None = None
    for index, option in enumerate(slot.options):
        if option["type"] != "move":
            continue
        score, target = _score_move(option, slot.number, observation, foes, hp)
        if best is None or score > best[0]:
            best = (score, index, target)
    if best is not None and best[0] > 0:
        return {"type": "move", "target": best[2], "_option": best[1]}

    for index, option in enumerate(slot.options):
        if option["type"] == "switch" and option.get("species") != taken:
            return {"type": "switch", "species": option.get("species"), "_option": index}
    if best is not None:
        return {"type": "move", "target": best[2], "_option": best[1]}
    if "pass" in kinds:
        return {"type": "pass", "_option": _index_of(slot, lambda o: o["type"] == "pass")}
    raise TemplateError(f"doubles_turn slot {slot.number} has no move, distinct switch, or pass: {slot.options!r}")


def _score_move(option: dict, slot_number: int, observation: dict, foes: dict, hp: dict) -> tuple[float, int | None]:
    """Rough expected damage, and the target that maximizes it."""
    details = _move_details(option, slot_number, observation)
    move_id = str(option.get("move_id") or "").lower()
    power = _number(option.get("base_power", details.get("base_power"))) or 0.0
    accuracy = _number(option.get("accuracy", details.get("accuracy")))
    accuracy = 1.0 if accuracy is None else (accuracy / 100.0 if accuracy > 1.0 else accuracy)
    move_type = _move_type(option, details)

    entries = _target_entries(option)
    if not entries:  # a field or self move: valued below any real attack
        return (_support_value(move_id), None)

    best: tuple[float, int | None] | None = None
    for entry in entries:
        target = entry.get("target")
        if not isinstance(target, int) or isinstance(target, bool):
            continue
        side = entry.get("side") or ("opponent" if target > 0 else "ally" if target < 0 else "none")
        multiplier = effectiveness(move_type, foes.get(entry.get("species") or "")) if side == "opponent" else 1.0
        if power <= 0:
            score = _support_value(move_id) if side != "ally" else 0.0
        elif side == "ally" or side == "self":
            score = 0.0  # never the heuristic's choice while anything else scores
        elif multiplier == 0.0:
            score = 0.0  # immune: strictly dominated (3.2)
        else:
            score = power * accuracy * multiplier
            if move_id in SPREAD_MOVES_HITTING_ALL:
                score *= 0.75  # spread reduction
            # Chip a weakened foe out rather than spread damage around.
            score *= 1.0 + 0.5 * (1.0 - hp.get(entry.get("species") or "", 1.0))
        if best is None or score > best[0]:
            best = (score, target)
    return best or (0.0, None)


def _support_value(move_id: str) -> float:
    """Where a non-damaging move sits against raw damage scores."""
    if move_id in SPEED_CONTROL:
        return 45.0
    if move_id in REDIRECTION:
        return 35.0
    if move_id in PROTECTION:
        return 25.0
    return 15.0


def _hp_fractions(actives: object) -> dict[str, float]:
    found: dict[str, float] = {}
    for entry in actives if isinstance(actives, list) else []:
        if not isinstance(entry, dict):
            continue
        species = entry.get("species") or entry.get("name")
        current, maximum = _number(entry.get("current_hp")), _number(entry.get("max_hp"))
        fraction = _number(entry.get("hp_fraction"))
        if fraction is None and current is not None and maximum:
            fraction = current / maximum
        if isinstance(species, str) and fraction is not None:
            found[species] = max(0.0, min(1.0, fraction))
    return found


# -- shared helpers -----------------------------------------------------------


def _template(action: LegalAction, expected: str) -> dict:
    template = action.input.get("action")
    if not isinstance(template, dict) or template.get("type") != expected:
        raise TemplateError(f"{expected} legal action has no {expected!r} template in input['action']: {template!r}")
    return template


def _observation(state: GameState) -> dict:
    observation = state.raw.get("observation")
    return observation if isinstance(observation, dict) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _index_of(slot: Slot, predicate) -> int:
    return next(index for index, option in enumerate(slot.options) if predicate(option))
