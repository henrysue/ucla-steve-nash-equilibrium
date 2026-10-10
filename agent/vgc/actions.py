"""Reading the server's action templates and building legal payloads.

GameAPI offers Team Preview and each doubles turn as a single legal action
whose ``input["action"]`` is a *template* to fill in, so this module owns
three jobs:

- parse and validate the template (never guess a payload);
- describe the options to the model the way Section 3.2 wants them — named
  targets, move power/accuracy, and the type multiplier against each legal
  target, with immune targets flagged as the dominated actions they are;
- turn an answer back into a payload GameAPI will accept, plus a
  guaranteed-legal floor for when everything else fails (the deterministic
  best move lives in lookahead.best_line).

The validation mirrors ``examples/llm/pokemon.py``: the model only ever
picks option *indices* the server offered, and ``build_*`` re-checks the
result against the server's own numbers before anything is sent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from altruagent import GameState, LegalAction

from . import abilities as ab
from .moverules import SlotRules
from .typechart import effectiveness

PASS = {"type": "pass"}

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


def build_doubles(slots: list[Slot], answer: object, rules: list[SlotRules] | None = None) -> dict:
    """Validate a ``{"slot_0": {...}, "slot_1": {...}}`` answer into a payload.

    Enforces the two cross-slot rules GameAPI would otherwise reject: the
    slots cannot switch the same Pokémon in, and they cannot both pass while
    a real option exists. With ``rules``, an option the move tracker knows
    will fail (Fake Out after the first turn) is rejected too.
    """
    if not isinstance(answer, dict):
        raise InvalidAnswer(f"a doubles turn answer must be an object with slot_0 and slot_1, got {answer!r}")
    slot_0 = _build_slot(slots[0], answer.get("slot_0"), _blocked(rules, 0))
    slot_1 = _build_slot(slots[1], answer.get("slot_1"), _blocked(rules, 1))
    if slot_0["type"] == slot_1["type"] == "switch" and slot_0["species"] == slot_1["species"]:
        raise InvalidAnswer(f"both slots cannot switch in the same Pokémon ({slot_0['species']})")
    if slot_0["type"] == slot_1["type"] == "pass" and not all(
        option["type"] == "pass" for slot in slots for option in slot.options
    ):
        raise InvalidAnswer("both slots cannot pass while a move or switch is available")
    return {"type": "doubles_turn", "slot_0": slot_0, "slot_1": slot_1}


def _blocked(rules: list[SlotRules] | None, number: int) -> dict[int, str]:
    return rules[number].blocked if rules else {}


def _build_slot(slot: Slot, answer: object, blocked: dict[int, str] | None = None) -> dict:
    if not isinstance(answer, dict):
        raise InvalidAnswer(f"slot_{slot.number} must be an object with 'option' and 'target'")
    index, target = answer.get("option"), answer.get("target")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(slot.options):
        raise InvalidAnswer(
            f"slot_{slot.number}.option must be an index 0..{len(slot.options) - 1}, got {index!r}"
        )
    if blocked and index in blocked:
        raise InvalidAnswer(f"slot_{slot.number} option {index}: {blocked[index]}")
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
        if target is None and len(targets) == 1:
            # Protect and other self moves list one target; the model often
            # leaves it null, and rejecting that dropped whole matrix rows.
            target = targets[0]
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


def describe_slots(
    slots: list[Slot],
    state: GameState,
    rules: list[SlotRules] | None = None,
    abilities=None,
    protect_odds=None,
    calculator=None,
) -> list[dict]:
    """The option space as the model sees it (3.2).

    Each move option carries its own numbers plus, per legal target, the
    type multiplier and an ``immune`` flag; a dominated line is labelled so
    the model does not spend a matrix row on it. With ``rules``, options the
    move tracker knows will fail are left out (indices keep the server's
    numbering) and history-dependent moves carry their rule. With
    ``abilities`` (species -> possible ability names), each opposing target
    also carries what its possible abilities would do to the move.
    """
    observation = _observation(state)
    foes = _species_types(observation.get("opponent_active_pokemon"))
    allies = _species_types(observation.get("active_pokemon"))
    described = []
    for slot in slots:
        rule = rules[slot.number] if rules else None
        options = []
        for index, option in enumerate(slot.options):
            if rule is not None and index in rule.blocked:
                continue
            shown = {
                "option": index,
                **_describe_option(option, slot, observation, foes, allies, abilities, protect_odds, calculator),
            }
            if rule is not None and option.get("type") == "move":
                shown.update(rule.note(str(option.get("move_id") or "").lower()))
            options.append(shown)
        described.append({
            "slot": slot.number, "active": slot.active, "force_switch": slot.force_switch, "options": options,
        })
    return described


def _describe_option(
    option: dict,
    slot: Slot,
    observation: dict,
    foes: dict,
    allies: dict,
    abilities=None,
    protect_odds=None,
    calculator=None,
) -> dict:
    if option.get("type") == "switch" and calculator is not None:
        shown = {key: value for key, value in option.items()}
        risk = calculator.incoming(option.get("species"))
        if risk:
            # Match 6a741e0d switched Flutter Mane into Tailwind-boosted
            # attacks and lost it before it moved: show the cost of entering.
            shown["switch_in_risk"] = risk
        return shown
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
    listed = movelist_lookup(move_id) or {}
    if move_id in _protecting():
        shown["family"] = "protection"
    elif "speed_control" in (listed.get("tags") or []):
        shown["family"] = "speed_control"
    elif move_id in ("followme", "ragepowder", "spotlight"):
        shown["family"] = "redirection"

    move = ab.Move(
        id=move_id,
        type=str(move_type or "").upper(),
        category=str(details.get("category") or option.get("category") or "").upper(),
        priority=int(_number(details.get("priority", option.get("priority"))) or 0),
    )
    entries = _target_entries(option)
    listed = movelist_lookup(move_id)
    if listed and listed["target"] in ("allAdjacentFoes", "allAdjacent"):
        # Spread moves list no targets: say who they hit and how hard.
        shown["hits"] = "both foes AND your own partner" if listed["target"] == "allAdjacent" else "both foes"
        shown["foe_type_multipliers"] = {foe: effectiveness(move.type, types) for foe, types in foes.items() if types}
        if listed["target"] == "allAdjacent":
            partner = next((types for name, types in allies.items() if name != _active_name(slot)), None)
            if partner:
                shown["partner_type_multiplier"] = effectiveness(move.type, partner)
    shown["targets"] = [
        _describe_target(entry, move_type, foes, allies, move, abilities, protect_odds) for entry in entries
    ]
    if calculator is not None and move.damaging:
        # PokéChamp's one-step lookahead: estimated damage and turns to KO.
        attacker = calculator.ours(slot.number)
        stats = {**details, "id": move_id, "type": move.type, "category": move.category}
        for described in shown["targets"]:
            species = next(
                (e.get("species") for e in entries if e.get("target") == described.get("target") and e.get("side") == "opponent"),
                None,
            )
            hit = calculator.damage(attacker, calculator.foe(species), stats) if species else None
            if hit:
                described["damage_estimate"] = hit
        if not entries:
            spread = {foe: calculator.damage(attacker, calculator.foe(foe), stats, spread=True) for foe in foes}
            spread = {foe: hit for foe, hit in spread.items() if hit}
            if spread:
                shown["damage_estimate_per_foe"] = spread
    if abilities is not None and not entries and move.damaging:
        # A spread move lists no targets but still hits every foe.
        per_foe = {foe: ab.warnings(move, abilities(foe) or []) for foe in foes}
        per_foe = {foe: found for foe, found in per_foe.items() if found}
        if per_foe:
            shown["foe_ability_warnings"] = per_foe
    return shown


def _protecting() -> frozenset:
    from . import movelist

    return movelist.PROTECTING


def movelist_lookup(move_id: str) -> dict | None:
    from . import movelist

    return movelist.lookup(move_id)


def _active_name(slot: Slot) -> str | None:
    active = slot.active
    if isinstance(active, dict):
        active = active.get("species") or active.get("name")
    return active if isinstance(active, str) else None


def _describe_target(
    entry: dict,
    move_type: str | None,
    foes: dict,
    allies: dict,
    move: ab.Move | None = None,
    abilities=None,
    protect_odds=None,
) -> dict:
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
    if side == "opponent" and protect_odds:
        chance = protect_odds.get(re.sub(r"[^a-z0-9]", "", str(species).lower()))
        if chance is not None:
            described["protect_likelihood"] = chance
    if side == "opponent" and move is not None and abilities is not None and species in foes:
        found = ab.warnings(move, abilities(species) or [])
        if found:
            described["ability_warnings"] = found
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


def move_details(option: dict, slot_number: int, observation: dict) -> dict:
    """The observation's data for one move option (type, power, category),
    with the option's own id so callers can key off it."""
    return {**_move_details(option, slot_number, observation), "id": option.get("move_id")}


def _move_details(option: dict, slot_number: int, observation: dict) -> dict:
    """Merge the template option with the observation's ``available_moves``."""
    from . import movelist

    # The move list fills whatever the observation leaves out (type, power,
    # category...); the observation wins wherever it has a value.
    known = movelist.details(option.get("move_id"))
    available = observation.get("available_moves")
    if not isinstance(available, list) or slot_number >= len(available):
        return known
    moves = available[slot_number]
    if not isinstance(moves, list):
        return known
    move_id = option.get("move_id")
    for move in moves:
        if isinstance(move, dict) and move.get("id") == move_id:
            return {**known, **{k: v for k, v in move.items() if v is not None}}
    return known


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


# -- the guaranteed-legal floor ------------------------------------------------


def guaranteed_action(state: GameState) -> dict:
    """A legal payload for the current decision that CANNOT raise.

    The floor beneath every other fallback. ``lookahead.best_line`` and
    ``build_lineup`` apply judgment and so can reject a malformed template
    with ``TemplateError``/``InvalidAnswer``; this one applies none. It reads
    the server's own first legal option per decision and returns it verbatim,
    so there is always *some* legal move to send rather than an escaping
    exception that forfeits the match (hard rule #1).

    Preference order, each strictly more defensive than the last:

    1. a ``doubles_turn`` built from each slot's first non-pass option
       (first move, else first switch, else pass), with that move's first
       listed target;
    2. a ``select_lineup`` built from the first four distinct roster species;
    3. the pass payload, which every doubles template permits.
    """
    universal_floor = {"type": "doubles_turn", "slot_0": dict(PASS), "slot_1": dict(PASS)}
    try:
        action = state.legal_actions[0] if state.legal_actions else None
        template = action.input.get("action") if action is not None and isinstance(action.input, dict) else None
        kind = template.get("type") if isinstance(template, dict) else None
        if kind == "doubles_turn":
            return _guaranteed_doubles(parse_doubles(action))
        if kind == "select_lineup":
            roster = parse_lineup(action)
            return build_lineup(roster, roster[:4], roster[:2])
    except Exception:  # noqa: BLE001 - this function's whole contract is "never raise"
        return universal_floor  # a torn template falls through to the floor
    return universal_floor


def _guaranteed_doubles(slots: list[Slot]) -> dict:
    """First non-pass option per slot, no scoring, no type reasoning."""

    def pick(slot: Slot, taken: str | None) -> dict:
        for option in slot.options:
            if option.get("type") == "move":
                targets = [t for t in option.get("targets") or [] if isinstance(t, int) and not isinstance(t, bool)]
                move = {"type": "move", "move_id": option.get("move_id")}
                if targets:
                    move["target"] = targets[0]
                return move
        for option in slot.options:
            if option.get("type") == "switch" and option.get("species") != taken:
                return {"type": "switch", "species": option.get("species")}
        return dict(PASS)

    slot_0 = pick(slots[0], taken=None)
    taken = slot_0.get("species") if slot_0["type"] == "switch" else None
    slot_1 = pick(slots[1], taken=taken)
    return {"type": "doubles_turn", "slot_0": slot_0, "slot_1": slot_1}


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
