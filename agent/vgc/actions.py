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

import re
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
# Fake Out: priority +3, flinches, but ONLY usable the first turn a Pokemon
# is on the field. The heuristic rewards it when that condition is visible
# (turn one, or an explicit per-active flag), never otherwise.
FAKE_OUT = "fakeout"
STAB_MULTIPLIER = 1.5


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


def heuristic_doubles(slots: list[Slot], state: GameState, *, base_speed=None) -> dict:
    """A legal, type-aware turn with no model call.

    Used when the clock runs out or the model never answers. Per slot it
    takes the highest-scoring damaging move at the best legal opponent
    (never an immunity, never our own side unless nothing else is legal),
    falling back to a status/protection move, then a switch, then pass.

    Beyond raw type-and-power scoring it accounts for the things a weak
    fallback would otherwise get punished on, but ONLY from data the
    observation actually shows: same-type-attack bonus, Fake Out's turn-one
    value, not re-using Protect when the field says it would fail, and (when
    ``base_speed`` is supplied) preferring a move from the faster Pokemon to
    finish a weakened foe. Every one of these degrades to the old behaviour
    when its data is absent, so it is never worse than before.
    """
    observation = _observation(state)
    foes = _species_types(observation.get("opponent_active_pokemon"))
    allies = _species_types(observation.get("active_pokemon"))
    hp = _hp_fractions(observation.get("opponent_active_pokemon"))
    context = _HeuristicContext(
        observation=observation,
        foes=foes,
        allies=allies,
        hp=hp,
        base_speed=base_speed,
    )

    slot_0 = _heuristic_slot(slots[0], context, taken=None)
    taken = slot_0.get("species") if slot_0["type"] == "switch" else None
    slot_1 = _heuristic_slot(slots[1], context, taken=taken)
    return build_doubles(slots, {
        "slot_0": {"option": slot_0.pop("_option"), "target": slot_0.get("target")},
        "slot_1": {"option": slot_1.pop("_option"), "target": slot_1.get("target")},
    })


@dataclass(frozen=True)
class _HeuristicContext:
    """Everything the per-slot heuristic reads, bundled so new signals do not
    grow every function's argument list."""

    observation: dict
    foes: dict
    allies: dict
    hp: dict
    base_speed: object  # callable species -> int | None, or None

    def active_entry(self, slot_number: int) -> dict:
        actives = self.observation.get("active_pokemon")
        if isinstance(actives, list) and 0 <= slot_number < len(actives) and isinstance(actives[slot_number], dict):
            return actives[slot_number]
        return {}

    def our_species(self, slot_number: int) -> str:
        entry = self.active_entry(slot_number)
        name = entry.get("species") or entry.get("name")
        return name if isinstance(name, str) else ""

    def our_types(self, slot_number: int) -> list[str]:
        return self.allies.get(self.our_species(slot_number), [])

    def can_fake_out(self, slot_number: int) -> bool:
        """Fake Out works only the first turn a Pokemon is on the field.

        True when the observation unambiguously shows that: it is turn one,
        or the active carries an explicit first-turn/just-switched flag.
        Absent that evidence we assume Fake Out is NOT usable, so we never
        over-reward a move that would fail.
        """
        if _number(self.observation.get("turn")) == 1.0:
            return True
        entry = self.active_entry(slot_number)
        for key in ("can_fake_out", "first_turn_active", "just_switched", "switched_in_this_turn"):
            if _truthy(entry.get(key)):
                return True
        return False

    def protect_would_fail(self, slot_number: int) -> bool:
        """True only when the field shows this slot already protected last
        turn (so another protection move is likely to fail). Conservative:
        with no such signal we do not suppress Protect."""
        entry = self.active_entry(slot_number)
        for key in ("protect_used_last_turn", "used_protect_last_turn", "protect_blocked"):
            if _truthy(entry.get(key)):
                return True
        counter = _number(entry.get("protect_counter") or entry.get("stall_counter"))
        if counter is not None and counter >= 1:
            return True
        last = entry.get("last_move") or entry.get("last_move_used")
        return isinstance(last, str) and re.sub(r"[^a-z0-9]", "", last.lower()) in PROTECTION


def guaranteed_action(state: GameState) -> dict:
    """A legal payload for the current decision that CANNOT raise.

    The floor beneath every other fallback. ``heuristic_doubles`` and
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


def _heuristic_slot(slot: Slot, context: _HeuristicContext, *, taken: str | None) -> dict:
    """Pick one slot's fallback action, with an explicit ANTI-PANIC-SWITCH rule.

    "Panic switching" is a documented failure mode of tactical agents (see
    PokeLLMon, arXiv:2402.01118): when a scary opponent is in front, an agent
    bleeds tempo by switching a healthy, capable Pokemon out instead of just
    attacking. This heuristic must never do that. The invariant enforced
    here is simple and testable:

        a VOLUNTARY switch is chosen ONLY when no move can usefully act
        (every move scores 0 because it is immune, hits only our own side,
        or is an unusable Fake Out), AND the slot is not forced to move.

    A ``force_switch`` slot (a fainted Pokemon's replacement) is the one case
    where switching is mandatory, and the server only offers switch options
    there anyway. Ordering: a positive-scoring move beats every switch; only
    if nothing scores do we switch; a zero-scoring move still beats a pass.
    """
    kinds = {option["type"] for option in slot.options}
    if kinds == {"pass"}:
        return {"type": "pass", "_option": _index_of(slot, lambda o: o["type"] == "pass")}

    best: tuple[float, int, int | None] | None = None
    for index, option in enumerate(slot.options):
        if option["type"] != "move":
            continue
        score, target = _score_move(option, slot.number, context)
        if best is None or score > best[0]:
            best = (score, index, target)

    # Anti-panic-switch: a move that can actually do something (score > 0)
    # always beats switching a healthy mon out. This is the whole guard.
    if best is not None and best[0] > 0:
        return {"type": "move", "target": best[2], "_option": best[1]}

    # No move scores. Switching is now legitimate repositioning, not panic:
    # either we are forced out, or every move we have is useless here.
    for index, option in enumerate(slot.options):
        if option["type"] == "switch" and option.get("species") != taken:
            return {"type": "switch", "species": option.get("species"), "_option": index}

    # Nothing scored and no distinct switch: a zero-scoring move (e.g. the
    # only move is resisted but not immune) still beats passing the turn.
    if best is not None:
        return {"type": "move", "target": best[2], "_option": best[1]}
    if "pass" in kinds:
        return {"type": "pass", "_option": _index_of(slot, lambda o: o["type"] == "pass")}
    raise TemplateError(f"doubles_turn slot {slot.number} has no move, distinct switch, or pass: {slot.options!r}")


def _score_move(option: dict, slot_number: int, context: _HeuristicContext) -> tuple[float, int | None]:
    """Rough expected damage, and the target that maximizes it."""
    details = _move_details(option, slot_number, context.observation)
    move_id = str(option.get("move_id") or "").lower()
    power = _number(option.get("base_power", details.get("base_power"))) or 0.0
    accuracy = _number(option.get("accuracy", details.get("accuracy")))
    accuracy = 1.0 if accuracy is None else (accuracy / 100.0 if accuracy > 1.0 else accuracy)
    move_type = _move_type(option, details)
    stab = STAB_MULTIPLIER if (move_type and move_type.upper() in {t.upper() for t in context.our_types(slot_number)}) else 1.0

    entries = _target_entries(option)
    if not entries:  # a field or self move: valued below any real attack
        return (_support_value(move_id, slot_number, context), None)

    best: tuple[float, int | None] | None = None
    for entry in entries:
        target = entry.get("target")
        if not isinstance(target, int) or isinstance(target, bool):
            continue
        side = entry.get("side") or ("opponent" if target > 0 else "ally" if target < 0 else "none")
        foe_species = entry.get("species") or ""
        # Resolve the defender's types. The server names the species on each
        # opponent target entry; if it is ever omitted and there is exactly
        # one opponent active, that one is unambiguously the target, so we can
        # still see an immunity. With two unnamed foes we do NOT guess — a
        # wrong inference would be a new mistake — and fall back to neutral.
        foe_types = context.foes.get(foe_species)
        if foe_types is None and side == "opponent" and not foe_species and len(context.foes) == 1:
            foe_species, foe_types = next(iter(context.foes.items()))
        multiplier = effectiveness(move_type, foe_types) if side == "opponent" else 1.0
        if power <= 0:
            # Fake Out is a 40-BP move but its real value is turn-one
            # priority + flinch; score it as support when it cannot connect
            # for damage reasons and only when it is actually usable.
            score = _support_value(move_id, slot_number, context) if side != "ally" else 0.0
        elif side == "ally" or side == "self":
            score = 0.0  # never the heuristic's choice while anything else scores
        elif multiplier == 0.0:
            score = 0.0  # immune: strictly dominated (3.2)
        elif move_id == FAKE_OUT and not context.can_fake_out(slot_number):
            score = 0.0  # Fake Out off turn one simply fails: never choose it
        else:
            score = power * accuracy * multiplier * stab
            if move_id in SPREAD_MOVES_HITTING_ALL:
                score *= 0.75  # spread reduction
            foe_hp = context.hp.get(foe_species, 1.0)
            # Chip a weakened foe out rather than spread damage around.
            score *= 1.0 + 0.5 * (1.0 - foe_hp)
            # Fake Out: guaranteed first (priority +3) and flinches, so on the
            # turn it is usable it is worth more than its tiny base power. The
            # bonus is additive and only applied when truly usable.
            if move_id == FAKE_OUT and context.can_fake_out(slot_number):
                score += 60.0
            # If we can confirm we outspeed this foe, a move that could finish
            # it is slightly preferred (we KO before it acts). Base-speed
            # prior only; never acts on a tie or unknown.
            elif context.base_speed is not None and foe_hp <= 0.35:
                from . import speed as _speed
                faster = _speed.faster_than(context.observation, context.base_speed,
                                            context.our_species(slot_number), str(foe_species))
                if faster is True:
                    score *= 1.1
        if best is None or score > best[0]:
            best = (score, target)
    return best or (0.0, None)


def _support_value(move_id: str, slot_number: int, context: _HeuristicContext) -> float:
    """Where a non-damaging (or priority-utility) move sits against damage.

    Fake Out is ranked above everything else on the turn it is usable (free
    flinch + chip), and ignored otherwise. A protection move is dropped to
    the floor when the field shows it would fail (consecutive-use), so the
    fallback does not waste a turn re-Protecting.
    """
    if move_id == FAKE_OUT:
        return 95.0 if context.can_fake_out(slot_number) else 0.0
    if move_id in SPEED_CONTROL:
        return 45.0
    if move_id in REDIRECTION:
        return 35.0
    if move_id in PROTECTION:
        return 1.0 if context.protect_would_fail(slot_number) else 25.0
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


def _truthy(value: object) -> bool:
    """A per-active flag counts as set when truthy or a positive count."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on", "active"}
    return False


def _index_of(slot: Slot, predicate) -> int:
    return next(index for index, option in enumerate(slot.options) if predicate(option))
