"""The move list (data/moves.json, from Showdown) and the moves it rules out.

The live match logs showed moves no player would make: Earthquake into
Gyarados and Tornadus (Flying: immune), Earthquake beside our own Iron
Hands (Earthquake hits the partner too, and Iron Hands takes 2x), Flash
Cannon into Gyarados (resisted) with a neutral Draco Meteor on the same
Pokemon. The type chart alone could not stop them: spread moves list no
targets, so nothing checked who they hit, and nothing knew Earthquake
hits an ally at all.

Two layers, both deterministic:

- ``option_blocks``: options withheld for a slot (the model never sees
  them, the heuristic skips them): a damaging move every opposing target is
  immune to; a status move every target ignores (Thunder Wave into Ground,
  Spore into Grass); a plain damaging move another of the slot's moves beats
  on every target by ``DOMINATED_BY`` or more.
- ``row_problem``: a whole joint action rejected: a move that also hits our
  partner (Earthquake, Surf, Discharge) when the partner is neither immune
  nor protecting that turn; an attack aimed at an immune target; anything
  hostile aimed at our own partner.

Moves with a reason beyond raw damage (priority, pivoting, speed control,
reliable stat drops, power that depends on more than base power...) are
never pruned as dominated.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .typechart import effectiveness

DEFAULT_MOVES = Path(__file__).resolve().parents[2] / "data" / "moves.json"
DOMINATED_BY = 1.3  # another move must do this much more on every target
RESISTED = 0.5  # an attack doing at most this to every foe is a wasted turn
# Reasons to use a resisted attack anyway.
STILL_USEFUL = frozenset({"pivot", "speed_control", "drops_target_stats", "priority", "variable_power", "status"})
STAMINA_CHIP_HP = 0.35  # above this, a hit on a Stamina user is a chip, not a KO
SINGLE_TARGETS = ("normal", "any", "adjacentFoe", "randomNormal")
UTILITY = frozenset({
    "priority", "pivot", "heal", "charge", "status", "flinch", "drops_target_stats", "speed_control",
    "self_boost", "field", "bypasses_protect", "ignores_ability", "ohko", "variable_power", "spread",
})
PROTECTING = frozenset({
    "protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "burningbulwark", "silktrap", "obstruct",
    "wideguard",
})
# Status moves that fail on a type: Thunder Wave on Ground/Electric, powder on Grass...
STATUS_IMMUNE_TYPES = {
    "thunderwave": {"GROUND", "ELECTRIC"}, "glare": set(), "willowisp": {"FIRE"},
    "toxic": {"POISON", "STEEL"}, "poisonpowder": {"POISON", "STEEL", "GRASS"}, "toxicthread": {"POISON", "STEEL"},
    "spore": {"GRASS"}, "sleeppowder": {"GRASS"}, "stunspore": {"GRASS", "ELECTRIC"}, "leechseed": {"GRASS"},
}
# Abilities that make a Pokemon immune to a type, used for our OWN partner only
# when every ability it can have grants the immunity.
IMMUNITY_ABILITIES = {
    "GROUND": {"Levitate", "Earth Eater"}, "WATER": {"Water Absorb", "Storm Drain", "Dry Skin"},
    "ELECTRIC": {"Volt Absorb", "Lightning Rod", "Motor Drive"}, "FIRE": {"Flash Fire", "Well-Baked Body"},
    "GRASS": {"Sap Sipper"},
}
# Moves that are meant to target an ally.
ALLY_OK = frozenset({
    "pollenpuff", "beatup", "helpinghand", "coaching", "decorate", "instruct", "afteryou", "allyswitch",
    "acupressure", "healpulse", "floralhealing", "lifedew", "skillswap", "entrainment", "aromaticmist",
    "holdhands", "quash", "ragepowder", "followme", "spotlight", "swagger", "flatter",
})


def _id(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


@lru_cache(maxsize=2)
def load(path: str = str(DEFAULT_MOVES)) -> dict[str, dict]:
    """move id -> data, or {} if the move list is missing."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("moves", {})
    except (OSError, ValueError):
        return {}


def lookup(move_id: object) -> dict | None:
    return load().get(_id(move_id))


def details(move_id: object) -> dict:
    """The move list's data in the observation's own key names, for filling gaps."""
    move = lookup(move_id)
    if not move:
        return {}
    accuracy = move["accuracy"]
    return {
        "type": move["type"],
        "category": move["category"],
        "base_power": move["power"],
        "accuracy": 1.0 if accuracy is None else accuracy / 100,
        "priority": move["priority"],
    }


# -- who is where ---------------------------------------------------------------------------


class Board:
    """Types and abilities of everyone on the field for one decision."""

    def __init__(self, slots, observation: dict, table) -> None:
        self.table = table
        self.slots = slots
        self.ours = [self._entry(observation.get("active_pokemon"), slot.number, slot) for slot in slots]
        foes = observation.get("opponent_active_pokemon") or []
        self.foe_hp = {
            _id(f.get("species") or f.get("name")): _fraction(f)
            for f in foes
            if isinstance(f, dict) and (f.get("species") or f.get("name"))
        }
        self.foes = {
            _id(f.get("species") or f.get("name")): self._types(f)
            for f in foes
            if isinstance(f, dict) and (f.get("species") or f.get("name"))
        }
        from .observed import field_state

        self.field = set((field_state(observation) or {}).get("field", []))

    def grounded(self, species: str | None) -> bool:
        """On the ground for terrain: not Flying, not Levitate, no Air Balloon."""
        types = self.foe_types(species) or []
        abilities = self.table.abilities(species)
        return not ("FLYING" in types or abilities == ["Levitate"] or _id(self.table.common_item(species)) == "airballoon")

    def prankster(self, number: int) -> bool:
        mine = self.ours[number] if number < len(self.ours) else {}
        return bool(mine) and self.table.abilities(mine.get("species")) == ["Prankster"]

    def blocked_on(self, number: int, move_id: str, move: dict, species: str | None) -> str | None:
        """Why this move fails on this foe because of the field or our ability."""
        if not species:
            return None
        if "psychicterrain" in self.field and (move["priority"] or 0) > 0 and _damaging(move) and self.grounded(species):
            return f"{move['name']} fails: Psychic Terrain blocks priority moves on a grounded {species}"
        if move["category"] == "STATUS" and self.prankster(number) and "DARK" in (self.foe_types(species) or []):
            return f"{move['name']} fails: Dark types ignore Prankster status moves ({species})"
        return None

    def _types(self, entry: dict) -> list[str]:
        types = entry.get("types")
        if isinstance(types, list) and types:
            return [str(t).upper() for t in types]
        return [t.upper() for t in self.table.species_types(entry.get("species") or entry.get("name"))]

    def _entry(self, actives, number: int, slot) -> dict:
        entry = actives[number] if isinstance(actives, list) and number < len(actives) and isinstance(actives[number], dict) else {}
        species = entry.get("species") or entry.get("name") or (slot.active if isinstance(slot.active, str) else None)
        if slot.force_switch or not species:
            return {}
        return {"species": species, "types": self._types({**entry, "species": species})}

    def ally_of(self, number: int) -> dict:
        other = 1 - number
        return self.ours[other] if 0 <= other < len(self.ours) else {}

    def foe_types(self, species: object) -> list[str] | None:
        return self.foes.get(_id(species))

    def ally_immune(self, number: int, move_type: str) -> bool:
        ally = self.ally_of(number)
        if not ally:
            return True  # no partner on the field: nobody to hit
        if ally["types"] and effectiveness(move_type, ally["types"]) == 0:
            return True
        abilities = set(self.table.abilities(ally["species"]))
        return bool(abilities) and abilities <= IMMUNITY_ABILITIES.get(move_type, set())

    def ally_multiplier(self, number: int, move_type: str) -> float | None:
        ally = self.ally_of(number)
        return effectiveness(move_type, ally["types"]) if ally and ally["types"] else None


def _foe_targets(option: dict) -> list[dict]:
    entries = option.get("target_options")
    if isinstance(entries, list) and entries:
        return [e for e in entries if isinstance(e, dict) and e.get("side") == "opponent"]
    return [{"target": t, "side": "opponent"} for t in option.get("targets") or [] if isinstance(t, int) and t > 0]


def _damaging(move: dict) -> bool:
    return move.get("category") in ("PHYSICAL", "SPECIAL") and (move.get("power") or 0) > 0


def _status_fails(move_id: str, types: list[str]) -> bool:
    return bool(set(types) & STATUS_IMMUNE_TYPES.get(move_id, set()))


def _strength(move_id: str, move: dict, our_types: list[str], our_stats: dict | None, foe: list[str]) -> float:
    """Rough expected damage of a plain attack on one target, for comparisons only."""
    stab = 1.5 if move["type"] in our_types else 1.0
    accuracy = 1.0 if move["accuracy"] is None else move["accuracy"] / 100
    hits = move["multihit"]
    hits = (sum(hits) / 2 + 0.6 if isinstance(hits, list) else hits) if hits else 1
    stat = 1.0
    if our_stats:
        stat = our_stats["atk" if move["category"] == "PHYSICAL" else "spa"] / 100
    return move["power"] * stab * effectiveness(move["type"], foe) * accuracy * hits * stat


# -- the rules -------------------------------------------------------------------------------


def option_blocks(board: Board) -> dict[int, dict[int, str]]:
    """slot number -> {option index: why it is withheld}."""
    found: dict[int, dict[int, str]] = {}
    for slot in board.slots:
        mine = board.ours[slot.number] if slot.number < len(board.ours) else {}
        blocked: dict[int, str] = {}
        plain: list[tuple[int, str, dict, list[dict]]] = []
        for index, option in enumerate(slot.options):
            if option.get("type") != "move":
                continue
            move_id = _id(option.get("move_id"))
            move = lookup(move_id)
            if not move:
                continue
            foes = _foe_targets(option)
            typed = [(e, board.foe_types(e.get("species"))) for e in foes]
            known = [t for _, t in typed if t]
            if move["target"] in ("allAdjacentFoes", "allAdjacent"):
                spread_foes = [t for t in board.foes.values() if t]
                if _damaging(move) and spread_foes and all(effectiveness(move["type"], t) == 0 for t in spread_foes):
                    blocked[index] = f"{move['name']} cannot affect either foe (type immunity)"
                continue
            if not foes or len(known) < len(foes):
                continue  # self/field move, or a target we cannot type: leave it to the model
            reasons = [board.blocked_on(slot.number, move_id, move, e.get("species")) for e in foes]
            if all(reasons):
                blocked[index] = reasons[0]
            elif _damaging(move) and all(effectiveness(move["type"], t) == 0 for t in known):
                blocked[index] = f"{move['name']} cannot affect any legal target (type immunity)"
            elif move["category"] == "STATUS" and all(_status_fails(move_id, t) for t in known):
                blocked[index] = f"{move['name']} fails on every legal target (type immunity)"
            elif _damaging(move) and move["target"] in SINGLE_TARGETS and not (set(move["tags"]) & UTILITY):
                plain.append((index, move_id, move, foes))
        # Resisted by every foe: match ec1752bf sat Choice-locked into Rock
        # Slide for five turns into Iron Treads (0.25x) and Ting-Lu (0.5x).
        # Withheld when the slot has a way out: a switch, or another attack
        # some foe does not resist.
        resisted, useful_attack = [], False
        for index, option in enumerate(slot.options):
            move = lookup(option.get("move_id")) if option.get("type") == "move" else None
            if index in blocked or not move or not _damaging(move):
                continue
            if move["target"] in ("allAdjacentFoes", "allAdjacent"):
                types = [t for t in board.foes.values() if t]
            else:
                types = [board.foe_types(e.get("species")) for e in _foe_targets(option)]
            if not types or not all(types):
                useful_attack = True  # cannot judge it: assume it is fine
                continue
            best = max(effectiveness(move["type"], t) for t in types)
            if best <= RESISTED and not (set(move["tags"]) & STILL_USEFUL):
                resisted.append((index, move, best))
            else:
                useful_attack = True
        can_switch = any(o.get("type") == "switch" for o in slot.options)
        if resisted and (can_switch or useful_attack):
            for index, move, best in resisted:
                blocked[index] = f"{move['name']} is resisted by every foe (best {best:g}x); switch or use another attack"
        # Dominated: another plain single-target attack from this slot does
        # DOMINATED_BY times more on every target this one could hit.
        stats = board.table.base_stats(mine.get("species")) if mine else None
        our_types = mine.get("types") or []
        for index, move_id, move, foes in plain:
            if index in blocked:
                continue
            for other_index, other_id, other, other_foes in plain:
                if other_index == index or other_index in blocked:
                    continue
                reachable = {_id(e.get("species")) for e in other_foes}
                if not all(_id(e.get("species")) in reachable for e in foes):
                    continue
                if all(
                    _strength(other_id, other, our_types, stats, board.foe_types(e.get("species")))
                    >= DOMINATED_BY * max(_strength(move_id, move, our_types, stats, board.foe_types(e.get("species"))), 1e-9)
                    for e in foes
                ):
                    blocked[index] = f"{move['name']} is outclassed by {other['name']} on every target"
                    break
        if blocked:
            found[slot.number] = blocked
    return found


def ally_hit_blocks(board: Board) -> dict[int, dict[int, str]]:
    """For the heuristic, which picks slots one at a time: moves that would
    hit a partner who is not immune (it cannot coordinate a Protect)."""
    found: dict[int, dict[int, str]] = {}
    for slot in board.slots:
        for index, option in enumerate(slot.options):
            move = lookup(option.get("move_id")) if option.get("type") == "move" else None
            if move and move["target"] == "allAdjacent" and _damaging(move) and not board.ally_immune(slot.number, move["type"]):
                ally = board.ally_of(slot.number)["species"]
                found.setdefault(slot.number, {})[index] = f"{move['name']} would also hit our {ally}"
    return found


def row_problem(board: Board, payload: dict) -> str | None:
    """Why a joint action is a blunder, or ``None``."""
    for number in (0, 1):
        choice = payload.get(f"slot_{number}") or {}
        if choice.get("type") != "move":
            continue
        move_id = _id(choice.get("move_id"))
        move = lookup(move_id)
        if not move:
            continue
        partner = payload.get(f"slot_{1 - number}") or {}
        partner_guards = partner.get("type") == "move" and _id(partner.get("move_id")) in PROTECTING
        if move["target"] == "allAdjacent" and _damaging(move) and not board.ally_immune(number, move["type"]):
            if not partner_guards:
                ally = board.ally_of(number)
                multiplier = board.ally_multiplier(number, move["type"])
                strength = f" ({multiplier:g}x)" if multiplier is not None else ""
                return f"{move['name']} also hits our own {ally['species']}{strength} and it is not protecting"
        target = choice.get("target")
        if isinstance(target, int) and target > 0 and _damaging(move):
            option = _chosen_option(board, number, move_id)
            entries = _foe_targets(option) if option else []
            entry = next((e for e in entries if e.get("target") == target), None)
            species = entry.get("species") if entry else None
            others = [e for e in entries if e is not entry and board.foe_types(e.get("species"))]
            if _stamina_multihit(board, move, species):
                return f"{move['name']} into {species}: every hit can raise its Defense (Stamina)"
            # Which foe is the better target (resisted here, Stamina chip) is
            # the race's call now: it sees the damage, the Defense Stamina gains
            # and whether the other foe is predicted to Protect. As a hard rule
            # it forced Ice Beam into a Tornadus read to Protect (match ccb570bb).
        if isinstance(target, int) and target < 0 and move_id not in ALLY_OK and move["target"] in SINGLE_TARGETS:
            return f"{move['name']} aimed at our own partner"
        if isinstance(target, int) and target > 0:
            option = _chosen_option(board, number, move_id)
            entry = next((e for e in _foe_targets(option) if e.get("target") == target), None) if option else None
            failing = board.blocked_on(number, move_id, move, entry.get("species")) if entry else None
            if failing:
                return failing
            types = board.foe_types(entry.get("species")) if entry else None
            if types and _damaging(move) and effectiveness(move["type"], types) == 0:
                return f"{move['name']} into an immune {entry.get('species')}"
    return None


def _stamina_multihit(board: Board, move: dict, species: str | None) -> bool:
    """A multi-hit move into Stamina: up to +5 Defense from one move. Always wrong."""
    return bool(species and move.get("multihit") and "Stamina" in board.table.abilities(species))


def _fraction(entry: dict) -> float:
    value = entry.get("hp_fraction")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    current, maximum = entry.get("current_hp"), entry.get("max_hp")
    if isinstance(current, (int, float)) and isinstance(maximum, (int, float)) and maximum:
        return current / maximum
    return 1.0


def _chosen_option(board: Board, number: int, move_id: str) -> dict | None:
    slot = next((s for s in board.slots if s.number == number), None)
    if slot is None:
        return None
    return next((o for o in slot.options if o.get("type") == "move" and _id(o.get("move_id")) == move_id), None)


def merge(rules, extra: dict[int, dict[int, str]], slots):
    """SlotRules with ``extra`` blocks added, never withholding every option."""
    from .moverules import SlotRules

    merged = []
    for slot, rule in zip(slots, rules):
        blocked = {**rule.blocked, **extra.get(slot.number, {})}
        if len(blocked) >= len(slot.options):
            blocked = dict(rule.blocked)
        merged.append(SlotRules(rule.species, rule.fresh, rule.protect_chain, rule.last_move, blocked))
    return merged
