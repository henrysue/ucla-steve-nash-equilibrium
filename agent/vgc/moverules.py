"""Moves whose success depends on battle history, and the tracker that remembers it.

The server offers every move a Pokemon knows, including ones that are certain
to fail this turn: Fake Out after its user has already moved, a Protect right
after a Protect (1/3). Neither the model nor the template can see that history, so this
module keeps it, per match, for every Pokemon on both sides:

- how many move actions ours have taken since entering (Fake Out, First
  Impression and Mat Block only work on the first one);
- each of ours' consecutive Protect-family uses (a second in a row is withheld);
- each of ours' last move (Blood Moon and Gigaton Hammer cannot repeat);
- whether each opposing Pokemon is on its first turn, and so still a Fake Out
  threat.

It produces one ``SlotRules`` per slot each decision: the option indices that
would fail (withheld from the model and skipped by the heuristic) plus the
facts behind them, which are logged with every decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable

from .typechart import effectiveness

# Fail unless it is the user's first move action since entering the field.
FIRST_TURN_ONLY = frozenset({"fakeout", "firstimpression", "matblock"})
# Share the consecutive-use counter: each repeat succeeds 1/3 as often.
PROTECT_CHAIN = frozenset({
    "protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "burningbulwark",
    "silktrap", "obstruct", "maxguard", "endure",
})
# Gen 9: cannot be selected two turns in a row.
NO_REPEAT = frozenset({"bloodmoon", "gigatonhammer"})
# Our side's timed effects -> base turns, counting the turn it is set. A
# re-use while up fails (Trick Room's ENDS it). Match 8a882a3f: Talonflame
# used Tailwind on turns 2, 5, 6 and 8 and never attacked. Base durations
# only (Light Clay's 8 is not assumed), so a re-set is never withheld after
# the effect has really ended. Weather and terrain are left out: the
# opponent can replace them, which this tracker cannot see.
MEANINGFUL_HIT = 0.08  # below this, a hit and Leftovers can cancel out: no inference
HEAL_ALLOWANCE = 0.07  # Leftovers / Black Sludge / Grassy Terrain, end of turn
TIMED_EFFECTS = {"tailwind": 4, "reflect": 5, "lightscreen": 5, "auroraveil": 5, "trickroom": 5}
# Never Protect twice in a row with the same Pokemon: the repeat succeeds only
# 1/3 of the time (match fb4dcbc1: Tornadus protected turns 5 and 6).
PROTECT_BLOCK_CHAIN = 1
# Opposing leads that stop Fake Out outright: Psychic Surge (Indeedee) blocks
# priority on grounded targets, Armor Tail (Farigiraf) blocks it on the side.
FAKE_OUT_BLOCKERS = frozenset({"indeedee", "indeedeef", "indeedeem", "farigiraf"})

# Conditions the tracker cannot check from its own history. Shown to the
# model next to the option so it prices them; never enforced in code.
NOTES = {
    "fakeout": "Only works on the user's first turn after entering the field.",
    "firstimpression": "Only works on the user's first turn after entering the field.",
    "matblock": "Only works on the user's first turn after entering the field.",
    "trickroom": "Using it while Trick Room is already up ENDS Trick Room.",
    "tailwind": "Fails while your side's Tailwind is already up.",
    "reflect": "Fails while your Reflect is already up.",
    "lightscreen": "Fails while your Light Screen is already up.",
    "auroraveil": "Fails unless snow is active, or while Aurora Veil is already up.",
    "suckerpunch": "Fails unless the target is using a damaging move this turn.",
    "thunderclap": "Fails unless the target is using a damaging move this turn.",
    "bloodmoon": "Cannot be used two turns in a row.",
    "gigatonhammer": "Cannot be used two turns in a row.",
    "encore": "Fails if the target has not moved yet this battle.",
    "spore": "Grass types, Overcoat and Safety Goggles are immune.",
    "ragepowder": "Grass types, Overcoat and Safety Goggles ignore the redirection.",
}
_PROTECT_NOTE = "Consecutive Protect-family uses succeed 1/3 as often each time."


def move_id(name: object) -> str:
    """Showdown's id form: lowercase, punctuation and spaces stripped."""
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


@dataclass
class OnField:
    """What one Pokemon has done since it last entered the field."""

    move_actions: int = 0
    protect_chain: int = 0
    last_move: str | None = None
    turns_seen: int = 0  # full turns it has been on the field for


@dataclass(frozen=True)
class SlotRules:
    """One slot's history-dependent facts for this decision."""

    species: str | None
    fresh: bool = True  # first-turn-only moves still work
    protect_chain: int = 0
    last_move: str | None = None
    blocked: dict[int, str] = field(default_factory=dict)  # option index -> why it fails
    targets: dict[int, tuple[int, ...]] = field(default_factory=dict)  # option index -> the only targets allowed

    @property
    def protect_success(self) -> float:
        return 1.0 / 3 ** self.protect_chain

    def note(self, move: str) -> dict:
        """Extra facts to show the model beside one move option."""
        extra = {}
        if move in NOTES:
            extra["rule"] = NOTES[move]
        if move in PROTECT_CHAIN:
            extra["rule"] = _PROTECT_NOTE
            extra["success_chance"] = round(self.protect_success, 3)
        return extra


class MoveTracker:
    """Per-match memory of on-field history for both sides.

    Call ``begin`` at the start of every doubles decision and ``record``
    with the payload actually sent. ``known_moves`` maps an opposing species
    to the moves it commonly runs (the reference table), for the Fake Out
    threat read; without it that read is simply absent. ``exact_moves`` maps
    an opposing species to its draft card's four moves when we have the card
    (``None`` otherwise): only then do we KNOW it has no Protect.
    ``ko_threats`` maps an observation to the opposing species ids that
    threaten a KO on one of ours this turn: worth a Fake Out even if they
    carry Protect.
    """

    def __init__(
        self,
        known_moves: Callable[[str], list[str]] | None = None,
        offense: Callable[[str], int | None] | None = None,
        exact_moves: Callable[[str], list[str] | None] | None = None,
        ko_threats: Callable[[dict], set[str]] | None = None,
    ) -> None:
        self._known_moves = known_moves
        self._offense = offense
        self._exact_moves = exact_moves
        self._ko_threats = ko_threats
        self._full_turns = 0
        self._ours: dict[str, OnField] = {}
        self._theirs: dict[str, OnField] = {}
        self._slot_species: list[str | None] = [None, None]
        self._full_turn = False
        self._decisions = 0
        self.used: list[dict] = []  # every rule-governed move we sent, in order
        # Opponent Protect tracking: which foes we attacked last decision and
        # their HP then, so a hit that took nothing shows they protected.
        self._foe_hp: dict[str, float] = {}
        self._foe_types: dict[str, list[str]] = {}
        self._targets: list[dict[int, str]] = [{}, {}]
        self._attacked: dict[str, set[str]] = {}  # foe id -> our attackers aimed at it
        self._sure_hit: dict[str, bool] = {}  # foe id -> one of those attacks cannot miss
        self.protected_last_turn: dict[str, str] = {}  # foe id -> how we know
        self._effects: dict[str, int] = {}  # timed effect -> full turn (0-based) it was set
        self._foe_known: dict[str, set[str]] = {}  # foe id -> moves the server has shown
        self._expected_low: dict[str, float] = {}  # foe id -> least damage (HP fraction) our last attacks should do
        self._pending: list[tuple[str, str, int]] = []  # (effect, setter, turn) awaiting confirmation
        # A fresh foe with Fake Out was out last decision: any of ours may have
        # been flinched, so "it took no damage" is no evidence of a Protect.
        self._flinch_risk = False

    def begin(self, slots: list, observation: dict) -> list[SlotRules]:
        """Update who is on the field and return each slot's rules."""
        self._decisions += 1
        self._full_turn = any(option.get("type") == "move" for slot in slots for option in slot.options)
        self._slot_species = [_slot_species(slot, observation) for slot in slots]
        self._update_effects(observation)
        from .observed import field_state

        self._their_trick_room = "trickroom" in ((field_state(observation) or {}).get("field") or []) \
            and self.effect_turns_left("trickroom") == 0
        self._ours = _carry(self._ours, [s for s in self._slot_species if s])
        self._theirs = _carry(self._theirs, _opponent_species(observation))
        self._read_opponent_protects(observation)
        self._targets = [_foe_targets(slot) for slot in slots]
        rules = [self._rules(slot, species) for slot, species in zip(slots, self._slot_species)]
        # The opening turn: our first full turn, unless the server's own turn
        # counter says otherwise (an agent restarted mid-match).
        server_turn = observation.get("turn")
        if self._full_turn and self._full_turns == 0 and server_turn in (None, 0, 1):
            rules = self._lead_with_fake_out(slots, rules, observation)
        self._flinch_risk = self._fake_out_live(observation)
        return rules

    def _fake_out_live(self, observation: dict) -> bool:
        """Some opposing active is on its first turn out and has Fake Out."""
        for entry in observation.get("opponent_active_pokemon") or []:
            if not isinstance(entry, dict) or _fainted(entry):
                continue
            name = entry.get("species") or entry.get("name")
            species = move_id(name)
            if self._theirs.get(species) is None or self._theirs[species].turns_seen:
                continue
            moves = set(_known_moves(entry))
            for source in (self._exact_moves, self._known_moves):
                if source is not None and name:
                    moves |= {move_id(m) for m in source(name) or []}
            if "fakeout" in moves:
                return True
        return False

    def _cannot_have_protect(self, name: str) -> bool:
        """Its draft card is known and has no Protect-family move."""
        exact = self._exact_moves(name) if self._exact_moves and name else None
        return bool(exact) and not {move_id(m) for m in exact} & PROTECT_CHAIN

    def record(self, payload: object) -> None:
        """Remember what each slot did with the payload actually sent."""
        if not isinstance(payload, dict):
            return
        for number, species in enumerate(self._slot_species):
            choice = payload.get(f"slot_{number}")
            if not species or not isinstance(choice, dict) or choice.get("type") != "move":
                continue
            entry = self._ours.setdefault(species, OnField())
            move = move_id(choice.get("move_id"))
            entry.move_actions += 1
            entry.protect_chain = entry.protect_chain + 1 if move in PROTECT_CHAIN else 0
            entry.last_move = move
            if move in NOTES or move in PROTECT_CHAIN:
                self.used.append({"decision": self._decisions, "species": species, "move": move})
            if move in TIMED_EFFECTS and self._full_turn and move not in self._effects:
                self._pending.append((move, species, self._full_turns))
        self._sure_hit = {}
        self._attacked = self._attacked_by(payload)
        if self._full_turn:
            self._full_turns += 1
            for entry in (*self._ours.values(), *self._theirs.values()):
                entry.turns_seen += 1

    def _read_opponent_protects(self, observation: dict) -> None:
        """Which foes protected on the previous turn, and how we know.

        Reported: the server names its last move. Inferred: we attacked it
        with a move it is not immune to and its HP did not drop.
        """
        found = {}
        hp_now = {}
        for entry in observation.get("opponent_active_pokemon") or []:
            if not isinstance(entry, dict):
                continue
            species = move_id(entry.get("species") or entry.get("name"))
            if not species:
                continue
            hp_now[species] = _hp_fraction(entry)
            last = move_id(entry.get("last_move") or entry.get("last_move_used") or "")
            if last:
                if last in PROTECT_CHAIN:
                    found[species] = f"reported: last move {last}"
                continue
            known = _known_moves(entry)
            new_moves = known - self._foe_known.get(species, set()) if species in self._foe_known else set()
            shown = new_moves & PROTECT_CHAIN
            self._foe_known[species] = known
            if shown:
                # Match ccb570bb: Archaludon's first Protect showed up in its
                # moves while Leftovers raised its HP, and went unnoticed.
                found[species] = f"revealed: {sorted(shown)[0]}"
                continue
            if new_moves - PROTECT_CHAIN:
                # It showed another move this turn, so it acted: no Protect,
                # whatever its HP says (match 915ba9e1: a U-turn read as Protect).
                continue
            if self._cannot_have_protect(entry.get("species") or entry.get("name")) or self._flinch_risk:
                # Match 666b209d: "Chi-Yu protected" - its card has no Protect;
                # Dragonite had been flinched by Rillaboom's Fake Out.
                continue
            before = self._foe_hp.get(species)
            # Unchanged HP only: a rise means healing (Leftovers, Grassy Terrain)
            # after a hit, which is not evidence of a Protect.
            # Our attacker must also still be on the field: one KO'd before it
            # moved never hit anything (match e4546428: Urshifu fell to a faster
            # Chi-Yu, and "Chi-Yu took no damage" was misread as a Protect).
            survivors = self._attacked.get(species, set()) & set(self._ours)
            if not self._sure_hit.get(species, True):
                # Every hit we aimed at it can miss (Scale Shot 90%, Rock Slide):
                # no damage may be a miss (match 41d776ed, turn 8).
                survivors = set()
            low = self._expected_low.get(species, 0.0)
            if survivors and before is not None and low >= MEANINGFUL_HIT:
                # Judged against the damage we expected: a Leftovers rise after
                # a real hit is still a drop of several %; after a blocked one
                # it is a pure rise (match ccb570bb).
                taken = before - hp_now[species] + (HEAL_ALLOWANCE if hp_now[species] > before + 0.005 else 0.0)
                if taken < 0.5 * low:
                    found[species] = f"inferred: took {max(0.0, before - hp_now[species]):.0%} where our attacks do {low:.0%}+"
            elif survivors and before is not None and abs(hp_now[species] - before) <= 0.005:
                found[species] = "inferred: our attack last turn did no damage"
        self.protected_last_turn = found
        self._foe_hp = hp_now
        self._foe_types = {
            move_id(e.get("species") or e.get("name")): [str(t).upper() for t in e.get("types") or []]
            for e in observation.get("opponent_active_pokemon") or []
            if isinstance(e, dict)
        }

    def _attacked_by(self, payload: dict) -> dict[str, set[str]]:
        """Foe id -> which of ours aimed a damaging move it is not immune to."""
        from . import movelist

        hit: dict[str, set[str]] = {}
        for number in (0, 1):
            choice = payload.get(f"slot_{number}")
            if not isinstance(choice, dict) or choice.get("type") != "move":
                continue
            move = movelist.lookup(choice.get("move_id"))
            if not move or not move["power"] or move["category"] == "STATUS":
                continue
            if move["target"] in ("allAdjacentFoes", "allAdjacent"):
                foes = list(self._foe_hp)
            else:
                target = self._targets[number].get(choice.get("target")) if number < len(self._targets) else None
                foes = [target] if target else []
            for foe in foes:
                types = self._foe_types.get(foe) or []
                if not types or effectiveness(move["type"], types) > 0:
                    hit.setdefault(foe, set()).add(self._slot_species[number])
                    sure = move["accuracy"] is None or move["accuracy"] >= 100
                    self._sure_hit[foe] = self._sure_hit.get(foe, False) or sure
        return hit

    def _update_effects(self, observation: dict) -> None:
        """Confirm last turn's effects and drop expired ones. An effect counts
        only if its setter is still out (so it moved; one KO'd first did not).
        When the server sends the field, it is the authority: an effect it
        shows is up, one it does not show is not (Trick Room only when we set
        it: the opponent's we may want to end)."""
        from .observed import field_state

        for move, setter, turn in self._pending:
            if setter in self._slot_species:
                self._effects[move] = turn
        self._pending = []
        shown = field_state(observation)
        if shown is not None:
            up = set(shown["ours"]) | ({"trickroom"} & set(shown["field"]))
            for move in TIMED_EFFECTS:
                ours = move != "trickroom" or move in self._effects
                if move in up and ours and self.effect_turns_left(move) == 0:
                    self._effects[move] = self._full_turns - 1  # shown up: at least this turn left
                elif move not in up:
                    self._effects.pop(move, None)
        self._effects = {m: t for m, t in self._effects.items() if self.effect_turns_left(m) > 0}

    def effect_turns_left(self, move: str) -> int:
        """Turns our timed effect has left, counting the coming one; 0 if not up."""
        if move not in self._effects:
            return 0
        return max(0, TIMED_EFFECTS[move] - (self._full_turns - self._effects[move]))

    def expect(self, damage: dict[str, float]) -> None:
        """Foe id -> least HP fraction our just-sent attacks should take off it."""
        self._expected_low = {move_id(k): v for k, v in damage.items()}

    def just_in(self) -> set[str]:
        """Our species ids that came in after the opening turn and have not had a
        full turn yet. Not the leads on turn 1: switching one out idles its slot
        once, not twice (match b0e756ca flagged Incineroar's turn-1 pivot)."""
        if self._full_turns == 0:
            return set()
        return {species for species, entry in self._ours.items() if entry.turns_seen == 0}

    def opponent_first_turn(self) -> dict[str, bool]:
        """species id -> whether that opposing active is on its first turn out."""
        return {species: entry.turns_seen == 0 for species, entry in self._theirs.items()}

    def snapshot(self, rules: list[SlotRules]) -> dict:
        """The tracker's view, for the model's prompt and the match log."""
        ours = {
            rule.species: {
                "first_turn_moves_work": rule.fresh,
                "protect_success_chance": round(rule.protect_success, 3),
                "last_move": rule.last_move,
            }
            for rule in rules
            if rule.species
        }
        theirs = {}
        for species, entry in self._theirs.items():
            first_turn = entry.turns_seen == 0
            view = {"first_turn_on_field": first_turn}
            if self._known_moves is not None:
                view["fake_out_threat"] = first_turn and "fakeout" in {
                    move_id(m) for m in self._known_moves(species) or []
                }
            theirs[species] = view
        withheld = [
            f"slot_{number} option {index}: {reason}"
            for number, rule in enumerate(rules)
            for index, reason in rule.blocked.items()
        ]
        effects = {m: self.effect_turns_left(m) for m in self._effects}
        return {"ours": ours, "theirs": theirs, "our_side_effects_turns_left": effects or "none",
                "withheld": withheld, "special_moves_used": list(self.used)}

    def _rules(self, slot, species: str | None) -> SlotRules:
        entry = self._ours.get(species) if species else None
        if entry is None:
            return SlotRules(species)
        fresh = entry.move_actions == 0
        blocked = {}
        for index, option in enumerate(slot.options):
            if option.get("type") != "move":
                continue
            move = move_id(option.get("move_id"))
            if move in FIRST_TURN_ONLY and not fresh:
                blocked[index] = f"{move} fails: {species} has already moved since it came in"
            elif move in PROTECT_CHAIN and entry.protect_chain >= PROTECT_BLOCK_CHAIN:
                blocked[index] = f"{move} would be {species}'s protect #{entry.protect_chain + 1} in a row (1/{3 ** entry.protect_chain} chance)"
            elif move == "trickroom" and not getattr(self, "_their_trick_room", False):
                # We do not play Trick Room; it is only used to END theirs.
                blocked[index] = "trickroom: we do not play Trick Room (only to end theirs)"
            elif move in NO_REPEAT and entry.last_move == move:
                blocked[index] = f"{move} cannot be used two turns in a row"
            elif (left := self.effect_turns_left(move)) > 0:
                effect = "would END our Trick Room" if move == "trickroom" else "fails: it is already up"
                blocked[index] = f"{move} {effect} ({left} turn(s) left)"
        # Never withhold everything: the server says each option is legal,
        # and a slot must still send something.
        if len(blocked) == len(slot.options):
            blocked = {}
        return SlotRules(species, fresh, entry.protect_chain, entry.last_move, blocked)


    def _lead_with_fake_out(self, slots: list, rules: list[SlotRules], observation: dict) -> list[SlotRules]:
        """Opening turn: Fake Out only into a foe we KNOW cannot Protect.

        Leads Protect on their first turn out, so a Fake Out into one that
        may is likely wasted (match 4e60a5a2: Fake Out and Thunder Wave both
        hit a Protect). A foe whose draft card, or all four revealed moves,
        has no Protect is a safe target: the lead with the weaker offense
        Fake Outs it and the other is free to act. No safe target (or a foe
        whose ability blocks priority): Fake Out waits for a later turn.
        A foe that threatens a KO on one of ours is a target too, Protect or
        not (match e1f70879: Fake Out was held, and Archaludon's Body Press
        took Chien-Pao on turn 1).
        """
        foes = _opponent_types(observation)
        safe = set() if any(s in FAKE_OUT_BLOCKERS for s in foes) else \
            self._cannot_protect(observation) | self._threatening(observation)
        users = []
        for slot, rule in zip(slots, rules):
            index = _fake_out_index(slot, rule)
            targets = _fake_out_targets(slot.options[index], foes, safe) if index is not None else ()
            if targets:
                users.append((self._offense(rule.species) or 0 if self._offense else 0, slot.number, index, targets))
        changed = list(rules)
        if not users:
            for slot, rule in zip(slots, rules):
                index = _fake_out_index(slot, rule)
                blocked = {**rule.blocked, index: "opening turn: every foe may Protect on its first turn out and "
                                                  "none threatens a KO; Fake Out only into one that cannot "
                                                  "Protect or one that threatens a KO"} if index is not None else {}
                if index is not None and len(blocked) < len(slot.options):
                    changed[slot.number] = replace(rule, blocked=blocked)
            return changed
        _, locked, fake_out, targets = min(users)
        for slot, rule in zip(slots, rules):
            blocked = dict(rule.blocked)
            aimed = dict(rule.targets)
            if slot.number == locked:
                reason = f"opening turn: {rule.species} leads with Fake Out"
                blocked.update({i: reason for i in range(len(slot.options)) if i != fake_out})
                aimed[fake_out] = targets
            else:
                partner = _fake_out_index(slot, rule)
                if partner is not None:
                    blocked[partner] = "opening turn: one Fake Out is enough; the partner attacks"
            if len(blocked) < len(slot.options):
                changed[slot.number] = replace(rule, blocked=blocked, targets=aimed)
        return changed

    def _threatening(self, observation: dict) -> set[str]:
        """Opposing actives that threaten a KO on one of ours this turn."""
        if self._ko_threats is None:
            return set()
        try:
            return {move_id(s) for s in self._ko_threats(observation) or ()}
        except Exception:  # noqa: BLE001 - no threat read: no extra target
            return set()

    def _cannot_protect(self, observation: dict) -> set[str]:
        """Opposing actives certain to have no Protect: their draft card, or all four revealed moves."""
        found = set()
        for entry in observation.get("opponent_active_pokemon") or []:
            if not isinstance(entry, dict) or _fainted(entry):
                continue
            name = entry.get("species") or entry.get("name")
            exact = self._exact_moves(name) if self._exact_moves and name else None
            moves = {move_id(m) for m in exact} if exact else _known_moves(entry)
            if (exact or len(moves) >= 4) and not moves & PROTECT_CHAIN:
                found.add(move_id(name))
        return found


def _fake_out_index(slot, rule: SlotRules) -> int | None:
    return next(
        (
            index
            for index, option in enumerate(slot.options)
            if option.get("type") == "move" and move_id(option.get("move_id")) == "fakeout" and index not in rule.blocked
        ),
        None,
    )


def _fake_out_targets(option: dict, foes: dict[str, list[str]], safe: set[str]) -> tuple[int, ...]:
    """Fake Out's legal targets among ``safe`` foes that a Normal move can hit."""
    found = []
    for entry in option.get("target_options") or []:
        if not isinstance(entry, dict) or entry.get("side") != "opponent":
            continue
        species = move_id(entry.get("species"))
        types = foes.get(species)
        if species in safe and (not types or effectiveness("NORMAL", types) > 0):
            found.append(entry.get("target"))
    return tuple(t for t in found if isinstance(t, int))


def _opponent_types(observation: dict) -> dict[str, list[str]]:
    found = {}
    for entry in observation.get("opponent_active_pokemon") or []:
        if isinstance(entry, dict) and not _fainted(entry):
            name = move_id(entry.get("species") or entry.get("name"))
            types = entry.get("types")
            if name:
                found[name] = [t for t in types if isinstance(t, str)] if isinstance(types, list) else []
    return found


def _carry(previous: dict[str, OnField], present: list[str]) -> dict[str, OnField]:
    """Keep history for those still on the field; start fresh for arrivals."""
    return {species: previous.get(species) or OnField() for species in dict.fromkeys(present)}


def _slot_species(slot, observation: dict) -> str | None:
    """The Pokemon able to act in this slot, or None if it is leaving."""
    if slot.force_switch:
        return None  # fainted or pivoting out: whoever replaces it is new
    active = slot.active
    if isinstance(active, dict):
        active = active.get("species") or active.get("name")
    if not active:
        actives = observation.get("active_pokemon")
        if isinstance(actives, list) and slot.number < len(actives) and isinstance(actives[slot.number], dict):
            entry = actives[slot.number]
            active = entry.get("species") or entry.get("name")
    return move_id(active) or None


def _opponent_species(observation: dict) -> list[str]:
    found = []
    for entry in observation.get("opponent_active_pokemon") or []:
        if not isinstance(entry, dict) or _fainted(entry):
            continue
        name = move_id(entry.get("species") or entry.get("name"))
        if name:
            found.append(name)
    return found


def _foe_targets(slot) -> dict[int, str]:
    """target number -> opposing species id, across this slot's options."""
    found = {}
    for option in slot.options:
        for entry in option.get("target_options") or []:
            if isinstance(entry, dict) and entry.get("side") == "opponent" and entry.get("species"):
                found[entry.get("target")] = move_id(entry["species"])
    return found


def _known_moves(entry: dict) -> set[str]:
    raw = entry.get("moves") or entry.get("known_moves") or entry.get("revealed_moves") or []
    items = raw.keys() if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    return {move_id(m.get("id") or m.get("name") if isinstance(m, dict) else m) for m in items} - {""}


def _hp_fraction(entry: dict) -> float:
    value = entry.get("hp_fraction")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    current, maximum = entry.get("current_hp"), entry.get("max_hp")
    if isinstance(current, (int, float)) and isinstance(maximum, (int, float)) and maximum:
        return current / maximum
    return 1.0


def _fainted(entry: dict) -> bool:
    if entry.get("fainted") is True:
        return True
    hp = entry.get("hp_fraction", entry.get("current_hp"))
    return isinstance(hp, (int, float)) and not isinstance(hp, bool) and hp <= 0
