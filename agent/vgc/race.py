"""Two-turn race: score a joint action by who is ahead after it plays out.

Scoring one turn on its own valued Protect by the damage it blocked and
Tailwind by a constant, so the agent kept doing things that dealt nothing:
in match 8a882a3f Talonflame went Protect, Tailwind, Protect, Tailwind,
Tailwind, Protect, Tailwind and never attacked. Here every action is worth
what it changes in the damage race:

- Turn 1: our pair against the prediction (each foe Protects, switches or
  attacks one of ours with the predicted chances). Actions resolve in speed
  order (priority first; Tailwind doubles Speed; Trick Room reverses it), so
  a Pokemon likely KO'd before it moves mostly does not act. Fake Out
  flinches, Protect blocks, a switch-in takes the hits aimed at its slot,
  Helping Hand boosts the partner, Follow Me / Rage Powder pull single
  hits, a hit into a fainted target moves to its partner. Tailwind, Trick
  Room, screens, Icy Wind-style speed drops, sleep, burns, paralysis and
  boosting moves change turn 2.
- Turns 2 and 3: everything still standing uses its best attack on its best
  target (no Protect, no switches). Two follow-up turns because speed control
  pays off once HP is low enough for moving first to mean a KO first, and
  Tailwind is still up on turn 3. Gale Wings and Prankster priority count.
- End: a Pokemon left in KO range of a faster foe is counted half lost: it
  will likely fall before acting. So a Protect that only delays the same
  hits gains nothing, while one that lets the partner remove an attacker,
  or outlasts their Tailwind, does.

Score = damage dealt minus damage taken, on the HP scale, with KO_BONUS per
expected KO (lookahead's scale). Expected values throughout, no branching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import movelist
from .typechart import effectiveness

DOOMED_SHARE = 0.5
# A priority move that KOs takes the target out before it acts: worth more
# than its damage (match 744c9e10: Baxcalibur's Ice Shard took a 27% Rillaboom
# that the sim had it Icicle Spear after our faster Chien-Pao KO'd it).
PRIORITY_KO_BONUS = 25.0
# When both of ours are in their combined KO range, Protecting one invites the
# KO on the other: this share of the hits aimed at the protector moves to its
# partner (match 1fe90ea3: Iron Valiant Protected, they KO'd Iron Hands).
FOCUS_SHIFT = 0.5
FOLLOW_UP_TURNS = 2
# What WE gain in each later turn counts this much of the turn before: it is
# less certain (the foe may Protect, switch, crit), so a KO now beats the same
# KO next turn. What we LOSE later counts in full, so delaying their hits with
# Protect earns nothing by itself.
FOLLOW_UP_WEIGHT = 0.85
SCREEN = 2732 / 4096  # doubles screen multiplier
FULL_PARALYSIS = 0.25
OTHER_STATUS_VALUE = 10.0  # Taunt, Encore, Parting Shot...: not simulated, a small credit
SLEEP = frozenset({"spore", "sleeppowder", "hypnosis", "lovelykiss", "sing", "darkvoid", "grasswhistle"})
BURN = frozenset({"willowisp"})
PARALYZE = frozenset({"thunderwave", "glare", "stunspore", "nuzzle"})
REDIRECT = frozenset({"followme", "ragepowder"})
SCREENS = {"reflect": ("PHYSICAL",), "lightscreen": ("SPECIAL",), "auroraveil": ("PHYSICAL", "SPECIAL")}
# Boosting moves -> (stat, stages) pairs that change turn-2 damage or Speed.
SETUP = {
    "swordsdance": (("atk", 2),), "nastyplot": (("spa", 2),), "calmmind": (("spa", 1),), "bulkup": (("atk", 1),),
    "dragondance": (("atk", 1), ("spe", 1)), "quiverdance": (("spa", 1), ("spe", 1)),
    "victorydance": (("atk", 1), ("spe", 1)), "shellsmash": (("atk", 2), ("spa", 2), ("spe", 2)),
    "tidyup": (("atk", 1), ("spe", 1)), "agility": (("spe", 2),), "rockpolish": (("spe", 2),),
}
# Recoil: share of the damage dealt the user takes back (match 07a3cb1f: two
# Wood Hammers cost Rillaboom about half its HP). Rocky Helmet: 1/6 of max HP
# on contact; Life Orb: 1/10 per attack.
RECOIL = {
    "woodhammer": 1 / 3, "bravebird": 1 / 3, "flareblitz": 1 / 3, "wavecrash": 1 / 3, "doubleedge": 1 / 3,
    "volttackle": 1 / 3, "headsmash": 1 / 2, "lightofruin": 1 / 2, "wildcharge": 1 / 4, "takedown": 1 / 4,
    "submission": 1 / 4, "headcharge": 1 / 4,
}
FIRST_TURN_ONLY = frozenset({"fakeout", "firstimpression"})
# Moves that switch the user out after they hit (or, for the status ones, after they work).
PIVOTS = frozenset({"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception", "shedtail"})
# Fail unless the target is about to use a damaging move.
NEEDS_TARGET_ATTACKING = frozenset({"suckerpunch", "thunderclap"})
GUARDS = {"quickguard": "quick", "wideguard": "wide"}  # +3: the user's side is safe from priority / spread moves
FLINCH_PROOF_ABILITIES = frozenset({"innerfocus", "shielddust"})
FIELD_MOVES = frozenset({"tailwind", "trickroom"}) | set(SCREENS) | set(SETUP) | set(GUARDS) | {"partingshot", "teleport"}
PROTECTS = frozenset({"protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "burningbulwark", "silktrap", "obstruct"})


def _id(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


@dataclass
class Mon:
    side: str  # "us" / "them"
    entry: dict
    hp: float  # expected HP left, % of max
    alive: float = 1.0  # chance it is still standing
    protecting: float = 0.0
    switching: float = 0.0
    flinch: float = 0.0
    asleep: float = 0.0
    burned: float = 0.0
    paralyzed: float = 0.0
    speed_mult: float = 1.0
    boost: dict = field(default_factory=dict)  # "atk"/"spa" -> expected damage multiplier
    slot_of: str = ""  # a switch-in: who the prediction aimed at in this slot
    sash: float = 0.0  # chance Focus Sash is intact and it is untouched: a single hit then leaves it at 1 HP
    def_stage: float = 0.0  # expected Defense stages gained (Stamina)
    presence: float = 1.0  # chance it is on the field (a benched foe: 0 until it may switch in)
    attacking: float = 1.0  # chance its action this turn is a damaging move (for Sucker Punch)
    pivot_to: "Mon | None" = None  # who comes in if it pivots out (U-turn, Parting Shot...)
    weight: float = 1.0  # what its HP is worth to the race: more for their win condition
    turn_hp: float | None = None  # its HP when this turn began
    turn_hits: list = field(default_factory=list)  # (chance, damage %) of the hits it took this turn

    @property
    def name(self) -> str:
        return _id(self.entry.get("species") or self.entry.get("name"))

    @property
    def aimed_as(self) -> str:
        """The name the prediction uses for whoever stands in this slot."""
        return self.slot_of or self.name

    def acts(self) -> float:
        return self.presence * self.alive * (1 - self.flinch) * (1 - self.asleep) * (1 - FULL_PARALYSIS * self.paralyzed)


@dataclass
class Board:
    """Side effects in force for a turn: chance-weighted, as everything here."""

    tailwind: dict = field(default_factory=lambda: {"us": 0.0, "them": 0.0})
    trick_room: float = 0.0
    screens: dict = field(default_factory=lambda: {"us": {}, "them": {}})  # side -> category -> chance
    helping: dict = field(default_factory=dict)  # id(mon) -> chance its partner helped it
    redirect: dict = field(default_factory=lambda: {"us": None, "them": None})  # side -> (mon, chance)
    guard: dict = field(default_factory=lambda: {"us": {}, "them": {}})  # side -> {"quick"/"wide": chance}


def score(pair: list, slots: list, per_slot: list, observation: dict, calc, expected: dict,
          bench: list[str] | None = None) -> float:
    """The two-turn race value of one joint action (``pair``: one row per slot).

    ``bench``: their Pokemon that could switch in. A foe predicted to switch
    takes only part of our hit; the rest lands on whichever of these takes it
    best, since that is who they switch to (match 0af0f5a5: Close Combat and
    Fake Out into Incineroar hit the Gholdengo it switched to; Dire Claw into
    Sylveon hit Archaludon)."""
    ours = [_mon("us", calc.ours(slot.number), calc._table) for slot in slots]
    theirs = [_mon("them", foe, calc._table) for foe in calc._theirs]
    from .lookahead import WIN_CONDITION_WEIGHT, win_conditions

    win = win_conditions(calc)
    for mon in theirs:
        if mon is not None and mon.name in win:
            mon.weight = WIN_CONDITION_WEIGHT
    benched: list[Mon] = []
    board = _board(observation)
    plans = []
    for slot, row, rule, mon in zip(slots, pair, per_slot, list(ours)):
        option = slot.options[row["option"]] if 0 <= row.get("option", -1) < len(slot.options) else {}
        if option.get("type") == "switch" and mon is not None:
            fresh = _mon("us", {"species": option.get("species"), "hp_fraction": _bench_hp(observation, option.get("species"))}, calc._table)
            fresh.slot_of = mon.name  # it takes the hits the prediction aimed at the slot (match 8e3a639c)
            benched.append(mon)
            ours[ours.index(mon)] = fresh
        elif option.get("type") == "move" and mon is not None:
            plans.append((slot, row, option, rule, mon))
    for mon in theirs:
        e = expected.get(mon.name)
        if e:
            mon.protecting, mon.switching = e.protect, e.switch
            mon.attacking = min(1.0, sum(e.attacks.values()))
    for slot, row, option, rule, mon in plans:
        listed = movelist.lookup(option.get("move_id")) or {}
        mon.attacking = 1.0 if listed.get("category") in ("PHYSICAL", "SPECIAL") else 0.0
    for mon in ours:
        if mon is not None and mon not in [p[4] for p in plans]:
            mon.attacking = 0.0  # switching in this turn
    from .lookahead import SLEEP_AWAKE, _status

    for mon in ours:
        if mon is not None and _status(mon.entry.get("status")) == "slp":
            mon.asleep = 1 - SLEEP_AWAKE  # our sleeper acts only if it wakes

    sim = _Sim(calc, board, ours, theirs, observation)
    # Our pivots: who comes in after U-turn / Parting Shot (the one their reads hurt least).
    for slot, row, option, rule, mon in plans:
        if _id(option.get("move_id")) in PIVOTS:
            names = [o.get("species") for o in slot.options if o.get("type") == "switch" and o.get("species")]
            if names:
                best = min(names, key=lambda n: (calc.incoming(n) or {"expected_pct": 50.0})["expected_pct"])
                fresh = _mon("us", {"species": best, "hp_fraction": _bench_hp(observation, best)}, calc._table)
                if fresh is not None:
                    fresh.presence, fresh.slot_of, mon.pivot_to = 0.0, mon.aimed_as, fresh
                    sim.ours.append(fresh)
                    benched.append(fresh)
    sim.bench = [m for s in bench or [] if (m := _mon("them", {"species": s, "hp_fraction": _foe_bench_hp(observation, s)}, calc._table))]
    for m in sim.bench:
        m.presence = 0.0
    bonus = 0.0
    actions = []
    for slot, row, option, rule, mon in plans:
        move_id = _id(option.get("move_id"))
        if move_id in PROTECTS:
            # Asleep, it Protects only if it wakes (match ead979d8: a sleeping
            # Dondozo "Protected" on turns 7 and 9 and took Sludge Bomb each time).
            mon.protecting = rule.protect_success * mon.alive * (1 - mon.asleep)
            continue
        found, extra = sim.our_action(slot, row, option, move_id, mon)
        actions.extend(found)
        bonus += extra
    for mon in theirs:
        actions.extend(sim.foe_actions(mon, expected.get(mon.name)))
    everyone = [m for m in ours + theirs + benched + sim.bench if m is not None]
    state = _state(everyone)
    sim.resolve(actions)
    sim.end_of_turn()
    gain, loss = _swing(everyone, state)
    value = gain - loss

    # Follow-up turns: no Protect, no switches; the effects set in turn 1 apply.
    # A foe that may have switched shares its place with its switch-in, which
    # attacks too (match 915ba9e1: Dragonite came in with Baxcalibur, 4x Ice,
    # on their bench, and fell to it the next turn).
    sim.bring_switch_ins()
    sim.board = sim.next_board
    # Turn 1 already scaled a sleeping foe's actions (lookahead.asleep); later
    # turns: it may sleep on (ours from turn 1, below).
    from .lookahead import SLEEP_AWAKE, _status

    for mon in theirs:
        if mon is not None and _status(mon.entry.get("status")) == "slp":
            mon.asleep = max(mon.asleep, 1 - SLEEP_AWAKE)
    weight = 1.0
    for _ in range(FOLLOW_UP_TURNS):
        weight *= FOLLOW_UP_WEIGHT
        for mon in sim.ours + sim.theirs + sim.bench:
            mon.protecting = mon.switching = mon.flinch = 0.0
            mon.attacking = 1.0
        sim.board.helping = {}
        sim.board.guard = {"us": {}, "them": {}}
        state = _state(everyone)
        sim.resolve([a for mon in sim.ours + sim.foes_present() if (a := sim.best_attack(mon)) is not None])
        sim.end_of_turn()
        gain, loss = _swing(everyone, state)
        value += weight * gain - loss
    value += weight * sum(sim.doomed(m) for m in theirs if m) - sum(sim.doomed(m) for m in ours if m)
    return value + bonus


def _state(mons: list) -> list[tuple[float, float]]:
    return [(m.hp, m.alive) for m in mons]


def _swing(mons: list, before: list[tuple[float, float]]) -> tuple[float, float]:
    """What one turn changed: (their losses, ours), as HP plus KO_BONUS per KO."""
    from .lookahead import KO_BONUS

    gain = loss = 0.0
    for mon, (hp, alive) in zip(mons, before):
        lost = (hp - mon.hp) + KO_BONUS * (alive - mon.alive)
        if mon.side == "them":
            gain += lost * mon.weight
        else:
            loss += lost
    return gain, loss


def _mon(side: str, entry: dict | None, table=None) -> Mon | None:
    from .lookahead import _fraction

    if not entry:
        return None
    hp = 100 * _fraction(entry)
    from .lookahead import held_item

    item = held_item(entry, table) if table is not None else (entry.get("item") or "")
    return Mon(side, dict(entry), hp, alive=1.0 if hp > 0 else 0.0, sash=1.0 if _id(item) == "focussash" and hp >= 99.5 else 0.0)


def lookahead_hits(move: dict) -> float:
    from .lookahead import expected_hits

    return expected_hits(move.get("id") or move.get("move_id") or "")


def _bench_hp(observation: dict, species: object) -> float:
    from .lookahead import _fraction

    team = observation.get("team")
    entries = team.values() if isinstance(team, dict) else team if isinstance(team, list) else []
    for entry in entries:
        if isinstance(entry, dict) and _id(entry.get("species") or entry.get("name")) == _id(species):
            return _fraction(entry)
    return 1.0


def _foe_bench_hp(observation: dict, species: object) -> float:
    from .lookahead import _fraction

    team = observation.get("opponent_team")
    entries = team.values() if isinstance(team, dict) else team if isinstance(team, list) else []
    for entry in entries:
        if isinstance(entry, dict) and _id(entry.get("species") or entry.get("name")) == _id(species):
            return _fraction(entry)
    return 1.0


def _board(observation: dict) -> Board:
    from .observed import field_state

    board = Board()
    shown = field_state(observation) or {"ours": [], "theirs": [], "field": []}
    for side, key in (("us", "ours"), ("them", "theirs")):
        up = set(shown[key])
        board.tailwind[side] = 1.0 if "tailwind" in up else 0.0
        for move, categories in SCREENS.items():
            if move in up:
                for category in categories:
                    board.screens[side][category] = 1.0
    board.trick_room = 1.0 if "trickroom" in shown["field"] else 0.0
    return board


class _Sim:
    def __init__(self, calc, board: Board, ours: list, theirs: list, observation: dict) -> None:
        self.calc = calc
        self.board = board
        self.next_board = Board(
            tailwind=dict(board.tailwind), trick_room=board.trick_room,
            screens={side: dict(v) for side, v in board.screens.items()},
        )
        self.ours = [m for m in ours if m is not None]
        self.theirs = [m for m in theirs if m is not None]
        self.bench: list[Mon] = []  # their switch-in candidates
        self._entered: dict[int, Mon] = {}  # id(foe) -> the switch-in our turn-1 hits went to
        self.observation = observation

    # -- building actions ------------------------------------------------------------

    def our_action(self, slot, row, option, move_id, mon) -> tuple[list, float]:
        from .actions import move_details

        details = {**move_details(option, slot.number, self.observation), "id": move_id}
        listed = movelist.lookup(move_id) or {}
        power = details.get("base_power") or 0
        category = str(details.get("category") or "").upper()
        priority = details.get("priority") or 0
        if power > 0 and category in ("PHYSICAL", "SPECIAL"):
            spread = listed.get("target") in ("allAdjacentFoes", "allAdjacent")
            targets = self.theirs if spread else [t for t in [self._foe_at(row.get("target"))] if t]
            if not targets:
                return [], 0.0
            move = {**details, "spread": spread, "bypass": "bypasses_protect" in (listed.get("tags") or [])}
            priority = self.priority(mon, move)
            kind = "fakeout" if move_id == "fakeout" else "attack_slow" if "speed_control" in (listed.get("tags") or []) else "attack"
            return [(kind, mon, move, [(t, 1.0) for t in targets], priority)], 0.0
        target = self._foe_at(row.get("target"))
        priority = self.priority(mon, {**details, "category": "STATUS"})
        if move_id in FIELD_MOVES or move_id in SLEEP | BURN | PARALYZE | REDIRECT or move_id == "helpinghand":
            return [("status", mon, {"id": move_id, "priority": priority}, [(target, 1.0)] if target else [], priority)], 0.0
        return [], OTHER_STATUS_VALUE * mon.alive

    def foe_actions(self, foe: Mon, e) -> list:
        if e is None:  # no read on it: it attacks where it hurts most
            best = self.best_attack(foe)
            return [best] if best else []
        found = []
        if getattr(e, "redirect", 0.0) > 0:
            # Rage Powder / Follow Me to cover a partner in our KO range (match 37a1ee8a:
            # Amoonguss pulled the Surging Strikes meant for a 36% Arcanine).
            found.append(("status", foe, {"id": "ragepowder", "priority": 2, "weight": e.redirect}, [], 2))
        spread_chance, spread_move = 0.0, None
        shifted = self._focus_shifts(e)  # id(mon) -> change in the chance it is attacked (never mutates e)
        for target in self.ours:
            chance = max(0.0, e.attacks.get(target.aimed_as, 0.0) + shifted.get(id(target), 0.0))
            if chance <= 0:
                continue
            fake = min(chance, e.fake_out.get(target.aimed_as, 0.0))
            if fake > 0:
                move = {"id": "fakeout", "base_power": 40, "type": "NORMAL", "category": "PHYSICAL", "spread": False, "priority": 3}
                found.append(("fakeout", foe, move, [(target, fake)], 3))
            if chance - fake > 0:
                move = self._best_move(foe, [target])
                if move and move.get("spread"):
                    # Make It Rain, Heat Wave...: one move hits both of ours (match
                    # 0af0f5a5: Whimsicott switched into a +2 Make It Rain "aimed at Sneasler").
                    spread_chance += chance - fake
                    spread_move = spread_move or move
                elif move:
                    found.append(("attack", foe, move, [(target, chance - fake)], self.priority(foe, move)))
        if spread_move is not None:
            weight = min(1.0, spread_chance)
            found.append(("attack", foe, spread_move, [(t, weight) for t in self.ours], self.priority(foe, spread_move)))
        return found

    def _focus_shifts(self, e) -> dict[int, float]:
        """id(mon) -> change in its attack chance: a share moved off a Protecting
        Pokemon onto its partner, when both of ours are in their combined KO range."""
        moved: dict[int, float] = {}
        live = [m for m in self.ours if m.alive > 0 and m.presence > 0]
        if len(live) != 2 or not all(self._in_combined_ko(m) for m in live):
            return moved
        for mon, partner in (live, live[::-1]):
            share = e.attacks.get(mon.aimed_as, 0.0) * mon.protecting * FOCUS_SHIFT
            if share > 0 and partner.protecting < 0.5:
                moved[id(mon)] = moved.get(id(mon), 0.0) - share
                moved[id(partner)] = moved.get(id(partner), 0.0) + share
        return moved

    def _in_combined_ko(self, mon: Mon) -> bool:
        """Their actives' best hits on ``mon`` add up to its HP."""
        total = 0.0
        for foe in self.theirs:
            if foe.alive <= 0:
                continue
            move = self._best_move(foe, [mon])
            hit = self._damage(foe, mon, move) if move else None
            total += hit["expected_pct"] if hit else 0.0
        return total >= mon.hp

    def best_attack(self, mon: Mon):
        """Turn 2: its best attack against the other side as it stands."""
        if mon.alive <= 0:
            return None
        enemies = self.foes_present() if mon.side == "us" else [m for m in self.ours if m.alive > 0]
        if not enemies:
            return None
        best, best_value = None, 0.0
        for move in self._attacks(mon):
            targets = enemies if move["spread"] else None
            for target in ([None] if targets else enemies):
                hit_on = targets or [target]
                value = self._choice_value(mon, hit_on, move)
                if value > best_value:
                    best, best_value = ("attack", mon, move, [(t, 1.0) for t in hit_on], self.priority(mon, move)), value
        return best

    def _best_move(self, attacker: Mon, targets: list) -> dict | None:
        best, best_value = None, 0.0
        for move in self._attacks(attacker):
            value = self._choice_value(attacker, targets, move)
            if value > best_value:
                best, best_value = move, value
        return best

    def _choice_value(self, attacker: Mon, targets: list, move: dict) -> float:
        """What picking this move is worth: its damage, plus PRIORITY_KO_BONUS per
        likely KO when it moves in a priority bracket (the target never acts)."""
        value = sum(self._value(attacker, t, move) for t in targets)
        if self.priority(attacker, move) > 0:
            for target in targets:
                hit = self._damage(attacker, target, move) if target.alive > 0 else None
                if hit:
                    value += PRIORITY_KO_BONUS * target.presence * target.alive * _ko(hit, target.hp)
        return value

    def _attacks(self, mon: Mon) -> list[dict]:
        """Its attacks for a plain turn: not Fake Out / First Impression, which
        only work on its first turn out (a foe's Fake Out is read from the prediction)."""
        return [m for m in self.calc.likely_attacks(mon.entry) if _id(m.get("id") or m.get("name")) not in FIRST_TURN_ONLY]

    def _foe_at(self, target: object) -> Mon | None:
        if isinstance(target, int) and 0 < target <= len(self.theirs):
            return self.theirs[target - 1]
        return None

    def priority(self, mon: Mon, move: dict) -> int:
        """Move priority, plus Gale Wings (Flying moves at full HP) and Prankster (status moves)."""
        value = move.get("priority") or 0
        abilities = {_id(a) for a in self.calc._abilities(mon.entry)}
        if len(abilities) == 1 and "galewings" in abilities and str(move.get("type")).upper() == "FLYING" and mon.hp >= 99.5:
            value += 1
        if len(abilities) == 1 and "prankster" in abilities and str(move.get("category")).upper() == "STATUS":
            value += 1
        if _id(move.get("id") or move.get("name")) == "grassyglide" and self.calc._field.terrain == "grassy" \
                and "FLYING" not in [str(t).upper() for t in (mon.entry.get("types") or [])]:
            value += 1  # +1 in Grassy Terrain (grounded user)
        return value

    def threat_switch_in(self) -> "Mon | None":
        """Their benched Pokemon that would hurt our side most."""
        options = [m for m in self.bench if m.alive > 0]
        if not options:
            return None
        return max(options, key=lambda m: max((self._value(m, ours, move) for ours in self.ours if ours.alive > 0
                                               for move in self._attacks(m)), default=0.0))

    # -- resolving -------------------------------------------------------------------

    def speed(self, mon: Mon) -> float:
        base = self.calc.speed(mon.entry) or 0.0
        return base * (1 + self.board.tailwind[mon.side]) * mon.speed_mult * (1 - 0.5 * mon.paralyzed)

    def resolve(self, actions: list) -> None:
        for mon in self.ours + self.theirs + self.bench:
            mon.turn_hp, mon.turn_hits = mon.hp, []
        reverse = self.board.trick_room > 0.5
        # Higher priority first, then faster (slower under Trick Room). A speed tie is a
        # coin flip in Showdown; we assume we lose it (they move first), the safe side.
        order = sorted(actions, key=lambda a: (-a[4], self.speed(a[1]) if reverse else -self.speed(a[1]),
                                               0 if a[1].side == "them" else 1))
        for kind, user, move, targets, priority in order:
            if kind == "status":
                self._status(user, move["id"], targets, move.get("weight", 1.0))
            else:
                self._hit(kind, user, move, targets, priority)
            if _id(move.get("id") or move.get("name")) in PIVOTS:
                self._pivot(user)

    def _pivot(self, user: Mon) -> None:
        """After a U-turn / Parting Shot: the user leaves as often as it acted."""
        if user.pivot_to is None and user.side == "them":
            user.pivot_to = self.threat_switch_in()  # their most dangerous benched Pokemon
        incoming = user.pivot_to
        if incoming is None or incoming.alive <= 0:
            return
        went = user.presence * user.alive * (1 - user.flinch) * (1 - user.asleep)
        incoming.presence = min(1.0, incoming.presence + went)
        user.presence *= 1 - went
        if user.side == "them" and incoming not in self.bench:
            self.bench.append(incoming)

    def _hit(self, kind: str, user: Mon, move: dict, targets: list, priority: int = 0) -> None:
        acts = user.acts()
        if acts <= 0:
            return
        boost = self.board.helping.get(id(user), 0.0)
        move_id = _id(move.get("id") or move.get("name"))
        from .lookahead import accuracy

        hit_chance = accuracy(move)
        for target, weight in targets:
            for mon, share in self._landing_spots(user, target, move):
                if mon.alive <= 0:
                    continue
                guard = self.board.guard.get(mon.side, {})
                if priority > 0 and guard.get("quick"):
                    share *= 1 - guard["quick"]  # Quick Guard: priority moves (Fake Out too) fail
                if move.get("spread") and guard.get("wide"):
                    share *= 1 - guard["wide"]
                if move_id in NEEDS_TARGET_ATTACKING:
                    share *= mon.attacking  # Sucker Punch fails unless the target attacks
                switch_in = self._switch_in(user, mon, move) if mon.switching else None
                if mon.switching and switch_in is not None:
                    self._land(kind, user, switch_in, move, acts * weight * share * mon.switching, boost)
                    stays = 1 - mon.switching
                else:
                    stays = 1 - 0.5 * mon.switching  # no bench known: half credit, as before
                land = weight * share * (1.0 if move.get("bypass") else 1 - mon.protecting) * stays * mon.presence
                land *= hit_chance  # it can miss
                if land > 0:
                    self._land(kind, user, mon, move, acts * land, boost)
        self._self_drop(user, move_id, acts)

    @staticmethod
    def _self_drop(user: Mon, move_id: str, acts: float) -> None:
        """Draco Meteor, Overheat...: the user's later hits are weaker as often
        as it fired (match 286bbc2e: two Draco Meteors left Archaludon at -4 SpA)."""
        from .principles import SELF_DROPS

        drop = SELF_DROPS.get(move_id)
        if drop and acts > 0:
            stat, stages = drop
            factor = 2 / (2 + stages)  # one stage at a time from neutral: -1 -> 2/3, -2 -> 1/2
            user.boost[stat] = user.boost.get(stat, 1.0) * (1 - acts * (1 - factor))

    def _switch_in(self, user: Mon, leaving: Mon, move: dict) -> Mon | None:
        """The benched foe that takes this move best: who they would switch to."""
        options = [m for m in self.bench if m.alive > 0 and m is not leaving]
        if not options:
            return None
        chosen = min(options, key=lambda m: (self._damage(user, m, move) or {"expected_pct": 0.0})["expected_pct"])
        self._entered.setdefault(id(leaving), chosen)
        return chosen

    def foes_present(self) -> list[Mon]:
        return [m for m in self.theirs + self.bench if m.alive > 0 and m.presence > 0]

    def bring_switch_ins(self) -> None:
        """After turn 1: each foe that may have switched hands that share of its
        place to its switch-in (the one our hits went to, else the one that
        threatens us most)."""
        for foe in self.theirs:
            share = foe.switching * foe.alive
            options = [m for m in self.bench if m.alive > 0]
            if share <= 0 or not options:
                continue
            chosen = self._entered.get(id(foe)) or max(
                options, key=lambda m: max((self._value(m, ours, move) for ours in self.ours if ours.alive > 0
                                            for move in self._attacks(m)), default=0.0))
            chosen.presence = min(1.0, chosen.presence + share)
            foe.presence *= 1 - foe.switching
            foe.pivot_to = foe.pivot_to or chosen  # hits aimed at its slot reach whoever is there

    def _land(self, kind: str, user: Mon, mon: Mon, move: dict, chance: float, boost: float) -> None:
        """One hit landing on ``mon`` with this chance."""
        hit = self._damage(user, mon, move)
        if hit is None:
            return
        factor = 1 + 0.5 * boost
        expected = hit["expected_pct"] * factor
        ko = self._focus_ko(mon, hit, factor)
        mon.turn_hits.append((chance, expected, ko))
        if mon.sash and lookahead_hits(move) == 1:
            # Focus Sash: an untouched holder survives at 1 HP.
            expected = mon.sash * min(expected, mon.hp - 1) + (1 - mon.sash) * expected
            ko *= 1 - mon.sash
        if mon.sash:
            mon.sash *= 1 - chance  # any landed hit breaks it (or uses it)
        if self._stamina(mon):
            # Each hit: +1 Defense, which weakens our physical hits and
            # powers its Body Press (match 48f23a42; ccb570bb).
            mon.def_stage = min(6.0, mon.def_stage + chance * lookahead_hits(move))
        dealt = chance * min(mon.hp, expected)
        mon.hp -= dealt
        mon.alive *= 1 - chance * ko
        self._self_damage(user, mon, move, dealt, chance)
        if kind == "fakeout" and not self._flinch_proof(mon):
            mon.flinch += (1 - mon.flinch) * chance
        elif kind == "attack_slow":
            mon.speed_mult *= 1 - chance / 3

    @staticmethod
    def _focus_ko(mon: Mon, hit: dict, factor: float) -> float:
        """Chance this hit KOs, given it survived the hits it already took this turn.

        By scenario, not from the average HP: two 60% hits that each land 60%
        of the time KO a full-HP target 36% of the time, where hitting the
        average (64% left) says never. Match 37a1ee8a: Ace Trainer Claude
        focused one of ours at a time."""
        start = mon.turn_hp if mon.turn_hp is not None else mon.hp
        if not mon.turn_hits or start is None:
            return _ko(hit, mon.hp, factor)
        landed = 1.0
        for chance, _, _ in mon.turn_hits:
            landed *= 1 - chance
        landed = 1 - landed  # some earlier hit landed
        if landed <= 0:
            return _ko(hit, start, factor)
        damage = sum(c * d for c, d, _ in mon.turn_hits) / landed  # what an earlier hit did, when one did
        first_ko = min(1.0, sum(c * k for c, _, k in mon.turn_hits) / landed)  # ...and how often it KO'd
        survived = landed * (1 - first_ko)
        after = max(0.0, start - damage)
        numerator = survived * _ko(hit, after, factor) + (1 - landed) * _ko(hit, start, factor)
        alive = survived + (1 - landed)
        return numerator / alive if alive > 0 else 1.0

    def _self_damage(self, user: Mon, target: Mon, move: dict, dealt: float, chance: float) -> None:
        """Recoil, Rocky Helmet and Life Orb: HP the attacker pays for the hit."""
        move_id = _id(move.get("id") or move.get("move_id"))
        cost = 0.0
        listed = movelist.lookup(move_id) or {}
        recoil = listed.get("recoil")
        share = recoil[0] / recoil[1] if isinstance(recoil, list) and len(recoil) == 2 and recoil[1] else RECOIL.get(move_id, 0.0)
        if share and dealt > 0:
            # Damage is in the target's HP %; recoil is in the user's.
            cost += share * dealt * self._max_hp(target) / max(1.0, self._max_hp(user))
        if "contact" in (listed.get("flags") or []) and _id(self._item(target)) == "rockyhelmet":
            cost += chance * 100 / 6
        sheer_force = listed.get("secondary") and "sheerforce" in {_id(a) for a in self.calc._abilities(user.entry)}
        if _id(self._item(user)) == "lifeorb" and not sheer_force:
            cost += chance * 10
        if cost:
            # Faints from it only when the hit lands and the full cost covers its HP.
            if chance > 0 and cost / chance >= user.hp:
                user.alive *= 1 - chance
            user.hp = max(0.0, user.hp - cost)

    def end_of_turn(self) -> None:
        """Residual HP at the end of a turn: Leftovers / Black Sludge, Grassy Terrain,
        sandstorm, burn and poison (match 9627e1fd: Archaludon regained ~12% a turn from
        Leftovers + Grassy Terrain while we left it alive)."""
        from .lookahead import floats

        weather, terrain = self.calc._field.weather, self.calc._field.terrain
        for mon in self.ours + self.theirs + self.bench:
            if mon.alive <= 0 or mon.presence <= 0 or mon.hp <= 0:
                continue
            types = [str(t).upper() for t in (mon.entry.get("types") or self.calc._table.species_types(
                mon.entry.get("species") or mon.entry.get("name")))]
            abilities = {_id(a) for a in self.calc._abilities(mon.entry)}
            magic_guard = abilities == {"magicguard"}
            item = _id(self._item(mon))
            delta = 0.0
            if item == "leftovers" or (item == "blacksludge" and "POISON" in types):
                delta += 100 / 16
            elif item == "blacksludge" and not magic_guard:
                delta -= 100 / 8
            if terrain == "grassy" and "FLYING" not in types and not floats(mon.entry, self.calc._table):
                delta += 100 / 16
            if weather == "sand" and not magic_guard and not (set(types) & {"ROCK", "GROUND", "STEEL"})                     and not abilities & {"overcoat", "sandveil", "sandrush", "sandforce"} and item != "safetygoggles":
                delta -= 100 / 16
            status = _id(mon.entry.get("status"))
            if not magic_guard:
                burn = max(mon.burned, 1.0 if status in ("brn", "burn") else 0.0)
                delta -= burn * 100 / 16
                if status in ("psn", "tox", "poison", "toxic") and not abilities & {"poisonheal"}:
                    delta -= 100 / 8
            if not delta:
                continue
            # hp is expected HP over the KO branches; only the living, on-field share heals or chips.
            change = delta * mon.presence * mon.alive
            if change > 0:
                mon.hp = min(100.0, mon.hp + change)
                continue
            if -delta >= mon.hp / mon.alive:
                mon.alive *= 1 - mon.presence  # the chip finishes it
            mon.hp = max(0.0, mon.hp + change)

    def _flinch_proof(self, mon: Mon) -> bool:
        """Inner Focus, Shield Dust or a Covert Cloak: Fake Out does not flinch it."""
        abilities = {_id(a) for a in self.calc._abilities(mon.entry)}
        return (len(abilities) == 1 and bool(abilities & FLINCH_PROOF_ABILITIES)) or _id(self._item(mon)) == "covertcloak"

    def _item(self, mon: Mon) -> str:
        from .lookahead import held_item

        return held_item(mon.entry, self.calc._table)

    def _max_hp(self, mon: Mon) -> float:
        from .lookahead import _hp

        base = self.calc.base_stats(mon.entry)
        if not base:
            return 100.0
        return float(self.calc._stat(mon.entry, base, "hp") or _hp(base["hp"], 132))

    def _landing_spots(self, user: Mon, target: Mon, move: dict) -> list:
        """Where a hit aimed at ``target`` lands: redirected by Follow Me, or
        moved to its partner if it has fainted (single-target moves only)."""
        if move.get("spread"):
            return [(target, 1.0)]
        side = target.side
        redirect = self.board.redirect.get(side)
        spots = []
        rest = 1.0
        if redirect and redirect[0] is not target and redirect[0].alive > 0:
            spots.append((redirect[0], redirect[1]))
            rest -= redirect[1]
        partner = next((m for m in (self.ours if side == "us" else self.theirs) if m is not target and m.alive > 0), None)
        if partner is not None:
            spots += [(target, rest * target.alive), (partner, rest * (1 - target.alive))]
        else:
            spots.append((target, rest))
        # A target that may have pivoted out: whoever came in takes the hit, by presence.
        expanded = []
        for mon, share in spots:
            expanded.append((mon, share))
            if mon.pivot_to is not None and mon.pivot_to.presence > 0:
                expanded.append((mon.pivot_to, share))
        return expanded

    def _stamina(self, mon: Mon) -> bool:
        return "Stamina" in self.calc._abilities(mon.entry)

    @staticmethod
    def _staged(mon: Mon) -> dict:
        """Its entry with the Defense stages it has gained in this simulation."""
        if not mon.def_stage:
            return mon.entry
        boosts = dict(mon.entry.get("boosts") or {})
        boosts["def"] = int(max(-6, min(6, round((boosts.get("def") or 0) + mon.def_stage))))
        return {**mon.entry, "boosts": boosts}

    def _damage(self, user: Mon, target: Mon, move: dict) -> dict | None:
        from .lookahead import HP_SCALING

        attacker = self._staged(user)
        if _id(move.get("id") or move.get("name")) in HP_SCALING:
            # Eruption / Water Spout: power from the HP it has NOW in the race,
            # so our hits on it this turn cut its next one (match 4e60a5a2).
            attacker = {**attacker, "hp_fraction": max(0.0, user.hp) / 100}
        hit = self.calc.damage(attacker, self._staged(target), move, spread=bool(move.get("spread")))
        if not hit or hit.get("immune"):
            return None
        category = str(move.get("category") or "").upper()
        factor = user.boost.get("atk" if category == "PHYSICAL" else "spa", 1.0)
        if category == "PHYSICAL":
            factor *= 1 - 0.5 * user.burned
        factor *= 1 - (1 - SCREEN) * self.board.screens[target.side].get(category, 0.0)
        scaled = {**hit, "expected_pct": hit["expected_pct"] * factor,
                  "range_pct": [hit["range_pct"][0] * factor, hit["range_pct"][1] * factor]}
        if hit.get("rolls_pct"):
            scaled["rolls_pct"] = [r * factor for r in hit["rolls_pct"]]
        return scaled

    def _value(self, user: Mon, target: Mon, move: dict) -> float:
        from .lookahead import KO_BONUS

        if target.alive <= 0:
            return 0.0
        hit = self._damage(user, target, move)
        if not hit:
            return 0.0
        from .lookahead import accuracy

        return target.presence * target.alive * accuracy(move) * (min(target.hp, hit["expected_pct"]) + KO_BONUS * _ko(hit, target.hp))

    def _status(self, user: Mon, move_id: str, targets: list, weight: float = 1.0) -> None:
        chance = user.acts() * weight
        if chance <= 0:
            return
        side = user.side
        nxt = self.next_board
        if move_id == "tailwind":
            nxt.tailwind[side] = max(nxt.tailwind[side], chance)
        elif move_id == "trickroom":
            nxt.trick_room = self.board.trick_room * (1 - chance) + (1 - self.board.trick_room) * chance
        elif move_id in SCREENS:
            # Up at once: every hit after it this turn is cut too (a Prankster
            # screen beats their attacks; match 286bbc2e skipped Grimmsnarl's).
            for board in (self.board, nxt):
                for category in SCREENS[move_id]:
                    now = board.screens[side].get(category, 0.0)
                    board.screens[side][category] = now + (1 - now) * chance
        elif move_id in SETUP:
            for stat, stages in SETUP[move_id]:
                gain = (2 + stages) / 2 - 1
                if stat == "spe":
                    user.speed_mult *= 1 + chance * gain
                else:
                    user.boost[stat] = user.boost.get(stat, 1.0) * (1 + chance * gain)
        elif move_id == "helpinghand":
            partner = next((m for m in (self.ours if side == "us" else self.theirs) if m is not user), None)
            if partner is not None:
                self.board.helping[id(partner)] = chance
        elif move_id in REDIRECT:
            self.board.redirect[side] = (user, chance)
        elif move_id in GUARDS:
            self.board.guard[side][GUARDS[move_id]] = chance
        for target, weight in targets:
            if target is None or target.alive <= 0:
                continue
            land = chance * weight * (1 - target.protecting)
            if move_id == "partingshot":  # -1 Attack and Special Attack, then the user pivots
                for stat in ("atk", "spa"):
                    target.boost[stat] = target.boost.get(stat, 1.0) * (1 - land / 3)
            types = [str(t).upper() for t in (target.entry.get("types") or self.calc._table.species_types(target.entry.get("species")))]
            if move_id in SLEEP and not ("GRASS" in types and "powder" in ((movelist.lookup(move_id) or {}).get("flags") or [])):
                target.asleep += (1 - target.asleep) * land
            elif move_id in BURN and "FIRE" not in types:
                target.burned += (1 - target.burned) * land
            elif move_id in PARALYZE and "ELECTRIC" not in types and not (move_id == "thunderwave" and effectiveness("ELECTRIC", types) == 0):
                target.paralyzed += (1 - target.paralyzed) * land

    def doomed(self, mon: Mon) -> float:
        """Expected loss still pending at the end: in KO range of a faster enemy."""
        from .lookahead import KO_BONUS

        if mon.alive <= 0 or mon.hp <= 0:
            return 0.0
        enemies = [m for m in self.ours if m.alive > 0] if mon.side == "them" else self.foes_present()
        worst = 0.0
        reverse = self.board.trick_room > 0.5
        for enemy in enemies:
            move = self._best_move(enemy, [mon])
            if not move:
                continue
            hit = self._damage(enemy, mon, move)
            faster = self.priority(enemy, move) > 0 or (
                self.speed(enemy) < self.speed(mon) if reverse else self.speed(enemy) > self.speed(mon))
            if hit and faster and _ko(hit, mon.hp) > 0:
                worst = max(worst, enemy.alive * enemy.acts() * _ko(hit, mon.hp))
        return DOOMED_SHARE * worst * mon.alive * (mon.hp + KO_BONUS)


def _ko(hit: dict, hp: float, factor: float = 1.0) -> float:
    """Chance this hit KOs from ``hp``: exact over Showdown's 16 rolls when we
    have them, else read off the damage range."""
    rolls = hit.get("rolls_pct")
    if rolls:
        return sum(r * factor >= hp for r in rolls) / len(rolls) if hp > 0 else 1.0
    return _ko_chance(hit["range_pct"][0] * factor, hit["range_pct"][1] * factor, hp)


def _ko_chance(low: float, high: float, hp: float) -> float:
    if hp <= 0:
        return 1.0
    if high <= hp:
        return 0.0
    if low >= hp or high <= low:
        return 1.0
    return (high - hp) / (high - low)
