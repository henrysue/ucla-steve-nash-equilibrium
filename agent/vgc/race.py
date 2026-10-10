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
FIELD_MOVES = frozenset({"tailwind", "trickroom"}) | set(SCREENS) | set(SETUP)
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

    @property
    def name(self) -> str:
        return _id(self.entry.get("species") or self.entry.get("name"))

    @property
    def aimed_as(self) -> str:
        """The name the prediction uses for whoever stands in this slot."""
        return self.slot_of or self.name

    def acts(self) -> float:
        return self.alive * (1 - self.flinch) * (1 - self.asleep) * (1 - FULL_PARALYSIS * self.paralyzed)


@dataclass
class Board:
    """Side effects in force for a turn: chance-weighted, as everything here."""

    tailwind: dict = field(default_factory=lambda: {"us": 0.0, "them": 0.0})
    trick_room: float = 0.0
    screens: dict = field(default_factory=lambda: {"us": {}, "them": {}})  # side -> category -> chance
    helping: dict = field(default_factory=dict)  # id(mon) -> chance its partner helped it
    redirect: dict = field(default_factory=lambda: {"us": None, "them": None})  # side -> (mon, chance)


def score(pair: list, slots: list, per_slot: list, observation: dict, calc, expected: dict) -> float:
    """The two-turn race value of one joint action (``pair``: one row per slot)."""
    ours = [_mon("us", calc.ours(slot.number), calc._table) for slot in slots]
    theirs = [_mon("them", foe, calc._table) for foe in calc._theirs]
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

    sim = _Sim(calc, board, ours, theirs, observation)
    bonus = 0.0
    actions = []
    for slot, row, option, rule, mon in plans:
        move_id = _id(option.get("move_id"))
        if move_id in PROTECTS:
            mon.protecting = rule.protect_success * mon.alive
            continue
        found, extra = sim.our_action(slot, row, option, move_id, mon)
        actions.extend(found)
        bonus += extra
    for mon in theirs:
        actions.extend(sim.foe_actions(mon, expected.get(mon.name)))
    everyone = [m for m in ours + theirs + benched if m is not None]
    state = _state(everyone)
    sim.resolve(actions)
    gain, loss = _swing(everyone, state)
    value = gain - loss

    # Follow-up turns: no Protect, no switches; the effects set in turn 1 apply.
    sim.board = sim.next_board
    weight = 1.0
    for _ in range(FOLLOW_UP_TURNS):
        weight *= FOLLOW_UP_WEIGHT
        for mon in sim.ours + sim.theirs:
            mon.protecting = mon.switching = mon.flinch = 0.0
        sim.board.helping = {}
        state = _state(everyone)
        sim.resolve([a for mon in sim.ours + sim.theirs if (a := sim.best_attack(mon)) is not None])
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
            gain += lost
        else:
            loss += lost
    return gain, loss


def _mon(side: str, entry: dict | None, table=None) -> Mon | None:
    from .lookahead import _fraction

    if not entry:
        return None
    hp = 100 * _fraction(entry)
    item = entry.get("item") or (table.common_item(entry.get("species") or entry.get("name")) if table is not None else "")
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
        for target in self.ours:
            chance = e.attacks.get(target.aimed_as, 0.0)
            if chance <= 0:
                continue
            fake = min(chance, e.fake_out.get(target.aimed_as, 0.0))
            if fake > 0:
                move = {"id": "fakeout", "base_power": 40, "type": "NORMAL", "category": "PHYSICAL", "spread": False, "priority": 3}
                found.append(("fakeout", foe, move, [(target, fake)], 3))
            if chance - fake > 0:
                move = self._best_move(foe, [target])
                if move:
                    found.append(("attack", foe, move, [(target, chance - fake)], self.priority(foe, move)))
        return found

    def best_attack(self, mon: Mon):
        """Turn 2: its best attack against the other side as it stands."""
        if mon.alive <= 0:
            return None
        enemies = [m for m in (self.theirs if mon.side == "us" else self.ours) if m.alive > 0]
        if not enemies:
            return None
        best, best_value = None, 0.0
        for move in self._attacks(mon):
            targets = enemies if move["spread"] else None
            for target in ([None] if targets else enemies):
                hit_on = targets or [target]
                value = sum(self._value(mon, t, move) for t in hit_on)
                if value > best_value:
                    best, best_value = ("attack", mon, move, [(t, 1.0) for t in hit_on], self.priority(mon, move)), value
        return best

    def _best_move(self, attacker: Mon, targets: list) -> dict | None:
        best, best_value = None, 0.0
        for move in self._attacks(attacker):
            value = sum(self._value(attacker, t, move) for t in targets)
            if value > best_value:
                best, best_value = move, value
        return best

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
        return value

    # -- resolving -------------------------------------------------------------------

    def speed(self, mon: Mon) -> float:
        base = self.calc.speed(mon.entry) or 0.0
        return base * (1 + self.board.tailwind[mon.side]) * mon.speed_mult * (1 - 0.5 * mon.paralyzed)

    def resolve(self, actions: list) -> None:
        reverse = self.board.trick_room > 0.5
        order = sorted(actions, key=lambda a: (-a[4], self.speed(a[1]) if reverse else -self.speed(a[1])))
        for kind, user, move, targets, _priority in order:
            if kind == "status":
                self._status(user, move["id"], targets)
            else:
                self._hit(kind, user, move, targets)

    def _hit(self, kind: str, user: Mon, move: dict, targets: list) -> None:
        acts = user.acts()
        if acts <= 0:
            return
        boost = self.board.helping.get(id(user), 0.0)
        for target, weight in targets:
            for mon, share in self._landing_spots(user, target, move):
                if mon.alive <= 0:
                    continue
                land = weight * share * (1.0 if move.get("bypass") else 1 - mon.protecting) * (1 - 0.5 * mon.switching)
                if land <= 0:
                    continue
                hit = self._damage(user, mon, move)
                if hit is None:
                    continue
                factor = 1 + 0.5 * boost
                expected, low, high = (hit["expected_pct"] * factor, hit["range_pct"][0] * factor, hit["range_pct"][1] * factor)
                chance = acts * land
                ko = _ko_chance(low, high, mon.hp)
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
                if kind == "fakeout":
                    mon.flinch += (1 - mon.flinch) * chance
                elif kind == "attack_slow":
                    mon.speed_mult *= 1 - chance / 3

    def _self_damage(self, user: Mon, target: Mon, move: dict, dealt: float, chance: float) -> None:
        """Recoil, Rocky Helmet and Life Orb: HP the attacker pays for the hit."""
        move_id = _id(move.get("id") or move.get("move_id"))
        cost = 0.0
        share = RECOIL.get(move_id, 0.0)
        if share and dealt > 0:
            # Damage is in the target's HP %; recoil is in the user's.
            cost += share * dealt * self._max_hp(target) / max(1.0, self._max_hp(user))
        listed = movelist.lookup(move_id) or {}
        if "contact" in (listed.get("flags") or []) and _id(self._item(target)) == "rockyhelmet":
            cost += chance * 100 / 6
        if _id(self._item(user)) == "lifeorb":
            cost += chance * 10
        if cost:
            # Faints from it only when the hit lands and the full cost covers its HP.
            if chance > 0 and cost / chance >= user.hp:
                user.alive *= 1 - chance
            user.hp = max(0.0, user.hp - cost)

    def _item(self, mon: Mon) -> str:
        return mon.entry.get("item") or self.calc._table.common_item(mon.entry.get("species") or mon.entry.get("name"))

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
        return spots

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
        hit = self.calc.damage(self._staged(user), self._staged(target), move, spread=bool(move.get("spread")))
        if not hit or hit.get("immune"):
            return None
        category = str(move.get("category") or "").upper()
        factor = user.boost.get("atk" if category == "PHYSICAL" else "spa", 1.0)
        if category == "PHYSICAL":
            factor *= 1 - 0.5 * user.burned
        factor *= 1 - (1 - SCREEN) * self.board.screens[target.side].get(category, 0.0)
        return {**hit, "expected_pct": hit["expected_pct"] * factor,
                "range_pct": [hit["range_pct"][0] * factor, hit["range_pct"][1] * factor]}

    def _value(self, user: Mon, target: Mon, move: dict) -> float:
        from .lookahead import KO_BONUS

        if target.alive <= 0:
            return 0.0
        hit = self._damage(user, target, move)
        if not hit:
            return 0.0
        return target.alive * (min(target.hp, hit["expected_pct"]) + KO_BONUS * _ko_chance(*hit["range_pct"], target.hp))

    def _status(self, user: Mon, move_id: str, targets: list) -> None:
        chance = user.acts()
        if chance <= 0:
            return
        side = user.side
        nxt = self.next_board
        if move_id == "tailwind":
            nxt.tailwind[side] = max(nxt.tailwind[side], chance)
        elif move_id == "trickroom":
            nxt.trick_room = self.board.trick_room * (1 - chance) + (1 - self.board.trick_room) * chance
        elif move_id in SCREENS:
            for category in SCREENS[move_id]:
                now = nxt.screens[side].get(category, 0.0)
                nxt.screens[side][category] = now + (1 - now) * chance
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
        for target, weight in targets:
            if target is None or target.alive <= 0:
                continue
            land = chance * weight * (1 - target.protecting)
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
        enemies = [m for m in (self.theirs if mon.side == "us" else self.ours) if m.alive > 0]
        worst = 0.0
        reverse = self.board.trick_room > 0.5
        for enemy in enemies:
            move = self._best_move(enemy, [mon])
            if not move:
                continue
            hit = self._damage(enemy, mon, move)
            faster = self.priority(enemy, move) > 0 or (
                self.speed(enemy) < self.speed(mon) if reverse else self.speed(enemy) > self.speed(mon))
            if hit and faster and _ko_chance(*hit["range_pct"], mon.hp) > 0:
                worst = max(worst, enemy.alive * enemy.acts() * _ko_chance(*hit["range_pct"], mon.hp))
        return DOOMED_SHARE * worst * mon.alive * (mon.hp + KO_BONUS)


def _ko_chance(low: float, high: float, hp: float) -> float:
    if hp <= 0:
        return 1.0
    if high <= hp:
        return 0.0
    if low >= hp or high <= low:
        return 1.0
    return (high - hp) / (high - low)
