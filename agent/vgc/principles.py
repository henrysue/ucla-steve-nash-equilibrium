"""Decision principles: what a line does right or wrong whatever the read says.

The read decides which foe does what; these decide whether our line is sound
first. Each check returns a flag the strategist sees beside the option, and
some cost value on the damage scale so the ranking already respects them:

1. Protect the plan. Never switch a Pokemon the plan cannot lose (the game
   plan's ``preserve``, or our speed-control setter while its effect is not
   up) into a predicted hit. Match 286bbc2e: Farigiraf, the Trick Room
   setter, switched into Close Combat, took 62% and never set Trick Room.
2. Do your job. A Prankster screen setter with no screens up, or a Trick
   Room setter on a slower side, that does something else is flagged.
   Match 286bbc2e: Grimmsnarl Thunder Waved turn 1 and died turn 2 with no
   screen ever set.
3. Stay robust. Both attacks into one foe that may Protect wastes the turn
   if it does; split them when the other foe is a target. Match 286bbc2e,
   turn 8: Flare Blitz and Flash Cannon both into a first-turn Iron Valiant,
   which Protected.
4. Mind stat drops. An attack at -2 or worse in the stat it uses is half a
   move, whatever lowered it: Draco Meteor (match 286bbc2e left Archaludon
   at -4 SpA) or Intimidate (Yuriko's Arcanine and Gyarados left Chien-Pao
   at -2 Atk, and its Sacred Sword did 15%). Switching out resets the drop,
   so it is no tempo loss.
5. Keep tempo. A healthy Pokemon does not switch out unless the switch-in
   takes clearly less of the hit coming; more so when it has a KO on the
   board, and most when it only came in last turn. Match e1f70879: Great
   Tusk switched out with Headlong Rush lined up on a 46% Archaludon; match
   60beb733: Archaludon -> Dragonite -> Archaludon gave Grimmsnarl two free
   turns to set Reflect, Thunder Wave and Light Screen.
6. Take the sure KO. A line that leaves alive a foe one of our attacks likely
   KOs this turn (and is not likely to Protect) pays for it: a KO now beats
   chip on a priority target (match ead979d8's sleep-user weight otherwise
   chipped a full Amoonguss over finishing a 15% Landorus).

Never raises: a principle that cannot be checked is simply not reported.
"""

from __future__ import annotations

import re

from . import movelist

PRESERVE_SHARE = 1.0  # a plan Pokemon switched into a hit: that damage counts again
PRESERVE_FROM = 25.0  # predicted damage (% HP) on the switch-in before it counts
MISSED_KO = 25.0  # leaving alive a foe we likely KO this turn
MISSED_KO_PROTECT = 0.5  # ...unless it is at least this likely to Protect
TEMPO_COST = 30.0  # switching a healthy Pokemon out of a KO it has on the board
TEMPO_SWITCH = 15.0  # switching a healthy Pokemon out at all: a turn handed to them
TEMPO_PING_PONG = 20.0  # on top: it only came in last turn
TEMPO_HEALTHY = 0.5  # HP fraction from which a Pokemon counts as healthy
TEMPO_RESISTS = 0.5  # unless the switch-in takes at most this share of the hit
WEAKENED_COST = 10.0  # attacking at -2 or worse in the stat the move uses
WEAKENED_AT = -2
ROLE_SKIPPED = 15.0  # a support not doing the one thing it was brought for
DOUBLE_INTO_PROTECT = 25.0  # x the foe's Protect chance, for both attacks into it
DOUBLE_FROM = 0.3  # Protect chance from which doubling into a foe is flagged
SPEED_CONTROL = frozenset({"trickroom", "tailwind"})
SCREEN_MOVES = frozenset({"reflect", "lightscreen", "auroraveil"})
# Moves that lower the user's attacking stat: id -> (stat, stages).
SELF_DROPS = {
    "dracometeor": ("spa", 2), "overheat": ("spa", 2), "leafstorm": ("spa", 2), "fleurcannon": ("spa", 2),
    "psychoboost": ("spa", 2), "makeitrain": ("spa", 1), "superpower": ("atk", 1),
}


def _id(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


class Judge:
    """Checks one turn's candidate lines against the principles.

    ``expected`` is foe id -> predict.FoeExpectation (the blended read);
    ``preserve`` the game plan's list; ``effects`` our side's and the field's
    effects now up (observed.field_state ids, plus the move tracker's).
    """

    def __init__(self, slots, observation: dict, calc, expected: dict, *, preserve=(), effects=(), just_in=()) -> None:
        self.slots = slots
        self.observation = observation
        self.calc = calc
        self.expected = expected or {}
        self.effects = {_id(e) for e in effects or ()}
        self.preserve = [_id(p) for p in preserve or () if p]
        self.just_in = {_id(s) for s in just_in or ()}  # ours that came in at the last decision

    # -- the one entry point ---------------------------------------------------------

    def judge(self, payload: dict) -> tuple[float, list[str]]:
        """(value to subtract, flags) for one checked payload."""
        cost, flags = 0.0, []
        for check in (self._plan_switches, self._tempo_switches, self._unused_roles, self._doubled_into_protect,
                      self._self_drops, self._missed_kos):
            try:
                more_cost, more = check(payload)
            except Exception:  # noqa: BLE001 - a principle we cannot check is not reported
                continue
            cost += more_cost
            flags.extend(more)
        return cost, flags

    # -- the principles ------------------------------------------------------------

    def _plan_switches(self, payload: dict) -> tuple[float, list[str]]:
        cost, flags = 0.0, []
        for slot in self.slots:
            choice = payload.get(f"slot_{slot.number}") or {}
            if choice.get("type") != "switch" or slot.force_switch:
                continue
            incoming = _id(choice.get("species"))
            why = self._plan_role(incoming)
            if not why:
                continue
            taken = self._damage_on_switch_in(slot, incoming)
            if taken >= PRESERVE_FROM:
                cost += PRESERVE_SHARE * taken
                flags.append(f"switches {incoming} ({why}) into ~{taken:.0f}% predicted damage")
        return cost, flags

    def _tempo_switches(self, payload: dict) -> tuple[float, list[str]]:
        cost, flags = 0.0, []
        for slot in self.slots:
            choice = payload.get(f"slot_{slot.number}") or {}
            active = _active(slot)
            if choice.get("type") != "switch" or not active:
                continue
            entry = self._entry(active)
            if _fraction(entry) < TEMPO_HEALTHY or _asleep(entry) or self._crippled(slot, entry):
                # A sleeper was not going to act (match ead979d8); an attacker at -2
                # resets its drops by leaving (Yuriko's Intimidates): no tempo lost.
                continue
            incoming = _id(choice.get("species"))
            staying = self._damage_on(entry, active)
            ping_pong = _id(active) in self.just_in
            # The switch-in takes the hit far better: a real pivot - but not twice
            # running, which leaves the slot idle two turns (match 60beb733).
            if not ping_pong and staying >= PRESERVE_FROM and \
                    self._damage_on_switch_in(slot, incoming) <= TEMPO_RESISTS * staying:
                continue
            ko = self._ko_on_board(slot)
            more = TEMPO_COST if ko else TEMPO_SWITCH
            why = f"switches out a healthy {active}" + (f" that has a KO on {ko}" if ko else "")
            if ping_pong:
                more += TEMPO_PING_PONG
                why += ", a turn after it came in"
            cost += more
            flags.append(why)
        return cost, flags

    def _ko_on_board(self, slot) -> str | None:
        """A foe one of this slot's attacks likely KOs this turn, or None."""
        found = self._kos(slot)
        return sorted(found)[0] if found else None

    def _kos(self, slot) -> set[str]:
        """Foes one of this slot's attacks likely KOs this turn."""
        from .actions import move_details

        attacker = self.calc.ours(slot.number)
        if slot.force_switch or _asleep(attacker or {}):
            return set()  # it would have to wake first
        found = set()
        for option in slot.options:
            if option.get("type") != "move":
                continue
            details = {**move_details(option, slot.number, self.observation), "id": option.get("move_id")}
            if not details.get("base_power"):
                continue
            for foe in getattr(self.calc, "_theirs", []):
                if _fraction(foe) <= 0:
                    continue
                hit = self.calc.damage(attacker, foe, details)
                if hit and hit.get("ko_this_turn_likely") and not hit.get("immune"):
                    found.add(_id(foe.get("species") or foe.get("name")))
        return found

    def _missed_kos(self, payload: dict) -> tuple[float, list[str]]:
        killable = set()
        for slot in self.slots:
            killable |= self._kos(slot)
        killable = {f for f in killable if (self.expected.get(f).protect if self.expected.get(f) else 0.0) < MISSED_KO_PROTECT}
        if not killable:
            return 0.0, []
        theirs = getattr(self.calc, "_theirs", [])
        aimed = set()
        for slot in self.slots:
            choice = payload.get(f"slot_{slot.number}") or {}
            move = movelist.lookup(choice.get("move_id")) if choice.get("type") == "move" else None
            if not move or not move["power"] or move["category"] == "STATUS":
                continue
            if move["target"] in ("allAdjacentFoes", "allAdjacent"):
                aimed |= {_id(f.get("species") or f.get("name")) for f in theirs}
            elif isinstance(choice.get("target"), int) and 0 < choice["target"] <= len(theirs):
                foe = theirs[choice["target"] - 1]
                aimed.add(_id(foe.get("species") or foe.get("name")))
        if killable & aimed:
            return 0.0, []
        return MISSED_KO, [f"leaves {sorted(killable)[0]} alive though we likely KO it this turn"]

    def _damage_on(self, entry: dict, name: str) -> float:
        """Predicted damage (% HP) on one of ours staying in."""
        total = 0.0
        for foe in getattr(self.calc, "_theirs", []):
            e = self.expected.get(_id(foe.get("species") or foe.get("name")))
            chance = e.attacks.get(_id(name), 0.0) if e else 0.0
            if chance:
                hit = self.calc.best_hit(foe, entry)
                total += chance * (hit["expected_pct"] if hit else 50.0)
        return total

    def _unused_roles(self, payload: dict) -> tuple[float, list[str]]:
        flags = []
        for slot in self.slots:
            active = _active(slot)
            if not active:
                continue
            used = _id((payload.get(f"slot_{slot.number}") or {}).get("move_id"))
            moves = {_id(o.get("move_id")) for o in slot.options if o.get("type") == "move"}
            screen = self._useful_screen((moves & SCREEN_MOVES) - self.effects)
            entry = self._entry(active)
            if screen and self._has_ability(entry, "prankster") and not self.effects & SCREEN_MOVES and used not in SCREEN_MOVES:
                flags.append(f"{active} skips its job: Prankster {screen} goes up before their attacks")
            if "trickroom" in moves and "trickroom" not in self.effects and used != "trickroom" and self._we_are_slower():
                flags.append(f"{active} skips its job: Trick Room would make our slower side move first")
        return ROLE_SKIPPED * len(flags), flags

    def _doubled_into_protect(self, payload: dict) -> tuple[float, list[str]]:
        aimed: dict[int, int] = {}
        for slot in self.slots:
            choice = payload.get(f"slot_{slot.number}") or {}
            move = movelist.lookup(choice.get("move_id")) if choice.get("type") == "move" else None
            target = choice.get("target")
            if move and move["power"] and move["category"] != "STATUS" and isinstance(target, int) and target > 0 \
                    and move["target"] not in ("allAdjacentFoes", "allAdjacent"):
                aimed[target] = aimed.get(target, 0) + 1
        foes = [f for f in getattr(self.calc, "_theirs", []) if _fraction(f) > 0]
        for target, count in aimed.items():
            if count < 2 or len(foes) < 2 or not 0 < target <= len(self.calc._theirs):
                continue
            foe = self.calc._theirs[target - 1]
            name = _id(foe.get("species") or foe.get("name"))
            e = self.expected.get(name)
            chance = e.protect if e else 0.0
            if chance >= DOUBLE_FROM:
                return DOUBLE_INTO_PROTECT * chance, [f"both attacks into {name}, which may Protect ({chance:.0%})"]
        return 0.0, []

    def _self_drops(self, payload: dict) -> tuple[float, list[str]]:
        cost, flags = 0.0, []
        for slot in self.slots:
            choice = payload.get(f"slot_{slot.number}") or {}
            active = _active(slot)
            if choice.get("type") != "move" or not active:
                continue
            stat = _attacking_stat(choice.get("move_id"))
            stage = _stage_of(self._entry(active), stat) if stat else 0
            if stat and stage <= WEAKENED_AT:
                cost += WEAKENED_COST
                flags.append(f"{active}'s {choice.get('move_id')} at {stage} {stat.upper()}: much weaker than shown")
        return cost, flags

    def _crippled(self, slot, entry: dict) -> bool:
        """Every attack this slot has is at -2 or worse in the stat it uses."""
        stats = {_attacking_stat(o.get("move_id")) for o in slot.options if o.get("type") == "move"} - {None}
        return bool(stats) and all(_stage_of(entry, s) <= WEAKENED_AT for s in stats)

    # -- facts -----------------------------------------------------------------------

    def _plan_role(self, species: str) -> str | None:
        if any(p == species or p.startswith(species) for p in self.preserve):
            return "the game plan preserves it"
        moves = {_id(m) for m in (self.calc._table.common_moves(species) or [])}
        setter = sorted((moves & SPEED_CONTROL) - self.effects)
        if setter:
            return f"our {setter[0]} setter"
        return None

    def _damage_on_switch_in(self, slot, incoming: str) -> float:
        leaving = _id(_active(slot))
        hp = 100 * _bench_fraction(self.observation, incoming)
        total = 0.0
        for foe in getattr(self.calc, "_theirs", []):
            e = self.expected.get(_id(foe.get("species") or foe.get("name")))
            chance = e.attacks.get(leaving, 0.0) if e else 0.0
            if chance:
                hit = self.calc.best_hit(foe, {"species": incoming})
                total += chance * (hit["expected_pct"] if hit else 50.0)
        return min(hp, total)

    def _useful_screen(self, screens: set[str]) -> str | None:
        """The screen that cuts their attacks: Aurora Veil, else Reflect against a
        mostly physical side, Light Screen against a mostly special one."""
        if not screens:
            return None
        if "auroraveil" in screens:
            return "auroraveil"
        physical = special = 0
        for foe in getattr(self.calc, "_theirs", []):
            if _fraction(foe) <= 0:
                continue
            for move in self.calc.likely_attacks(foe) or []:
                category = str(move.get("category") or "").upper()
                physical += category == "PHYSICAL"
                special += category == "SPECIAL"
        wanted = "reflect" if physical >= special else "lightscreen"
        return wanted if wanted in screens else None

    def _we_are_slower(self) -> bool:
        ours = [self.calc.speed(e) for e in getattr(self.calc, "_ours", []) if _fraction(e) > 0]
        theirs = [self.calc.speed(e) for e in getattr(self.calc, "_theirs", []) if _fraction(e) > 0]
        ours, theirs = [s for s in ours if s], [s for s in theirs if s]
        return bool(ours and theirs) and sum(ours) / len(ours) < sum(theirs) / len(theirs)

    def _entry(self, species: str) -> dict:
        for entry in getattr(self.calc, "_ours", []):
            if _id(entry.get("species") or entry.get("name")) == _id(species):
                return entry
        return {"species": species}

    def _has_ability(self, entry: dict, ability: str) -> bool:
        return {_id(a) for a in self.calc._abilities(entry) or []} == {ability}


def _active(slot) -> str:
    active = slot.active
    if isinstance(active, dict):
        active = active.get("species") or active.get("name")
    return "" if slot.force_switch else str(active or "")


def _fraction(entry: dict) -> float:
    from .lookahead import _fraction as fraction

    return fraction(entry)


def _attacking_stat(move_id: object) -> str | None:
    """The stat a damaging move hits with: atk / spa (Body Press: def), or None."""
    move = movelist.lookup(move_id)
    if not move or not move["power"] or move["category"] == "STATUS" or _id(move_id) == "foulplay":
        return None
    if _id(move_id) == "bodypress":
        return "def"
    return "atk" if move["category"] == "PHYSICAL" else "spa"


def _stage_of(entry: dict, stat: str | None) -> int:
    stage = (entry.get("boosts") or {}).get(stat) if stat else 0
    return stage if isinstance(stage, int) and not isinstance(stage, bool) else 0


def _asleep(entry: dict) -> bool:
    from .lookahead import _status

    return _status(entry.get("status")) == "slp"


def _bench_fraction(observation: dict, species: str) -> float:
    from .race import _bench_hp

    return _bench_hp(observation, species)
