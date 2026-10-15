"""One-step lookahead: estimated damage, turns to KO, and the best line.

PokéChamp (Karten, Nguyen and Jin, arXiv 2503.04094) gives its LLM a
one-step-lookahead damage table ("turns to KO" per move) and adds a
code-computed candidate to the LLM's own proposals. This module is that
tool for VGC doubles: the damage facts on every option, the opponents'
threat to us, and ``best_line``: our best legal joint action against the
predicted opponent (predict.py), which is the move we play.

The calculator is the Gen 9 formula at level 50, in expected value (as
PokéChamp does, rather than enumerating rolls):

    floor(floor(22 * BP * A / D) / 50) + 2, then x spread 0.75, x roll
    0.85-1.00, x STAB 1.5, x type, x burn 0.5, x item

Every draft card is an exact set (cards.py), so for a Pokemon whose card we
have seen its stats come from the card's EVs and nature, and its moves, item
and ability are the card's. Otherwise hidden stats are estimated: an
attacker has 252 EVs in its attacking stat; a defender's expected bulk is a
middling spread, with a range from bulky (252 HP / 252+ Def) to frail
(0 / 0). Real stats, boosts, items and status are used whenever the
observation carries them. Abilities,
weather, terrain, Tera and crits are left to the model and to
``abilities.py``'s warnings; this is a prior, labelled as one.
"""

from __future__ import annotations

import math
import re

from .abilities import MULTI_HIT
from .cards import nature_factor
from .typechart import effectiveness

LEVEL = 50
ROLL_LOW, ROLL_MID, ROLL_HIGH = 0.85, 0.925, 1.0
GENERIC_STAB_POWER = 90  # the opponent's unseen attack, per STAB type
KO_CAP = 9
# Expected hits: 2-5 hit moves average 3.1; fixed-count moves are exact.
_TWO = {"doublehit", "dualwingbeat", "doublekick", "twinbeam", "dragondarts", "bonemerang", "doubleironbash", "tachyoncutter"}
_THREE = {"tripleaxel", "surgingstrikes", "triplekick"}


def expected_hits(move_id: str) -> float:
    move_id = _id(move_id)
    if move_id in _TWO:
        return 2.0
    if move_id in _THREE:
        return 3.0
    if move_id == "populationbomb":
        return 6.0
    return 3.1 if move_id in MULTI_HIT else 1.0

_OFFENSE = {"PHYSICAL": "atk", "SPECIAL": "spa"}
HITS_PHYSICAL_DEFENSE = frozenset({"psyshock", "psystrike", "secretsword"})  # special moves against Defense
# Halve one super-effective hit of their type (Chilan: any Normal hit). Consumption is not tracked.
RESIST_BERRIES = {
    "chopleberry": "FIGHTING", "occaberry": "FIRE", "passhoberry": "WATER", "wacanberry": "ELECTRIC",
    "rindoberry": "GRASS", "yacheberry": "ICE", "kebiaberry": "POISON", "shucaberry": "GROUND", "cobaberry": "FLYING",
    "payapaberry": "PSYCHIC", "tangaberry": "BUG", "chartiberry": "ROCK", "kasibberry": "GHOST", "habanberry": "DRAGON",
    "colburberry": "DARK", "babiriberry": "STEEL", "roseliberry": "FAIRY", "chilanberry": "NORMAL",
}
_DEFENSE = {"PHYSICAL": "def", "SPECIAL": "spd"}
# (HP EVs, defensive EVs, nature) for the defender estimates.
_BULKY, _MIDDLE, _FRAIL = (252, 252, 1.1), (132, 60, 1.0), (0, 0, 1.0)


def _id(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _stat(base: int, ev: int, nature: float = 1.0) -> int:
    return math.floor((math.floor((2 * base + 31 + ev // 4) * LEVEL / 100) + 5) * nature)


def _hp(base: int, ev: int) -> int:
    return math.floor((2 * base + 31 + ev // 4) * LEVEL / 100) + LEVEL + 10


def _stage(n: int) -> float:
    n = max(-6, min(6, n))
    return (2 + n) / 2 if n >= 0 else 2 / (2 - n)


def held_item(entry: dict, table) -> str:
    """The item it holds now: what the server shows, "" once it is gone
    (an Air Balloon popped, a berry eaten, Knock Off), else the draft card's
    (or the table's usual) item while still unrevealed."""
    if "item" in entry and isinstance(entry.get("item"), str) and not entry["item"].strip():
        return ""
    item = entry.get("item")
    if not item or _id(item) in ("unknownitem", "unknown"):
        return table.common_item(entry.get("species") or entry.get("name")) or ""
    return str(item)


def floats(entry: dict, table) -> bool:
    """Immune to Ground: Levitate (when it is its only ability) or an Air Balloon still up."""
    ability = entry.get("ability")
    abilities = [ability] if isinstance(ability, str) and ability.strip() else table.abilities(entry.get("species") or entry.get("name"))
    return abilities == ["Levitate"] or _id(held_item(entry, table)) == "airballoon"


def _item_power(item: str, category: str, multiplier: float) -> float:
    item = _id(item)
    if item == "choiceband" and category == "PHYSICAL":
        return 1.5
    if item == "choicespecs" and category == "SPECIAL":
        return 1.5
    if item == "lifeorb":
        return 1.3
    if item == "expertbelt" and multiplier > 1:
        return 1.2
    return 1.0


def _base_damage(power: float, attack: float, defense: float) -> float:
    return math.floor(math.floor((2 * LEVEL / 5 + 2) * power * attack / defense) / 50) + 2


class Calculator:
    """Damage for one decision, from the observation and the table.

    When both Pokemon are known draft cards (EVs, nature, item, ability),
    damage is Showdown's exactly (showdown.py: all 16 rolls, verified against
    Showdown's own calculator); otherwise it is the estimate below.
    """

    def __init__(self, observation: dict, table) -> None:
        self._table = table
        self._ours = _by_species(observation.get("active_pokemon"))
        self._theirs = _by_species(observation.get("opponent_active_pokemon"))
        self._field = _engine_field(observation, self._ours + self._theirs, table)
        self._cache: dict = {}

    def damage(self, attacker: dict, defender: dict, move: dict, *, spread: bool = False) -> dict | None:
        """Expected and ranged damage as % of the defender's max HP, plus
        turns to KO from its current HP; ``None`` when data is missing.
        Exact (Showdown's rolls) when both sets are known, else estimated."""
        exact = self._exact(attacker, defender, move, spread)
        if exact is not None:
            return exact
        return self._estimate(attacker, defender, move, spread=spread)

    def _battler(self, entry: dict):
        """A showdown.Battler for a known set, or ``None``."""
        from . import showdown

        card = self._card(entry)
        name = entry.get("species") or entry.get("name")
        if not card or not isinstance(card.get("evs"), dict) or not card.get("nature") or not showdown.species(card["species"]):
            return None
        ability = entry.get("ability")
        ability = ability if isinstance(ability, str) and ability.strip() and _id(ability) not in ("unknown", "unknownability") \
            else card.get("ability", "")
        types = entry.get("types") if isinstance(entry.get("types"), list) and entry.get("types") else None
        boosts = {k: v for k, v in (entry.get("boosts") or {}).items() if isinstance(v, int) and not isinstance(v, bool)}
        return showdown.Battler(card["species"], held_item(entry, self._table), ability, card.get("nature", ""),
                                card.get("evs") or {}, card.get("ivs") or {}, boosts=boosts,
                                status=_status(entry.get("status")), hp_fraction=_fraction(entry), types=types)

    def _partner(self, entry: dict) -> dict | None:
        """Who stands beside this Pokemon on its side of the field."""
        name = _id(entry.get("species") or entry.get("name"))
        for side in (self._ours, self._theirs):
            names = [_id(e.get("species") or e.get("name")) for e in side]
            if name in names:
                return next((e for e in side if _id(e.get("species") or e.get("name")) != name and _fraction(e) > 0), None)
        return None

    def _partner_field(self, attacker: dict, defender: dict):
        """The field, plus the partners' Battery / Power Spot (attacker) and Friend Guard (defender)."""
        from dataclasses import replace

        flags = {}
        ally = self._partner(attacker)
        if ally:
            names = {_id(a) for a in self._abilities(ally)}
            flags["battery"] = names == {"battery"}
            flags["power_spot"] = names == {"powerspot"}
        guard = self._partner(defender)
        if guard:
            flags["friend_guard"] = {_id(a) for a in self._abilities(guard)} == {"friendguard"}
        return replace(self._field, **flags) if any(flags.values()) else self._field

    def _exact(self, attacker: dict, defender: dict, move: dict, spread: bool) -> dict | None:
        from . import showdown

        move_id = _id(move.get("id") or move.get("move_id") or move.get("name"))
        if not move_id or not showdown.Move.get(move_id):
            return None
        key = (_entry_key(attacker), _entry_key(defender), move_id, spread)
        if key in self._cache:
            return self._cache[key]
        a, d = self._battler(attacker), self._battler(defender)
        if a is None or d is None:
            return None
        rolls = _expected_rolls(a, d, move_id, self._partner_field(attacker, defender), spread)
        if not rolls:
            result = {"expected_pct": 0.0, "range_pct": [0.0, 0.0], "rolls_pct": [0.0] * 16, "turns_to_ko": None,
                      "ko_this_turn_likely": False, "immune": True}
        else:
            pct = [100 * r / d.max_hp for r in rolls]
            mid = sum(pct) / len(pct)
            remaining = 100 * _fraction(defender)
            result = {
                "expected_pct": round(mid, 1),
                "range_pct": [round(min(pct), 1), round(max(pct), 1)],
                "rolls_pct": [round(x, 2) for x in pct],
                "turns_to_ko": KO_CAP if mid <= 0 else min(KO_CAP, math.ceil(remaining / mid)),
                "ko_this_turn_likely": mid >= remaining,
                "ko_chance": sum(x >= remaining for x in pct) / len(pct),
                "exact": True,
            }
        self._cache[key] = result
        return result

    def _estimate(self, attacker: dict, defender: dict, move: dict, *, spread: bool = False) -> dict | None:
        """The estimate for a set we do not know: 252 EVs to attack, a range of bulk."""
        category = str(move.get("category") or "").upper()
        power = move.get("base_power") or 0
        if _id(move.get("id") or move.get("move_id") or move.get("name")) in HP_SCALING and power:
            power = max(1, int(150 * _fraction(attacker)))  # Eruption / Water Spout: power follows HP
        move_type = str(move.get("type") or "").upper()
        a_name, d_name = attacker.get("species") or attacker.get("name"), defender.get("species") or defender.get("name")
        a_base, d_base = self.base_stats(attacker), self.base_stats(defender)
        if category not in _OFFENSE or power <= 0 or not move_type or not a_base or not d_base:
            return None
        a_types = _types(attacker, self._table)
        d_types = _types(defender, self._table)
        multiplier = effectiveness(move_type, d_types) if d_types else 1.0
        if move_type == "GROUND" and floats(defender, self._table):
            multiplier = 0.0  # Air Balloon / Levitate (match 0af0f5a5: Earthquake into Gholdengo's balloon)
        if multiplier == 0:
            return {"expected_pct": 0.0, "range_pct": [0.0, 0.0], "turns_to_ko": None, "ko_this_turn_likely": False, "immune": True}

        off_key, def_key = _OFFENSE[category], _DEFENSE[category]
        move_id = _id(move.get("id") or move.get("move_id"))
        def_key = "def" if move_id in HITS_PHYSICAL_DEFENSE else def_key
        if move_id == "bodypress":  # the user's Defense (match 8e3a639c: Archaludon's was read as Attack, half the damage)
            attack = (self._stat(attacker, a_base, "def") or _stat(a_base["def"], 252)) * _stage(_boost(attacker, "def"))
        elif move_id == "foulplay":  # the target's Attack
            attack = (self._stat(defender, d_base, "atk") or _stat(d_base["atk"], 252)) * _stage(_boost(defender, "atk"))
        else:
            attack = (self._stat(attacker, a_base, off_key) or _stat(a_base[off_key], 252)) * _stage(_boost(attacker, off_key))
        modifier = 1.5 if move_type in a_types else 1.0
        modifier *= multiplier * (0.75 if spread else 1.0)
        if category == "PHYSICAL" and _id(attacker.get("status")) in ("brn", "burn"):
            modifier *= 0.5
        modifier *= _item_power(held_item(attacker, self._table), category, multiplier)
        modifier *= _ruin(self._abilities(attacker), self._abilities(defender), category)
        modifier *= expected_hits(move.get("id") or move.get("move_id") or "")

        d_item = _id(held_item(defender, self._table))
        if RESIST_BERRIES.get(d_item) == move_type and (multiplier > 1 or move_type == "NORMAL"):
            modifier *= 0.5
        stage = _stage(_boost(defender, def_key))
        vest = 1.5 if d_item == "assaultvest" and def_key == "spd" else 1.0
        vest *= 1.5 if d_item == "eviolite" else 1.0  # both defences (match ccb570bb: Porygon2 read as twice as frail)
        exact_def, exact_hp = self._stat(defender, d_base, def_key), self._stat(defender, d_base, "hp")
        if exact_def and exact_hp:
            # Known bulk: only the damage roll varies.
            pct = 100 * _base_damage(power, attack, exact_def * stage * vest) * modifier / exact_hp
            low, mid, high = pct * ROLL_LOW, pct * ROLL_MID, pct * ROLL_HIGH
        else:
            percents = []
            for hp_ev, def_ev, nature in (_BULKY, _MIDDLE, _FRAIL):
                defense = _stat(d_base[def_key], def_ev, nature) * stage * vest
                hp = _hp(d_base["hp"], hp_ev)
                base = _base_damage(power, attack, defense) * modifier
                percents.append(100 * base / hp)
            low, mid, high = percents[0] * ROLL_LOW, percents[1] * ROLL_MID, percents[2] * ROLL_HIGH
        remaining = 100 * _fraction(defender)
        turns = KO_CAP if mid <= 0 else min(KO_CAP, math.ceil(remaining / mid))
        return {
            "expected_pct": round(mid, 1),
            "range_pct": [round(low, 1), round(high, 1)],
            "turns_to_ko": turns,
            "ko_this_turn_likely": mid >= remaining,
        }

    def base_stats(self, entry: dict) -> dict | None:
        """The table's base stats, else the observation's own ``base_stats``
        (the server sends them; a species the table lacks still gets numbers)."""
        found = self._table.base_stats(entry.get("species") or entry.get("name"))
        if found:
            return found
        sent = entry.get("base_stats")
        if not isinstance(sent, dict):
            return None
        stats = {("spe" if k == "speed" else k): v for k, v in sent.items()}
        keys = ("hp", "atk", "def", "spa", "spd", "spe")
        return {k: int(stats[k]) for k in keys} if all(isinstance(stats.get(k), (int, float)) for k in keys) else None

    def _abilities(self, entry: dict) -> list[str]:
        """A revealed ability, else the card's, else the table's options."""
        ability = entry.get("ability")
        if isinstance(ability, str) and ability.strip():
            return [ability]
        return self._table.abilities(entry.get("species") or entry.get("name"))

    def _card(self, entry: dict) -> dict | None:
        card_set = getattr(self._table, "card_set", None)
        return card_set(entry.get("species") or entry.get("name")) if card_set else None

    def _stat(self, entry: dict, base: dict, key: str) -> int | None:
        """A real stat: from the observation, else from the draft card's EVs
        and nature; ``None`` when neither is known."""
        # Not max_hp: the server may report a foe's as a percentage.
        real = entry.get("stats") if isinstance(entry.get("stats"), dict) else {}
        if real.get(key):
            return real[key]
        card = self._card(entry)
        if not card:
            return None
        evs = card.get("evs") if isinstance(card.get("evs"), dict) else {}
        ev = evs.get(key) or 0
        if key == "hp":
            return _hp(base["hp"], ev)
        return _stat(base[key], ev, nature_factor(card.get("nature"), key))

    def speed(self, entry: dict) -> float | None:
        """Effective Speed: the real or card stat (else a 252-EV estimate,
        which orders Pokemon the same as base Speed), x boosts, x Choice Scarf."""
        base = self.base_stats(entry)
        if not base:
            return None
        value = self._stat(entry, base, "spe") or _stat(base["spe"], 252)
        item = held_item(entry, self._table)
        scarf = 1.5 if _id(item) == "choicescarf" else 1.0
        return value * _stage(_boost(entry, "spe")) * scarf * self._quark_speed(entry)

    def _quark_speed(self, entry: dict) -> float:
        """1.5 when Protosynthesis / Quark Drive is active (Booster Energy, sun,
        Electric Terrain) and Speed is the stat it boosts. Yuriko's Booster
        Energy Iron Valiant moved at ~276, not 184, and outsped Chien-Pao."""
        from . import showdown

        try:
            battler = self._battler(entry)
            if battler is not None and showdown.qp_active(battler, self._field) and showdown.qp_stat(battler) == "spe":
                return 1.5
        except Exception:  # noqa: BLE001 - a missing fact leaves Speed as it is
            pass
        return 1.0

    def ours(self, slot_number: int) -> dict:
        return self._ours[slot_number] if slot_number < len(self._ours) else {}

    def foe(self, species: str) -> dict:
        return next((f for f in self._theirs if _id(f.get("species") or f.get("name")) == _id(species)), {})

    def likely_attacks(self, foe: dict) -> list[dict]:
        """The attacks a foe has: its draft card's four moves (exact) plus any
        revealed; without a card, revealed moves, else its common set from
        the table, else - no data at all - a 90-power STAB attack per type
        from its stronger attacking stat. Match e4546428: the generic
        stand-in had Chi-Yu barely scratching Urshifu; its real Overheat
        with Beads of Ruin KO'd it."""
        from . import movelist

        name = foe.get("species") or foe.get("name")
        revealed = foe.get("revealed_moves") or foe.get("known_moves") or foe.get("moves") or []
        names = [m.get("id") or m.get("name") if isinstance(m, dict) else m for m in revealed if m]
        card = self._card(foe)
        names = list(dict.fromkeys(names + list(card["moves"]))) if card else names or self._table.common_moves(name)
        found = []
        for move_name in names:
            move = movelist.lookup(move_name)
            if move and move["power"] and move["category"] in ("PHYSICAL", "SPECIAL"):
                found.append({
                    "id": _id(move_name), "name": move["name"], "type": move["type"], "base_power": move["power"],
                    "category": move["category"], "spread": move["target"] in ("allAdjacentFoes", "allAdjacent"),
                    "priority": move["priority"] or 0,
                })
        if found:
            return found
        base = self.base_stats(foe)
        if not base:
            return []
        category = "PHYSICAL" if base["atk"] >= base["spa"] else "SPECIAL"
        return [
            {"id": "", "name": f"{t.title()} STAB", "type": t, "base_power": GENERIC_STAB_POWER, "category": category, "spread": False}
            for t in _types(foe, self._table)
        ]

    def best_hit(self, foe: dict, target: dict) -> dict | None:
        """The foe's hardest likely hit on ``target``, with the move named."""
        best = None
        for move in self.likely_attacks(foe):
            hit = self.damage(foe, target, move, spread=move["spread"])
            if hit and (best is None or hit["expected_pct"] > best["expected_pct"]):
                best = {**hit, "move": move["name"], "type": move["type"].title(), "spread": move["spread"],
                        "priority": move.get("priority", 0)}
        return best

    def incoming(self, species: str) -> dict | None:
        """The hardest likely hit a Pokemon would take on entry."""
        worst = None
        for foe in self._theirs:
            hit = self.best_hit(foe, {"species": species})
            if hit and (worst is None or hit["expected_pct"] > worst["expected_pct"]):
                worst = {"from": foe.get("species") or foe.get("name"), "move": hit["move"],
                         "expected_pct": hit["expected_pct"], "range_pct": hit["range_pct"]}
        return worst

    def threats(self) -> dict:
        """Each foe's hardest likely hit on each of our actives."""
        found = {}
        for foe in self._theirs:
            row = {}
            for ally in self._ours:
                hit = self.best_hit(foe, ally)
                if hit:
                    row[ally.get("species") or ally.get("name")] = hit
            if row:
                found[foe.get("species") or foe.get("name")] = row
        return found


# Values on the damage scale (% of a target's HP), so every option compares directly.
FAKE_OUT_VALUE = 30.0  # the flinch itself, on top of the damage it prevents
PROTECT_PROOF_VALUE = 150.0  # scaled by the chance every foe Protects: a free setup turn
REDIRECTION_VALUE = 25.0  # at most: the damage it pulls off our partner
# A sleeping foe wakes (and acts) about one turn in three; match e1f70879
# predicted two sleeping foes to attack and Rage Powdered them.
SLEEP_AWAKE = 1 / 3
SPEED_CONTROL_VALUE = 40.0
OTHER_STATUS_VALUE = 10.0
SWITCH_VALUE = 1.0
SWITCH_SHARE = 0.6  # a switch is worth this share of the damage it moves onto a safer Pokemon
# A KO removes a whole Pokemon, not just its last few HP: match ec1752bf
# Protected three times with Talonflame instead of Brave Bird into a 9% Hatterene.
KO_BONUS = 50.0
# Protecting from a KO only delays it a turn while the threat stays: a
# quarter of the bonus. Every logged game overused Protect (double Protects,
# a lone Talonflame protecting three times), so this errs toward acting.
DELAYED_KO_SHARE = 0.25
# A Protect while our other slot does nothing (a lone active, or both slots
# protecting) changes nothing: next turn the same hits come at the same HP.
# Match e4546428: a lone Tornadus alternated Protect and Bleakwind Storm into
# two full-HP foes; ec1752bf: two double Protects in a row. Field timers it
# could stall out (their Tailwind, Trick Room) are not in the observation.
STALL_SHARE = 0.25
# Their win condition is worth more than its raw damage. Match 4e60a5a2: a
# full-HP sun Torkoal's Eruption took 83%, 54% and 93% off ours and it never
# took a hit: every point off an Eruption user also cuts its next Eruption,
# and a weather setter keeps its partners' boost up until it falls.
WIN_CONDITION_WEIGHT = 1.5
HP_SCALING = frozenset({"eruption", "waterspout", "dragonenergy"})
HP_SCALING_FROM = 0.5  # below half HP its Eruption is already weak
# Match 666b209d: a Specs Chi-Yu, named their win condition in our own game
# plan, ranked level with a Rillaboom for Close Combat.
CHOICE_NUKES = frozenset({"choicespecs", "choiceband"})
# Match 60beb733: Grimmsnarl's Prankster Reflect, Thunder Wave and Light
# Screen went up while we hit Iron Hands; both screens then halved our hits.
SCREEN_MOVES = frozenset({"reflect", "lightscreen", "auroraveil"})
# Match ead979d8: Amoonguss's Spore took Iron Hands and Dondozo out of the game.
SLEEP_MOVES = frozenset({"spore", "sleeppowder", "hypnosis", "lovelykiss", "sing", "darkvoid", "grasswhistle", "yawn"})
WEATHER_SETTERS = frozenset({"drought", "drizzle", "sandstream", "snowwarning", "orichalcumpulse",
                             "desolateland", "primordialsea"})
CANDIDATES_PER_SLOT = 6
RACE_CANDIDATES = 24  # legal pairs (best one-turn values first) the two-turn race re-ranks
# Moves Protect cannot stop: they target our own side or the field.
PROTECT_PROOF = frozenset({
    "tailwind", "trickroom", "reflect", "lightscreen", "auroraveil", "swordsdance", "nastyplot", "calmmind",
    "dragondance", "quiverdance", "bulkup", "irondefense", "shellsmash", "bellydrum", "growth", "workup",
    "victorydance", "tidyup", "shiftgear", "agility", "rockpolish", "coaching", "howl", "substitute",
    "electricterrain", "grassyterrain", "psychicterrain", "mistyterrain", "sunnyday", "raindance", "snowscape",
    "sandstorm", "wish", "lifedew",
})
REDIRECTION = frozenset({"followme", "ragepowder", "spotlight"})


def best_line(turn_rules, observation: dict, calc: "Calculator", expected: dict | None = None) -> dict | None:
    """Step 3: our best legal joint action against the predicted opponent."""
    ranked = ranked_lines(turn_rules, observation, calc, expected, limit=1)
    return ranked[0][1] if ranked else None


def ranked_lines(turn_rules, observation: dict, calc: "Calculator", expected: dict | None = None, *, limit: int = 5,
                 bench: list[str] | None = None, cost=None):
    """Legal joint actions, best first, scored against ``expected``.

    ``expected`` is foe species id -> predict.FoeExpectation. Each slot's
    options are scored on the damage scale: an attack is worth its damage
    (a likely KO counts as the target's remaining HP and half again) times
    the chance the target does not Protect, and half as much into a likely
    switch; Fake Out also counts the damage it prevents; Protect counts the
    damage it blocks; a switch counts the damage it moves onto a safer
    Pokemon; a Protect-proof setup move counts when every foe is predicted
    to Protect. Pairs are tried best first and kept only if they pass
    ``turn_rules.check``. On a turn with moves, the best RACE_CANDIDATES of
    those are re-ranked by race.py's two-turn damage race, and that is the
    value returned. ``cost`` (payload -> value), when given, is subtracted
    before the cut: principles.Judge's price for breaking a decision
    principle. Returns [(value, answer)] in the model's row format.
    """
    from . import movelist

    context = _Context(calc, expected or {})
    ranked = [
        _slot_choices(slot, turn_rules.per_slot[slot.number], observation, context, movelist)
        for slot in turn_rules.slots
    ]
    pairs = sorted(
        ((_pair_value(turn_rules.slots, a, b, movelist), a[1], b[1]) for a in ranked[0] for b in ranked[1]),
        key=lambda item: -item[0],
    )
    racing = any(o.get("type") == "move" for slot in turn_rules.slots for o in slot.options)
    found = []
    for value, first, second in pairs:
        answer = {"slot_0": first, "slot_1": second}
        if turn_rules.legal(answer):
            found.append((round(value, 1), answer))
            if len(found) >= (RACE_CANDIDATES if racing else limit):
                break
    if racing:
        from . import race

        try:
            found = sorted(
                ((round(race.score([a["slot_0"], a["slot_1"]], turn_rules.slots, turn_rules.per_slot,
                                   observation, calc, context.expected, bench), 1), a) for _, a in found),
                key=lambda item: -item[0],
            )
        except Exception:  # noqa: BLE001 - the one-turn ranking still stands
            pass
    if cost is not None:
        try:
            found = sorted(((round(v - cost(turn_rules.check(a)), 1), a) for v, a in found), key=lambda item: -item[0])
        except Exception:  # noqa: BLE001 - the unpriced ranking still stands
            pass
    return found[:limit]


def _pair_value(slots, first, second, movelist) -> float:
    """Both slots' values, with Protect scaled by STALL_SHARE when neither
    slot does anything else this turn."""
    kinds = [_choice_kind(slot, choice[1], movelist) for slot, choice in zip(slots, (first, second))]
    if "protect" in kinds and "act" not in kinds:
        return sum(value * (STALL_SHARE if kind == "protect" else 1.0) for (value, _), kind in zip((first, second), kinds))
    return first[0] + second[0]


def _choice_kind(slot, choice: dict, movelist) -> str:
    """'protect', 'idle' (pass, or no option), or 'act'."""
    index = choice.get("option")
    option = slot.options[index] if isinstance(index, int) and 0 <= index < len(slot.options) else {}
    if option.get("type") == "move":
        return "protect" if _id(option.get("move_id")) in movelist.PROTECTING else "act"
    return "act" if option.get("type") == "switch" else "idle"


class _Context:
    """The prediction, and the damage figures every option is scored with."""

    def __init__(self, calc, expected: dict) -> None:
        self.calc = calc
        self.expected = asleep(calc, expected)
        self.threats = _threats_by_id(calc)
        self.win = win_conditions(calc)

    def foe_id(self, foe: dict) -> str:
        return _id(foe.get("species") or foe.get("name"))

    def protects(self, foe_id: str) -> float:
        e = self.expected.get(foe_id)
        return e.protect if e else 0.0

    def switches(self, foe_id: str) -> float:
        e = self.expected.get(foe_id)
        return e.switch if e else 0.0

    def incoming(self, ours: dict) -> float:
        """Damage the prediction sends at one of ours this turn (HP scale)."""
        name = _id(ours.get("species") or ours.get("name"))
        remaining = 100 * _fraction(ours)
        total = 0.0
        for foe_id, e in self.expected.items():
            chance = e.attacks.get(name, 0.0)
            hit = self.threats.get(foe_id, {}).get(name)
            if chance and hit:
                saved = remaining + KO_BONUS * DELAYED_KO_SHARE if hit["ko_this_turn_likely"] else min(remaining, hit["expected_pct"])
                total += chance * saved
        return total

    def incoming_on_switch(self, active: dict, species: str) -> float:
        """The same predicted attacks, landing on a Pokemon switching in."""
        name = _id(active.get("species") or active.get("name"))
        total = 0.0
        for foe_id, e in self.expected.items():
            chance = e.attacks.get(name, 0.0)
            foe = next((f for f in self.calc._theirs if _id(f.get("species") or f.get("name")) == foe_id), None)
            if chance and foe:
                hit = self.calc.best_hit(foe, {"species": species})
                total += chance * (hit["expected_pct"] if hit else 50.0)
        return total

    def survives_to_act(self, ours: dict) -> float:
        """Chance our Pokemon is still standing when its (non-priority) move
        comes up: not KO'd first by a predicted attacker that outspeeds it
        (card or estimated Speed, boosts, Choice Scarf; Tailwind and Trick
        Room are not modelled)."""
        name = _id(ours.get("species") or ours.get("name"))
        our_speed = self.calc.speed(ours)
        if our_speed is None:
            return 1.0
        alive = 1.0
        for foe_id, e in self.expected.items():
            chance = e.attacks.get(name, 0.0)
            hit = self.threats.get(foe_id, {}).get(name)
            foe = next((f for f in self.calc._theirs if _id(f.get("species") or f.get("name")) == foe_id), {})
            foe_speed = self.calc.speed(foe) if foe else None
            if chance and hit and hit["ko_this_turn_likely"] and foe_speed is not None and foe_speed > our_speed:
                alive *= 1 - chance
        return alive

    def prevented(self, foe: dict) -> float:
        """Damage this foe was predicted to deal, which a flinch cancels."""
        foe_id = self.foe_id(foe)
        e = self.expected.get(foe_id)
        if not e:
            return 0.0
        total = 0.0
        for name, chance in e.attacks.items():
            hit = self.threats.get(foe_id, {}).get(name)
            if hit:
                total += chance * hit["expected_pct"]
        return total


def _slot_choices(slot, rule, observation: dict, ctx: _Context, movelist) -> list:
    from .actions import move_details

    blocked = rule.blocked

    forced = not any(o.get("type") == "move" for o in slot.options)
    active = ctx.calc.ours(slot.number)
    choices = []
    for index, option in enumerate(slot.options):
        if index in blocked:
            continue
        kind = option.get("type")
        if kind == "pass":
            choices.append((0.0, {"option": index, "target": None}))
        elif kind == "switch":
            if forced:
                value = replacement_value(ctx.calc, observation, option.get("species"))
            else:
                saved = ctx.incoming(active) - ctx.incoming_on_switch(active, option.get("species"))
                value = SWITCH_VALUE + SWITCH_SHARE * max(0.0, saved)
            choices.append((value, {"option": index, "target": None}))
        elif kind == "move":
            details = move_details(option, slot.number, observation)
            move_id = _id(option.get("move_id"))
            choices.extend(_move_choices(slot, index, option, move_id, details, active, ctx, movelist, rule))
    choices.sort(key=lambda item: -item[0])
    return choices[:CANDIDATES_PER_SLOT] or [(0.0, {"option": 0, "target": None})]


def _move_choices(slot, index, option, move_id, details, active, ctx: _Context, movelist, rule) -> list:
    listed = movelist.lookup(move_id) or {}
    foes = [e for e in option.get("target_options") or [] if isinstance(e, dict) and e.get("side") == "opponent"]
    if not foes:
        foes = [{"target": t, "side": "opponent", "species": None} for t in option.get("targets") or [] if isinstance(t, int) and t > 0]
    power = details.get("base_power") or 0
    category = str(details.get("category") or "").upper()
    # A slower Pokemon predicted to be KO'd first never gets its move off.
    acts = 1.0 if (details.get("priority") or 0) > 0 else ctx.survives_to_act(active)
    if power > 0 and category in ("PHYSICAL", "SPECIAL"):
        stats = {**details, "id": move_id}
        bypasses = "bypasses_protect" in (listed.get("tags") or [])
        if listed.get("target") in ("allAdjacentFoes", "allAdjacent") or (not foes and not option.get("targets")):
            value = sum(_hit_value(ctx, slot.number, foe, stats, bypasses, spread=True) for foe in ctx.calc._theirs)
            return [(value * acts, {"option": index, "target": None})]
        out = []
        aimed = rule.targets.get(index)
        for entry in foes:
            if aimed and entry.get("target") not in aimed:
                continue  # the rules allow this move only into certain foes (opening Fake Out)
            foe = ctx.calc.foe(entry["species"]) if entry.get("species") else (
                ctx.calc._theirs[entry["target"] - 1] if 0 < entry["target"] <= len(ctx.calc._theirs) else {}
            )
            value = _hit_value(ctx, slot.number, foe, stats, bypasses, spread=False)
            if move_id == "fakeout" and foe:
                landing = 1 - ctx.protects(ctx.foe_id(foe))
                value += landing * (FAKE_OUT_VALUE + ctx.prevented(foe))
            out.append((value * acts, {"option": index, "target": entry.get("target")}))
        return out
    # Status, self and field moves.
    if move_id in movelist.PROTECTING:
        # The predicted damage it blocks, times its chance of working: a
        # second Protect in a row works 1 time in 3.
        value = ctx.incoming(active) * rule.protect_success
        if _status(active.get("status")) == "slp":
            value *= SLEEP_AWAKE  # asleep: it Protects only if it wakes
    elif move_id in PROTECT_PROOF:
        all_protect = 1.0
        for foe in ctx.calc._theirs:
            all_protect *= ctx.protects(ctx.foe_id(foe))
        base = SPEED_CONTROL_VALUE if move_id in ("tailwind", "trickroom") else OTHER_STATUS_VALUE
        value = PROTECT_PROOF_VALUE * all_protect + base
    elif move_id in REDIRECTION:
        # Worth what it pulls off our partner: nothing in a 1-vs-1, little
        # against sleeping foes (match e1f70879 Rage Powdered both).
        partner = next((e for e in ctx.calc._ours if e is not active and _fraction(e) > 0), None)
        value = min(REDIRECTION_VALUE, ctx.incoming(partner)) if partner else 0.0
    elif "speed_control" in (listed.get("tags") or []):
        value = SPEED_CONTROL_VALUE
    else:
        value = OTHER_STATUS_VALUE
    if move_id not in movelist.PROTECTING:
        value *= acts
    if foes:
        best = min(foes, key=lambda e: ctx.protects(_id(e.get("species"))))
        return [(value * (1 - ctx.protects(_id(best.get("species")))), {"option": index, "target": best.get("target")})]
    targets = [t for t in option.get("targets") or [] if isinstance(t, int)]
    return [(value, {"option": index, "target": targets[0] if len(targets) == 1 else None})]


def accuracy(move: dict) -> float:
    """Chance a move hits: the observation's or the move list's accuracy (a
    fraction or a percentage), 1.0 for moves that never miss. Icicle Crash and
    Draco Meteor (90%) missed three times across Yuriko's games while valued
    as sure hits."""
    value = move.get("accuracy") if isinstance(move, dict) else None
    if value is None and isinstance(move, dict):
        from . import movelist

        listed = movelist.lookup(move.get("id") or move.get("move_id") or move.get("name"))
        value = (listed or {}).get("accuracy")
        if value is None:
            return 1.0
    if value is True or not isinstance(value, (int, float)):
        return 1.0
    value = float(value) / 100 if value > 1 else float(value)
    return max(0.0, min(1.0, value)) or 1.0


def _hit_value(ctx: _Context, slot_number: int, foe: dict, move: dict, bypasses: bool, *, spread: bool) -> float:
    """Expected damage on the HP scale (a likely KO is the rest of its HP and
    half again), times the chance it lands: not into a predicted Protect, and
    half into a predicted switch."""
    if not foe:
        return 0.0
    hit = ctx.calc.damage(ctx.calc.ours(slot_number), foe, move, spread=spread)
    remaining = 100 * _fraction(foe)
    if hit is None:
        value = min(remaining, (move.get("base_power") or 0) * 0.35)
    else:
        value = remaining + KO_BONUS if hit["ko_this_turn_likely"] else min(remaining, hit["expected_pct"])
    foe_id = ctx.foe_id(foe)
    landing = 1.0 if bypasses else 1 - ctx.protects(foe_id)
    weight = WIN_CONDITION_WEIGHT if foe_id in ctx.win else 1.0
    return value * weight * landing * accuracy(move) * (1 - 0.5 * ctx.switches(foe_id))


def asleep(calc, expected: dict) -> dict:
    """``expected`` with each sleeping foe's actions scaled to the chance it wakes
    this turn (it can still switch). Never raises: unknown status is awake."""
    from .predict import FoeExpectation

    found = dict(expected)
    for foe in getattr(calc, "_theirs", None) or []:
        try:
            name = _id(foe.get("species") or foe.get("name"))
            e = found.get(name)
            if e is None or _status(foe.get("status")) != "slp":
                continue
            found[name] = FoeExpectation(
                protect=e.protect * SLEEP_AWAKE, switch=e.switch,
                attacks={o: v * SLEEP_AWAKE for o, v in e.attacks.items()},
                fake_out={o: v * SLEEP_AWAKE for o, v in e.fake_out.items()},
            )
        except Exception:  # noqa: BLE001
            continue
    return found


REPLACEMENT_SECOND_HIT = 0.5  # share of the second foe's best hit also aimed at the newcomer


def replacement_value(calc, observation: dict, species: object) -> float:
    """A Pokemon sent in after a faint: its HP left after their best hits, minus
    KO_BONUS if a faster foe likely KOs it before it moves. From its CURRENT
    HP (Multiscale and Focus Sash only hold at full), against every foe, not
    only the read's targets. Match 60beb733: an 82% Dragonite came in to a
    faster Flutter Mane's Moonblast and fell without acting."""
    from .race import _bench_hp

    hp = _bench_hp(observation, species)
    entry = {"species": species, "hp_fraction": hp}
    mine = calc.speed(entry) or 0.0
    trick_room = "trickroom" in ((observed_field(observation) or {}).get("field") or [])
    hits, first = [], 0.0
    for foe in getattr(calc, "_theirs", []):
        if _fraction(foe) <= 0:
            continue
        hit = calc.best_hit(foe, entry)
        if not hit:
            continue
        hits.append(hit["expected_pct"])
        theirs = calc.speed(foe) or 0.0
        faster = theirs < mine if trick_room else theirs > mine
        if faster and hit.get("ko_this_turn_likely"):
            first = 1.0
    hits.sort(reverse=True)
    taken = (hits[0] if hits else 50.0) + REPLACEMENT_SECOND_HIT * (hits[1] if len(hits) > 1 else 0.0)
    return 100.0 * hp - taken - KO_BONUS * first


def observed_field(observation: dict):
    from .observed import field_state

    return field_state(observation)


def win_conditions(calc) -> dict[str, str]:
    """foe id -> why it is their win condition: an Eruption / Water Spout /
    Dragon Energy user at half HP or more (each hit cuts that move's power),
    a Pokemon certain to set weather, a Choice Specs / Band attacker (its
    draft card or a revealed item), a Prankster screen setter, or a sleep user. Never raises: a missing fact is no flag."""
    found = {}
    table = getattr(calc, "_table", None)
    for foe in getattr(calc, "_theirs", None) or []:
        try:
            name = foe.get("species") or foe.get("name")
            if not name or _fraction(foe) <= 0:
                continue
            moves = {_id(m) for m in (table.common_moves(name) if table is not None else []) or []}
            moves |= {_id(m.get("id") or m.get("name") if isinstance(m, dict) else m)
                      for m in foe.get("revealed_moves") or foe.get("moves") or [] if m}
            scaling = sorted(moves & HP_SCALING)
            abilities = {_id(a) for a in calc._abilities(foe) or []}
            if scaling and _fraction(foe) >= HP_SCALING_FROM:
                found[_id(name)] = f"{name}'s {scaling[0]} loses power with every point of HP we take off"
            elif abilities and abilities <= WEATHER_SETTERS:
                found[_id(name)] = f"{name} sets the weather its team plays around"
            elif table is not None and _id(held_item(foe, table)) in CHOICE_NUKES:
                # held_item: a shown "unknown_item" falls back to the draft card's
                found[_id(name)] = f"{name} holds a Choice Specs / Band: it nukes every turn it stays in"
            elif abilities == {"prankster"} and moves & SCREEN_MOVES:
                found[_id(name)] = f"{name} sets Prankster screens that halve our hits"
            elif moves & SLEEP_MOVES:
                found[_id(name)] = f"{name} puts ours to sleep ({sorted(moves & SLEEP_MOVES)[0]})"
        except Exception:  # noqa: BLE001 - a priority hint, never a lost move
            continue
    return found


def _threats_by_id(calc) -> dict:
    """foe id -> our id -> its best estimated STAB hit on that Pokemon."""
    return {_id(foe): {_id(ours): hit for ours, hit in row.items()} for foe, row in (calc.threats() or {}).items()}


def _ruin(attacker_abilities: list[str], defender_abilities: list[str], category: str) -> float:
    """The Ruin abilities: Beads/Sword lower everyone else's SpD/Def by 25%,
    Tablets/Vessel lower everyone else's Atk/SpA by 25%. Their holders have
    only that ability, so the table's list is certain."""
    factor = 1.0
    if category == "SPECIAL":
        factor *= 1 / 0.75 if "Beads of Ruin" in attacker_abilities else 1.0
        factor *= 0.75 if "Vessel of Ruin" in defender_abilities else 1.0
    else:
        factor *= 1 / 0.75 if "Sword of Ruin" in attacker_abilities else 1.0
        factor *= 0.75 if "Tablets of Ruin" in defender_abilities else 1.0
    return factor


def _status(value: object) -> str:
    """brn / par / psn / tox / slp / frz from however the server names it."""
    text = _id(str(value or "").split(".")[-1])
    for code, names in (("brn", ("brn", "burn")), ("par", ("par", "paralysis")), ("tox", ("tox", "toxic")),
                        ("psn", ("psn", "poison")), ("slp", ("slp", "sleep")), ("frz", ("frz", "freeze"))):
        if text in names:
            return code
    return ""


def _entry_key(entry: dict) -> tuple:
    boosts = tuple(sorted((k, v) for k, v in (entry.get("boosts") or {}).items() if isinstance(v, int) and v))
    types = tuple(entry.get("types") or ())
    return (_id(entry.get("species") or entry.get("name")), str(entry.get("item")), str(entry.get("ability")),
            boosts, str(entry.get("status")), round(_fraction(entry), 3), types)


# Hits of a 2-5 hit move: 2 and 3 at 35% each, 4 and 5 at 15%; Loaded Dice 4-5; Skill Link 5.
_HIT_ODDS = {(2, 5): [(2, 0.35), (3, 0.35), (4, 0.15), (5, 0.15)]}


def _expected_rolls(attacker, defender, move_id: str, fld, spread: bool) -> list[float]:
    """The 16 rolls; for a 2-5 hit move, averaged over how many hits it makes."""
    from . import showdown

    move = showdown.Move.get(move_id)
    hits = move.multihit if move else None
    if isinstance(hits, list) and tuple(hits) in _HIT_ODDS and not attacker.has("skilllink"):
        odds = [(4, 0.5), (5, 0.5)] if attacker.holds("loadeddice") else _HIT_ODDS[tuple(hits)]
        total = [0.0] * 16
        for n, p in odds:
            rolls = showdown.damage_rolls(attacker, defender, move_id, fld, hits=n, spread=spread)
            if not rolls:
                return []
            total = [t + p * r for t, r in zip(total, rolls)]
        return total
    return showdown.damage_rolls(attacker, defender, move_id, fld, spread=spread)


_WEATHER = {"sunnyday": "sun", "desolateland": "sun", "raindance": "rain", "primordialsea": "rain",
            "sandstorm": "sand", "snow": "snow", "snowscape": "snow", "hail": "snow"}
_TERRAIN = {"electricterrain": "electric", "grassyterrain": "grassy", "psychicterrain": "psychic", "mistyterrain": "misty"}
_RUIN = {"swordofruin", "beadsofruin", "tabletsofruin", "vesselofruin"}


def _engine_field(observation: dict, actives: list[dict], table):
    """Weather, terrain, gravity and Ruin abilities on the field. Screens and
    Helping Hand are left out: the race applies those itself."""
    from . import showdown
    from .observed import field_state

    shown = set((field_state(observation) or {}).get("field", []))
    weather = next((w for k, w in _WEATHER.items() if k in shown), "")
    terrain = next((t for k, t in _TERRAIN.items() if k in shown), "")
    ruin = set()
    for entry in actives:
        ability = entry.get("ability") if isinstance(entry.get("ability"), str) else ""
        names = [ability] if ability.strip() else table.abilities(entry.get("species") or entry.get("name"))
        if len(names) == 1 and _id(names[0]) in _RUIN:
            ruin.add(_id(names[0]))
    return showdown.Field(weather=weather, terrain=terrain, gravity="gravity" in shown, ruin=frozenset(ruin))


def _by_species(actives: object) -> list[dict]:
    return [a if isinstance(a, dict) else {} for a in actives] if isinstance(actives, list) else []


def _types(entry: dict, table) -> list[str]:
    types = entry.get("types")
    if isinstance(types, list) and types:
        return [str(t).upper() for t in types]
    return [t.upper() for t in table.species_types(entry.get("species") or entry.get("name"))]


def _boost(entry: dict, key: str) -> int:
    boosts = entry.get("boosts")
    value = boosts.get(key) if isinstance(boosts, dict) else 0
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _fraction(entry: dict) -> float:
    fraction = entry.get("hp_fraction")
    if isinstance(fraction, (int, float)) and not isinstance(fraction, bool):
        return max(0.0, min(1.0, float(fraction)))
    current, maximum = entry.get("current_hp"), entry.get("max_hp")
    if isinstance(current, (int, float)) and isinstance(maximum, (int, float)) and maximum:
        return max(0.0, min(1.0, current / maximum))
    return 1.0
