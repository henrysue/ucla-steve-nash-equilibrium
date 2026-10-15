"""Pokémon Showdown's Gen 9 damage formula: exact damage rolls for known sets.

The platform runs real Showdown battles, and Showdown is the authority on
damage. Every Pokémon here is a fixed draft card (EVs, nature, item,
ability), so damage need not be estimated: this module reproduces Showdown's
formula and returns the 16 possible damage rolls of a hit, the way Showdown
draws one of them.

It mirrors @smogon/calc's Gen 7-9 mechanics (``gen789.js``; the calculator
Showdown's damage page uses, written to match the battle simulator) stage by
stage: exact stats, base power and its modifier chain, attack and defense
with their chains, the base damage, spread / weather / critical hits, then
per roll the random factor, STAB, type effectiveness, burn and the final
modifier chain, with Showdown's 4096-based rounding throughout. It covers the
abilities, items and moves found in VGC draft pools; an effect it does not
know falls back to the plain formula.

Checked against @smogon/calc on every card-vs-card attack the agent has seen:
``tests/fixtures/damage_reference.json`` (regenerate with
``scripts/check_damage.py`` when Node and @smogon/calc are installed).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field as dc_field
from functools import lru_cache
from pathlib import Path

from . import movelist
from .typechart import effectiveness

POKEDEX = Path(__file__).resolve().parents[2] / "data" / "pokedex.json"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
NATURES = {
    "lonely": ("atk", "def"), "brave": ("atk", "spe"), "adamant": ("atk", "spa"), "naughty": ("atk", "spd"),
    "bold": ("def", "atk"), "relaxed": ("def", "spe"), "impish": ("def", "spa"), "lax": ("def", "spd"),
    "timid": ("spe", "atk"), "hasty": ("spe", "def"), "jolly": ("spe", "spa"), "naive": ("spe", "spd"),
    "modest": ("spa", "atk"), "mild": ("spa", "def"), "quiet": ("spa", "spe"), "rash": ("spa", "spd"),
    "calm": ("spd", "atk"), "gentle": ("spd", "def"), "sassy": ("spd", "spe"), "careful": ("spd", "spa"),
}
ALWAYS_CRIT = frozenset({"surgingstrikes", "wickedblow", "flowertrick", "frostbreath", "stormthrow", "zippyzap"})
HITS_DEFENSE = frozenset({"psyshock", "psystrike", "secretsword"})
IGNORES_DEFENSE_BOOSTS = frozenset({"sacredsword", "darkestlariat", "chipaway"})
WEIGHT_BASED = frozenset({"lowkick", "grassknot", "heavyslam", "heatcrash"})
ITEM_TYPE = {
    "silkscarf": "NORMAL", "blackbelt": "FIGHTING", "sharpbeak": "FLYING", "poisonbarb": "POISON", "softsand": "GROUND",
    "hardstone": "ROCK", "silverpowder": "BUG", "spelltag": "GHOST", "metalcoat": "STEEL", "charcoal": "FIRE",
    "mysticwater": "WATER", "miracleseed": "GRASS", "magnet": "ELECTRIC", "twistedspoon": "PSYCHIC",
    "nevermeltice": "ICE", "dragonfang": "DRAGON", "blackglasses": "DARK", "fairyfeather": "FAIRY",
    "seaincense": "WATER", "waveincense": "WATER", "roseincense": "GRASS", "oddincense": "PSYCHIC", "rockincense": "ROCK",
    "flameplate": "FIRE", "splashplate": "WATER", "meadowplate": "GRASS", "zapplate": "ELECTRIC", "icicleplate": "ICE",
    "fistplate": "FIGHTING", "toxicplate": "POISON", "earthplate": "GROUND", "skyplate": "FLYING", "mindplate": "PSYCHIC",
    "insectplate": "BUG", "stoneplate": "ROCK", "spookyplate": "GHOST", "dracoplate": "DRAGON", "dreadplate": "DARK",
    "ironplate": "STEEL", "pixieplate": "FAIRY",
}
RESIST_BERRY = {
    "chopleberry": "FIGHTING", "occaberry": "FIRE", "passhoberry": "WATER", "wacanberry": "ELECTRIC", "rindoberry": "GRASS",
    "yacheberry": "ICE", "kebiaberry": "POISON", "shucaberry": "GROUND", "cobaberry": "FLYING", "payapaberry": "PSYCHIC",
    "tangaberry": "BUG", "chartiberry": "ROCK", "kasibberry": "GHOST", "habanberry": "DRAGON", "colburberry": "DARK",
    "babiriberry": "STEEL", "roseliberry": "FAIRY", "chilanberry": "NORMAL",
}
ATE = {"aerilate": "FLYING", "galvanize": "ELECTRIC", "pixilate": "FAIRY", "refrigerate": "ICE"}
# Defender abilities that make a move fail outright (type, flag or priority).
IMMUNE_TYPE_ABILITY = {
    "sapsipper": "GRASS", "flashfire": "FIRE", "wellbakedbody": "FIRE", "dryskin": "WATER", "stormdrain": "WATER",
    "waterabsorb": "WATER", "lightningrod": "ELECTRIC", "motordrive": "ELECTRIC", "voltabsorb": "ELECTRIC",
    "levitate": "GROUND", "eartheater": "GROUND",
}
MOLD_BREAKERS = frozenset({"moldbreaker", "teravolt", "turboblaze"})
UNBREAKABLE = frozenset({"fullmetalbody", "prismarmor", "shadowshield", "tabletsofruin", "vesselofruin"})


def _id(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def poke_round(num: float) -> int:
    """Showdown's rounding: .5 rounds down."""
    return math.ceil(num) if num % 1 > 0.5 else math.floor(num)


def chain(mods: list[int], low: int, high: int) -> int:
    m = 4096
    for mod in mods:
        if mod != 4096:
            m = (m * mod + 2048) >> 12
    return max(min(m, high), low)


def apply_chain(value: int, mods: list[int], low: int, high: int) -> int:
    return max(1, poke_round(value * chain(mods, low, high) / 4096))


@lru_cache(maxsize=1)
def pokedex() -> dict:
    try:
        return json.loads(POKEDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def species(name: object) -> dict | None:
    return pokedex().get(_id(name))


def modified_stat(stat: int, stage: int) -> int:
    stage = max(-6, min(6, stage))
    num, den = (2 + stage, 2) if stage >= 0 else (2, 2 - stage)
    return stat * num // den


@dataclass
class Battler:
    """One Pokémon as the formula needs it: a full set plus its battle state."""

    species: str
    item: str = ""
    ability: str = ""
    nature: str = ""
    evs: dict = dc_field(default_factory=dict)
    ivs: dict = dc_field(default_factory=dict)
    level: int = 50
    boosts: dict = dc_field(default_factory=dict)
    status: str = ""
    hp_fraction: float = 1.0
    types: list | None = None  # the board's types win (a type change), else the Pokédex's
    quark: bool = True  # Protosynthesis / Quark Drive counted when its trigger holds (Booster Energy, sun, terrain)

    def __post_init__(self) -> None:
        data = species(self.species) or {}
        self.data = data
        self.types = [t.upper() for t in (self.types or data.get("types") or [])]
        base = data.get("base") or {}
        self.raw = {s: self._stat(s, base.get(s, 0)) for s in STATS}
        self.max_hp = self.raw["hp"]
        self.cur_hp = max(0, min(self.max_hp, round(self.hp_fraction * self.max_hp)))

    def _stat(self, stat: str, base: int) -> int:
        iv = self.ivs.get(stat, 31) if isinstance(self.ivs, dict) else 31
        ev = (self.evs or {}).get(stat, 0)
        core = (2 * base + iv + ev // 4) * self.level // 100
        if stat == "hp":
            return core + self.level + 10 if base != 1 else 1
        up, down = NATURES.get(_id(self.nature), (None, None))
        n = 1.0 if up == down else 1.1 if stat == up else 0.9 if stat == down else 1.0
        return math.floor((core + 5) * n)

    def has(self, *abilities: str) -> bool:
        return _id(self.ability) in {_id(a) for a in abilities}

    def holds(self, *items: str) -> bool:
        return _id(self.item) in {_id(i) for i in items}

    def boost(self, stat: str) -> int:
        value = (self.boosts or {}).get(stat, 0)
        return value if isinstance(value, int) else 0

    def stat(self, s: str) -> int:
        return modified_stat(self.raw[s], self.boost(s))

    @property
    def weight(self) -> float:
        w = self.data.get("weightkg") or 0
        if self.has("heavymetal"):
            w *= 2
        elif self.has("lightmetal"):
            w /= 2
        if self.holds("floatstone"):
            w /= 2
        return max(0.1, w)


@dataclass
class Field:
    weather: str = ""  # sun / rain / sand / snow
    terrain: str = ""  # electric / grassy / psychic / misty
    doubles: bool = True
    gravity: bool = False
    reflect: bool = False  # the defender's side
    light_screen: bool = False
    aurora_veil: bool = False
    friend_guard: bool = False  # the defender's partner has Friend Guard
    helping_hand: bool = False  # the attacker's side
    battery: bool = False  # the attacker's partner has Battery (special moves)
    power_spot: bool = False  # the attacker's partner has Power Spot
    ruin: frozenset = frozenset()  # Ruin abilities on the field beside these two


def grounded(mon: Battler, fld: Field) -> bool:
    return fld.gravity or mon.holds("ironball") or not ("FLYING" in mon.types or mon.has("levitate") or mon.holds("airballoon"))


def qp_active(mon: Battler, fld: Field) -> bool:
    if not mon.quark:
        return False
    return (mon.has("protosynthesis") and (fld.weather == "sun" or mon.holds("boosterenergy"))) or \
        (mon.has("quarkdrive") and (fld.terrain == "electric" or mon.holds("boosterenergy")))


def qp_stat(mon: Battler) -> str:
    best = "atk"
    for s in ("def", "spa", "spd", "spe"):
        if mon.stat(s) > mon.stat(best):
            best = s
    return best


@dataclass
class Move:
    id: str
    name: str
    type: str
    category: str
    power: int
    target: str
    priority: int
    flags: frozenset
    multihit: object
    secondary: bool = False  # has secondary effects (Sheer Force)
    recoil: object = None

    @classmethod
    def get(cls, move_id: str) -> "Move | None":
        data = movelist.lookup(move_id)
        if not data:
            return None
        return cls(_id(move_id), data["name"], data["type"], data["category"], data["power"] or 0, data["target"],
                   data["priority"] or 0, frozenset(data.get("flags") or []), data.get("multihit"),
                   bool(data.get("secondary")), data.get("recoil"))

    @property
    def spread(self) -> bool:
        return self.target in ("allAdjacent", "allAdjacentFoes")


def default_hits(move: Move, attacker: Battler) -> int:
    """Hits a multi-hit move is assumed to make (integer; Skill Link 5, Loaded Dice 4, else 3)."""
    hits = move.multihit
    if isinstance(hits, int):
        return hits
    if isinstance(hits, list) and len(hits) == 2:
        if attacker.has("skilllink"):
            return hits[1]
        if attacker.holds("loadeddice") and hits[1] >= 5:
            return 4
        return 3 if hits[1] >= 5 else hits[0]
    return 1


def _effectiveness(move: Move, defender: Battler, attacker: Battler, fld: Field, move_type: str) -> float:
    total = 1.0
    for t in defender.types:
        if t == "GHOST" and move_type in ("NORMAL", "FIGHTING") and (attacker.has("scrappy", "mindseye") or defender.holds("ringtarget")):
            e = 1.0
        elif t == "FLYING" and move_type == "GROUND" and (fld.gravity or defender.holds("ringtarget")):
            e = 1.0
        elif move.id == "freezedry" and t == "WATER":
            e = 2.0
        elif move.id == "flyingpress":
            e = effectiveness("FIGHTING", [t]) * effectiveness("FLYING", [t])
        else:
            e = effectiveness(move_type, [t])
        total *= e
    if total == 0 and move_type == "GROUND" and defender.holds("ironball"):
        total = 1.0
    return total


def damage_rolls(attacker: Battler, defender: Battler, move_id: str, fld: Field | None = None, *,
                 crit: bool = False, hits: int | None = None, spread: bool | None = None) -> list[int]:
    """The 16 damage rolls (HP) of one use of the move; ``[]`` when it cannot hit."""
    fld = fld or Field()
    move = Move.get(move_id)
    if move is None or move.category not in ("PHYSICAL", "SPECIAL"):
        return []
    attacker = _copy(attacker)
    defender = _copy(defender)
    if attacker.has(*MOLD_BREAKERS) and not defender.has(*UNBREAKABLE):
        defender.ability = ""
    crit = (crit or move.id in ALWAYS_CRIT) and not defender.has("battlearmor", "shellarmor")

    # Type, and the abilities that change it.
    move_type = move.type
    ate = False
    if move.id == "weatherball":
        move_type = {"sun": "FIRE", "rain": "WATER", "sand": "ROCK", "snow": "ICE"}.get(fld.weather, "NORMAL")
    elif move.id == "terrainpulse" and grounded(attacker, fld) and fld.terrain:
        move_type = {"electric": "ELECTRIC", "grassy": "GRASS", "misty": "FAIRY", "psychic": "PSYCHIC"}[fld.terrain]
    elif move.id not in ("weatherball", "terrainpulse", "struggle"):
        ate_type = ATE.get(_id(attacker.ability))
        if ate_type and move_type == "NORMAL":
            move_type, ate = ate_type, True
        elif attacker.has("liquidvoice") and "sound" in move.flags:
            move_type = "WATER"
        elif attacker.has("normalize"):
            move_type, ate = "NORMAL", True
    priority = move.priority
    if attacker.has("galewings") and move_type == "FLYING" and attacker.cur_hp == attacker.max_hp:
        priority = 1

    eff = _effectiveness(move, defender, attacker, fld, move_type)
    if eff == 0:
        return []
    ability = _id(defender.ability)
    if (ability == "wonderguard" and eff <= 1) or IMMUNE_TYPE_ABILITY.get(ability) == move_type and not (
            move_type == "GROUND" and (fld.gravity or defender.holds("ironball"))) \
            or ("bullet" in move.flags and ability == "bulletproof") or ("sound" in move.flags and ability == "soundproof") \
            or (priority > 0 and ability in ("queenlymajesty", "dazzling", "armortail")) \
            or ("wind" in move.flags and ability == "windrider"):
        return []
    if move_type == "GROUND" and not fld.gravity and defender.holds("airballoon"):
        return []
    if priority > 0 and fld.terrain == "psychic" and grounded(defender, fld):
        return []

    if move.id in ("seismictoss", "nightshade"):
        return [attacker.level] * 16

    n_hits = hits or default_hits(move, attacker)
    is_spread = (move.spread if spread is None else spread) and fld.doubles
    total = [0] * 16
    for hit in range(1, n_hits + 1):
        bp = _base_power(attacker, defender, move, move_type, fld, ate, hit)
        if bp == 0:
            return []
        attack = _attack(attacker, defender, move, move_type, fld, crit)
        defense = _defense(attacker, defender, move, fld, crit)
        base = (2 * attacker.level // 5 + 2) * bp * attack // defense // 50 + 2
        if is_spread:
            base = poke_round(base * 3072 / 4096)
        if not defender.holds("utilityumbrella"):
            if (fld.weather == "sun" and move_type == "FIRE") or (fld.weather == "rain" and move_type == "WATER"):
                base = poke_round(base * 6144 / 4096)
            elif (fld.weather == "sun" and move_type == "WATER") or (fld.weather == "rain" and move_type == "FIRE"):
                base = poke_round(base * 2048 / 4096)
        if crit:
            base = math.floor(base * 1.5)
        stab = _stab(attacker, move_type)
        burn = attacker.status == "brn" and move.category == "PHYSICAL" and not attacker.has("guts") and move.id != "facade"
        final = chain(_final_mods(attacker, defender, move, move_type, fld, crit, eff, hit - 1), 41, 131072)
        for i in range(16):
            amount = math.floor(base * (85 + i) / 100)
            if stab != 4096:
                amount = amount * stab / 4096
            amount = math.floor(poke_round(amount) * eff)
            if burn:
                amount = amount // 2
            total[i] += poke_round(max(1, amount * final / 4096))
        _after_hit(attacker, defender, move)
    return total


def _copy(mon: Battler) -> Battler:
    clone = Battler.__new__(Battler)
    clone.__dict__.update(mon.__dict__)
    clone.boosts = dict(mon.boosts or {})
    return clone


def _after_hit(attacker: Battler, defender: Battler, move: Move) -> None:
    """Between hits of one multi-hit move."""
    if defender.has("stamina") and not attacker.has("unaware"):
        defender.boosts["def"] = min(6, defender.boost("def") + 1)
    elif defender.has("watercompaction") and move.type == "WATER" and not attacker.has("unaware"):
        defender.boosts["def"] = min(6, defender.boost("def") + 2)
    elif defender.has("weakarmor") and not attacker.has("unaware"):
        defender.boosts["def"] = max(-6, defender.boost("def") - 1)


def _stab(attacker: Battler, move_type: str) -> int:
    stab = 4096
    if move_type in attacker.types:
        stab += 2048
    elif attacker.has("protean", "libero"):
        stab += 2048
    if attacker.has("adaptability") and move_type in attacker.types:
        stab += 2048
    return stab


def _base_power(attacker: Battler, defender: Battler, move: Move, move_type: str, fld: Field, ate: bool, hit: int) -> int:
    bp = move.power
    turn_last = attacker.stat("spe") <= defender.stat("spe")
    mid = move.id
    if mid in ("lowkick", "grassknot"):
        w = defender.weight
        bp = 120 if w >= 200 else 100 if w >= 100 else 80 if w >= 50 else 60 if w >= 25 else 40 if w >= 10 else 20
    elif mid in ("heavyslam", "heatcrash"):
        r = attacker.weight / defender.weight
        bp = 120 if r >= 5 else 100 if r >= 4 else 80 if r >= 3 else 60 if r >= 2 else 40
    elif mid in ("eruption", "waterspout", "dragonenergy"):
        bp = max(1, 150 * attacker.cur_hp // attacker.max_hp)
    elif mid in ("hex", "infernalparade"):
        bp *= 2 if defender.status else 1
    elif mid == "acrobatics":
        bp *= 2 if (not attacker.item or qp_active(attacker, fld)) else 1
    elif mid == "electroball":
        r = attacker.stat("spe") // max(1, defender.stat("spe"))
        bp = 150 if r >= 4 else 120 if r >= 3 else 80 if r >= 2 else 60 if r >= 1 else 40
    elif mid == "gyroball":
        bp = min(150, 25 * defender.stat("spe") // max(1, attacker.stat("spe")) + 1)
    elif mid in ("storedpower", "powertrip"):
        bp = 20 + 20 * sum(v for v in (attacker.boosts or {}).values() if isinstance(v, int) and v > 0)
    elif mid == "weatherball":
        bp *= 2 if fld.weather else 1
    elif mid == "terrainpulse":
        bp *= 2 if grounded(attacker, fld) and fld.terrain else 1
    elif mid == "risingvoltage":
        bp *= 2 if grounded(defender, fld) and fld.terrain == "electric" else 1
    elif mid == "psyblade":
        bp = math.floor(bp * (1.5 if fld.terrain == "electric" else 1))
    elif mid in ("payback",):
        bp *= 2 if turn_last else 1
    elif mid in ("boltbeak", "fishiousrend"):
        bp *= 1 if turn_last else 2
    elif mid == "tripleaxel":
        bp = hit * 20
    elif mid == "triplekick":
        bp = hit * 10
    elif mid in ("crushgrip", "wringout"):
        p = 100 * (defender.cur_hp * 4096 // defender.max_hp)
        bp = (120 * p + 2047) // 4096 // 100 or 1
    elif mid == "hardpress":
        p = 100 * (defender.cur_hp * 4096 // defender.max_hp)
        bp = (100 * p + 2047) // 4096 // 100 or 1
    if bp == 0:
        return 0

    mods = []
    knock_off_ok = bool(defender.item) and not qp_active(defender, fld)
    if (mid == "facade" and attacker.status in ("brn", "par", "psn", "tox")) or \
            (mid == "brine" and defender.cur_hp <= defender.max_hp / 2) or \
            (mid == "venoshock" and defender.status in ("psn", "tox")):
        mods.append(8192)
    elif mid == "expandingforce" and grounded(attacker, fld) and fld.terrain == "psychic":
        mods.append(6144)
    elif (mid == "knockoff" and knock_off_ok) or (mid == "mistyexplosion" and grounded(attacker, fld) and fld.terrain == "misty"):
        mods.append(6144)
    elif mid in ("solarbeam", "solarblade") and fld.weather in ("rain", "sand", "snow"):
        mods.append(2048)
    elif mid in ("collisioncourse", "electrodrift") and _effectiveness(move, defender, attacker, fld, move_type) >= 2:
        mods.append(5461)
    if fld.helping_hand:
        mods.append(6144)
    if grounded(attacker, fld) and ((fld.terrain == "electric" and move_type == "ELECTRIC")
                                    or (fld.terrain == "grassy" and move_type == "GRASS")
                                    or (fld.terrain == "psychic" and move_type == "PSYCHIC")):
        mods.append(5325)
    if grounded(defender, fld) and ((fld.terrain == "misty" and move_type == "DRAGON")
                                    or (fld.terrain == "grassy" and mid in ("bulldoze", "earthquake"))):
        mods.append(2048)
    if (attacker.has("technician") and bp <= 60) or (attacker.has("megalauncher") and "pulse" in move.flags) \
            or (attacker.has("strongjaw") and "bite" in move.flags) or (attacker.has("steelyspirit") and move_type == "STEEL") \
            or (attacker.has("sharpness") and "slicing" in move.flags) \
            or (attacker.has("flareboost") and attacker.status == "brn" and move.category == "SPECIAL") \
            or (attacker.has("toxicboost") and attacker.status in ("psn", "tox") and move.category == "PHYSICAL"):
        mods.append(6144)
    if (attacker.has("sheerforce") and move.secondary) \
            or (attacker.has("sandforce") and fld.weather == "sand" and move_type in ("ROCK", "GROUND", "STEEL")) \
            or (attacker.has("analytic") and turn_last) or (attacker.has("toughclaws") and "contact" in move.flags) \
            or (attacker.has("punkrock") and "sound" in move.flags):
        mods.append(5325)
    if fld.battery and move.category == "SPECIAL":
        mods.append(5325)
    if fld.power_spot:
        mods.append(5325)
    if ate:
        mods.append(4915)
    if (attacker.has("reckless") and (move.recoil or mid in ("highjumpkick", "jumpkick", "axekick", "supercellslam"))) \
            or (attacker.has("ironfist") and "punch" in move.flags):
        mods.append(4915)
    if attacker.holds("punchingglove") and "punch" in move.flags:
        mods.append(4506)
    if defender.has("dryskin") and move_type == "FIRE":
        mods.append(5120)
    item = _id(attacker.item)
    if item.endswith("gem") and item[:-3].upper() == move_type:
        mods.append(5325)
    elif ITEM_TYPE.get(item) == move_type:
        mods.append(4915)
    elif (item == "muscleband" and move.category == "PHYSICAL") or (item == "wiseglasses" and move.category == "SPECIAL"):
        mods.append(4505)
    return max(1, poke_round(bp * chain(mods, 41, 2097152) / 4096))


def _attack(attacker: Battler, defender: Battler, move: Move, move_type: str, fld: Field, crit: bool) -> int:
    stat = "def" if move.id == "bodypress" else "spa" if move.category == "SPECIAL" else "atk"
    source = defender if move.id == "foulplay" else attacker
    stage = source.boost(stat)
    if stage == 0 or (crit and stage < 0) or defender.has("unaware"):
        attack = source.raw[stat]
    else:
        attack = modified_stat(source.raw[stat], stage)
    if attacker.has("hustle") and move.category == "PHYSICAL":
        attack = poke_round(attack * 3 / 2)
    mods = []
    physical, special = move.category == "PHYSICAL", move.category == "SPECIAL"
    if attacker.has("defeatist") and attacker.cur_hp <= attacker.max_hp / 2:
        mods.append(2048)
    elif attacker.has("solarpower") and fld.weather == "sun" and special:
        mods.append(6144)
    elif attacker.has("gorillatactics") and physical:
        mods.append(6144)
    elif (attacker.has("guts") and attacker.status and physical) or (
            attacker.cur_hp <= attacker.max_hp / 3 and (
            (attacker.has("overgrow") and move_type == "GRASS") or (attacker.has("blaze") and move_type == "FIRE")
            or (attacker.has("torrent") and move_type == "WATER") or (attacker.has("swarm") and move_type == "BUG"))):
        mods.append(6144)
    elif (attacker.has("steelworker") and move_type == "STEEL") or (attacker.has("dragonsmaw") and move_type == "DRAGON") \
            or (attacker.has("rockypayload") and move_type == "ROCK"):
        mods.append(6144)
    elif attacker.has("transistor") and move_type == "ELECTRIC":
        mods.append(5325)
    elif (attacker.has("waterbubble") and move_type == "WATER") or (attacker.has("hugepower", "purepower") and physical):
        mods.append(8192)
    if (defender.has("thickfat") and move_type in ("FIRE", "ICE")) or (defender.has("waterbubble") and move_type == "FIRE") \
            or (defender.has("purifyingsalt") and move_type == "GHOST"):
        mods.append(2048)
    if defender.has("heatproof") and move_type == "FIRE":
        mods.append(2048)
    ruin = fld.ruin | {_id(defender.ability)}
    if ("tabletsofruin" in ruin and not attacker.has("tabletsofruin") and physical) or \
            ("vesselofruin" in ruin and not attacker.has("vesselofruin") and special):
        mods.append(3072)
    if qp_active(attacker, fld) and ((physical and qp_stat(attacker) == "atk") or (special and qp_stat(attacker) == "spa")):
        mods.append(5325)
    if (attacker.has("hadronengine") and special and fld.terrain == "electric") or \
            (attacker.has("orichalcumpulse") and physical and fld.weather == "sun" and not attacker.holds("utilityumbrella")):
        mods.append(5461)
    if (attacker.holds("choiceband") and physical) or (attacker.holds("choicespecs") and special):
        mods.append(6144)
    return apply_chain(attack, mods, 410, 131072)


def _defense(attacker: Battler, defender: Battler, move: Move, fld: Field, crit: bool) -> int:
    physical = move.category == "PHYSICAL" or move.id in HITS_DEFENSE
    stat = "def" if physical else "spd"
    stage = defender.boost(stat)
    if stage == 0 or (crit and stage > 0) or move.id in IGNORES_DEFENSE_BOOSTS or attacker.has("unaware"):
        defense = defender.raw[stat]
    else:
        defense = modified_stat(defender.raw[stat], stage)
    if fld.weather == "sand" and "ROCK" in defender.types and not physical:
        defense = poke_round(defense * 3 / 2)
    if fld.weather == "snow" and "ICE" in defender.types and physical:
        defense = poke_round(defense * 3 / 2)
    mods = []
    if defender.has("marvelscale") and defender.status and physical:
        mods.append(6144)
    elif defender.has("grasspelt") and fld.terrain == "grassy" and physical:
        mods.append(6144)
    elif defender.has("furcoat") and physical:
        mods.append(8192)
    ruin = fld.ruin | {_id(attacker.ability)}
    if ("swordofruin" in ruin and not defender.has("swordofruin") and physical) or \
            ("beadsofruin" in ruin and not defender.has("beadsofruin") and not physical):
        mods.append(3072)
    if qp_active(defender, fld) and ((physical and qp_stat(defender) == "def") or (not physical and qp_stat(defender) == "spd")):
        mods.append(5324)
    if (defender.holds("eviolite") and (defender.data.get("nfe") or _id(defender.species) == "dipplin")) or \
            (not physical and defender.holds("assaultvest")):
        mods.append(6144)
    return apply_chain(defense, mods, 410, 131072)


def _final_mods(attacker: Battler, defender: Battler, move: Move, move_type: str, fld: Field, crit: bool,
                eff: float, hit_index: int) -> list[int]:
    mods = []
    screen = 2732 if fld.doubles else 2048
    if not crit:
        if fld.aurora_veil:
            mods.append(screen)
        elif (fld.reflect and move.category == "PHYSICAL") or (fld.light_screen and move.category == "SPECIAL"):
            mods.append(screen)
    if attacker.has("neuroforce") and eff > 1:
        mods.append(5120)
    elif attacker.has("sniper") and crit:
        mods.append(6144)
    elif attacker.has("tintedlens") and eff < 1:
        mods.append(8192)
    if defender.has("multiscale", "shadowshield") and defender.cur_hp == defender.max_hp and hit_index == 0:
        mods.append(2048)
    if defender.has("fluffy") and "contact" in move.flags and not attacker.has("longreach"):
        mods.append(2048)
    elif (defender.has("punkrock") and "sound" in move.flags) or (defender.has("icescales") and move.category == "SPECIAL"):
        mods.append(2048)
    if defender.has("solidrock", "filter", "prismarmor") and eff > 1:
        mods.append(3072)
    if fld.friend_guard:
        mods.append(3072)
    if defender.has("fluffy") and move_type == "FIRE":
        mods.append(8192)
    if attacker.holds("expertbelt") and eff > 1:
        mods.append(4915)
    elif attacker.holds("lifeorb"):
        mods.append(5324)
    if RESIST_BERRY.get(_id(defender.item)) == move_type and (eff > 1 or move_type == "NORMAL") and hit_index == 0 \
            and not attacker.has("unnerve"):
        mods.append(1024 if defender.has("ripen") else 2048)
    return mods
