"""Target abilities that punish a move: absorb it, ignore it, or grow from it.

Type multipliers alone miss these. In match dac4104d Dragonite used Scale
Shot (2-5 hits) into Archaludon, whose Stamina raised its Defense on every
hit; by the next turn it was at +6 Defense, which also powers its Body Press.

We cannot see the opponent's ability, only the ones its species can have,
so every warning says "may have". Each rule is (abilities, test, warning);
``test`` gets the move as ``Move``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Move:
    id: str
    type: str  # upper case, "" when unknown
    category: str  # PHYSICAL / SPECIAL / STATUS, "" when unknown
    priority: int

    @property
    def damaging(self) -> bool:
        return self.category in ("PHYSICAL", "SPECIAL")


MULTI_HIT = frozenset({
    "scaleshot", "bulletseed", "rockblast", "iciclespear", "tailslap", "surgingstrikes", "populationbomb",
    "dualwingbeat", "doublehit", "dragondarts", "tripleaxel", "watershuriken", "bonemerang", "doublekick",
    "twinbeam", "tachyoncutter", "pinmissile", "armthrust", "furyswipes", "triplekick", "doubleironbash",
})
STAT_DROPPING = frozenset({
    "icywind", "electroweb", "snarl", "partingshot", "bulldoze", "rocktomb", "strugglebug", "breakingswipe",
    "lunge", "trailblaze", "mudshot", "nobleroar", "faketears", "screech", "charm", "growl", "tickle",
    "spiritbreak", "lowsweep", "scaryface", "featherdance", "eerieimpulse", "metalsound",
})
BALL_BOMB = frozenset({
    "shadowball", "sludgebomb", "energyball", "focusblast", "aurasphere", "electroball", "mudbomb",
    "seedbomb", "gyroball", "rockblast", "pollenpuff", "weatherball", "zapcannon", "pyroball", "acidspray",
    "octazooka", "magnetbomb", "searingshot", "mistball",
})
WIND = frozenset({
    "hurricane", "heatwave", "icywind", "bleakwindstorm", "sandsearstorm", "springtidestorm", "wildboltstorm",
    "petalblizzard", "aircutter", "blizzard", "gust", "twister", "fairywind", "whirlwind",
})
POWDER = frozenset({"spore", "sleeppowder", "stunspore", "poisonpowder", "ragepowder", "magicpowder"})
SOUND = frozenset({"hypervoice", "boomburst", "bugbuzz", "snarl", "overdrive", "partingshot", "torchsong", "alluringvoice", "psychicnoise"})


def _type(*types: str) -> Callable[[Move], bool]:
    return lambda move: move.type in types


RULES: list[tuple[tuple[str, ...], Callable[[Move], bool], str]] = [
    (("Stamina",), lambda m: m.damaging and m.id in MULTI_HIT, "Stamina: +1 Defense PER HIT (a multi-hit move can give +5)"),
    (("Stamina",), lambda m: m.damaging and m.id not in MULTI_HIT, "Stamina: +1 Defense when hit"),
    (("Weak Armor",), lambda m: m.category == "PHYSICAL", "Weak Armor: -1 Defense but +2 Speed when hit"),
    (("Justified",), lambda m: m.damaging and m.type == "DARK", "Justified: +1 Attack from Dark hits"),
    (("Steam Engine",), lambda m: m.damaging and m.type in ("FIRE", "WATER"), "Steam Engine: +6 Speed"),
    (("Water Compaction",), lambda m: m.damaging and m.type == "WATER", "Water Compaction: +2 Defense"),
    (("Anger Point",), lambda m: m.damaging, "Anger Point: maxes Attack if crit"),
    (("Flash Fire", "Well-Baked Body"), _type("FIRE"), "immune to Fire"),
    (("Water Absorb", "Storm Drain", "Dry Skin"), _type("WATER"), "immune to Water (Storm Drain also redirects it)"),
    (("Volt Absorb", "Lightning Rod", "Motor Drive"), _type("ELECTRIC"), "immune to Electric (Lightning Rod redirects it)"),
    (("Sap Sipper",), _type("GRASS"), "immune to Grass, +1 Attack"),
    (("Levitate", "Earth Eater"), _type("GROUND"), "immune to Ground"),
    (("Thick Fat",), _type("FIRE", "ICE"), "Thick Fat: takes half from Fire and Ice"),
    (("Purifying Salt",), _type("GHOST"), "Purifying Salt: takes half from Ghost"),
    (("Heatproof", "Water Bubble"), _type("FIRE"), "takes half from Fire"),
    (("Fluffy",), _type("FIRE"), "Fluffy: takes double from Fire"),
    (("Good as Gold",), lambda m: m.category == "STATUS", "Good as Gold: status moves fail"),
    (("Magic Bounce",), lambda m: m.category == "STATUS", "Magic Bounce: status moves bounce back"),
    (("Defiant",), lambda m: m.id in STAT_DROPPING, "Defiant: +2 Attack if a stat drops"),
    (("Competitive",), lambda m: m.id in STAT_DROPPING, "Competitive: +2 Sp. Atk if a stat drops"),
    (("Mirror Armor",), lambda m: m.id in STAT_DROPPING, "Mirror Armor: stat drops reflect back"),
    (("Clear Body", "Full Metal Body", "White Smoke"), lambda m: m.id in STAT_DROPPING, "ignores stat drops"),
    (("Armor Tail", "Queenly Majesty", "Dazzling"), lambda m: m.priority > 0, "blocks priority moves"),
    (("Bulletproof",), lambda m: m.id in BALL_BOMB, "Bulletproof: immune to ball/bomb moves"),
    (("Wind Rider",), lambda m: m.id in WIND, "Wind Rider: immune to wind moves, +1 Attack"),
    (("Overcoat",), lambda m: m.id in POWDER, "Overcoat: immune to powder moves"),
    (("Soundproof",), lambda m: m.id in SOUND, "Soundproof: immune to sound moves"),
    (("Inner Focus",), lambda m: m.id == "fakeout", "Inner Focus: cannot be flinched"),
    (("Multiscale", "Shadow Shield"), lambda m: m.damaging, "takes half damage at full HP"),
]


def warnings(move: Move, possible: list[str]) -> list[str]:
    """What the target's possible abilities do to this move, "may have" each."""
    have = set(possible)
    found = []
    for names, test, text in RULES:
        hit = [name for name in names if name in have]
        if hit and test(move):
            found.append(f"may have {'/'.join(hit)} - {text}")
    return found
