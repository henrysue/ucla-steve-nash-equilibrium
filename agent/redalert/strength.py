"""What a mix of units is worth against another mix — not just its price.

From the OpenRA RA rules (mods/ra/rules, mods/ra/weapons; 25 ticks/s):
hit points, armor class (infantry "none", jeeps and artillery "light",
tanks and APCs "heavy") and damage per second against each armor class
(warhead damage x the weapon's ``Versus`` modifier / reload). A rifleman
does 1875/s to infantry but 125/s to a tank; a rocket soldier 2500/s to a
tank and 250/s to infantry; a medium tank 2300/s to tanks, 600/s to
infantry. Per credit, rockets kill tanks ~7x faster than riflemen do.

:func:`edge` is a Lanchester-style comparison: each side's damage aimed at
the other's armor mix (weighted by its hit points) times its own hit
points; the square root of the ratio reads like a value ratio (1.0 = even).
Match 52a77fb1: 16 riflemen, 3 light tanks and 3 rockets (5000) against 7
infantry and 3 medium tanks (~4000) looked 1.25:1 by price; by strength it
was about even, and it was fought split and outside our turrets.
"""

from __future__ import annotations

from . import config as C

ARMOR_CLASSES = ("none", "light", "heavy")
# type: (hp, armor, {armor: damage per second})
UNITS: dict[str, tuple[int, str, dict[str, float]]] = {
    "e1": (5000, "none", {"none": 1875, "light": 500, "heavy": 125}),
    "e2": (5000, "none", {"none": 1500, "light": 625, "heavy": 625}),
    "e3": (4500, "none", {"none": 250, "light": 850, "heavy": 2500}),
    "e4": (4000, "none", {"none": 3000, "light": 1500, "heavy": 750}),
    "dog": (1800, "none", {"none": 5000, "light": 0, "heavy": 0}),
    "shok": (5000, "none", {"none": 1500, "light": 1500, "heavy": 1500}),
    "1tnk": (23000, "heavy", {"none": 952, "light": 3452, "heavy": 1429}),
    "2tnk": (46000, "heavy", {"none": 600, "light": 1500, "heavy": 2300}),
    "3tnk": (60000, "heavy", {"none": 857, "light": 2143, "heavy": 3286}),
    "4tnk": (90000, "heavy", {"none": 1500, "light": 2500, "heavy": 3833}),
    "ttnk": (40000, "heavy", {"none": 1500, "light": 2000, "heavy": 2500}),
    "ctnk": (40000, "heavy", {"none": 1000, "light": 2000, "heavy": 3000}),
    "apc": (35000, "heavy", {"none": 4000, "light": 1250, "heavy": 417}),
    "jeep": (15000, "light", {"none": 4000, "light": 1250, "heavy": 417}),
    "ftrk": (15000, "light", {"none": 1600, "light": 2400, "heavy": 400}),
    "arty": (10000, "light", {"none": 1500, "light": 1500, "heavy": 1500}),
    "v2rl": (20000, "light", {"none": 1500, "light": 1500, "heavy": 1500}),
}
_DEFAULT = (15000, "none", {"none": 1000, "light": 1000, "heavy": 1000})


def stats(kind: str):
    return UNITS.get(kind, _DEFAULT)


def armor_share(kinds) -> dict[str, float]:
    """Each armor class's share of the hit points in ``kinds``."""
    hp = {a: 0.0 for a in ARMOR_CLASSES}
    for k in kinds:
        h, armor, _ = stats(k)
        hp[armor] += h
    total = sum(hp.values())
    return {a: (v / total if total else 1.0 / len(ARMOR_CLASSES)) for a, v in hp.items()}


def power(ours, theirs) -> float:
    """Our damage per second aimed at their armor mix, times our hit points."""
    ours = [k for k in ours]
    if not ours:
        return 0.0
    share = armor_share(theirs)
    dps = sum(sum(share[a] * stats(k)[2][a] for a in ARMOR_CLASSES) for k in ours)
    hp = sum(stats(k)[0] for k in ours)
    return dps * hp


def edge(ours, theirs) -> float:
    """Our strength over theirs (1.0 = even), read like a value ratio."""
    ours, theirs = list(ours), list(theirs)
    a, b = power(ours, theirs), power(theirs, ours)
    if b <= 0:
        return float("inf") if a > 0 else 1.0
    return (a / b) ** 0.5


def matchup(view, ours, theirs) -> float:
    """How much better (>1) or worse (<1) our mix fights theirs than the
    prices say: :func:`edge` over the value ratio, kept within
    ``MATCHUP_LIMITS``. 1.0 when either side is too small to judge."""
    ours, theirs = list(ours), list(theirs)
    if len(ours) < C.MATCHUP_MIN_UNITS or len(theirs) < C.MATCHUP_MIN_UNITS:
        return 1.0
    cost = lambda k: view.unit_costs.get(k, C.KNOWN_COSTS.get(k, C.DEFAULT_UNIT_COST))  # noqa: E731
    value = sum(cost(k) for k in ours) / max(1.0, sum(cost(k) for k in theirs))
    lo, hi = C.MATCHUP_LIMITS
    return max(lo, min(hi, edge(ours, theirs) / value))


def rifles_per_rocket(theirs) -> int:
    """Riflemen per rocket soldier against this enemy: the usual 3:1 against
    infantry, down to 1:1 as their hit points shift into armor (rockets do
    20x a rifleman's damage to a tank)."""
    heavy = armor_share(theirs)["heavy"] if theirs else 0.0
    return max(1, round(C.RIFLE_TO_ROCKET - (C.RIFLE_TO_ROCKET - 1) * min(1.0, heavy / C.ARMOR_HEAVY_SHARE)))
