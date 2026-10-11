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
:func:`odds` is how every fight is judged — attacks, retreats, meeting a
push, raids, the base's defense — so they all read the same numbers.
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
    # Turrets: 40000 hp, heavy armor; damage per second approximated from
    # their weapons (pillbox machine gun, flame tower, cannon turret, tesla).
    # They fight in a battle like a unit that doesn't move.
    "pbox": (40000, "heavy", {"none": 4000, "light": 1200, "heavy": 400}),
    "hbox": (40000, "heavy", {"none": 4000, "light": 1200, "heavy": 400}),
    "ftur": (40000, "heavy", {"none": 6000, "light": 2000, "heavy": 1000}),
    "gun": (40000, "heavy", {"none": 800, "light": 2000, "heavy": 3500}),
    "tsla": (40000, "heavy", {"none": 6000, "light": 4000, "heavy": 4000}),
}
_DEFAULT = (15000, "none", {"none": 1000, "light": 1000, "heavy": 1000})


def stats(kind: str):
    return UNITS.get(kind, _DEFAULT)


def _weights(kinds) -> dict[str, float]:
    """``kinds`` as {type: count}; a dict passes through (counts may be fractions)."""
    if isinstance(kinds, dict):
        return {k: float(n) for k, n in kinds.items() if n > 0}
    out: dict[str, float] = {}
    for k in kinds:
        out[k] = out.get(k, 0.0) + 1.0
    return out


def armor_share(kinds) -> dict[str, float]:
    """Each armor class's share of the hit points in ``kinds``."""
    hp = {a: 0.0 for a in ARMOR_CLASSES}
    for k, n in _weights(kinds).items():
        h, armor, _ = stats(k)
        hp[armor] += h * n
    total = sum(hp.values())
    return {a: (v / total if total else 1.0 / len(ARMOR_CLASSES)) for a, v in hp.items()}


def power(ours, theirs) -> float:
    """Our damage per second aimed at their armor mix, times our hit points."""
    ours = _weights(ours)
    if not ours:
        return 0.0
    share = armor_share(theirs)
    dps = sum(n * sum(share[a] * stats(k)[2][a] for a in ARMOR_CLASSES) for k, n in ours.items())
    hp = sum(n * stats(k)[0] for k, n in ours.items())
    return dps * hp


def edge(ours, theirs) -> float:
    """Our strength over theirs (1.0 = even), read like a value ratio."""
    ours, theirs = _weights(ours), _weights(theirs)
    a, b = power(ours, theirs), power(theirs, ours)
    if b <= 0:
        return float("inf") if a > 0 else 1.0
    return (a / b) ** 0.5


def cost(view, kind: str) -> float:
    return view.unit_costs.get(kind) or C.KNOWN_COSTS.get(kind, C.DEFAULT_UNIT_COST)


def value(view, kinds) -> float:
    """The price of ``kinds``."""
    return sum(n * cost(view, k) for k, n in _weights(kinds).items())


def odds(view, ours, theirs, believed: float = 0.0) -> float:
    """The one way every fight is judged: our strength over theirs (1.0 =
    even, read like a value ratio), from what each side is made of
    (:func:`edge`), turrets included. ``believed``: the enemy army value we
    believe in (seen lately) — what we can't place among the units we know
    counts as a mirror of our own mix at that value, so with little known
    this reads as the value ratio and with everything known as the real
    matchup. Nothing of theirs at all: inf (0 without an army of ours)."""
    ours, theirs = _weights(ours), _weights(theirs)
    if not ours:
        return 1.0 if not theirs and believed <= 0 else 0.0   # nothing against nothing: even
    unseen = believed - value(view, {k: n for k, n in theirs.items() if k not in TURRETS})
    if unseen > 0:
        scale = unseen / max(1.0, value(view, ours))
        theirs = dict(theirs)
        for k, n in ours.items():
            theirs[k] = theirs.get(k, 0.0) + n * scale
    if not theirs:
        return float("inf")
    return edge(ours, theirs)


def enemy_kinds(view) -> list[str]:
    """The enemy fighters we see or remember, by type (no harvesters or MCVs)."""
    skip = set(C.HARVESTER) | {C.MCV}
    seen = {a.id: a.type for a in list(view.known_enemy_units) + list(view.enemy_units)
            if a.id is not None and not a.wreck and a.type not in skip}
    return list(seen.values())


def army_odds(view, turrets=(), busy_ids=frozenset(), busy_value: float = 0.0) -> float:
    """Our whole army against theirs: the units we see or remember, the
    most we saw lately (``enemy_army_peak``), plus ``turrets`` (the enemy
    turrets defending a target), less ``busy_ids`` (worth ``busy_value``:
    their units held elsewhere by our raid squads)."""
    believed = max(view.enemy_army_value, view.enemy_army_peak, 0.0) - busy_value
    if not view.combat_units:
        # No unit list (only the server's figures): the value ratio.
        theirs_value = max(0.0, believed) + value(view, turrets)
        if theirs_value <= 0:
            return float("inf") if view.army_value > 0 else 1.0
        return view.army_value / theirs_value
    skip = set(C.HARVESTER) | {C.MCV}
    theirs = {a.id: a.type for a in list(view.known_enemy_units) + list(view.enemy_units)
              if a.id is not None and not a.wreck and a.type not in skip and a.id not in busy_ids}
    return odds(view, [u.type for u in view.combat_units], list(theirs.values()) + list(turrets), believed)


TURRETS = frozenset(C.TURRETS)


def rifles_per_rocket(theirs) -> int:
    """Riflemen per rocket soldier against this enemy: the usual 3:1 against
    infantry, down to 1:1 as their hit points shift into armor (rockets do
    20x a rifleman's damage to a tank)."""
    heavy = armor_share(theirs)["heavy"] if theirs else 0.0
    return max(1, round(C.RIFLE_TO_ROCKET - (C.RIFLE_TO_ROCKET - 1) * min(1.0, heavy / C.ARMOR_HEAVY_SHARE)))
