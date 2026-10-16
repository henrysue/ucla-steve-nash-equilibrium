"""Draft pick priority: deny, counter, must picks, our first pick's partners, then strength.

Each draft pick follows these rules, in order:

0. DENY. A card in DENY is taken on sight, before anyone else can: Archaludon's
   draft card (Stamina + Leftovers Body Press) 2HKOs 30 of the pool's 46 other
   cards and only 13 2HKO it; opponents draft it first, and we were 2-9 against
   it (data/card_matchups.json; the ladder's 44% is a different set).
0b. COUNTER. When the opponent holds a card in COUNTER_TARGETS, a card on the
   board whose best hit one-hit KOs it (expected >= 100%) is taken: Chi-Yu's
   Overheat does 177% to Archaludon. (Sylveon's card has Hyper Voice, not Hyper
   Beam: 53% spread / 70% single, and it takes 47% back from Flash Cannon.)
1. MUST PICK. An easy-to-play, strong card with few weaknesses is taken on
   sight: its easy score (``easy_strength``: strength below, with weaknesses
   counted twice, pure supports and Trick Room setters marked down, Fake Out
   up) clears MUST_EASY_AT after the draft adjustments. We play a simple game
   plan - self-sufficient attackers, Fake Out, Intimidate, Tailwind or Icy
   Wind speed control - and never Trick Room (match f14f1652: a 44.6%
   Hatterene, our only setter, was forced into the four and set Trick Room at
   24% HP). The ladder win rate still counts, inside the score.
2. PARTNER. Otherwise, a card that pairs well with our FIRST pick is taken:
   the two win together at least PARTNER_MIN_WIN over PARTNER_MIN_GAMES games,
   and better than the first pick does on its own (only an easy card).
3. STRONGEST. Otherwise the generally strongest easy card: stats, typing
   (immunities, resistances vs weaknesses), a strong ability, its win rate when
   known (Dragonite: 600 total, Ground immunity, Multiscale), how its card does
   against the whole pool by exact damage (matchups.pool_net), how it trades
   with the opponent's drafted cards (matchups.versus: physical vs special,
   real stats, real moves), nudged by how well it fits the team so far
   (draft.score: plan, roles, shared weaknesses).

Every pick, rules 1-3, is adjusted for the bring (``draft_adjustment``): a
card our lineup rules would bench beside our picks (a third Pokemon weak to
one type, a second sharing a STAB type their picks resist) is marked down
(match f14f1652 drafted Great Tusk and Garchomp, then benched both); a
Tailwind / Icy Wind user is marked up once we hold none; physical attackers
are marked down once they hold two Intimidate users.

Every pick returns its reason; the strategist keeps them as the draft's
reasoning for Team Preview and the battle.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import matchups
from .typechart import TYPES, effectiveness

STATS_PATH = Path(__file__).resolve().parents[2] / "data" / "draft_stats.json"
MUST_PICK_AT = 51.5  # lower 90% bound on win rate, %
Z90 = 1.645
PARTNER_MIN_WIN, PARTNER_MIN_GAMES, PARTNER_EDGE = 52.0, 1000, 1.0
FIT_WEIGHT = 0.35  # rule 3: strength + this x team fit
POOL_WEIGHT = 10.0  # strength: net 2HKOs as a share of the pool (Iron Hands +28/46 = +6.1)
VERSUS_WEIGHT = 1.5  # rule 3: per KO-sized edge over one of their drafted cards
# The simple game plan (see rule 1).
MUST_EASY_AT = 5.5  # easy score a card needs to be taken on sight
EXTRA_WEAK, EXTRA_QUAD = 0.4, 0.8  # on top of strength's -0.4 / -0.4: weaknesses count twice
TRICK_ROOM_PENALTY = 5.0  # we do not play Trick Room
SUPPORT_PENALTY = 2.0  # one attack or fewer: needs a partner to do anything
FAKE_OUT_BONUS = 1.0  # easy, strong utility
PARTNER_MIN_EASY = 1.0
STACKED_PENALTY = 3.0  # would be our third Pokemon weak to a type (or second 4x)
SHARED_ATTACK_PENALTY = 1.5  # shares a STAB type with a pick, which half their picks resist
SPEED_CONTROL_BONUS = 2.0  # Tailwind / Icy Wind once we hold none (after two picks)
SPEED_CONTROL_MOVES = frozenset({"tailwind", "icywind", "electroweb"})
INTIMIDATE_PHYSICAL = 1.5  # physical attacker vs two of their Intimidates
DENY_MARGIN = 2.0  # a must pick this much stronger than the deny card is taken instead
DENY = ("archaludon",)
COUNTER_TARGETS = ("archaludon",)
STRONG_ABILITIES = {
    "multiscale", "intimidate", "regenerator", "levitate", "goodasgold", "prankster", "unseenfist", "armortail",
    "swordofruin", "beadsofruin", "tabletsofruin", "vesselofruin", "drought", "drizzle", "sandstream", "snowwarning",
    "psychicsurge", "grassysurge", "electricsurge", "mistysurge", "thickfat", "friendguard", "adaptability",
    "hugepower", "sheerforce", "galewings", "stamina", "wellbakedbody", "eartheater", "sapsipper", "flashfire",
    "waterabsorb", "voltabsorb", "lightningrod", "stormdrain", "protosynthesis", "quarkdrive", "guts", "toughclaws",
    "technician", "sharpness", "purifyingsalt", "innerfocus", "clearbody", "shadowshield", "dauntlessshield",
}


def _id(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


@lru_cache(maxsize=1)
def stats() -> dict:
    try:
        data = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"pokemon": {}, "_source": {}}
    return data


def entry(species: object) -> dict | None:
    return stats()["pokemon"].get(_id(species))


def games(species: object) -> int:
    e = entry(species)
    battles = stats().get("_source", {}).get("battles") or 0
    return int((e or {}).get("usage", 0) / 100 * battles * 2)


def win_floor(species: object) -> float | None:
    """Lower end of a 90% confidence interval on its win rate (%), or None without data."""
    e, n = entry(species), games(species)
    if not e or n <= 0:
        return None
    p = e["win_rate"] / 100
    return 100 * (p - Z90 * math.sqrt(p * (1 - p) / n))


def is_must_pick(species: object) -> bool:
    floor = win_floor(species)
    return floor is not None and floor >= MUST_PICK_AT


def pair(a: object, b: object) -> tuple[float, int] | None:
    """(pair win rate %, games together), from either Pokemon's partner lists."""
    best = None
    for x, y in ((a, b), (b, a)):
        for key in ("partners", "strong_partners", "weak_partners"):
            for row in (entry(x) or {}).get(key, []):
                if row["species"] == _id(y) and (best is None or row["games"] > best[1]):
                    best = (row["win_rate"], row["games"])
    return best


def complements(first: object, candidate: object) -> tuple[float, int] | None:
    """The pair's record if ``candidate`` is a proven partner for ``first``."""
    found = pair(first, candidate)
    if not found:
        return None
    win, n = found
    alone = max((entry(first) or {}).get("win_rate", 50.0), (entry(candidate) or {}).get("win_rate", 50.0))
    if win >= PARTNER_MIN_WIN and n >= PARTNER_MIN_GAMES and win >= alone + PARTNER_EDGE:
        return found
    return None


def strength(species: object, table, card: dict | None = None) -> tuple[float, list[str]]:
    """How strong a Pokemon is on its own: stats, typing, ability, win rate."""
    from . import showdown

    why = []
    data = showdown.species(species) or {}
    base = data.get("base") or table.base_stats(species) or {}
    total = sum(base.values()) if base else 0
    score = (total - 500) / 40 if total else 0.0
    if total:
        why.append(f"{total} base stat total")
    types = [t.upper() for t in (data.get("types") or table.species_types(species))]
    if types:
        immune = [t for t in TYPES if effectiveness(t, types) == 0]
        resists = [t for t in TYPES if 0 < effectiveness(t, types) < 1]
        weak = [t for t in TYPES if effectiveness(t, types) > 1]
        quad = [t for t in TYPES if effectiveness(t, types) >= 4]
        score += 1.0 * len(immune) + 0.2 * len(resists) - 0.4 * len(weak) - 0.4 * len(quad)
        if immune:
            why.append("immune to " + "/".join(t.title() for t in immune))
    ability = (card or {}).get("ability")
    abilities = [ability] if ability else (table.abilities(species) or data.get("abilities") or [])
    good = [a for a in abilities if _id(a) in STRONG_ABILITIES]
    if good and (ability or len(abilities) == 1):
        score += 1.0
        why.append(good[0])
    e = entry(species)
    if e:
        score += (e["win_rate"] - 50) * 0.5
        why.append(f"{e['win_rate']:.1f}% win rate")
    pool = matchups.pool_net(species)
    if pool and matchups.pool_size() > 1:
        net, kills, killed_by = pool
        score += POOL_WEIGHT * net / (matchups.pool_size() - 1)
        why.append(f"vs the pool: 2HKOs {kills}, 2HKO'd by {killed_by}")
    return round(score, 2), why


def _card_moves(species: object, table, card: dict | None = None) -> set[str]:
    card = card or (table.card_set(species) if hasattr(table, "card_set") else None) or {}
    moves = card.get("moves") or table.common_moves(species) or []
    return {_id(m) for m in moves}


def easy_strength(species: object, table, card: dict | None = None) -> tuple[float, list[str]]:
    """``strength`` for our simple game plan: weaknesses count twice, a pure
    support or a Trick Room setter is marked down, Fake Out is marked up."""
    from . import movelist, showdown

    score, why = strength(species, table, card)
    data = showdown.species(species) or {}
    types = [t.upper() for t in (data.get("types") or table.species_types(species))]
    if types:
        weak = [t for t in TYPES if effectiveness(t, types) > 1]
        quad = [t for t in TYPES if effectiveness(t, types) >= 4]
        score -= EXTRA_WEAK * len(weak) + EXTRA_QUAD * len(quad)
        why.append(f"{len(weak)} weakness{'es' if len(weak) != 1 else ''}" + (f" ({len(quad)} at 4x)" if quad else ""))
    moves = _card_moves(species, table, card)
    if "trickroom" in moves:
        score -= TRICK_ROOM_PENALTY
        why.append("a Trick Room setter (we do not play Trick Room)")
    attacks = [m for m in moves if (movelist.lookup(m) or {}).get("power")]
    if moves and len(attacks) <= 1:
        score -= SUPPORT_PENALTY
        why.append("a support (one attack)")
    if "fakeout" in moves:
        score += FAKE_OUT_BONUS
        why.append("Fake Out")
    return round(score, 2), why


def _physical(species: object, table, card: dict | None = None) -> bool | None:
    """Its attacks are mostly physical (True) or special (False); None when unknown."""
    from . import movelist

    kinds = [(movelist.lookup(m) or {}).get("category") for m in _card_moves(species, table, card)]
    physical, special = kinds.count("PHYSICAL"), kinds.count("SPECIAL")
    return None if physical == special else physical > special


def draft_adjustment(species: object, mine: list[str], theirs: list[str], table,
                     card: dict | None = None) -> tuple[float, list[str]]:
    """Points for a card given what we and they hold: bench risk, speed control, Intimidate."""
    from . import showdown

    def types_of(s):
        data = showdown.species(s) or {}
        return [t.upper() for t in (data.get("types") or table.species_types(s))]

    score, why = 0.0, []
    mine_types = {s: types_of(s) for s in mine}
    own = types_of(species)
    their_attacks = {t for s in theirs for t in types_of(s)}  # their STAB types: what the lineup rule checks
    if own:
        for attack in TYPES:
            if effectiveness(attack, own) <= 1 or (theirs and attack not in their_attacks):
                continue
            weak = [s for s, t in mine_types.items() if t and effectiveness(attack, t) > 1]
            quad = [s for s, t in mine_types.items() if t and effectiveness(attack, t) >= 4]
            if len(weak) >= 2 or (quad and effectiveness(attack, own) >= 4):
                score -= STACKED_PENALTY
                why.append(f"a third Pokemon weak to {attack.title()}, which they carry ({', '.join(weak)}): "
                           f"our lineup rules would bench one")
                break
        their_types = [t for t in (types_of(s) for s in theirs) if t]
        if len(their_types) >= 2:
            for attack in own:
                sharing = [s for s, t in mine_types.items() if attack in t]
                resist = sum(1 for t in their_types if effectiveness(attack, t) < 1)
                if sharing and resist * 2 >= len(their_types):
                    score -= SHARED_ATTACK_PENALTY
                    why.append(f"shares {attack.title()} with {', '.join(sharing)}, which {resist} of their "
                               f"{len(their_types)} resist")
                    break
    if len(mine) >= 2 and not any(_card_moves(s, table) & SPEED_CONTROL_MOVES for s in mine) \
            and _card_moves(species, table, card) & SPEED_CONTROL_MOVES:
        score += SPEED_CONTROL_BONUS
        why.append("speed control (Tailwind / Icy Wind), which we lack")
    intimidators = [s for s in theirs if _id((table.card_set(s) or {}).get("ability") if hasattr(table, "card_set") else "")
                    == "intimidate"]
    if len(intimidators) >= 2:
        kind = _physical(species, table, card)
        if kind is True:
            score -= INTIMIDATE_PHYSICAL
            why.append(f"physical into {len(intimidators)} Intimidates")
        elif kind is False:
            score += INTIMIDATE_PHYSICAL / 2
            why.append(f"special: their {len(intimidators)} Intimidates do not touch it")
    return round(score, 2), why


@dataclass
class Pick:
    species: str
    rule: str  # "deny" / "counter" / "must pick" / "partner" / "strongest"
    reason: str


def choose(board: list[str], mine: list[str], table, fit: dict[str, float] | None = None,
           cards: dict[str, dict] | None = None, theirs: list[str] | None = None) -> Pick | None:
    """Our pick from ``board`` (species on offer) given our roster so far and theirs."""
    if not board:
        return None
    fit = fit or {}
    cards = cards or {}
    theirs = list(theirs or [])
    denied = [s for s in board if _id(s) in DENY]
    if denied:
        # Match 37a1ee8a: denying Archaludon (5.5) gave them Iron Hands (9.6).
        scored = {s: easy_strength(s, table, cards.get(s))[0] + draft_adjustment(s, list(mine), theirs, table, cards.get(s))[0]
                  for s in board}
        stronger = [s for s in board if s not in denied and scored[s] >= MUST_EASY_AT
                    and scored[s] >= scored[denied[0]] + DENY_MARGIN]
        if stronger:
            denied = []
    if denied:
        best = denied[0]
        pool = matchups.pool_net(best)
        detail = f"2HKOs {pool[1]} pool cards, 2HKO'd by {pool[2]}; " if pool else ""
        return Pick(best, "deny", f"deny: {detail}taken before an opponent can draft it")
    targets = [t for t in theirs if _id(t) in COUNTER_TARGETS]
    counters = [(s, t, hit) for t in targets for s in board if (hit := matchups.one_hit_ko(s, t, table))]
    if counters:
        best, target, hit = max(counters, key=lambda c: (c[2]["pct"], fit.get(c[0], 0.0)))
        return Pick(best, "counter", f"counter: {hit['move']} one-hit KOs their {target} ({hit['pct']:.0f}%, "
                                     f"{str(hit.get('category', '')).lower() or 'attack'})")
    easy = {s: easy_strength(s, table, cards.get(s)) for s in board}
    adjust = {s: draft_adjustment(s, mine, theirs, table, cards.get(s)) for s in board}
    total = {s: easy[s][0] + adjust[s][0] for s in board}

    def notes(s):
        return "; ".join(easy[s][1][:5] + adjust[s][1])

    musts = [s for s in board if total[s] >= MUST_EASY_AT]
    if musts:
        best = max(musts, key=lambda s: (total[s], fit.get(s, 0.0)))
        return Pick(best, "must pick", f"must pick: easy to play and strong ({notes(best)}; score {total[best]:.1f})")
    if mine:
        first = mine[0]
        partners = [(s, rec) for s in board if total[s] >= PARTNER_MIN_EASY and (rec := complements(first, s))]
        if partners:
            best, (win, n) = max(partners, key=lambda item: (item[1][0], fit.get(item[0], 0.0)))
            return Pick(best, "partner", f"partner for our first pick {first}: the pair wins {win:.1f}% over {n:,} games")
    trades = {s: matchups.versus(s, theirs, table) for s in board} if theirs else {}
    best = max(board, key=lambda s: total[s] + VERSUS_WEIGHT * trades.get(s, (0.0, []))[0] + FIT_WEIGHT * fit.get(s, 0.0))
    reasons = [notes(best)]
    if best in trades:
        value, edges = trades[best]
        reasons.append(f"vs their picks {value:+.1f}" + (f": {'; '.join(edges)}" if edges else ""))
    return Pick(best, "strongest", f"strongest available ({', '.join(reasons)}; team fit {fit.get(best, 0.0):+.1f})")


def priority_table(species_list: list[str], table) -> list[dict]:
    """One row per species, for planning drafts: must pick?, win rate, best partners."""
    rows = []
    for s in species_list:
        e = entry(s) or {}
        partners = sorted(((p, rec) for p in species_list if p != s and (rec := complements(s, p))),
                          key=lambda item: -item[1][0])
        score, why = strength(s, table)
        pool = matchups.pool_net(s)
        rows.append({
            "species": s,
            "must_pick": is_must_pick(s),
            "win_rate": e.get("win_rate"),
            "win_rate_floor_90": None if win_floor(s) is None else round(win_floor(s), 2),
            "usage_pct": e.get("usage"),
            "games_est": games(s) or None,
            "tier": e.get("tier"),
            "strength": score,
            "strength_why": "; ".join(why),
            "partners": "; ".join(f"{p} ({w:.1f}% / {n:,} games)" for p, (w, n) in partners[:5]),
            "deny": _id(s) in DENY,
            "pool_2hkos": pool[1] if pool else None,
            "pool_2hkod_by": pool[2] if pool else None,
        })
    rows.sort(key=lambda r: (not r["deny"], not r["must_pick"], -(r["win_rate_floor_90"] or 0), -r["strength"]))
    return rows
