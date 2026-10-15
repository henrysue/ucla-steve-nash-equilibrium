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
1. MUST PICK. A card whose win rate stands out (data/draft_stats.json: VGC
   Regulation I ladder, ~112k battles) is taken on sight. "Stands out" uses
   the lower end of a 90% confidence interval, so a high rate from few games
   does not qualify: lower bound >= MUST_PICK_AT.
2. PARTNER. Otherwise, a card that pairs well with our FIRST pick is taken:
   the two win together at least PARTNER_MIN_WIN over PARTNER_MIN_GAMES games,
   and better than the first pick does on its own.
3. STRONGEST. Otherwise the generally strongest card: stats, typing
   (immunities, resistances vs weaknesses), a strong ability, its win rate when
   known (Dragonite: 600 total, Ground immunity, Multiscale), how its card does
   against the whole pool by exact damage (matchups.pool_net), how it trades
   with the opponent's drafted cards (matchups.versus: physical vs special,
   real stats, real moves), nudged by how well it fits the team so far
   (draft.score: plan, roles, shared weaknesses).

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
    musts = [s for s in board if is_must_pick(s)]
    if musts:
        best = max(musts, key=lambda s: (win_floor(s) or 0, fit.get(s, 0.0)))
        e = entry(best)
        return Pick(best, "must pick", f"must pick: {e['win_rate']:.1f}% win rate over ~{games(best):,} games "
                                       f"(at least {win_floor(best):.1f}% with 90% confidence)")
    if mine:
        first = mine[0]
        partners = [(s, rec) for s in board if (rec := complements(first, s))]
        if partners:
            best, (win, n) = max(partners, key=lambda item: (item[1][0], fit.get(item[0], 0.0)))
            return Pick(best, "partner", f"partner for our first pick {first}: the pair wins {win:.1f}% over {n:,} games")
    scored = {s: strength(s, table, cards.get(s)) for s in board}
    trades = {s: matchups.versus(s, theirs, table) for s in board} if theirs else {}
    best = max(board, key=lambda s: scored[s][0] + VERSUS_WEIGHT * trades.get(s, (0.0, []))[0]
               + FIT_WEIGHT * fit.get(s, 0.0))
    reasons = scored[best][1][:5]
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
