"""Draft cards against each other, by exact damage (data/card_matchups.json).

The ladder's win rates describe each species' usual ladder set; the draft hands
out one fixed card per species, sometimes a very different set (Archaludon's
card is Bold Stamina + Leftovers Body Press, not the ladder's rain attacker: 44%
on the ladder, yet only 13 of 46 cards 2HKO it while it 2HKOs 30). Exact damage
also sees physical vs special: a high Special Attack into a low Special Defense
(Chi-Yu's Overheat does 177% to Archaludon's 85 SpD; Body Press bounces off 200 Def).

- pool_net: cards it 2HKOs minus cards that 2HKO it, over the whole pool;
- best_hit: one card's best hit on another (grid, else live from the table);
- versus: how a card trades with the opponent's drafted roster.

Built by scripts/build_card_matchups.py; re-run it when new cards appear.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "data" / "card_matchups.json"
TWO_HIT = 50.0
ONE_HIT = 100.0


def _id(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


@lru_cache(maxsize=1)
def _grid() -> dict:
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"damage": {}, "stats": {}}
    damage = {_id(a): {_id(d): hit for d, hit in row.items()} for a, row in (data.get("damage") or {}).items()}
    return {"damage": damage, "stats": {_id(k): v for k, v in (data.get("stats") or {}).items()},
            "names": {_id(k): k for k in data.get("damage") or {}}}


def known(species: object) -> bool:
    return _id(species) in _grid()["damage"]


def stats(species: object) -> dict | None:
    """The card's actual level-50 stats (its EVs and nature)."""
    return _grid()["stats"].get(_id(species))


def best_hit(attacker: object, defender: object, table=None) -> dict | None:
    """{"pct", "move", "category"}: the attacker card's best hit on the defender card."""
    hit = _grid()["damage"].get(_id(attacker), {}).get(_id(defender))
    if hit is not None or table is None:
        return hit
    return _live(attacker, defender, table)


def _live(attacker: object, defender: object, table) -> dict | None:
    """A card outside the grid: the battle calculator, from the table's cards."""
    from . import lookahead, movelist

    try:
        calc = lookahead.Calculator({"turn": 1, "active_pokemon": [], "opponent_active_pokemon": []}, table)
        hit = calc.best_hit({"species": str(attacker)}, {"species": str(defender)})
    except Exception:  # noqa: BLE001 - no data: no matchup
        return None
    if not hit:
        return None
    return {"pct": round(hit["expected_pct"], 1), "move": hit["move"],
            "category": (movelist.lookup(hit["move"]) or {}).get("category", "")}


def pool_net(species: object) -> tuple[int, int, int] | None:
    """(net, 2HKOs, 2HKO'd by) over every other card in the pool; None if unknown."""
    grid = _grid()["damage"]
    me = _id(species)
    if me not in grid:
        return None
    kills = sum(hit["pct"] >= TWO_HIT for hit in grid[me].values())
    killed_by = sum(row.get(me, {}).get("pct", 0) >= TWO_HIT for other, row in grid.items() if other != me)
    return kills - killed_by, kills, killed_by


def pool_size() -> int:
    return len(_grid()["damage"])


def one_hit_ko(attacker: object, defender: object, table=None) -> dict | None:
    """The attacker's hit when it is an expected one-hit KO on the defender."""
    hit = best_hit(attacker, defender, table)
    return hit if hit and hit["pct"] >= ONE_HIT else None


def _faster(a: object, b: object) -> bool | None:
    """Does card ``a`` outspeed card ``b``? None on a tie or without stats."""
    sa, sb = stats(a), stats(b)
    if not sa or not sb or sa["spe"] == sb["spe"]:
        return None
    return sa["spe"] > sb["spe"]


def versus(species: object, theirs: list[str], table=None) -> tuple[float, list[str]]:
    """How this card trades with each of their drafted cards: our best hit minus
    theirs on us (each capped at a KO), in KOs per foe, summed. Reasons name
    the strongest edges, with physical / special."""
    value, edges = 0.0, []
    for foe in theirs:
        ours = best_hit(species, foe, table)
        back = best_hit(foe, species, table)
        if ours is None and back is None:
            continue
        dealt = min(ONE_HIT, (ours or {}).get("pct", 0.0))
        taken = min(ONE_HIT, (back or {}).get("pct", 0.0))
        # A faster one-hit KO lands first: the slower side never hits back
        # (Chi-Yu's Overheat beats Archaludon's Draco Meteor to it).
        faster = _faster(species, foe)
        if dealt >= ONE_HIT and faster is True:
            taken = 0.0
        elif taken >= ONE_HIT and faster is False:
            dealt = 0.0
        edge = (dealt - taken) / ONE_HIT
        value += edge
        if ours and abs(edge) >= 0.25:
            kind = str(ours.get("category", "")).lower()
            edges.append((abs(edge), f"{ours['move']} {ours['pct']:.0f}% into {foe}" + (f" ({kind})" if kind else "")
                          + f", takes {taken:.0f}%"))
    edges.sort(reverse=True)
    return round(value, 2), [text for _, text in edges[:2]]
