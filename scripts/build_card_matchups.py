"""Every draft card against every other: exact damage grid plus a stats table.

Usage:
    .venv/Scripts/python.exe scripts/build_card_matchups.py

Reads data/card_sets.json (every draft card seen) and writes:
- data/card_matchups.json: for each attacker -> defender, the attacker's best hit as
  % of the defender's HP (Showdown's exact formula, neutral field, full HP, spread
  moves at the doubles 0.75), the move, and its category. pick_priority reads it.
- data/card_stats.csv: one row per card with base stats, the card's actual level-50
  stats (its EVs and nature), whether it attacks physically or specially, which of
  its defences is softer, and its record against the pool (cards it 2HKOs / is
  2HKO'd by).

Re-run it when new draft cards appear in data/card_sets.json.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.vgc import lookahead, meta, movelist, showdown  # noqa: E402
from agent.vgc.cards import CardSets  # noqa: E402

CARDS = ROOT / "data" / "card_sets.json"
GRID_OUT = ROOT / "data" / "card_matchups.json"
STATS_OUT = ROOT / "data" / "card_stats.csv"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
TWO_HIT = 50.0


def _style(card: dict) -> str:
    kinds = {(movelist.lookup(m) or {}).get("category") for m in card.get("moves") or []}
    physical, special = "PHYSICAL" in kinds, "SPECIAL" in kinds
    return "mixed" if physical and special else "physical" if physical else "special" if special else "support"


def main() -> None:
    cards = list(json.loads(CARDS.read_text(encoding="utf-8")).values())
    sets = CardSets(persist=False)
    sets.remember(cards)
    table = sets.overlay(meta.table())
    calc = lookahead.Calculator({"turn": 1, "active_pokemon": [], "opponent_active_pokemon": []}, table)
    names = [c["species"] for c in cards]

    damage: dict[str, dict] = {}
    for a in names:
        row = {}
        for d in names:
            if d == a:
                continue
            hit = calc.best_hit({"species": a}, {"species": d})
            if hit:
                category = (movelist.lookup(hit["move"]) or {}).get("category", "")
                row[d] = {"pct": round(hit["expected_pct"], 1), "move": hit["move"], "category": category}
        damage[a] = row

    stats = {}
    for card in cards:
        mon = showdown.Battler(card["species"], card.get("item", ""), card.get("ability", ""), card.get("nature", ""),
                               card.get("evs") or {}, card.get("ivs") or {})
        stats[card["species"]] = mon.raw
    GRID_OUT.write_text(json.dumps({
        "_source": {"built": date.today().isoformat(), "cards": len(names),
                    "how": "best hit, exact Showdown damage, neutral field, full HP, spread moves x0.75"},
        "stats": stats,
        "damage": damage,
    }, indent=1), encoding="utf-8")

    rows = []
    for card in cards:
        name = card["species"]
        data = showdown.species(name) or {}
        base = data.get("base") or {}
        actual = stats[name]
        kills = sum(v["pct"] >= TWO_HIT for v in damage[name].values())
        killed_by = sum(damage[a].get(name, {}).get("pct", 0) >= TWO_HIT for a in names if a != name)
        taken = [damage[a][name]["pct"] for a in names if a != name and name in damage[a]]
        item = showdown._id(card.get("item"))
        # Assault Vest x1.5 SpD; Eviolite x1.5 both (on a Pokemon that can still evolve).
        physical_bulk = actual["hp"] * actual["def"] * (1.5 if item == "eviolite" else 1.0)
        special_bulk = actual["hp"] * actual["spd"] * (1.5 if item in ("assaultvest", "eviolite") else 1.0)
        softer = "Def (hit it physically)" if physical_bulk < 0.85 * special_bulk else \
            "SpD (hit it specially)" if special_bulk < 0.85 * physical_bulk else "even"
        rows.append({
            "species": name,
            "types": "/".join(t.title() for t in data.get("types") or []),
            "attacks": _style(card),
            "item": card.get("item", ""), "ability": card.get("ability", ""), "nature": card.get("nature", ""),
            "evs": " / ".join(f"{v} {k}" for k, v in (card.get("evs") or {}).items() if v),
            "moves": ", ".join(card.get("moves") or []),
            **{f"base_{s}": base.get(s) for s in STATS},
            "base_total": sum(base.values()) if base else None,
            **{f"lv50_{s}": actual[s] for s in STATS},
            "physical_bulk": round(physical_bulk / 1000, 1),
            "special_bulk": round(special_bulk / 1000, 1),
            "softer_side": softer,
            "pool_2hkos": kills,
            "pool_2hkod_by": killed_by,
            "pool_net": kills - killed_by,
            "avg_pct_taken": round(sum(taken) / len(taken), 1) if taken else None,
        })
    rows.sort(key=lambda r: -r["pool_net"])
    with STATS_OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(names)} cards -> {GRID_OUT.name}, {STATS_OUT.name}")


if __name__ == "__main__":
    main()
