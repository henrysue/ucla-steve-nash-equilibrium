"""Export data/pick_priority.csv: the draft's must picks, partners and strength, per species.

Usage:
    .venv/Scripts/python.exe scripts/export_pick_priority.py [out.csv]

Rows: every species seen in a draft pool (data/card_sets.json, data/card_pool.csv)
plus every Pokemon in data/draft_stats.json, sorted must picks first, then by
win rate (90% lower bound) and strength. 'partners' lists the proven partners
(pair win rate >= 52% over 1,000+ games, better than either alone) among them.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.vgc import meta, pick_priority  # noqa: E402

OUT = ROOT / "data" / "pick_priority.csv"


def main(out: Path = OUT) -> int:
    seen = set(json.loads((ROOT / "data" / "card_sets.json").read_text(encoding="utf-8")))
    pool = ROOT / "data" / "card_pool.csv"
    if pool.exists():
        seen |= {meta.species_id(r["species"]) for r in csv.DictReader(pool.open(encoding="utf-8"))}
    in_stats = set(pick_priority.stats()["pokemon"])
    species = sorted(seen | {s for s in in_stats if s in seen or not s.endswith(("shadow", "ice"))})
    table = meta.table()
    rows = pick_priority.priority_table(species, table)
    for row in rows:
        row["seen_in_our_pools"] = row["species"] in seen
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} species -> {out}  (must picks: {', '.join(r['species'] for r in rows if r['must_pick'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT))
