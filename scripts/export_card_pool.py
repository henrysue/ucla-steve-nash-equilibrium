"""Export every Pokemon seen in a VGC draft pool to a CSV, for planning drafts ahead.

Usage:
    .venv/Scripts/python.exe scripts/export_card_pool.py [out.csv]

Sources:
- data/card_sets.json: the exact set of every draft card the agent has seen on a
  board (species, item, ability, nature, EVs, four moves).
- matchlogs/*.jsonl and *.summary.json: which species we drafted, which we faced,
  the result of each match, and (from 2026-10-09 on) every draft board offered.
- The reference table for the format these species belong to (types, base stats,
  roles, usage tier, draft points).

Our own self-play matches (both seats logged for one session) are left out of the
win/loss columns, since we were on both sides.
"""

from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.vgc import meta  # noqa: E402
from agent.vgc.cards import CardSets  # noqa: E402

LOGS = ROOT / "matchlogs"
OUT = ROOT / "data" / "card_pool.csv"
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


def _session(path: Path) -> str | None:
    match = re.search(r"pokemon-vgc-doubles-draft-([0-9a-f-]{36})-seat(\d)", path.name)
    return match.group(1) if match else None


def _logs() -> dict:
    """Per species id: matches we picked it, faced it, boards it was on, and results."""
    picked, faced, boards = collections.defaultdict(set), collections.defaultdict(set), collections.Counter()
    record = collections.defaultdict(lambda: {"win": 0, "loss": 0})
    seats = collections.defaultdict(set)
    for path in LOGS.glob("*.jsonl"):
        session = _session(path)
        if session:
            seats[session].add(path.name[-11:-6])
    for path in sorted(LOGS.glob("*.jsonl")):
        session = _session(path)
        if not session:
            continue  # Red Alert, or an old test artifact
        ours = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            payload = row.get("payload")
            if row.get("kind") == "draft_pick" and isinstance(payload, str) and payload.startswith("draft_pick:vgc-"):
                ours.add(meta.species_id(payload.split("vgc-", 1)[1]))
            for card in row.get("board") or []:
                if isinstance(card, dict) and card.get("species"):
                    boards[meta.species_id(card["species"])] += 1
            prediction = ((row.get("analysis") or {}).get("prediction") or {})
            for foe in prediction:
                faced[meta.species_id(foe)].add(session)
        for species in ours:
            picked[species].add(session)
        summary = path.with_name(path.name.replace(".jsonl", ".summary.json"))
        if summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))
            for foe in data.get("opponent_species") or []:
                faced[meta.species_id(foe)].add(session)
            outcome = data.get("outcome")
            if outcome in ("win", "loss") and len(seats[session]) == 1:  # not self-play
                for species in ours:
                    record[species][outcome] += 1
    return {"picked": picked, "faced": faced, "boards": boards, "record": record}


def _evs(evs: object) -> str:
    if not isinstance(evs, dict):
        return ""
    names = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}
    return " / ".join(f"{v} {names.get(k, k)}" for k, v in evs.items() if v)


def main(out: Path = OUT) -> int:
    cards = CardSets(persist=False)
    cards._saved = CardSets()._saved  # read the saved catalog without ever writing it
    seen = _logs()
    species = set(cards._saved) | set(seen["picked"]) | set(seen["faced"]) | set(seen["boards"])
    names = [(cards.get(s) or {}).get("species") or s for s in species]
    table = cards.overlay(meta.best_table(names, meta.table()))

    rows = []
    for key in sorted(species):
        card = cards.get(key) or {}
        row = table.row(card.get("species") or key) or {}
        name = row.get("Pokemon") or card.get("species") or key  # the table's spelling, not an id
        moves = list(card.get("moves") or []) + [""] * 4
        types = table.species_types(name) + ["", ""]
        stats = table.base_stats(name) or {}
        roles = [r for r, on in table.roles(name).items() if on] if row else []
        wins, losses = seen["record"][key]["win"], seen["record"][key]["loss"]
        rows.append({
            "species": name,
            "card_id": card.get("card_id", ""),
            "set_source": "draft card" if card.get("card_id") else "team preview roster" if card else "species only (seen before sets were saved)",
            "item": card.get("item", ""),
            "ability": card.get("ability", ""),
            "nature": card.get("nature", ""),
            "evs": _evs(card.get("evs")),
            "move_1": moves[0], "move_2": moves[1], "move_3": moves[2], "move_4": moves[3],
            "type_1": types[0].title(), "type_2": types[1].title(),
            **{k: stats.get(k, "") for k in STAT_KEYS},
            "bst": sum(stats.values()) if stats else "",
            "roles": ", ".join(roles),
            "restricted": "yes" if table.is_restricted(name) else "",
            "usage_tier": row.get("Usage Tier", ""),
            "draft_points": table.draft_points(name) if row else "",
            "matches_we_picked_it": len(seen["picked"][key]),
            "matches_we_faced_it": len(seen["faced"][key]),
            "our_record_with_it": f"{wins}-{losses}" if wins or losses else "",
            "boards_seen_on": seen["boards"][key],
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} species -> {out} (reference table: {Path(table.source or '').name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT))
