"""Build data/draft_stats.json: win rates, partners and matchups for VGC Regulation I.

Usage:
    .venv/Scripts/python.exe scripts/build_draft_stats.py --cache <dir> [--refresh]

Source: showdowntier.com's "VGC 2026 Regulation I" pages (the same ruleset as
the platform's gen9vgc2025regi), built from ~112k Showdown battles. Per
Pokemon: win rate, usage, tier; partners with how often they are paired and the
pair's win rate ("Frequent Partners", "Strong Partners", "Challenging
Partners"); and the opponents it beats or struggles against. Pages are cached
in --cache, so a rebuild does not refetch them.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "draft_stats.json"
BASE = "https://showdowntier.com/formats/di/"
SECTIONS = {
    "Frequent Partners": "partners", "Strong Partners": "strong_partners", "Challenging Partners": "weak_partners",
    "Favorable Matchups": "beats", "Difficult Matchups": "loses_to",
}


def species_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def fetch(url: str, cache: Path, refresh: bool) -> str:
    path = cache / (re.sub(r"[^a-z0-9]+", "_", url.lower()) + ".html")
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (VGC draft research)"})
    with urllib.request.urlopen(request, timeout=40) as response:
        text = response.read().decode("utf-8", errors="replace")
    path.write_text(text, encoding="utf-8")
    time.sleep(0.6)  # be polite
    return text


def ranked(page: str) -> list[dict]:
    rows = []
    for match in re.finditer(r'<tr[^>]*data-name="([^"]+)"[^>]*data-tier="([^"]*)"[^>]*data-usage="([^"]*)"'
                             r'[^>]*data-win="([^"]*)"[^>]*data-rating="([^"]*)"[^>]*>(.*?)</tr>', page, re.S):
        name, tier, usage, win, rating, body = match.groups()
        href = re.search(r'href="(pokemon/[^"]+\.html)"', body)
        rows.append({"name": html.unescape(name), "tier": tier, "usage": float(usage), "win_rate": float(win),
                     "rating": float(rating), "href": href.group(1) if href else None})
    return rows


def sections(page: str) -> dict:
    found = {}
    heads = list(re.finditer(r"<h2[^>]*>([^<]+)</h2>", page))
    for i, head in enumerate(heads):
        key = SECTIONS.get(head.group(1).strip())
        if not key:
            continue
        body = page[head.end(): heads[i + 1].start() if i + 1 < len(heads) else len(page)]
        rows = []
        for row in re.finditer(r'/assets/images/pokemon/([a-z0-9-]+)\.png.*?<small>([\d,]+) games[^<]*?([\d.]+)% appear'
                               r'.*?<strong class="text-right[^"]*">([\d.]+)%</strong>', body, re.S):
            slug, games, appear, win = row.groups()
            rows.append({"species": species_id(slug), "games": int(games.replace(",", "")), "appear": float(appear),
                         "win_rate": float(win)})
        found[key] = rows
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    index = fetch(BASE + "index.html", cache, args.refresh)
    window = re.search(r"Analyze (\d+) battles from ([\d/]+) to ([\d/]+)", index)
    pokemon = {}
    for row in ranked(index):
        entry = {k: row[k] for k in ("name", "tier", "usage", "win_rate", "rating")}
        if row["href"]:
            try:
                entry.update(sections(fetch(BASE + row["href"], cache, args.refresh)))
            except OSError as exc:
                print(f"  could not fetch {row['name']}: {exc}", file=sys.stderr)
        pokemon[species_id(row["name"].replace("*", ""))] = entry
    OUT.write_text(json.dumps({
        "_source": {
            "site": BASE + "index.html", "format": "VGC Regulation I (gen9vgc2025regi ruleset)",
            "battles": int(window.group(1)) if window else None,
            "window": f"{window.group(2)} - {window.group(3)}" if window else None,
            "fetched": date.today().isoformat(),
        },
        "pokemon": pokemon,
    }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(pokemon)} Pokemon -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
