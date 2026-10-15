"""Build data/pokedex.json, the damage engine's species data, from Pokémon Showdown's.

Usage:
    curl -L -o <somewhere>/pokedex.json https://play.pokemonshowdown.com/data/pokedex.json
    .venv/Scripts/python.exe -I scripts/build_pokedex.py <somewhere>/pokedex.json

Showdown runs the battles, so its data is the authority: base stats, types,
abilities, weight (Low Kick, Heavy Slam...) and whether a species can still
evolve (Eviolite). Only those fields are kept.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "pokedex.json"


def main(source: str) -> int:
    raw = json.loads(Path(source).read_text(encoding="utf-8"))
    out = {}
    for key, mon in raw.items():
        if not isinstance(mon, dict) or not mon.get("baseStats") or (mon.get("num") or 0) <= 0:
            continue
        out[key] = {
            "name": mon["name"],
            "types": [t.upper() for t in mon.get("types") or []],
            "base": mon["baseStats"],
            "abilities": sorted(set((mon.get("abilities") or {}).values())),
            "weightkg": mon.get("weightkg") or 0,
            "nfe": bool(mon.get("evos")),
        }
    OUT.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(out)} species -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
