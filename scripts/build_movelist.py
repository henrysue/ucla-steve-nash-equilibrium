"""Build data/moves.json, the agent's move list, from Pokémon Showdown's move data.

Usage:
    curl -L -o <somewhere>/moves.json https://play.pokemonshowdown.com/data/moves.json
    .venv/Scripts/python.exe -I scripts/build_movelist.py <somewhere>/moves.json

The game runs on Showdown, so its data is the authority on each move's type,
power, accuracy, priority and, above all, WHO a move hits: "allAdjacent"
moves (Earthquake, Surf, Discharge...) hit our own partner too. Only the
fields the agent uses are kept, plus tags for anything beyond plain damage,
so a move with a reason to exist is never pruned as "dominated".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RELIABLE = 50  # % chance for a secondary effect to count
OUT = Path(__file__).resolve().parents[1] / "data" / "moves.json"
KEEP_FLAGS = ("contact", "sound", "bullet", "wind", "powder", "punch", "slicing", "pulse", "bite", "charge", "protect")
# Damage that depends on more than base power (HP, stats, items, history, field):
# never comparable on base power alone, so never "dominated".
VARIABLE_POWER = {
    "lastrespects", "ragefist", "weatherball", "eruption", "waterspout", "dragonenergy", "bodypress", "foulplay",
    "psyshock", "psystrike", "secretsword", "knockoff", "hex", "acrobatics", "facade", "storedpower", "powertrip",
    "heavyslam", "heatcrash", "lowkick", "grassknot", "gyroball", "electroball", "risingvoltage", "expandingforce",
    "terrainpulse", "boltbeak", "fishiousrend", "payback", "avalanche", "revenge", "brine", "venoshock", "ficklebeam",
    "temperflare", "stompingtantrum", "supercellslam", "hardpress", "crushgrip", "wringout", "beatup", "seismictoss",
    "nightshade", "superfang", "ruination", "finalgambit", "endeavor", "counter", "mirrorcoat", "metalburst",
    "comeuppance", "bulletpunch", "collisioncourse", "electrodrift", "solarbeam", "solarblade", "meteorbeam",
    "electroshot", "tripleaxel", "populationbomb", "terablast", "photongeyser", "luminacrash",
}


def tags(move_id: str, move: dict) -> list[str]:
    found = set()
    target = move.get("target")
    secondaries = list(move.get("secondaries") or []) + ([move["secondary"]] if move.get("secondary") else [])
    if move.get("priority", 0) > 0:
        found.add("priority")
    if target in ("allAdjacentFoes", "allAdjacent"):
        found.add("spread")
    if target == "allAdjacent":
        found.add("hits_ally")
    if move.get("selfSwitch"):
        found.add("pivot")
    if move.get("drain") or (move.get("flags") or {}).get("heal") or move.get("heal"):
        found.add("heal")
    if move.get("recoil") or move.get("mindBlownRecoil"):
        found.add("recoil")
    if move.get("multihit"):
        found.add("multihit")
    if (move.get("flags") or {}).get("charge"):
        found.add("charge")
    if (move.get("self") or {}).get("volatileStatus") == "mustrecharge":
        found.add("recharge")
    if any(v < 0 for v in ((move.get("self") or {}).get("boosts") or {}).values()) or any(
        v < 0 for v in (move.get("selfBoost") or {}).get("boosts", {}).values()
    ):
        found.add("self_drop")
    for sec in secondaries:
        # A 10% Flash Cannon drop is not a reason to use Flash Cannon: only
        # effects that land at least half the time count as utility.
        if sec.get("chance", 100) < RELIABLE:
            continue
        if sec.get("status"):
            found.add("status")
        if sec.get("volatileStatus") == "flinch":
            found.add("flinch")
        if any(v < 0 for v in (sec.get("boosts") or {}).values()):
            found.add("drops_target_stats")
            if (sec.get("boosts") or {}).get("spe", 0) < 0:
                found.add("speed_control")
        if any(v > 0 for v in ((sec.get("self") or {}).get("boosts") or {}).values()):
            found.add("self_boost")
    if move.get("category") == "Status":
        found.add("status_move")
        if move.get("status"):
            found.add("status")
        if any(v < 0 for v in (move.get("boosts") or {}).values()):
            found.add("drops_target_stats")
    if move.get("weather") or move.get("terrain") or move.get("pseudoWeather") or move.get("sideCondition"):
        found.add("field")
    if move_id in ("tailwind", "trickroom", "icywind", "electroweb", "thunderwave", "bulldoze", "rocktomb"):
        found.add("speed_control")
    if not (move.get("flags") or {}).get("protect") and move.get("category") != "Status" and target != "self":
        found.add("bypasses_protect")
    if move.get("ignoreAbility"):
        found.add("ignores_ability")
    if move.get("ohko"):
        found.add("ohko")
    if move_id in VARIABLE_POWER or move.get("damage"):
        found.add("variable_power")
    return sorted(found)


def trim(move_id: str, move: dict) -> dict:
    accuracy = move.get("accuracy")
    multihit = move.get("multihit")
    return {
        "name": move.get("name"),
        "type": (move.get("type") or "").upper(),
        "category": (move.get("category") or "").upper(),
        "power": move.get("basePower") or 0,
        "accuracy": None if accuracy is True else accuracy,  # None = never misses
        "priority": move.get("priority", 0),
        "target": move.get("target"),
        "multihit": multihit,
        "flags": sorted(f for f in KEEP_FLAGS if (move.get("flags") or {}).get(f)),
        "tags": tags(move_id, move),
        # Sheer Force boosts moves with secondary effects (and drops them).
        "secondary": bool(move.get("secondary") or move.get("secondaries")) or move_id in ("jetpunch", "orderup"),
        "recoil": move.get("recoil"),  # [numerator, denominator] of the damage dealt, or None
        "nonstandard": move.get("isNonstandard"),
    }


def main(source: str) -> None:
    raw = json.loads(Path(source).read_text(encoding="utf-8"))
    moves = {move_id: trim(move_id, move) for move_id, move in sorted(raw.items()) if isinstance(move, dict)}
    OUT.write_text(
        json.dumps({"_source": "play.pokemonshowdown.com/data/moves.json", "moves": moves}, indent=0, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(moves)} moves to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
