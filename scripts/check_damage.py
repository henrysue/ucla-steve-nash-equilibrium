"""Check agent/vgc/showdown.py against @smogon/calc (Showdown's damage calculator), roll by roll.

Usage:
    .venv/Scripts/python.exe scripts/check_damage.py --node <node.exe> --calc <dir with node_modules/@smogon/calc>
        [--write-fixture] [--show N]

Builds every attack among the draft cards in data/card_sets.json (each card's
damaging moves into every other card), plus a sample under weather, terrain,
screens and boosts, runs them through @smogon/calc and through our engine, and
compares all 16 damage rolls. ``--write-fixture`` saves the reference results
to tests/fixtures/damage_reference.json, which the test suite checks without
needing Node.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.vgc import movelist, showdown  # noqa: E402
from agent.vgc.cards import CardSets  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "damage_reference.json"
REF_JS = r"""
const {calculate, Generations, Pokemon, Move, Field} = require('@smogon/calc');
const gen = Generations.get(9);
function mon(s) {
  return new Pokemon(gen, s.species, {level: 50, item: s.item || undefined, ability: s.ability || undefined,
    nature: s.nature || 'Serious', evs: s.evs || {}, ivs: s.ivs && Object.keys(s.ivs).length ? s.ivs : undefined,
    boosts: s.boosts || undefined, status: s.status || '', boostedStat: 'auto'});
}
function flat(d) {
  if (typeof d === 'number') return Array(16).fill(d);
  if (Array.isArray(d) && Array.isArray(d[0])) return d[0].map((_, i) => d.reduce((t, h) => t + h[i], 0));
  return d;
}
let input = '';
process.stdin.on('data', c => (input += c));
process.stdin.on('end', () => {
  process.stdout.write(JSON.stringify(JSON.parse(input).map(c => {
    try {
      const a = mon(c.attacker), d = mon(c.defender);
      const r = calculate(gen, a, d, new Move(gen, c.move), new Field(Object.assign({gameType: 'Doubles'}, c.field || {})));
      return {rolls: flat(r.damage), hits: r.move.hits, attacker_boosts: r.attacker.boosts, defender_boosts: r.defender.boosts};
    } catch (e) { return {error: String(e.message || e)}; }
  })));
});
"""
SMOGON_FIELD = {  # our Field -> @smogon/calc Field options
    "weather": {"sun": "Sun", "rain": "Rain", "sand": "Sand", "snow": "Snow"},
    "terrain": {"electric": "Electric", "grassy": "Grassy", "psychic": "Psychic", "misty": "Misty"},
}
SITUATIONS = [  # extra conditions, applied to a sample of the cases
    {"weather": "sun"}, {"weather": "rain"}, {"weather": "sand"}, {"weather": "snow"},
    {"terrain": "electric"}, {"terrain": "grassy"}, {"terrain": "psychic"}, {"terrain": "misty"},
    {"reflect": True}, {"light_screen": True}, {"helping_hand": True},
    {"atk_boost": 2}, {"def_boost": -1}, {"status": "brn"}, {"hp": 0.4},
    {"battery": True}, {"power_spot": True}, {"friend_guard": True},
]
# Attackers beyond the cards, for abilities no card has had yet.
EXTRA_ATTACKERS = [
    {"species": "Landorus", "item": "Life Orb", "ability": "Sheer Force", "nature": "Modest",
     "evs": {"hp": 4, "spa": 252, "spe": 252}, "moves": ["Earth Power", "Sludge Bomb", "Sandsear Storm", "Psychic"]},
    {"species": "Nidoking", "item": "Life Orb", "ability": "Sheer Force", "nature": "Timid",
     "evs": {"spa": 252, "spe": 252}, "moves": ["Earth Power", "Sludge Wave", "Ice Beam", "Flamethrower"]},
    {"species": "Staraptor", "item": "Choice Band", "ability": "Reckless", "nature": "Jolly",
     "evs": {"atk": 252, "spe": 252}, "moves": ["Brave Bird", "Double-Edge", "Close Combat", "Quick Attack"]},
]


def cases() -> list[dict]:
    sets = [v for v in CardSets(ROOT / "data" / "card_sets.json")._saved.values() if v.get("card_id") and v.get("evs")]
    found = []
    for a in sets + EXTRA_ATTACKERS:
        for d in sets:
            if a is d:
                continue
            for name in a["moves"]:
                move = movelist.lookup(name)
                if move and move["category"] in ("PHYSICAL", "SPECIAL"):
                    found.append({"attacker": a, "defender": d, "move": move["name"], "situation": {}})
    for i, case in enumerate(list(found[::7])):
        found.append({**case, "situation": SITUATIONS[i % len(SITUATIONS)]})
    return found


def _smogon_case(case: dict) -> dict:
    s = case["situation"]
    attacker = {**case["attacker"]}
    defender = {**case["defender"]}
    if "atk_boost" in s:
        attacker["boosts"] = {"atk": s["atk_boost"], "spa": s["atk_boost"]}
    if "def_boost" in s:
        defender["boosts"] = {"def": s["def_boost"], "spd": s["def_boost"]}
    if "status" in s:
        attacker["status"] = s["status"]
    field = {}
    if "weather" in s:
        field["weather"] = SMOGON_FIELD["weather"][s["weather"]]
    if "terrain" in s:
        field["terrain"] = SMOGON_FIELD["terrain"][s["terrain"]]
    if s.get("reflect"):
        field["defenderSide"] = {"isReflect": True}
    if s.get("light_screen"):
        field["defenderSide"] = {"isLightScreen": True}
    if s.get("helping_hand"):
        field["attackerSide"] = {"isHelpingHand": True}
    if s.get("battery"):
        field["attackerSide"] = {"isBattery": True}
    if s.get("power_spot"):
        field["attackerSide"] = {"isPowerSpot": True}
    if s.get("friend_guard"):
        field["defenderSide"] = {"isFriendGuard": True}
    out = {"attacker": attacker, "defender": defender, "move": case["move"], "field": field}
    if "hp" in s:
        out["defender"] = {**defender, "curHP_fraction": s["hp"]}  # not passed to smogon: see engine_rolls
    return out


def _boosts(case: dict) -> tuple[dict, dict]:
    """The boosts @smogon/calc starts from: the situation's, plus Download, which it
    always applies on 'switch-in' (read back from it, they would already include
    what a multi-hit move did to Stamina)."""
    s = case["situation"]
    atk = {"atk": s["atk_boost"], "spa": s["atk_boost"]} if "atk_boost" in s else {}
    dfn = {"def": s["def_boost"], "spd": s["def_boost"]} if "def_boost" in s else {}
    for mine, other, boosts in ((case["attacker"], case["defender"], atk), (case["defender"], case["attacker"], dfn)):
        if mine.get("ability") == "Download":
            them = showdown.Battler(other["species"], other.get("item", ""), other.get("ability", ""), other.get("nature", ""),
                                    other.get("evs") or {}, other.get("ivs") or {})
            stat = "spa" if them.raw["spd"] <= them.raw["def"] else "atk"
            boosts[stat] = min(6, boosts.get(stat, 0) + 1)
    return atk, dfn


def engine_rolls(case: dict, ref: dict) -> list[int]:
    s = case["situation"]
    a, d = case["attacker"], case["defender"]
    status = s.get("status", "")
    atk_boosts, def_boosts = _boosts(case)
    attacker = showdown.Battler(a["species"], a.get("item", ""), a.get("ability", ""), a.get("nature", ""), a.get("evs") or {},
                                a.get("ivs") or {}, boosts=atk_boosts, status=status)
    defender = showdown.Battler(d["species"], d.get("item", ""), d.get("ability", ""), d.get("nature", ""), d.get("evs") or {},
                                d.get("ivs") or {}, boosts=def_boosts)
    fld = showdown.Field(weather=s.get("weather", ""), terrain=s.get("terrain", ""), reflect=bool(s.get("reflect")),
                         light_screen=bool(s.get("light_screen")), helping_hand=bool(s.get("helping_hand")),
                         battery=bool(s.get("battery")), power_spot=bool(s.get("power_spot")),
                         friend_guard=bool(s.get("friend_guard")))
    return showdown.damage_rolls(attacker, defender, movelist.lookup(case["move"])["name"], fld, hits=ref.get("hits") or None) or [0] * 16


def reference(node: str, calc_dir: str, todo: list[dict]) -> list[dict]:
    script = Path(calc_dir) / "_check_damage_ref.js"
    script.write_text(REF_JS, encoding="utf-8")
    payload = [_smogon_case(c) for c in todo if "hp" not in c["situation"]]
    run = subprocess.run([node, str(script)], input=json.dumps(payload), capture_output=True, text=True, cwd=calc_dir, check=True)
    results = iter(json.loads(run.stdout))
    return [next(results) if "hp" not in c["situation"] else None for c in todo]


def compare(todo: list[dict], refs: list[dict | None], show: int) -> tuple[int, int, list]:
    exact = total = 0
    misses = []
    for case, ref in zip(todo, refs):
        if not ref or "error" in ref:
            continue
        total += 1
        ours = engine_rolls(case, ref)
        if list(ours) == list(ref["rolls"]):
            exact += 1
        else:
            misses.append((case, ours, ref["rolls"]))
    for case, ours, theirs in misses[:show]:
        print(f"  MISS {case['attacker']['species']} {case['move']} -> {case['defender']['species']} {case['situation']}: "
              f"ours {ours[0]}-{ours[-1]} ref {theirs[0]}-{theirs[-1]}")
    return exact, total, misses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True)
    parser.add_argument("--calc", required=True, help="directory with node_modules/@smogon/calc")
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()
    todo = cases()
    refs = reference(args.node, args.calc, todo)
    exact, total, _ = compare(todo, refs, args.show)
    print(f"{exact}/{total} attacks match Showdown's damage calculator on all 16 rolls ({exact / max(1, total):.1%})")
    if args.write_fixture:
        # Compact: each card once, cases by species id, only what the test needs.
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        cards, rows = {}, []
        for c, r in zip(todo, refs):
            if not r or "error" in r:
                continue
            for side in ("attacker", "defender"):
                cards[showdown._id(c[side]["species"])] = {k: c[side][k] for k in ("species", "item", "ability", "nature", "evs", "ivs")
                                                          if k in c[side]}
            rows.append([showdown._id(c["attacker"]["species"]), showdown._id(c["defender"]["species"]), c["move"],
                         c["situation"], r["hits"], r["rolls"]])
        FIXTURE.write_text(json.dumps({"cards": cards, "cases": rows}, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"wrote {len(rows)} reference cases -> {FIXTURE}")
    return 0 if exact == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
