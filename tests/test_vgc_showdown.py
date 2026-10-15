"""agent/vgc/showdown.py against @smogon/calc (Showdown's damage calculator), all 16 rolls.

The reference rolls in tests/fixtures/damage_reference.json were produced by
@smogon/calc 0.10.0 (scripts/check_damage.py --write-fixture): every attack
among the draft cards seen, plus a sample under weather, terrain, screens,
Helping Hand, Battery, Power Spot, Friend Guard, boosts, burns,
and Sheer Force / Reckless attackers.
"""

import json
from pathlib import Path

import pytest

from agent.vgc import movelist, showdown

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "damage_reference.json").read_text(encoding="utf-8"))


def _battler(card: dict, boosts: dict, status: str = "") -> showdown.Battler:
    return showdown.Battler(card["species"], card.get("item", ""), card.get("ability", ""), card.get("nature", ""),
                            card.get("evs") or {}, card.get("ivs") or {}, boosts=boosts, status=status)


def _download(mine: dict, other: dict, boosts: dict) -> None:
    if mine.get("ability") == "Download":  # @smogon/calc applies it as on switch-in
        them = _battler(other, {})
        stat = "spa" if them.raw["spd"] <= them.raw["def"] else "atk"
        boosts[stat] = min(6, boosts.get(stat, 0) + 1)


def _rolls(row) -> tuple[list[int], list[int], str]:
    a_id, d_id, move, s, hits, ref = row
    a, d = FIXTURE["cards"][a_id], FIXTURE["cards"][d_id]
    atk = {"atk": s["atk_boost"], "spa": s["atk_boost"]} if "atk_boost" in s else {}
    dfn = {"def": s["def_boost"], "spd": s["def_boost"]} if "def_boost" in s else {}
    _download(a, d, atk)
    _download(d, a, dfn)
    fld = showdown.Field(weather=s.get("weather", ""), terrain=s.get("terrain", ""), reflect=bool(s.get("reflect")),
                         light_screen=bool(s.get("light_screen")), helping_hand=bool(s.get("helping_hand")),
                         battery=bool(s.get("battery")), power_spot=bool(s.get("power_spot")),
                         friend_guard=bool(s.get("friend_guard")))
    ours = showdown.damage_rolls(_battler(a, atk, s.get("status", "")), _battler(d, dfn), movelist.lookup(move)["name"], fld,
                                 hits=hits or None) or [0] * 16
    return ours, ref, f"{a['species']} {move} -> {d['species']} {s}"


def test_every_reference_attack_matches_showdown_roll_for_roll():
    misses = [label for ours, ref, label in map(_rolls, FIXTURE["cases"]) if ours != ref]
    assert not misses, f"{len(misses)} of {len(FIXTURE['cases'])} differ, e.g. {misses[:5]}"


@pytest.mark.parametrize("move,defender,expected", [
    ("freezedry", "pelipper", "super effective on Water"),
    ("hypervoice", "sableye", "Pixilate makes it Fairy, so it hits a Ghost"),
])
def test_the_special_cases_that_were_wrong_before(move, defender, expected):
    cards = FIXTURE["cards"]
    attacker = _battler(cards["ironbundle" if move == "freezedry" else "sylveon"], {})
    rolls = showdown.damage_rolls(attacker, _battler(cards[defender], {}), move)
    assert rolls and max(rolls) > 0, expected


def test_immunities_return_no_rolls():
    gholdengo = showdown.Battler("Gholdengo", "Air Balloon", "Good as Gold", "Modest", {"spa": 252})
    tinglu = showdown.Battler("Ting-Lu", "Leftovers", "Vessel of Ruin", "Careful", {"hp": 252})
    assert showdown.damage_rolls(tinglu, gholdengo, "earthquake") == []  # Air Balloon
    popped = showdown.Battler("Gholdengo", "", "Good as Gold", "Modest", {"spa": 252})
    assert showdown.damage_rolls(tinglu, popped, "earthquake")
    farigiraf = showdown.Battler("Farigiraf", "Sitrus Berry", "Armor Tail", "Quiet", {"hp": 252})
    chienpao = showdown.Battler("Chien-Pao", "Focus Sash", "Sword of Ruin", "Jolly", {"atk": 252})
    assert showdown.damage_rolls(chienpao, farigiraf, "suckerpunch") == []  # Armor Tail blocks priority
