"""Draft by exact card-vs-card damage: deny Archaludon, counter it, physical vs special.

Numbers come from data/card_matchups.json (scripts/build_card_matchups.py).
"""

import csv
from pathlib import Path

from agent.vgc import matchups, meta, pick_priority as pp

TABLE = meta.table()
ROOT = Path(__file__).resolve().parents[1]


def test_archaludon_s_card_is_strong_against_the_pool_despite_its_ladder_rate():
    net, kills, killed_by = matchups.pool_net("Archaludon")
    assert kills >= 25 and killed_by <= 15
    assert pp.entry("Archaludon")["win_rate"] < 45  # the ladder's (different) set
    # Same 600 total and ladder rate as Tyranitar; the card is what differs.
    assert pp.strength("Archaludon", TABLE)[0] > pp.strength("Tyranitar", TABLE)[0] + 4


def test_archaludon_is_taken_on_sight_even_over_a_must_pick():
    pick = pp.choose(["Grimmsnarl", "Chi-Yu", "Archaludon"], ["Iron Hands"], TABLE)
    assert (pick.species, pick.rule) == ("Archaludon", "deny")


def test_their_archaludon_is_countered_by_a_card_that_one_hit_kos_it():
    pick = pp.choose(["Grimmsnarl", "Chi-Yu", "Sylveon"], ["Iron Hands"], TABLE, theirs=["Archaludon"])
    assert (pick.species, pick.rule) == ("Chi-Yu", "counter")
    assert "Overheat" in pick.reason and "special" in pick.reason


def test_sylveon_s_card_does_not_one_hit_ko_archaludon():
    """Its card runs Hyper Voice (spread), not Hyper Beam."""
    assert matchups.best_hit("Sylveon", "Archaludon")["move"] == "Hyper Voice"
    assert matchups.one_hit_ko("Sylveon", "Archaludon") is None
    pick = pp.choose(["Sylveon", "Garchomp"], ["Iron Hands"], TABLE, theirs=["Archaludon"])
    assert pick.rule != "counter"


def test_special_attackers_are_drafted_into_low_special_defense():
    """Archaludon: 200 Def, 85 SpD. Torkoal's special Eruption does twice what
    Arcanine's physical Flare Blitz does - both Fire, neutral."""
    torkoal, arcanine = matchups.best_hit("Torkoal", "Archaludon"), matchups.best_hit("Arcanine", "Archaludon")
    assert torkoal["category"] == "SPECIAL" and arcanine["category"] == "PHYSICAL"
    assert torkoal["pct"] > 2 * arcanine["pct"]
    assert matchups.versus("Torkoal", ["Archaludon"])[0] > matchups.versus("Arcanine", ["Archaludon"])[0]


def test_a_faster_one_hit_ko_wins_the_trade():
    value, _ = matchups.versus("Chi-Yu", ["Archaludon"])
    assert value == 1.0  # Overheat lands first; Draco Meteor never does


def test_their_picks_steer_the_strongest_card():
    """Dondozo and Flutter Mane are level on strength; Dondozo's physical hits
    bounce off Archaludon's 200 Def, so their Archaludon tips it."""
    board = ["Dondozo", "Flutter Mane"]
    assert pp.choose(board, ["Iron Hands"], TABLE).species == "Dondozo"
    pick = pp.choose(board, ["Iron Hands"], TABLE, theirs=["Archaludon"])
    assert pick.species == "Flutter Mane" and "vs their picks" in pick.reason


def test_the_stats_table_has_every_card_with_real_level_50_stats():
    rows = {r["species"]: r for r in csv.DictReader((ROOT / "data" / "card_stats.csv").open(encoding="utf-8"))}
    assert len(rows) == matchups.pool_size()
    archaludon = rows["Archaludon"]
    assert (archaludon["lv50_def"], archaludon["lv50_spd"]) == ("200", "85")
    assert archaludon["softer_side"].startswith("SpD")
