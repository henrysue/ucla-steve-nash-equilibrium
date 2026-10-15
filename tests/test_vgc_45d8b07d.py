"""Regression tests from match 45d8b07d (2026-10-11, seat 0): Ice-weak picks into Iron Bundle."""

from agent.vgc import draft, meta
from agent.vgc.cards import CardSets

from test_vgc_strategist import _agent

CARDS = [
    {"card_id": "vgc-iron-bundle", "species": "Iron Bundle", "item": "Booster Energy", "ability": "Quark Drive",
     "moves": ["Freeze-Dry", "Hydro Pump", "Encore", "Protect"]},
    {"card_id": "vgc-salamence", "species": "Salamence", "item": "Lum Berry", "ability": "Intimidate",
     "moves": ["Draco Meteor", "Hurricane", "Tailwind", "Protect"]},
    {"card_id": "vgc-garchomp", "species": "Garchomp", "item": "Focus Sash", "ability": "Rough Skin",
     "moves": ["Earthquake", "Rock Slide", "Iron Head", "Protect"]},
    {"card_id": "vgc-urshifu-rapid-strike", "species": "Urshifu-Rapid-Strike", "item": "Mystic Water", "ability": "Unseen Fist",
     "moves": ["Surging Strikes", "Close Combat", "Aqua Jet", "Protect"]},
    {"card_id": "vgc-whimsicott", "species": "Whimsicott", "item": "Focus Sash", "ability": "Prankster",
     "moves": ["Tailwind", "Moonblast", "Encore", "Light Screen"]},
]
MINE = ["salamence", "incineroar", "rillaboom", "urshifurapidstrike", "irontreads"]  # before pick 12
THEIRS = ["whimsicott", "archaludon", "ironbundle", "tornadus", "ironhands", "arcanine"]


def _table():
    sets = CardSets(persist=False)
    sets.remember(CARDS)
    return sets.overlay(meta.table())


def test_stacking_a_weakness_their_attacks_can_hit_costs_more():
    """Garchomp stacked Ice (Salamence, Rillaboom) and Fairy; Iron Bundle carries
    Freeze-Dry and Whimsicott Moonblast, while the Fire/Rock it 'covers' barely mattered."""
    table = _table()
    known, detail = draft._coverage("garchomp", MINE, table, THEIRS)
    blind, _ = draft._coverage("garchomp", MINE, table)
    assert "Ice" in detail["they_can_hit"]
    assert known < blind <= 0


def test_a_four_x_threat_outweighs_hitting_the_rest():
    table = _table()
    value, detail = draft.versus("garchomp", ["ironbundle"], table)
    assert detail["threat_moves"]["ironbundle"] == "Freeze-Dry 4x"
    assert value < 0  # Iron Head hits it 2x, but Freeze-Dry hits Garchomp 4x


def _lineup_agent(opponent):
    agent = _agent()
    agent._cards.remember(CARDS)
    agent._preview_opponent = opponent
    return agent


ROSTER = ["salamence", "incineroar", "rillaboom", "urshifurapidstrike", "irontreads", "garchomp"]


def test_two_four_x_ice_weaknesses_are_not_brought_into_iron_bundle():
    agent = _lineup_agent(THEIRS)
    problems = agent._lineup_problems(ROSTER, ["salamence", "incineroar", "irontreads", "garchomp"], ["incineroar", "salamence"])
    assert any("Ice" in p and "garchomp (4x)" in p for p in problems)
    best = agent._best_four(ROSTER)
    assert not ({"salamence", "garchomp"} <= set(best))


def test_the_weakness_rule_needs_an_attacker_of_that_type():
    agent = _lineup_agent(["archaludon", "ironhands", "arcanine"])  # no Ice attack among them
    assert not any("Ice" in p for p in agent._lineup_problems(
        ROSTER, ["salamence", "incineroar", "irontreads", "garchomp"], ["incineroar", "salamence"]))


def test_the_weakness_rule_never_rules_out_every_lineup():
    agent = _lineup_agent(THEIRS)
    roster = ["salamence", "garchomp", "dragonite", "rillaboom"]  # every four stacks Ice
    assert not any("weakness" in p for p in agent._lineup_problems(roster, roster, roster[:2]))
