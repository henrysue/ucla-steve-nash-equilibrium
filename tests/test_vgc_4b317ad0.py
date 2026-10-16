"""Regression tests from match 4b317ad0 (2026-10-15), a loss to Ace Trainer Claude.

Turn 1 the code switched Iron Valiant out to Dragonite under a predicted Flash
Cannon + sun Eruption; the strategist vetoed ("Iron Valiant stays out and eats
the hit") for a line 154 points worse, offered because the code's pick broke
"protect the plan" - and Iron Valiant fainted. Turn 9 Incineroar used Parting
Shot "to pivot out" with no one left to switch in, and fainted.
"""

import random

from agent.vgc import actions as act
from agent.vgc import decisions, lookahead, meta, principles, rules
from agent.vgc.cards import CardSets
from agent.vgc.predict import FoeExpectation
from agent.vgc.strategist import CLEAN_LINE_MARGIN, VGCDraftStrategist, _name_payload

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import BATTLE_OBSERVATION, POKEMON, FakeProvider, _doubles_action, _move, _state

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "iv", "species": "Iron Valiant", "ability": "Quark Drive", "item": "Booster Energy", "nature": "Timid",
     "evs": {"spa": 252, "spd": 4, "spe": 252}, "moves": ["Moonblast", "Close Combat", "Thunderbolt", "Protect"]},
    {"card_id": "dn", "species": "Dragonite", "ability": "Multiscale", "item": "Loaded Dice", "nature": "Adamant",
     "evs": {"hp": 252, "atk": 252, "spe": 4}, "moves": ["Extreme Speed", "Scale Shot", "Tailwind", "Protect"]},
    {"card_id": "ar", "species": "Archaludon", "ability": "Stamina", "item": "Leftovers", "nature": "Bold",
     "evs": {"hp": 252, "def": 252, "spa": 4}, "moves": ["Body Press", "Draco Meteor", "Flash Cannon", "Protect"]},
    {"card_id": "tk", "species": "Torkoal", "ability": "Drought", "item": "Charcoal",
     "moves": ["Eruption", "Heat Wave", "Earth Power", "Protect"]},
])
TABLE = SETS.overlay(meta.table())
F = [{"target": 1, "side": "opponent", "species": "archaludon"}, {"target": 2, "side": "opponent", "species": "torkoal"}]


def _turn_1():
    observation = {"turn": 1, "weather": {"SUNNYDAY": 5},
                   "active_pokemon": [{"species": "incineroar", "types": ["FIRE", "DARK"], "hp_fraction": 1.0},
                                      {"species": "ironvaliant", "types": ["FAIRY", "FIGHTING"], "hp_fraction": 1.0}],
                   "opponent_active_pokemon": [{"species": "archaludon", "types": ["STEEL", "DRAGON"], "hp_fraction": 1.0},
                                               {"species": "torkoal", "types": ["FIRE"], "hp_fraction": 1.0}],
                   "team": [{"species": s, "hp_fraction": 1.0} for s in ("incineroar", "ironvaliant", "dragonite", "pelipper")]}
    slots = act.parse_doubles(_as_legal(_doubles(
        "incineroar", [_move("fakeout", [1, 2], F)],
        "ironvaliant", [_move("closecombat", [1, 2], F), {"type": "switch", "species": "dragonite"}])))
    return observation, slots


# -- 3. covering a teammate is not throwing away the plan ------------------------------------------


def test_switching_dragonite_in_to_cover_iron_valiant_is_not_flagged():
    observation, slots = _turn_1()
    calc = lookahead.Calculator(observation, TABLE)
    expected = {"archaludon": FoeExpectation(attacks={"ironvaliant": 0.6}), "torkoal": FoeExpectation(attacks={"ironvaliant": 0.6})}
    judge = principles.Judge(slots, observation, calc, expected, preserve=["dragonite"])
    switch = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "fakeout", "target": 2},
              "slot_1": {"type": "switch", "species": "dragonite"}}
    cost, flags = judge.judge(switch)
    assert not any("game plan preserves" in f for f in flags)


# -- 2. the clean line beyond the margin is capped --------------------------------------------------


def test_a_far_worse_clean_line_is_not_offered():
    provider = FakeProvider({"verdict": "play", "play_option": "1", "why": "", "reasoning_summary": ""})
    agent = VGCDraftStrategist(provider, log=[].append, rng=random.Random(1), clock=lambda: 0.0)
    agent._line_flags = [["switches dragonite (the game plan preserves it)"], ["x"], []]
    facts = {k: None for k in ("our_actives", "their_actives", "their_threats", "our_threats", "move_order",
                               "their_possible_switch_ins", "game_plan", "recent_turns")}
    ranked = [(-31.8, {}), (-60.0, {}), (-31.8 - CLEAN_LINE_MARGIN - 70, {})]  # 154 worse, as in turn 1
    played, review = agent._review(ranked, {}, "", facts, _state(_doubles_action(), observation=BATTLE_OBSERVATION),
                                   POKEMON, 0.0)
    options = provider.calls[0]["messages"][1]["content"] if provider.calls else ""
    assert '"breaks_principles": "none"' not in options  # the far-worse clean line was not shown
    assert played in (0, 1)


# -- 4. labels: a pivot with no one to switch in, and what a switch-in does --------------------------


def test_parting_shot_with_no_bench_is_labelled():
    stuck = act.parse_doubles(_as_legal(_doubles("incineroar", [_move("partingshot", [1, 2], F)], "dragonite",
                                                 [_move("scaleshot", [1, 2], F)])))
    payload = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "partingshot", "target": 1},
               "slot_1": {"type": "move", "move_id": "scaleshot", "target": 2}}
    assert "Parting Shot (no one to switch in: it stays)" in _name_payload(payload, stuck)
    free = act.parse_doubles(_as_legal(_doubles("incineroar", [_move("partingshot", [1, 2], F),
                                                               {"type": "switch", "species": "pelipper"}],
                                                "dragonite", [_move("scaleshot", [1, 2], F)])))
    assert "no one to switch in" not in _name_payload(payload, free)


def test_a_switch_is_labelled_with_who_takes_the_hits():
    observation, slots = _turn_1()
    payload = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "fakeout", "target": 2},
               "slot_1": {"type": "switch", "species": "dragonite"}}
    assert "switch to dragonite (it takes the hits aimed here)" in _name_payload(payload, slots)


# -- 1. the final call knows how switching works ------------------------------------------------------


def test_the_final_call_knows_the_switch_in_takes_the_hits():
    text = decisions.review_decision([{"option": 1, "value": 0, "line": "x"}], {}).prompt["instructions"]
    assert "switching in takes every hit aimed at that slot" in text and "the one leaving takes none" in text
