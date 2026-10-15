"""Regression tests from match 286bbc2e (2026-10-13), built from its real boards.

Lost 2 KOs to 4 with reads that were mostly right (attack reads 85%): the
losses were decisions. Grimmsnarl never set a screen, Farigiraf (the Trick
Room setter) was switched into Close Combat, both attacks went into a
first-turn Iron Valiant that Protected, a fresh Sneasler's Fake Out was read
as Protect, and two Draco Meteors left Archaludon at -4 SpA.
"""

import json
import random

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, predict, principles, protect, race, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker
from agent.vgc.strategist import VETO_MARGIN, VGCDraftStrategist

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import BATTLE_OBSERVATION, POKEMON, FakeProvider, _doubles_action, _move, _state

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "g", "species": "Grimmsnarl", "ability": "Prankster", "moves": ["Spirit Break", "Thunder Wave", "Reflect", "Light Screen"]},
    {"card_id": "i", "species": "Incineroar", "ability": "Intimidate", "moves": ["Fake Out", "Knock Off", "Flare Blitz", "Parting Shot"]},
    {"card_id": "b", "species": "Basculegion", "ability": "Adaptability", "moves": ["Wave Crash", "Liquidation", "Aqua Jet", "Protect"]},
    {"card_id": "c", "species": "Cresselia", "ability": "Levitate", "moves": ["Icy Wind", "Moonblast", "Helping Hand", "Protect"]},
    {"card_id": "a", "species": "Archaludon", "ability": "Stamina", "moves": ["Body Press", "Draco Meteor", "Flash Cannon", "Protect"]},
    {"card_id": "f", "species": "Farigiraf", "ability": "Armor Tail", "moves": ["Trick Room", "Psychic", "Dazzling Gleam", "Helping Hand"]},
    {"card_id": "s", "species": "Sneasler", "ability": "Unburden", "moves": ["Fake Out", "Close Combat", "Dire Claw", "Protect"]},
    {"card_id": "v", "species": "Iron Valiant", "ability": "Quark Drive", "moves": ["Moonblast", "Close Combat", "Thunderbolt", "Protect"]},
])
TABLE = SETS.overlay(meta.table())
TYPES = {"grimmsnarl": ["DARK", "FAIRY"], "incineroar": ["FIRE", "DARK"], "basculegion": ["WATER", "GHOST"],
         "cresselia": ["PSYCHIC"], "archaludon": ["STEEL", "DRAGON"], "farigiraf": ["NORMAL", "PSYCHIC"],
         "sneasler": ["FIGHTING", "POISON"], "ironvaliant": ["FAIRY", "FIGHTING"]}
PRESERVE = ["grimmsnarl", "farigiraf"]  # the match's game plan
REVIEW_FACTS = {k: None for k in ("our_actives", "their_actives", "their_threats", "our_threats", "move_order",
                                  "their_possible_switch_ins", "game_plan", "recent_turns")}


def _obs(ours, theirs, hp=None, turn=1, boosts=None):
    hp, boosts = hp or {}, boosts or {}
    return {"turn": turn,
            "active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0), "boosts": boosts.get(s, {})}
                               for s in ours],
            "opponent_active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in theirs],
            "team": [{"species": s, "hp_fraction": hp.get(s, 1.0)} for s in ("grimmsnarl", "incineroar", "archaludon", "farigiraf")]}


def _turn(action, observation, read, first_turn):
    """(judge, ranked payloads, code guess) the way the strategist builds them."""
    slots = act.parse_doubles(_as_legal(action))
    tracker = MoveTracker(known_moves=TABLE.common_moves, exact_moves=lambda s: (TABLE.card_set(s) or {}).get("moves"))
    turn_rules = rules.build(slots, observation, tracker, TABLE, lambda line: None)
    calc = lookahead.Calculator(observation, TABLE)
    odds = protect.odds(observation, first_turn=first_turn, common_moves=TABLE.common_moves, card_set=TABLE.card_set)
    ours = [e["species"] for e in observation["active_pokemon"]]
    base = predict.code_guess(observation, calc, tracker, TABLE, odds)
    expected = predict.blend(predict.expectations(read, ours), predict.expectations(base, ours))
    judge = principles.Judge(slots, observation, calc, expected, preserve=PRESERVE)
    ranked = lookahead.ranked_lines(turn_rules, observation, calc, expected, cost=lambda p: judge.judge(p)[0])
    return judge, [turn_rules.check(a) for _, a in ranked], base


F1 = [{"target": 1, "side": "opponent", "species": "basculegion"}, {"target": 2, "side": "opponent", "species": "cresselia"}]
TURN_1 = _doubles(
    "grimmsnarl", [_move("spiritbreak", [1, 2], F1), _move("thunderwave", [1, 2], F1), _move("reflect", []), _move("lightscreen", [])],
    "incineroar", [_move("fakeout", [1, 2], F1), _move("knockoff", [1, 2], F1), _move("flareblitz", [1, 2], F1),
                   {"type": "switch", "species": "archaludon"}, {"type": "switch", "species": "farigiraf"}])
F4 = [{"target": 1, "side": "opponent", "species": "sneasler"}, {"target": 2, "side": "opponent", "species": "cresselia"}]
TURN_5 = _doubles("incineroar", [_move("knockoff", [1, 2], F4), _move("flareblitz", [1, 2], F4), {"type": "switch", "species": "farigiraf"}],
                  "archaludon", [_move("bodypress", [1, 2], F4), _move("dracometeor", [1, 2], F4), _move("protect", [])])
F8 = [{"target": 1, "side": "opponent", "species": "ironvaliant"}, {"target": 2, "side": "opponent", "species": "cresselia"}]
TURN_8 = _doubles(
    "incineroar", [_move("fakeout", [1, 2], F8), _move("knockoff", [1, 2], F8), _move("flareblitz", [1, 2], F8)],
    "archaludon", [_move("bodypress", [1, 2], F8), _move("dracometeor", [1, 2], F8), _move("flashcannon", [1, 2], F8), _move("protect", [])])


def _move_row(move_id, target):
    return {"type": "move", "move_id": move_id, "target": target}


# -- principle 1: protect the plan ---------------------------------------------------------


def test_turn_5_never_switches_the_trick_room_setter_into_close_combat():
    observation = _obs(["incineroar", "archaludon"], ["sneasler", "cresselia"], hp={"incineroar": 0.79, "sneasler": 0.6},
                       turn=5, boosts={"archaludon": {"spa": -2}})
    read = {"sneasler": {"action": "attack", "target": "incineroar", "confidence": 0.6},
            "cresselia": {"action": "attack", "target": "archaludon", "confidence": 0.55}}
    judge, lines, _ = _turn(TURN_5, observation, read, {})
    switch = {"type": "doubles_turn", "slot_0": {"type": "switch", "species": "farigiraf"}, "slot_1": _move_row("bodypress", 1)}
    cost, flags = judge.judge(switch)
    assert cost > 0 and "farigiraf" in flags[0]
    assert all(line["slot_0"].get("species") != "farigiraf" for line in lines)


# -- principle 2: do your job --------------------------------------------------------------


def test_turn_1_flags_a_prankster_screen_setter_that_does_something_else():
    read = {"basculegion": {"action": "attack", "target": "incineroar", "confidence": 0.55},
            "cresselia": {"action": "support", "confidence": 0.4}}
    judge, _, _ = _turn(TURN_1, _obs(["grimmsnarl", "incineroar"], ["basculegion", "cresselia"]), read,
                        {"basculegion": True, "cresselia": True})
    played = {"type": "doubles_turn", "slot_0": _move_row("thunderwave", 1), "slot_1": {"type": "switch", "species": "archaludon"}}
    screen = {**played, "slot_0": _move_row("reflect", 0)}
    assert any("Prankster reflect" in f for f in judge.judge(played)[1])  # Basculegion attacks physically
    assert judge.judge(screen) == (0.0, [])


def test_a_prankster_screen_cuts_hits_the_same_turn():
    calc = lookahead.Calculator(_obs(["grimmsnarl", "incineroar"], ["basculegion", "cresselia"]), TABLE)
    sim = race._Sim(calc, race.Board(), [], [], {})
    sim._status(race.Mon("us", {"species": "grimmsnarl"}, 100.0), "reflect", [])
    assert sim.board.screens["us"]["PHYSICAL"] == 1.0 and sim.next_board.screens["us"]["PHYSICAL"] == 1.0


# -- principle 3: stay robust; reads come second -------------------------------------------


def test_a_read_cannot_erase_a_first_turn_protect():
    read = {"ironvaliant": {"action": "attack", "target": "incineroar", "confidence": 0.55}}
    base = {"ironvaliant": {"action": "protect", "target": None, "confidence": 0.7}}
    ours = ["incineroar", "archaludon"]
    mixed = predict.blend(predict.expectations(read, ours), predict.expectations(base, ours))
    assert abs(mixed["ironvaliant"].protect - predict.READ_WEIGHT * 0 - (1 - predict.READ_WEIGHT) * 0.7) < 1e-9


def test_turn_8_does_not_put_both_attacks_into_a_first_turn_iron_valiant():
    observation = _obs(["incineroar", "archaludon"], ["ironvaliant", "cresselia"],
                       hp={"incineroar": 0.79, "archaludon": 0.83, "cresselia": 0.68}, turn=8,
                       boosts={"archaludon": {"spa": -4, "def": 3}})
    read = {"ironvaliant": {"action": "attack", "target": "incineroar", "confidence": 0.55},
            "cresselia": {"action": "attack", "target": "archaludon", "confidence": 0.55}}
    judge, lines, _ = _turn(TURN_8, observation, read, {"ironvaliant": True})
    doubled = {"type": "doubles_turn", "slot_0": _move_row("flareblitz", 1), "slot_1": _move_row("flashcannon", 1)}
    likely = principles.Judge(judge.slots, observation, judge.calc,
                              {"ironvaliant": predict.FoeExpectation(protect=0.6)}, preserve=PRESERVE)
    assert "may Protect" in likely.judge(doubled)[1][0]
    best = lines[0]
    attacks = [row for row in (best["slot_0"], best["slot_1"]) if row.get("move_id") not in (None, "fakeout", "protect")]
    assert not (len(attacks) == 2 and all(row.get("target") == 1 for row in attacks))  # Fake Out + attack is fine


def test_a_fresh_fake_out_user_is_read_as_fake_out_not_protect():
    """Turn 4: Sneasler came in with Fake Out; the code read Protect and Incineroar flinched."""
    _, _, base = _turn(TURN_5, _obs(["incineroar", "archaludon"], ["sneasler", "cresselia"], turn=4), {}, {"sneasler": True})
    assert base["sneasler"]["action"] == "fake_out"


# -- principle 5: self-drops ---------------------------------------------------------------


def test_draco_meteor_weakens_the_users_later_hits_in_the_race():
    user = race.Mon("us", {"species": "archaludon"}, 100.0)
    race._Sim._self_drop(user, "dracometeor", 1.0)
    assert user.boost["spa"] == 0.5
    race._Sim._self_drop(user, "flashcannon", 1.0)
    assert user.boost["spa"] == 0.5


def test_a_self_drop_move_at_minus_two_is_flagged():
    observation = _obs(["incineroar", "archaludon"], ["sneasler", "cresselia"], turn=5, boosts={"archaludon": {"spa": -2}})
    judge, _, _ = _turn(TURN_5, observation, {}, {})
    line = {"type": "doubles_turn", "slot_0": _move_row("knockoff", 1), "slot_1": _move_row("dracometeor", 2)}
    assert any("-2 SPA" in f for f in judge.judge(line)[1])


# -- the strategist is shown the principled line -------------------------------------------


def test_a_pick_that_breaks_a_principle_offers_the_best_clean_line_beyond_the_margin():
    provider = FakeProvider({"verdict": "veto", "play_option": "3", "why": "keep the plan", "reasoning_summary": ""})
    agent = VGCDraftStrategist(provider, log=[].append, rng=random.Random(1), clock=lambda: 0.0)
    agent._line_flags = [["grimmsnarl skips its job"], ["also bad"], [], []]
    ranked = [(100.0, {}), (90.0, {}), (100.0 - VETO_MARGIN - 30, {}), (0.0, {})]
    state = _state(_doubles_action(), observation=BATTLE_OBSERVATION)
    played, review = agent._review(ranked, {}, "", REVIEW_FACTS, state, POKEMON, 0.0)
    options = json.loads(provider.calls[0]["messages"][1]["content"])["options"]
    assert [o["breaks_principles"] for o in options] == [["grimmsnarl skips its job"], ["also bad"], "none"]
    assert (played, review["verdict"], review["play_option"]) == (2, "veto", 3)


def test_a_clean_pick_keeps_the_plain_margin():
    provider = FakeProvider({"verdict": "play", "play_option": "1", "why": "", "reasoning_summary": ""})
    agent = VGCDraftStrategist(provider, log=[].append, rng=random.Random(1), clock=lambda: 0.0)
    agent._line_flags = [[], ["bad"], []]
    played, review = agent._review([(100.0, {}), (100.0 - VETO_MARGIN - 1, {}), (0.0, {})], {}, "", REVIEW_FACTS,
                                   None, POKEMON, 0.0)
    assert (played, review) == (0, None) and not provider.calls
