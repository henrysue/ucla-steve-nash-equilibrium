"""Regression tests from match ead979d8 (2026-10-15), a loss to Spore and Pollen Puff.

Amoonguss put Iron Hands and Dondozo to sleep and healed Archaludon 33% -> 89%.
The tempo rule flagged switching the sleeping Iron Hands out, so the strategist
kept it in "to make the KO" and it fell without moving; the sleeping Dondozo
"Protected" twice and took Sludge Bomb each time. Dondozo and Basculegion
(Water) were brought into Archaludon, Rillaboom and Amoonguss, who resist Water.
"""

import random

from agent.vgc import actions as act
from agent.vgc import decisions, lookahead, meta, predict, principles, race, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker
from agent.vgc.strategist import VGCDraftStrategist

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "am", "species": "Amoonguss", "ability": "Regenerator", "item": "Rocky Helmet", "nature": "Bold",
     "evs": {"hp": 244, "def": 172, "spd": 92}, "moves": ["Spore", "Rage Powder", "Pollen Puff", "Sludge Bomb"]},
    {"card_id": "ar", "species": "Archaludon", "ability": "Stamina", "item": "Leftovers", "nature": "Bold",
     "evs": {"hp": 252, "def": 252, "spa": 4}, "moves": ["Body Press", "Draco Meteor", "Flash Cannon", "Protect"]},
    {"card_id": "dz", "species": "Dondozo", "ability": "Unaware", "item": "Leftovers",
     "moves": ["Wave Crash", "Earthquake", "Liquidation", "Protect"]},
    {"card_id": "ih", "species": "Iron Hands", "ability": "Quark Drive", "item": "Assault Vest", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 252, "spe": 20}, "moves": ["Fake Out", "Close Combat", "Wild Charge", "Drain Punch"]},
    {"card_id": "in", "species": "Incineroar", "ability": "Intimidate", "item": "Safety Goggles",
     "moves": ["Fake Out", "Knock Off", "Flare Blitz", "Parting Shot"]},
])
TABLE = SETS.overlay(meta.table())
TYPES = {"amoonguss": ["GRASS", "POISON"], "archaludon": ["STEEL", "DRAGON"], "dondozo": ["WATER"],
         "ironhands": ["FIGHTING", "ELECTRIC"], "incineroar": ["FIRE", "DARK"], "basculegion": ["WATER", "GHOST"]}
F = [{"target": 1, "side": "opponent", "species": "amoonguss"}, {"target": 2, "side": "opponent", "species": "archaludon"}]


def _obs(ours, hp=None, status=None, turn=3):
    hp, status = hp or {}, status or {}
    side = lambda names: [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0), "status": status.get(s, "")}
                          for s in names]  # noqa: E731
    return {"turn": turn, "active_pokemon": side(ours), "opponent_active_pokemon": side(["amoonguss", "archaludon"])}


# -- 1. a sleeping Pokemon's Protect works only if it wakes -----------------------------------


def test_a_sleeping_dondozos_protect_mostly_fails():
    action = _doubles("basculegion", [_move("aquajet", [1, 2], F)], "dondozo", [_move("protect", []), _move("earthquake", [])])
    slots = act.parse_doubles(_as_legal(action))
    expected = predict.expectations({"amoonguss": {"action": "attack", "target": "dondozo", "confidence": 0.9}},
                                    ["basculegion", "dondozo"])
    protect_line = [{"option": 0, "target": 1}, {"option": 0, "target": None}]

    def score(status):
        observation = _obs(["basculegion", "dondozo"], status=status, turn=7)
        turn_rules = rules.build(slots, observation, MoveTracker(), TABLE, lambda line: None)
        calc = lookahead.Calculator(observation, TABLE)
        race.FOLLOW_UP_TURNS, saved = 0, race.FOLLOW_UP_TURNS
        try:
            return race.score(protect_line, slots, turn_rules.per_slot, observation, calc, expected)
        finally:
            race.FOLLOW_UP_TURNS = saved

    assert score({"dondozo": "slp"}) < score({})  # asleep, the Sludge Bomb mostly lands


# -- 2. switching out a sleeper costs no tempo ------------------------------------------------


def test_switching_out_a_sleeping_iron_hands_is_not_a_tempo_loss():
    observation = _obs(["ironhands", "incineroar"], hp={"archaludon": 0.33}, status={"ironhands": "slp"})
    action = _doubles("ironhands", [_move("closecombat", [1, 2], F), {"type": "switch", "species": "basculegion"}],
                      "incineroar", [_move("flareblitz", [1, 2], F)])
    slots = act.parse_doubles(_as_legal(action))
    calc = lookahead.Calculator(observation, TABLE)
    judge = principles.Judge(slots, observation, calc, {}, just_in={"ironhands"})
    switch = {"type": "doubles_turn", "slot_0": {"type": "switch", "species": "basculegion"},
              "slot_1": {"type": "move", "move_id": "flareblitz", "target": 1}}
    cost, flags = judge.judge(switch)
    assert not any("healthy ironhands" in f for f in flags)
    assert judge._ko_on_board(slots[0]) is None  # asleep: its Close Combat is no KO on the board
    awake_obs = _obs(["ironhands", "incineroar"], hp={"archaludon": 0.33})
    awake = principles.Judge(slots, awake_obs, lookahead.Calculator(awake_obs, TABLE), {})
    assert awake._ko_on_board(slots[0]) == "archaludon"


# -- 3. the final call knows our sleepers do nothing -----------------------------------------


def test_the_final_call_knows_a_sleeper_of_ours_does_nothing():
    text = decisions.review_decision([{"option": 1, "value": 0, "line": "x"}], {}).prompt["instructions"]
    assert "never keep it in to make a KO" in text and "switching it out costs no tempo" in text


# -- 4. a sleep user is their win condition ---------------------------------------------------


def test_a_spore_amoonguss_is_their_win_condition():
    calc = lookahead.Calculator(_obs(["ironhands", "incineroar"]), TABLE)
    assert "puts ours to sleep (spore)" in lookahead.win_conditions(calc)["amoonguss"]


def test_a_sure_ko_still_beats_chip_on_the_sleep_user():
    observation = _obs(["ironhands", "incineroar"], hp={"archaludon": 0.33})
    action = _doubles("ironhands", [_move("closecombat", [1, 2], F)], "incineroar", [_move("flareblitz", [1, 2], F)])
    slots = act.parse_doubles(_as_legal(action))
    calc = lookahead.Calculator(observation, TABLE)
    judge = principles.Judge(slots, observation, calc, {})
    chip = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "closecombat", "target": 1},
            "slot_1": {"type": "move", "move_id": "flareblitz", "target": 1}}
    cost, flags = judge.judge(chip)
    assert cost >= principles.MISSED_KO and any("leaves archaludon alive" in f for f in flags)
    finish = {**chip, "slot_0": {"type": "move", "move_id": "closecombat", "target": 2}}
    assert not any("leaves" in f for f in judge.judge(finish)[1])


# -- 5. team preview: no stacked attack type their team resists --------------------------------


def test_dondozo_and_basculegion_together_into_water_resists_is_a_lineup_problem():
    agent = VGCDraftStrategist(None, log=lambda line: None, rng=random.Random(1))
    agent._preview_opponent = ["Archaludon", "Rillaboom", "Amoonguss", "Kingambit", "Porygon2", "Landorus-Therian"]
    found = agent._stacked_attack(["ironhands", "incineroar", "dondozo", "basculegion"])
    assert found and "Water" in found and "3 of their 6" in found
    assert agent._stacked_attack(["ironhands", "incineroar", "dondozo", "screamtail"]) is None
