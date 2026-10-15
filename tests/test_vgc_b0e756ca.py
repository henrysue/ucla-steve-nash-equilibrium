"""Regression tests from match b0e756ca (2026-10-15), a loss to Ace Trainer Claude.

Its Pokemon in our KO range escaped every time: Basculegion Protected at 15%
then switched; Arcanine Protected at 16% then switched to Flutter Mane (immune
to Extreme Speed). Four finishing hits were wasted while the read said they
would attack. Turn 11 the strategist vetoed Earthquake ("Ting-Lu will Protect")
for a single-target hit into Arcanine, which Protected.
"""

from agent.vgc import actions as act
from agent.vgc import decisions, lookahead, meta, predict, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker
from agent.vgc.predict import FoeExpectation
from agent.vgc.strategist import _name_payload
from agent.vgc.style import OpponentStyle

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "ac", "species": "Arcanine", "ability": "Intimidate", "item": "Safety Goggles", "nature": "Adamant",
     "evs": {"hp": 252, "atk": 252, "spe": 4}, "moves": ["Flare Blitz", "Extreme Speed", "Snarl", "Protect"]},
    {"card_id": "bg", "species": "Basculegion", "ability": "Adaptability", "item": "Life Orb", "nature": "Adamant",
     "evs": {"hp": 4, "atk": 252, "spe": 252}, "moves": ["Wave Crash", "Aqua Jet", "Liquidation", "Protect"]},
    {"card_id": "cp", "species": "Chien-Pao", "ability": "Sword of Ruin", "item": "Focus Sash", "nature": "Jolly",
     "evs": {"hp": 4, "atk": 252, "spe": 252}, "moves": ["Icicle Crash", "Sacred Sword", "Sucker Punch", "Protect"]},
    {"card_id": "dn", "species": "Dragonite", "ability": "Multiscale", "item": "Loaded Dice", "nature": "Adamant",
     "evs": {"hp": 252, "atk": 252, "spe": 4}, "moves": ["Extreme Speed", "Scale Shot", "Tailwind", "Protect"]},
])
TABLE = SETS.overlay(meta.table())
TYPES = {"arcanine": ["FIRE"], "basculegion": ["WATER", "GHOST"], "chienpao": ["DARK", "ICE"],
         "dragonite": ["DRAGON", "FLYING"], "irontreads": ["GROUND", "STEEL"], "tinglu": ["DARK", "GROUND"]}


# -- 1. the KO-range escape rate is learned, and moves the expectation ---------------------------


def _resolve(style, foe, row, turn):
    style.watch_ko({foe})
    style.resolve({"theirs": {foe: row}}, turn)


def test_ace_trainer_claudes_escapes_raise_the_rate():
    style = OpponentStyle()
    assert abs(style.ko_escape_rate - 0.3) < 1e-9
    _resolve(style, "basculegion", {"hp_change": 0.0, "revealed": ["protect"]}, 3)  # Protected at 15%
    _resolve(style, "basculegion", {"left_field": "switched or fainted"}, 4)  # then switched out
    _resolve(style, "arcanine", {"hp_change": 0.0, "revealed": ["protect"]}, 4)  # Protected at 16%
    assert style.ko_spots == 3 and style.ko_escapes == 3
    assert style.ko_escape_rate > 0.6
    _resolve(style, "fluttermane", {"left_field": "fainted"}, 9)  # we got it: no evidence either way
    assert style.ko_spots == 3


def test_a_foe_that_stays_and_fights_lowers_it():
    style = OpponentStyle()
    _resolve(style, "chiyu", {"hp_change": -0.2, "revealed": ["overheat"]}, 2)
    assert style.ko_escape_rate < 0.3


def test_escape_goes_to_protect_when_it_can_and_switch_when_it_cannot():
    expected = {"basculegion": FoeExpectation(attacks={"chienpao": 0.75}),
                "arcanine": FoeExpectation(attacks={"chienpao": 0.65})}
    predict.escape(expected, {"basculegion", "arcanine"}, {"basculegion": False, "arcanine": True}, 0.7, has_bench=True)
    assert abs(expected["basculegion"].switch - 0.7) < 1e-9 and expected["basculegion"].protect == 0  # just Protected
    assert abs(expected["arcanine"].protect - 0.65) < 1e-9  # all of its attack share, at most
    stuck = {"basculegion": FoeExpectation(attacks={"chienpao": 0.75})}
    predict.escape(stuck, {"basculegion"}, {"basculegion": False}, 0.7, has_bench=False)
    assert stuck["basculegion"].switch == 0  # nowhere to go


def test_turn_4_stops_chasing_the_low_foes_against_an_escaping_opponent():
    observation = {"turn": 4,
                   "active_pokemon": [{"species": "chienpao", "types": TYPES["chienpao"], "hp_fraction": 0.006, "boosts": {"atk": -1}},
                                      {"species": "dragonite", "types": TYPES["dragonite"], "hp_fraction": 1.0,
                                       "boosts": {"atk": -1, "def": -2, "spe": 2}}],
                   "opponent_active_pokemon": [{"species": "basculegion", "types": TYPES["basculegion"], "hp_fraction": 0.15, "boosts": {"atk": -1}},
                                               {"species": "arcanine", "types": TYPES["arcanine"], "hp_fraction": 0.16}]}
    F = [{"target": 1, "side": "opponent", "species": "basculegion"}, {"target": 2, "side": "opponent", "species": "arcanine"}]
    action = _doubles("chienpao", [_move("suckerpunch", [1, 2], F), _move("iciclecrash", [1, 2], F), _move("protect", [])],
                      "dragonite", [_move("extremespeed", [1, 2], F), _move("scaleshot", [1, 2], F), _move("protect", [])])
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(), TABLE, lambda line: None)
    calc = lookahead.Calculator(observation, TABLE)
    read = {"basculegion": {"action": "attack", "target": "chienpao", "confidence": 0.75},
            "arcanine": {"action": "attack", "target": "chienpao", "confidence": 0.65}}
    chase = [{"option": 0, "target": 1}, {"option": 0, "target": 2}]  # Sucker Punch + Extreme Speed: what we played

    def value(rate):
        expected = predict.expectations(read, ["chienpao", "dragonite"])
        predict.escape(expected, {"basculegion", "arcanine"}, {"basculegion": False, "arcanine": True}, rate, has_bench=True)
        from agent.vgc import race

        return race.score(chase, slots, turn_rules.per_slot, observation, calc, expected)

    assert value(0.75) < value(0.0)  # an escaping opponent makes the chase worth less


# -- 2. spread moves are labelled, and the final call knows how they work -------------------------


def test_earthquake_is_labelled_as_hitting_both_foes():
    F = [{"target": 1, "side": "opponent", "species": "tinglu"}, {"target": 2, "side": "opponent", "species": "arcanine"}]
    action = _doubles("irontreads", [_move("earthquake", []), _move("highhorsepower", [1, 2], F)], "dragonite", [{"type": "pass"}])
    slots = act.parse_doubles(_as_legal(action))
    line = _name_payload({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "earthquake"},
                          "slot_1": {"type": "pass"}}, slots)
    assert "Earthquake (hits both foes and our partner)" in line


def test_the_final_call_knows_protect_on_one_foe_does_not_stop_a_spread_move():
    text = decisions.review_decision([{"option": 1, "value": 0, "line": "x"}], {}).prompt["instructions"]
    assert "still hits a foe that is not protecting when the other one Protects" in text
