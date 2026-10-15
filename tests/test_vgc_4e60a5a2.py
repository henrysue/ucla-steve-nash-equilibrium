"""Regression tests from match 4e60a5a2 (2026-10-13), a loss to sun Torkoal.

Torkoal's Eruption took 83%, 54% and 93% off ours and it never took a hit:
the Protect odds sat near 58% all game (a style layer saw its Protects but
not its attacks), the opening Fake Out went into a Protect, nothing valued
weakening an Eruption user, and a veto went 121 points down the list.
"""

import random

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, predict, protect, race, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker
from agent.vgc.predict import FoeExpectation
from agent.vgc.strategist import VETO_MARGIN, VGCDraftStrategist

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import POKEMON, FakeProvider, _move

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "vgc-torkoal", "species": "Torkoal", "ability": "Drought", "item": "Charcoal",
     "moves": ["Eruption", "Heat Wave", "Earth Power", "Protect"]},
    {"card_id": "vgc-thundurus", "species": "Thundurus", "ability": "Prankster",
     "moves": ["Thunderbolt", "Sludge Bomb", "Taunt", "Protect"]},
])
TABLE = SETS.overlay(meta.table())
FOES = [{"target": 1, "side": "opponent", "species": "thundurus"}, {"target": 2, "side": "opponent", "species": "torkoal"}]
TYPES = {"grimmsnarl": ["DARK", "FAIRY"], "lucario": ["FIGHTING", "STEEL"], "thundurus": ["ELECTRIC", "FLYING"],
         "torkoal": ["FIRE"]}


def _obs(hp=None, turn=6):
    hp = {"grimmsnarl": 0.074, **(hp or {})}
    return {"turn": turn, "weather": {"SUNNYDAY": 4},
            "active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in ("grimmsnarl", "lucario")],
            "opponent_active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in ("thundurus", "torkoal")]}


TURN_6 = _doubles("grimmsnarl", [_move("spiritbreak", [1, 2], FOES), _move("thunderwave", [1, 2], FOES)],
                  "lucario", [_move("closecombat", [1, 2], FOES), _move("poisonjab", [1, 2], FOES)])
BOTH_ATTACK = {"thundurus": {"action": "attack", "target": "grimmsnarl", "confidence": 0.55},
               "torkoal": {"action": "attack", "target": "grimmsnarl", "confidence": 0.5}}


# -- 1. one simple Protect table -------------------------------------------------------------


def _odds(observation, first_turn=None, protected=()):
    return protect.odds(observation, first_turn=first_turn or {}, common_moves=TABLE.common_moves,
                        card_set=TABLE.card_set, protected_last_turn=set(protected))


def test_a_first_turn_out_is_the_only_turn_protect_is_raised():
    assert _odds(_obs(), first_turn={"torkoal": True})["torkoal"]["likelihood"] == protect.FIRST_TURN
    assert _odds(_obs())["torkoal"]["likelihood"] == protect.NORMAL
    assert protect.NORMAL < protect.FIRST_TURN < protect.LIKELY  # e1f70879: 6 of 16 first turns, not 70%


def test_right_after_a_protect_the_chance_is_zero():
    row = _odds(_obs(), protected={"thundurus"})["thundurus"]
    assert row["likelihood"] == 0.0 and "will not try again" in row["why"][-1]


def test_a_read_of_a_repeat_protect_becomes_an_attack():
    expected = {"thundurus": FoeExpectation(protect=0.6, attacks={"grimmsnarl": 0.2, "lucario": 0.2})}
    predict.no_repeat_protect(expected, {"thundurus": "revealed: protect"})
    assert expected["thundurus"].protect == 0.0
    assert abs(sum(expected["thundurus"].attacks.values()) - 1.0) < 1e-9


# -- 3. their win condition comes first ------------------------------------------------------


def test_a_full_hp_eruption_user_is_their_win_condition():
    calc = lookahead.Calculator(_obs(), TABLE)
    assert "torkoal" in lookahead.win_conditions(calc)
    assert "thundurus" not in lookahead.win_conditions(calc)
    weak = lookahead.Calculator(_obs({"torkoal": 0.3}), TABLE)
    assert "eruption" not in lookahead.win_conditions(weak).get("torkoal", "")  # already weak: no Eruption flag


def test_our_hits_on_torkoal_cut_its_next_eruption_in_the_race():
    calc = lookahead.Calculator(_obs(), TABLE)
    sim = race._Sim(calc, race.Board(), [], [], _obs())
    eruption = next(m for m in calc.likely_attacks({"species": "torkoal"}) if m["id"] == "eruption")
    target = race.Mon("us", {"species": "lucario", "types": TYPES["lucario"], "hp_fraction": 1.0}, 100.0)
    full = sim._damage(race.Mon("them", {"species": "torkoal", "types": ["FIRE"], "hp_fraction": 1.0}, 100.0), target, eruption)
    half = sim._damage(race.Mon("them", {"species": "torkoal", "types": ["FIRE"], "hp_fraction": 1.0}, 50.0), target, eruption)
    assert half["expected_pct"] < 0.6 * full["expected_pct"]


def test_turn_6_close_combat_goes_into_torkoal_not_thundurus():
    """Played: Spirit Break + Close Combat into Thundurus; Torkoal stayed at full HP."""
    slots = act.parse_doubles(_as_legal(TURN_6))
    turn_rules = rules.build(slots, _obs(), MoveTracker(known_moves=TABLE.common_moves), TABLE, lambda line: None)
    calc = lookahead.Calculator(_obs(), TABLE)
    expected = predict.expectations(BOTH_ATTACK, ["grimmsnarl", "lucario"])
    payload = turn_rules.check(lookahead.best_line(turn_rules, _obs(), calc, expected))
    assert payload["slot_1"] == {"type": "move", "move_id": "closecombat", "target": 2}


# -- 4. a veto stays close to the code's pick ------------------------------------------------


def test_no_review_when_nothing_is_close_enough():
    """Turn 1 vetoed a 51.9 line for a -69.0 one; now that line is never offered."""
    agent = VGCDraftStrategist(FakeProvider(), log=[].append, rng=random.Random(1), clock=lambda: 0.0)
    played, review = agent._review([(100.0, {}), (100.0 - VETO_MARGIN - 1, {})], {}, "", {}, None, POKEMON, 0.0)
    assert (played, review) == (0, None)
