"""Regression tests from match 37a1ee8a (2026-10-15), a loss to Ace Trainer Claude.

It focused one of ours at a time (Urshifu, then Basculegion, then Archaludon),
and Amoonguss's Rage Powder pulled Urshifu's Surging Strikes off a 36%
Arcanine. The race scored a second hit against the average HP, so two hits
that KO together counted as nearly never; no read predicted redirection; and
the draft denied Archaludon (5.5) while Iron Hands (9.6) went to them.
"""

from agent.vgc import lookahead, meta, pick_priority as pp, predict, race
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker
from agent.vgc.predict import FoeExpectation
from agent.vgc.style import OpponentStyle

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "am", "species": "Amoonguss", "ability": "Regenerator", "item": "Rocky Helmet",
     "moves": ["Spore", "Rage Powder", "Pollen Puff", "Sludge Bomb"]},
    {"card_id": "ac", "species": "Arcanine", "ability": "Intimidate", "item": "Safety Goggles",
     "moves": ["Flare Blitz", "Extreme Speed", "Snarl", "Protect"]},
    {"card_id": "ih", "species": "Iron Hands", "ability": "Quark Drive", "item": "Assault Vest", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 252, "spe": 20}, "moves": ["Fake Out", "Close Combat", "Wild Charge", "Drain Punch"]},
    {"card_id": "ar", "species": "Archaludon", "ability": "Stamina", "item": "Leftovers", "nature": "Bold",
     "evs": {"hp": 252, "def": 252, "spa": 4}, "moves": ["Body Press", "Draco Meteor", "Flash Cannon", "Protect"]},
])
TABLE = SETS.overlay(meta.table())


# -- 1. the race scores focus fire by scenario ---------------------------------------------------


def test_two_hits_that_ko_together_are_scored_as_a_ko():
    """Two 60% hits, each landing 60% of the time, KO a full-HP target 36% of the time."""
    mon = race.Mon("us", {"species": "urshifurapidstrike"}, 100.0)
    mon.turn_hp, mon.turn_hits = 100.0, [(0.6, 60.0, 0.0)]  # the first hit: no KO alone
    hit = {"expected_pct": 60.0, "range_pct": [55.0, 65.0]}
    second = race._Sim._focus_ko(mon, hit, 1.0)
    assert abs(second - 0.6) < 1e-9  # it KOs whenever the first one landed
    alive = 1 - 0.6 * second
    assert abs(alive - 0.64) < 1e-9
    assert race._ko(hit, mon.turn_hp - 0.6 * 60.0) <= 0.1  # the old average-HP view: 10% at most, not 60%


def test_a_first_hit_is_scored_as_before():
    mon = race.Mon("us", {"species": "urshifurapidstrike"}, 100.0)
    mon.turn_hp, mon.turn_hits = 100.0, []
    assert race._Sim._focus_ko(mon, {"expected_pct": 60.0, "range_pct": [55.0, 65.0]}, 1.0) == 0.0


# -- 2. the focus-fire read --------------------------------------------------------------------


def test_the_focus_target_is_who_they_ko_together():
    threats = {"Iron Hands": {"Urshifu": {"expected_pct": 45.0}, "Archaludon": {"expected_pct": 20.0}},
               "Arcanine": {"Urshifu": {"expected_pct": 30.0}, "Archaludon": {"expected_pct": 15.0}}}
    target, why = predict.focus_target(threats, {"urshifu": 0.61, "archaludon": 0.83})
    assert target == "urshifu" and "together they KO Urshifu" in why
    assert predict.focus_target(threats, {"urshifu": 1.0, "archaludon": 1.0})[0] is None  # 75% < 100%


def test_focus_moves_their_attacks_onto_the_target():
    expected = {"ironhands": FoeExpectation(attacks={"urshifu": 0.3, "archaludon": 0.3}),
                "arcanine": FoeExpectation(attacks={"archaludon": 0.6})}
    predict.focus(expected, "urshifu", 0.5)
    assert abs(expected["ironhands"].attacks["urshifu"] - 0.45) < 1e-9
    assert abs(expected["arcanine"].attacks["urshifu"] - 0.3) < 1e-9 and abs(expected["arcanine"].attacks["archaludon"] - 0.3) < 1e-9


def test_the_focus_rate_is_learned_from_our_hp():
    style = OpponentStyle()
    assert abs(style.focus_rate - 0.5) < 1e-9
    focused = {"theirs": {"ironhands": {"hp_change": -0.05}, "arcanine": {"hp_change": 0.0, "revealed": ["flareblitz"]}},
               "ours": {"urshifu": {"hp_change": -0.49}, "archaludon": {"hp_change": 0.06}}}
    for turn in (4, 7, 9):
        style.resolve(focused, turn)
    assert style.focus_turns == 3 and style.focused == 3 and style.focus_rate > 0.7
    spread = {"theirs": focused["theirs"], "ours": {"urshifu": {"hp_change": -0.2}, "archaludon": {"hp_change": -0.2}}}
    style.resolve(spread, 10)
    assert style.focus_turns == 4 and style.focused == 3


# -- 3. the redirection read -------------------------------------------------------------------


def _turn_5():
    return {"turn": 5,
            "active_pokemon": [{"species": "urshifurapidstrike", "types": ["FIGHTING", "WATER"], "hp_fraction": 0.114},
                               {"species": "archaludon", "types": ["STEEL", "DRAGON"], "hp_fraction": 0.888}],
            "opponent_active_pokemon": [{"species": "amoonguss", "types": ["GRASS", "POISON"], "hp_fraction": 0.69},
                                        {"species": "arcanine", "types": ["FIRE"], "hp_fraction": 0.36}]}


def test_amoonguss_is_read_to_rage_powder_for_a_partner_in_ko_range():
    observation = _turn_5()
    calc = lookahead.Calculator(observation, TABLE)
    guess = predict.code_guess(observation, calc, MoveTracker(), TABLE, {}, in_range={"arcanine"})
    assert guess["amoonguss"]["action"] == "support" and guess["amoonguss"]["move"] == "Rage Powder"
    assert "arcanine" in guess["amoonguss"]["why"]
    plain = predict.code_guess(observation, calc, MoveTracker(), TABLE, {}, in_range=set())
    assert plain["amoonguss"]["action"] != "support"  # nothing to cover


def test_a_redirection_read_reaches_the_race():
    expected = predict.expectations({"amoonguss": {"action": "support", "move": "Rage Powder", "confidence": 0.5}},
                                    ["urshifurapidstrike", "archaludon"])
    assert expected["amoonguss"].redirect == 0.5
    calc = lookahead.Calculator(_turn_5(), TABLE)
    sim = race._Sim(calc, race.Board(), [race._mon("us", e, TABLE) for e in calc._ours],
                    [race._mon("them", e, TABLE) for e in calc._theirs], _turn_5())
    actions = sim.foe_actions(sim.theirs[0], expected["amoonguss"])
    assert any(a[0] == "status" and a[2]["id"] == "ragepowder" and a[2]["weight"] == 0.5 for a in actions)


# -- 5. the draft: deny does not beat a much stronger must pick --------------------------------


def test_iron_hands_beats_denying_archaludon():
    board = ["Archaludon", "Iron Hands", "Arcanine"]
    pick = pp.choose(board, [], TABLE, cards={b: TABLE.card_set(b) for b in board})
    assert pick.species == "Iron Hands"
    alone = pp.choose(["Archaludon", "Arcanine"], [], TABLE, cards={b: TABLE.card_set(b) for b in ["Archaludon", "Arcanine"]})
    assert (alone.species, alone.rule) == ("Archaludon", "deny")
