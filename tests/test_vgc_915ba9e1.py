"""Regression tests from match 915ba9e1 (2026-10-12, seat 1, a win)."""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, observed, predict, race, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

CARDS = [
    {"card_id": "vgc-dragonite", "species": "Dragonite", "item": "Loaded Dice", "ability": "Multiscale", "nature": "Adamant",
     "evs": {"atk": 252, "spe": 252}, "moves": ["Extreme Speed", "Scale Shot", "Tailwind", "Protect"]},
    {"card_id": "vgc-baxcalibur", "species": "Baxcalibur", "item": "Choice Band", "ability": "Thermal Exchange", "nature": "Jolly",
     "evs": {"atk": 252, "spe": 252}, "moves": ["Icicle Spear", "Glaive Rush", "Ice Shard", "Protect"]},
    {"card_id": "vgc-landorus-therian", "species": "Landorus-Therian", "item": "Choice Scarf", "ability": "Intimidate",
     "nature": "Jolly", "evs": {"atk": 252, "spe": 252}, "moves": ["Earthquake", "Rock Slide", "U-turn", "Protect"]},
    {"card_id": "vgc-iron-hands", "species": "Iron Hands", "item": "Assault Vest", "ability": "Quark Drive", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 252}, "moves": ["Fake Out", "Close Combat", "Wild Charge", "Drain Punch"]},
]
FOES = [{"target": 1, "side": "opponent", "species": "landorustherian"}]


def _table():
    sets = CardSets(persist=False)
    sets.remember(CARDS)
    return sets.overlay(meta.table())


def test_a_revealed_u_turn_is_not_read_as_a_protect():
    """Turn 2: Landorus used U-turn and took no damage; the HP inference called it a
    Protect, and the read 'Landorus protects' was logged as a hit."""
    action = _doubles("ironhands", [_move("closecombat", [1], FOES)], "dragonite", [_move("extremespeed", [1], FOES)])
    slots = act.parse_doubles(_as_legal(action))
    tracker = MoveTracker()
    before = {"turn": 2, "active_pokemon": [{"species": "ironhands"}, {"species": "dragonite"}],
              "opponent_active_pokemon": [{"species": "landorustherian", "hp_fraction": 1.0, "moves": ["earthquake"]}]}
    tracker.begin(slots, before)
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "closecombat", "target": 1},
                    "slot_1": {"type": "move", "move_id": "extremespeed", "target": 1}})
    tracker.expect({"landorustherian": 0.6})
    after = {**before, "turn": 3, "opponent_active_pokemon": [
        {"species": "landorustherian", "hp_fraction": 1.0, "moves": ["earthquake", "uturn"]}]}
    tracker.begin(slots, after)
    assert "landorustherian" not in tracker.protected_last_turn
    read = observed.verdict({"action": "protect"}, {"revealed": ["uturn"], "protected": "inferred"}, {})
    assert read["result"] == "miss"


def test_a_likely_switch_in_attacks_in_the_follow_up_turns():
    """Turn 2-3: Dragonite (4x Ice) came in while Baxcalibur sat on their bench; the
    race only let switch-ins absorb our hits, never hit back."""
    table = _table()
    # Our turn 1 lands nothing (Tailwind, Protect), so only the switch-in's threat differs.
    action = _doubles("dragonite", [_move("tailwind", [])], "ironhands", [_move("protect", [])])
    observation = {"turn": 2,
                   "active_pokemon": [{"species": "dragonite", "types": ["DRAGON", "FLYING"]},
                                      {"species": "ironhands", "types": ["FIGHTING", "ELECTRIC"]}],
                   "opponent_active_pokemon": [{"species": "landorustherian", "types": ["GROUND", "FLYING"]}]}
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(known_moves=table.common_moves), table, lambda line: None)
    calc = lookahead.Calculator(observation, table)
    expected = {"landorustherian": predict.FoeExpectation(switch=0.5, attacks={"ironhands": 0.5})}
    row = [{"option": 0, "target": None}, {"option": 0, "target": None}]
    no_threat = race.score(row, turn_rules.slots, turn_rules.per_slot, observation, calc, expected, [])
    ice_behind = race.score(row, turn_rules.slots, turn_rules.per_slot, observation, calc, expected, ["baxcalibur"])
    assert ice_behind < no_threat - 20
