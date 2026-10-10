"""Regression tests from match ccb570bb (2026-10-10, seat 1, a win), with its real draft cards."""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, movelist, predict, race, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

CARDS = [
    {"card_id": "vgc-porygon2", "species": "Porygon2", "item": "Eviolite", "ability": "Download", "nature": "Quiet",
     "evs": {"hp": 252, "spa": 252, "spd": 4}, "moves": ["Trick Room", "Tri Attack", "Ice Beam", "Recover"]},
    {"card_id": "vgc-sylveon", "species": "Sylveon", "item": "Wiki Berry", "ability": "Pixilate", "nature": "Modest",
     "evs": {"hp": 244, "spa": 252, "spd": 12}, "moves": ["Hyper Voice", "Shadow Ball", "Calm Mind", "Protect"]},
    {"card_id": "vgc-garchomp", "species": "Garchomp", "item": "Focus Sash", "ability": "Rough Skin", "nature": "Jolly",
     "evs": {"hp": 4, "atk": 252, "spe": 252}, "moves": ["Earthquake", "Rock Slide", "Iron Head", "Protect"]},
    {"card_id": "vgc-flutter-mane", "species": "Flutter Mane", "item": "Booster Energy", "ability": "Protosynthesis",
     "nature": "Timid", "evs": {"spa": 252, "spd": 4, "spe": 252}, "moves": ["Moonblast", "Shadow Ball", "Icy Wind", "Protect"]},
    {"card_id": "vgc-great-tusk", "species": "Great Tusk", "item": "Booster Energy", "ability": "Protosynthesis", "nature": "Jolly",
     "evs": {"hp": 252, "atk": 252, "spe": 4}, "moves": ["Headlong Rush", "Close Combat", "Rock Slide", "Protect"]},
    {"card_id": "vgc-archaludon", "species": "Archaludon", "item": "Leftovers", "ability": "Stamina", "nature": "Bold",
     "evs": {"hp": 252, "def": 252, "spa": 4}, "moves": ["Body Press", "Draco Meteor", "Flash Cannon", "Protect"]},
    {"card_id": "vgc-tornadus", "species": "Tornadus", "item": "Mental Herb", "ability": "Prankster", "nature": "Timid",
     "evs": {"hp": 252, "spd": 4, "spe": 252}, "moves": ["Bleakwind Storm", "Taunt", "Tailwind", "Protect"]},
]


def _table():
    sets = CardSets(persist=False)
    sets.remember(CARDS)
    return sets.overlay(meta.table())


def _cc(calc, defender):
    move = movelist.lookup("closecombat")
    return calc.damage({"species": "greattusk"}, {"species": defender},
                       {"id": "closecombat", "base_power": move["power"], "type": move["type"], "category": move["category"]})


def test_eviolite_makes_porygon2_bulky():
    """Close Combat read as 129% on Porygon2 without Eviolite."""
    calc = lookahead.Calculator({"active_pokemon": [], "opponent_active_pokemon": []}, _table())
    assert _cc(calc, "porygon2")["expected_pct"] < 100


def test_focus_sash_keeps_an_untouched_holder_alive_through_one_hit():
    observation = {"active_pokemon": [{"species": "garchomp", "hp_fraction": 1.0}],
                   "opponent_active_pokemon": [{"species": "archaludon", "hp_fraction": 1.0}]}
    calc = lookahead.Calculator(observation, _table())
    sim = race._Sim(calc, race.Board(), [], [], observation)
    garchomp = race._mon("us", {"species": "garchomp", "hp_fraction": 1.0}, calc._table)
    archaludon = race._mon("them", {"species": "archaludon", "hp_fraction": 1.0}, calc._table)
    sim.ours, sim.theirs = [garchomp], [archaludon]
    draco = {"id": "dracometeor", "base_power": 130, "type": "DRAGON", "category": "SPECIAL", "spread": False}
    sim._hit("attack", archaludon, draco, [(garchomp, 1.0)])
    assert garchomp.alive == 1.0 and 0 < garchomp.hp <= 1.0 and garchomp.sash == 0.0


FOES_1 = [{"target": 1, "side": "opponent", "species": "greattusk"}, {"target": 2, "side": "opponent", "species": "archaludon"}]


def test_turn_one_sets_trick_room_instead_of_double_switching_into_the_hits():
    """Played: both leads switched out; Garchomp came in at 22%, Flutter Mane at 35%."""
    table = _table()
    action = _doubles("porygon2", [_move("trickroom", []), _move("icebeam", [1, 2], FOES_1), _move("recover", []),
                                   {"type": "switch", "species": "garchomp"}, {"type": "switch", "species": "fluttermane"}],
                      "sylveon", [_move("hypervoice", []), _move("shadowball", [1, 2], FOES_1), _move("calmmind", []), _move("protect", []),
                                  {"type": "switch", "species": "garchomp"}, {"type": "switch", "species": "fluttermane"}])
    observation = {"turn": 1,
                   "active_pokemon": [{"species": "porygon2", "types": ["NORMAL"]}, {"species": "sylveon", "types": ["FAIRY"]}],
                   "opponent_active_pokemon": [{"species": "greattusk", "types": ["GROUND", "FIGHTING"]},
                                               {"species": "archaludon", "types": ["STEEL", "DRAGON"]}],
                   "team": [{"species": "garchomp", "hp_fraction": 1.0}, {"species": "fluttermane", "hp_fraction": 1.0}]}
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(known_moves=table.common_moves), table, lambda line: None)
    expected = predict.expectations({"greattusk": {"action": "attack", "target": "porygon2", "confidence": 0.55},
                                     "archaludon": {"action": "protect", "confidence": 0.5}}, ["porygon2", "sylveon"])
    payload = turn_rules.check(lookahead.best_line(turn_rules, observation, lookahead.Calculator(observation, table), expected))
    assert payload["slot_0"] == {"type": "move", "move_id": "trickroom"}
    assert payload["slot_1"]["type"] == "move"


FOES_5 = [{"target": 1, "side": "opponent", "species": "tornadus"}, {"target": 2, "side": "opponent", "species": "archaludon"}]


def test_ice_beam_into_stamina_archaludon_is_allowed():
    """Turn 5: the Stamina rule banned Ice Beam into Archaludon (a special hit,
    which its Defense does not touch), forcing it into the Tornadus read to Protect."""
    table = _table()
    action = _doubles("sylveon", [_move("hypervoice", [])], "porygon2", [_move("icebeam", [1, 2], FOES_5), _move("trickroom", [])])
    observation = {"turn": 5,
                   "active_pokemon": [{"species": "sylveon", "types": ["FAIRY"]}, {"species": "porygon2", "types": ["NORMAL"]}],
                   "opponent_active_pokemon": [{"species": "tornadus", "types": ["FLYING"]},
                                               {"species": "archaludon", "types": ["STEEL", "DRAGON"], "hp_fraction": 0.87}]}
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(known_moves=table.common_moves), table, lambda line: None)
    assert turn_rules.legal({"slot_0": {"option": 0, "target": None}, "slot_1": {"option": 0, "target": 2}})


def _tracker_turn(tracker, archaludon_hp, moves=None):
    action = _doubles("sylveon", [_move("shadowball", [2], FOES_5)], "porygon2", [_move("icebeam", [2], FOES_5)])
    slots = act.parse_doubles(_as_legal(action))
    archaludon = {"species": "archaludon", "types": ["STEEL", "DRAGON"], "hp_fraction": archaludon_hp}
    if moves:
        archaludon["moves"] = moves
    tracker.begin(slots, {"turn": 7, "active_pokemon": [{"species": "sylveon"}, {"species": "porygon2"}],
                          "opponent_active_pokemon": [archaludon]})
    return slots


def test_a_newly_revealed_protect_counts_as_protected_last_turn():
    """Turn 7: Archaludon's Protect showed in its moves while Leftovers raised its HP; we missed it."""
    tracker = MoveTracker()
    _tracker_turn(tracker, 0.41, ["bodypress", "dracometeor"])
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "hypervoice"},
                    "slot_1": {"type": "move", "move_id": "icebeam", "target": 2}})
    _tracker_turn(tracker, 0.47, ["bodypress", "dracometeor", "protect"])
    assert tracker.protected_last_turn["archaludon"].startswith("revealed")


def test_a_leftovers_rise_where_our_attack_should_have_hit_hard_means_protect():
    tracker = MoveTracker()
    _tracker_turn(tracker, 0.18)
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "shadowball", "target": 2},
                    "slot_1": {"type": "move", "move_id": "icebeam", "target": 2}})
    tracker.expect({"archaludon": 0.30})  # our attacks should take at least 30%
    _tracker_turn(tracker, 0.24)  # +6%: Leftovers, no damage
    assert "archaludon" in tracker.protected_last_turn
    # A small hit that Leftovers cancels out is not a Protect.
    tracker = MoveTracker()
    _tracker_turn(tracker, 0.50)
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "shadowball", "target": 2},
                    "slot_1": {"type": "move", "move_id": "icebeam", "target": 2}})
    tracker.expect({"archaludon": 0.09})
    _tracker_turn(tracker, 0.51)
    assert "archaludon" not in tracker.protected_last_turn
