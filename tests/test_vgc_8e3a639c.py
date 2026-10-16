"""Regression tests from match 8e3a639c (2026-10-09, seat 0), with its real draft cards."""

from agent.vgc import actions as act
from agent.vgc import draft, lookahead, meta, movelist, observed, predict, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

CARDS = [
    {"card_id": "vgc-archaludon", "species": "Archaludon", "item": "Leftovers", "ability": "Stamina", "nature": "Bold",
     "level": 50, "evs": {"hp": 252, "def": 252, "spa": 4}, "moves": ["Body Press", "Draco Meteor", "Flash Cannon", "Protect"]},
    {"card_id": "vgc-incineroar", "species": "Incineroar", "item": "Safety Goggles", "ability": "Intimidate", "nature": "Careful",
     "level": 50, "evs": {"hp": 244, "def": 12, "spd": 252}, "moves": ["Fake Out", "Knock Off", "Parting Shot", "Flare Blitz"]},
    {"card_id": "vgc-tyranitar", "species": "Tyranitar", "item": "Chople Berry", "ability": "Sand Stream", "nature": "Careful",
     "level": 50, "evs": {"hp": 252, "atk": 4, "spd": 252}, "moves": ["Rock Slide", "Knock Off", "Low Kick", "Protect"]},
    {"card_id": "vgc-iron-hands", "species": "Iron Hands", "item": "Assault Vest", "ability": "Quark Drive", "nature": "Adamant",
     "level": 50, "evs": {"hp": 236, "atk": 252, "spe": 20}, "moves": ["Fake Out", "Close Combat", "Wild Charge", "Drain Punch"]},
    {"card_id": "vgc-torkoal", "species": "Torkoal", "item": "Charcoal", "ability": "Drought", "nature": "Quiet",
     "level": 50, "evs": {"hp": 252, "spa": 252, "spd": 4}, "moves": ["Eruption", "Heat Wave", "Yawn", "Protect"]},
    {"card_id": "vgc-cresselia", "species": "Cresselia", "item": "Sitrus Berry", "ability": "Levitate", "nature": "Bold",
     "level": 50, "evs": {"hp": 252, "def": 252, "spd": 4}, "moves": ["Icy Wind", "Moonblast", "Helping Hand", "Protect"]},
]


def _table():
    sets = CardSets(persist=False)
    sets.remember(CARDS)
    return sets.overlay(meta.table())


def _hit(calc, attacker, defender, move_id, **attacker_extra):
    move = movelist.lookup(move_id)
    data = {"id": move_id, "base_power": move["power"], "type": move["type"], "category": move["category"]}
    return calc.damage({"species": attacker, **attacker_extra}, {"species": defender}, data)


def test_body_press_hits_from_archaludons_defense():
    """Turn 1: Body Press took ~51% from Incineroar; read from Attack it was 34%."""
    calc = lookahead.Calculator({"active_pokemon": [], "opponent_active_pokemon": []}, _table())
    hit = _hit(calc, "archaludon", "incineroar", "bodypress", boosts={"atk": -1})  # Intimidate does not lower it
    assert 50 <= hit["expected_pct"] <= 65


def test_foul_play_hits_from_the_targets_attack():
    calc = lookahead.Calculator({"active_pokemon": [], "opponent_active_pokemon": []}, meta.table())
    move = movelist.lookup("foulplay")
    data = {"id": "foulplay", "base_power": move["power"], "type": move["type"], "category": move["category"]}
    plain = calc.damage({"species": "grimmsnarl"}, {"species": "ironhands"}, data)
    boosted = calc.damage({"species": "grimmsnarl"}, {"species": "ironhands", "boosts": {"atk": 2}}, data)
    lowered = calc.damage({"species": "grimmsnarl", "boosts": {"atk": -2}}, {"species": "ironhands"}, data)
    assert boosted["expected_pct"] > 1.8 * plain["expected_pct"]  # the target's +2 Attack doubles it
    assert lowered["expected_pct"] == plain["expected_pct"]  # the user's own Attack does not matter


def test_a_resist_berry_halves_the_super_effective_hit():
    calc = lookahead.Calculator({"active_pokemon": [], "opponent_active_pokemon": []}, _table())
    with_berry = _hit(calc, "archaludon", "tyranitar", "bodypress")
    plain = lookahead.Calculator({"active_pokemon": [], "opponent_active_pokemon": []}, meta.table())
    assert with_berry["expected_pct"] < 0.75 * _hit(plain, "archaludon", "tyranitar", "bodypress")["expected_pct"] * 2.5


FOES_9 = [{"target": 1, "side": "opponent", "species": "torkoal"}, {"target": 2, "side": "opponent", "species": "archaludon"}]


def test_a_lone_cresselia_may_moonblast_when_every_target_is_dispreferred():
    """Turns 9-10: "resisted on Torkoal, hit Archaludon" and "Stamina chip on
    Archaludon, hit Torkoal" banned Moonblast outright; it used Icy Wind twice."""
    table = _table()
    action = _doubles("tyranitar", [{"type": "pass"}],
                      "cresselia", [_move("icywind", []), _move("moonblast", [1, 2], FOES_9), _move("helpinghand", [-1]),
                                    _move("protect", [])], force=(True, False))
    observation = {"turn": 9,
                   "active_pokemon": [{"species": "tyranitar", "hp_fraction": 0.0},
                                      {"species": "cresselia", "types": ["PSYCHIC"], "hp_fraction": 0.47}],
                   "opponent_active_pokemon": [{"species": "torkoal", "types": ["FIRE"], "hp_fraction": 1.0},
                                               {"species": "archaludon", "types": ["STEEL", "DRAGON"], "hp_fraction": 0.89}]}
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(known_moves=table.common_moves), table, lambda line: None)
    for target in (1, 2):
        assert turn_rules.legal({"slot_0": {"option": 0, "target": None}, "slot_1": {"option": 1, "target": target}})
    expected = predict.expectations({"torkoal": {"action": "attack", "target": "cresselia", "confidence": 0.65},
                                     "archaludon": {"action": "attack", "target": "cresselia", "confidence": 0.55}},
                                    ["tyranitar", "cresselia"])
    # Moonblast is back among the lines (it was banned); against Icy Wind it is now a near-tie
    # once two hits on a 47% Cresselia are scored by scenario (match 37a1ee8a), so either may lead.
    ranked = lookahead.ranked_lines(turn_rules, observation, lookahead.Calculator(observation, table), expected)
    assert any(turn_rules.check(a)["slot_1"]["move_id"] == "moonblast" for _, a in ranked[:3])


def test_our_own_switch_out_is_not_damage_taken():
    """Turn 1: Tyranitar switched out and Archaludon's Body Press hit Incineroar;
    the read "Archaludon attacks Tyranitar" was logged as a hit."""
    before = observed.snapshot({"active_pokemon": [{"species": "incineroar", "hp_fraction": 1.0},
                                                   {"species": "tyranitar", "hp_fraction": 1.0}],
                                "opponent_active_pokemon": [{"species": "archaludon", "hp_fraction": 1.0}]})
    after = observed.snapshot({"active_pokemon": [{"species": "incineroar", "hp_fraction": 0.42},
                                                  {"species": "cresselia", "hp_fraction": 1.0}],
                               "opponent_active_pokemon": [{"species": "archaludon", "hp_fraction": 1.0, "moves": ["bodypress"]}]})
    report = observed.last_turn(before, after, {"archaludon": {"action": "attack", "target": "tyranitar"}}, {}, {"tyranitar"})
    assert report["ours"]["tyranitar"] == {"left_field": "switched"}
    assert report["ours"]["cresselia"] == {"entered": True, "hp": 1.0}
    assert report["reads"]["archaludon"]["result"] == "hit, other target"


def test_the_matchup_sees_body_press_and_drain_punch_on_tyranitar():
    _, detail = draft.versus("tyranitar", ["ironhands", "archaludon", "torkoal"], _table())
    assert "Body Press 4x" in detail["threat_moves"]["archaludon"]
    assert detail["threat_moves"]["ironhands"].endswith("4x")


FOES_2 = [{"target": 1, "side": "opponent", "species": "ironhands"}, {"target": 2, "side": "opponent", "species": "archaludon"}]
BAXCALIBUR = {"card_id": "vgc-baxcalibur", "species": "Baxcalibur", "item": "Loaded Dice", "ability": "Thermal Exchange",
              "nature": "Jolly", "evs": {"hp": 4, "atk": 252, "spe": 252}, "moves": ["Icicle Spear", "Glaive Rush", "Ice Shard", "Protect"]}


def test_a_switch_in_takes_the_hit_the_prediction_aimed_at_its_slot():
    """Turn 2: Baxcalibur switched into Iron Hands' predicted Close Combat (~99%):
    the race keyed that attack by Incineroar's name, so after the switch it vanished."""
    table = _table()
    table.cards.remember([BAXCALIBUR])
    action = _doubles("incineroar", [_move("knockoff", [1, 2], FOES_2), _move("flareblitz", [1, 2], FOES_2),
                                     {"type": "switch", "species": "baxcalibur"}],
                      "cresselia", [_move("icywind", []), _move("moonblast", [1, 2], FOES_2), _move("protect", [])])
    observation = {"turn": 2,
                   "active_pokemon": [{"species": "incineroar", "types": ["FIRE", "DARK"], "hp_fraction": 0.42},
                                      {"species": "cresselia", "types": ["PSYCHIC"], "hp_fraction": 1.0}],
                   "opponent_active_pokemon": [
                       {"species": "ironhands", "types": ["FIGHTING", "ELECTRIC"], "hp_fraction": 0.87, "boosts": {"atk": -1}},
                       {"species": "archaludon", "types": ["STEEL", "DRAGON"], "hp_fraction": 1.0, "boosts": {"atk": -1}}],
                   "team": [{"species": "baxcalibur", "hp_fraction": 1.0}]}
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(known_moves=table.common_moves), table, lambda line: None)
    expected = predict.expectations({"ironhands": {"action": "attack", "target": "incineroar", "confidence": 0.8},
                                     "archaludon": {"action": "attack", "target": "cresselia", "confidence": 0.5}},
                                    ["incineroar", "cresselia"])
    payload = turn_rules.check(lookahead.best_line(turn_rules, observation, lookahead.Calculator(observation, table), expected))
    assert payload["slot_0"]["type"] != "switch"
