"""Regression tests from match 9627e1fd (2026-10-13, seat 1, a loss).

Archaludon (Leftovers) sat in Grassy Terrain and regained ~12% a turn while the
race ignored end-of-turn effects, and on turn 3 we double-switched Incineroar
(41%) and Thundurus into its Draco Meteor and Rillaboom's Wood Hammer instead of
pressing Close Combat into a 36% Archaludon.
"""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, predict, race, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

CARDS = [
    {"card_id": "vgc-gyarados", "species": "Gyarados", "item": "Sitrus Berry", "ability": "Intimidate", "nature": "Adamant",
     "evs": {"hp": 252, "atk": 252, "spe": 4}, "moves": ["Waterfall", "Earthquake", "Taunt", "Protect"]},
    {"card_id": "vgc-iron-hands", "species": "Iron Hands", "item": "Assault Vest", "ability": "Quark Drive", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 252, "spe": 20}, "moves": ["Fake Out", "Close Combat", "Wild Charge", "Drain Punch"]},
    {"card_id": "vgc-incineroar", "species": "Incineroar", "item": "Safety Goggles", "ability": "Intimidate", "nature": "Careful",
     "evs": {"hp": 244, "def": 12, "spd": 252}, "moves": ["Fake Out", "Knock Off", "Parting Shot", "Flare Blitz"]},
    {"card_id": "vgc-thundurus", "species": "Thundurus", "item": "Life Orb", "ability": "Defiant", "nature": "Timid",
     "evs": {"spa": 252, "spd": 4, "spe": 252}, "moves": ["Thunderbolt", "Sludge Bomb", "Taunt", "Protect"]},
    {"card_id": "vgc-archaludon", "species": "Archaludon", "item": "Leftovers", "ability": "Stamina", "nature": "Bold",
     "evs": {"hp": 252, "def": 252, "spa": 4}, "moves": ["Body Press", "Draco Meteor", "Flash Cannon", "Protect"]},
    {"card_id": "vgc-rillaboom", "species": "Rillaboom", "item": "Assault Vest", "ability": "Grassy Surge", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 116, "def": 4, "spd": 4, "spe": 148}, "moves": ["Fake Out", "Wood Hammer", "Grassy Glide", "U-turn"]},
    {"card_id": "vgc-chi-yu", "species": "Chi-Yu", "item": "Choice Specs", "ability": "Beads of Ruin", "nature": "Timid",
     "evs": {"hp": 4, "spa": 252, "spe": 252}, "moves": ["Heat Wave", "Dark Pulse", "Overheat", "Snarl"]},
]
TYPES = {"gyarados": ["WATER", "FLYING"], "ironhands": ["FIGHTING", "ELECTRIC"], "incineroar": ["FIRE", "DARK"],
         "thundurus": ["ELECTRIC", "FLYING"], "archaludon": ["STEEL", "DRAGON"], "rillaboom": ["GRASS"]}
SAND_AND_GRASS = {"weather": {"SANDSTORM": 2}, "fields": {"GRASSY_TERRAIN": 1}}


def _table():
    sets = CardSets(persist=False)
    sets.remember(CARDS)
    return sets.overlay(meta.table())


def _sim(ours, theirs, extra=None):
    observation = {"turn": 3, "active_pokemon": [{"species": s, "types": TYPES[s]} for s in ours],
                   "opponent_active_pokemon": [{"species": s, "types": TYPES[s]} for s in theirs], **(extra or {})}
    calc = lookahead.Calculator(observation, _table())
    mons = [[race._mon(side, e, calc._table) for e in observation[key]]
            for side, key in (("us", "active_pokemon"), ("them", "opponent_active_pokemon"))]
    return race._Sim(calc, race.Board(), mons[0], mons[1], observation)


def test_leftovers_and_grassy_terrain_heal_at_the_end_of_the_turn():
    sim = _sim(["gyarados"], ["archaludon", "rillaboom"], SAND_AND_GRASS)
    archaludon, rillaboom = sim.theirs
    archaludon.hp = rillaboom.hp = 36.0
    sim.end_of_turn()
    assert archaludon.hp == 36.0 + 12.5  # Leftovers + Grassy Terrain; Steel ignores the sand
    assert rillaboom.hp == 36.0  # Grassy Terrain +6.25, sandstorm -6.25


def test_sandstorm_chips_unless_immune_and_flyers_miss_the_terrain():
    sim = _sim(["gyarados", "incineroar"], ["archaludon"], SAND_AND_GRASS)
    gyarados, incineroar = sim.ours
    gyarados.hp = incineroar.hp = 50.0
    sim.end_of_turn()
    assert gyarados.hp == 50.0 - 6.25  # airborne: sand chip, no terrain heal
    assert incineroar.hp == 50.0 + 6.25  # Safety Goggles: terrain heal only


def test_sandstorm_chip_can_finish_a_pokemon():
    sim = _sim(["gyarados"], ["archaludon"], {"weather": {"SANDSTORM": 2}})
    sim.ours[0].hp = 5.0
    sim.end_of_turn()
    assert sim.ours[0].alive == 0.0 and sim.ours[0].hp == 0.0


def _turn_3():
    foes = [{"target": 1, "side": "opponent", "species": "archaludon"}, {"target": 2, "side": "opponent", "species": "rillaboom"}]
    switches = [{"type": "switch", "species": "incineroar"}, {"type": "switch", "species": "thundurus"}]
    action = _doubles("gyarados", [_move("earthquake", []), _move("taunt", [1, 2], foes), _move("protect", [])] + switches,
                      "ironhands", [_move("closecombat", [1, 2], foes), _move("drainpunch", [1, 2], foes)] + switches)
    observation = {
        "turn": 3, **SAND_AND_GRASS,
        "active_pokemon": [{"species": "gyarados", "types": TYPES["gyarados"], "hp_fraction": 0.683},
                           {"species": "ironhands", "types": TYPES["ironhands"], "hp_fraction": 0.714}],
        "opponent_active_pokemon": [{"species": "archaludon", "types": TYPES["archaludon"], "hp_fraction": 0.36,
                                     "boosts": {"def": 1}},
                                    {"species": "rillaboom", "types": TYPES["rillaboom"], "hp_fraction": 1.0}],
        "team": [{"species": "incineroar", "hp_fraction": 0.413}, {"species": "thundurus", "hp_fraction": 1.0},
                 {"species": "gyarados", "hp_fraction": 0.683}, {"species": "ironhands", "hp_fraction": 0.714}],
    }
    table = _table()
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(known_moves=table.common_moves), table, lambda line: None)
    return slots, turn_rules, observation, lookahead.Calculator(observation, table)


def _is_switch(slot, choice):
    return slot.options[choice["option"]].get("type") == "switch"


def test_turn_3_does_not_double_switch_out_of_a_winning_position():
    """The model read 'Archaludon Protects 0.55; Rillaboom Wood Hammers Gyarados'.
    The race played a double switch: Incineroar at 41% took Draco Meteor down to 8%."""
    slots, turn_rules, observation, calc = _turn_3()
    read = {"archaludon": {"action": "protect", "confidence": 0.55},
            "rillaboom": {"action": "attack", "target": "gyarados", "confidence": 0.65}}
    expected = predict.expectations(read, ["gyarados", "ironhands"])
    _, best = lookahead.ranked_lines(turn_rules, observation, calc, expected, limit=1, bench=["chiyu"])[0]
    assert not (_is_switch(slots[0], best["slot_0"]) and _is_switch(slots[1], best["slot_1"]))
    assert not _is_switch(slots[1], best["slot_1"])  # Iron Hands stays in and attacks


def test_finishing_a_healing_foe_gains_on_hitting_the_other_one():
    """Close Combat into the 36% Archaludon vs into Rillaboom: a foe left alive
    regains 12.5% a turn (Leftovers + Grassy Terrain), so with residuals the KO
    line gains ground on the line that leaves it healing."""
    slots, turn_rules, observation, calc = _turn_3()
    read = {"archaludon": {"action": "attack", "target": "gyarados", "confidence": 0.6},
            "rillaboom": {"action": "attack", "target": "gyarados", "confidence": 0.65}}
    expected = predict.expectations(read, ["gyarados", "ironhands"])

    def gap():
        rows = [[{"option": 2, "target": None}, {"option": 0, "target": target}] for target in (1, 2)]  # Gyarados Protects
        archaludon, rillaboom = (race.score(row, turn_rules.slots, turn_rules.per_slot, observation, calc, expected, ["chiyu"])
                                 for row in rows)
        return archaludon - rillaboom

    with_residuals = gap()
    original = race._Sim.end_of_turn
    race._Sim.end_of_turn = lambda self: None
    try:
        without = gap()
    finally:
        race._Sim.end_of_turn = original
    assert with_residuals > without > 0
