"""Regression tests from match 1fe90ea3 (2026-10-16), a loss to Ace Trainer Claude.

Turn 2 switched a healthy Iron Hands into a 2x Earthquake (-59%) to save a 37%
Grimmsnarl that had already set Reflect. Turn 5, with both of ours in their
combined KO range, Iron Valiant Protected and they KO'd Iron Hands instead.
The draft took and led Grimmsnarl again through the screens plan.
"""

from agent.vgc import actions as act
from agent.vgc import archetypes, lookahead, meta, principles, race
from agent.vgc.cards import CardSets
from agent.vgc.predict import FoeExpectation

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "lt", "species": "Landorus-Therian", "ability": "Intimidate", "item": "Choice Scarf", "nature": "Jolly",
     "evs": {"atk": 252, "def": 4, "spe": 252}, "moves": ["Earthquake", "Rock Slide", "U-turn", "Protect"]},
    {"card_id": "cr", "species": "Cresselia", "ability": "Levitate", "item": "Sitrus Berry", "nature": "Bold",
     "evs": {"hp": 252, "def": 252, "spd": 4}, "moves": ["Icy Wind", "Moonblast", "Helping Hand", "Protect"]},
    {"card_id": "if", "species": "Indeedee-F", "ability": "Psychic Surge", "item": "Rocky Helmet",
     "moves": ["Follow Me", "Helping Hand", "Dazzling Gleam", "Psychic"]},
    {"card_id": "ih", "species": "Iron Hands", "ability": "Quark Drive", "item": "Assault Vest", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 252, "spe": 20}, "moves": ["Fake Out", "Close Combat", "Wild Charge", "Drain Punch"]},
    {"card_id": "iv", "species": "Iron Valiant", "ability": "Quark Drive", "item": "Booster Energy", "nature": "Timid",
     "evs": {"spa": 252, "spd": 4, "spe": 252}, "moves": ["Moonblast", "Close Combat", "Thunderbolt", "Protect"]},
    {"card_id": "gs", "species": "Grimmsnarl", "ability": "Prankster", "item": "Light Clay",
     "moves": ["Light Screen", "Reflect", "Spirit Break", "Thunder Wave"]},
])
TABLE = SETS.overlay(meta.table())
TYPES = {"grimmsnarl": ["DARK", "FAIRY"], "basculegion": ["WATER", "GHOST"], "ironhands": ["FIGHTING", "ELECTRIC"],
         "ironvaliant": ["FAIRY", "FIGHTING"], "cresselia": ["PSYCHIC"], "landorustherian": ["GROUND", "FLYING"],
         "indeedeef": ["PSYCHIC", "NORMAL"]}


# -- 1. don't switch a healthy attacker into a big hit to save a spent, low teammate --------------


def test_turn_2_iron_hands_is_not_switched_into_earthquake_to_save_grimmsnarl():
    observation = {"turn": 2,
                   "active_pokemon": [{"species": "basculegion", "types": TYPES["basculegion"], "hp_fraction": 0.66},
                                      {"species": "grimmsnarl", "types": TYPES["grimmsnarl"], "hp_fraction": 0.37}],
                   "opponent_active_pokemon": [{"species": "cresselia", "types": TYPES["cresselia"], "hp_fraction": 1.0},
                                               {"species": "landorustherian", "types": TYPES["landorustherian"], "hp_fraction": 1.0}],
                   "team": [{"species": s, "hp_fraction": hp} for s, hp in
                            (("basculegion", 0.66), ("grimmsnarl", 0.37), ("ironhands", 1.0), ("ironvaliant", 1.0))]}
    F = [{"target": 1, "side": "opponent", "species": "cresselia"}, {"target": 2, "side": "opponent", "species": "landorustherian"}]
    slots = act.parse_doubles(_as_legal(_doubles(
        "basculegion", [_move("aquajet", [1, 2], F)],
        "grimmsnarl", [_move("thunderwave", [1, 2], F), {"type": "switch", "species": "ironhands"}])))
    calc = lookahead.Calculator(observation, TABLE)
    expected = {"cresselia": FoeExpectation(attacks={"grimmsnarl": 0.6}),
                "landorustherian": FoeExpectation(attacks={"grimmsnarl": 0.55})}
    judge = principles.Judge(slots, observation, calc, expected)
    rescue = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "aquajet", "target": 2},
              "slot_1": {"type": "switch", "species": "ironhands"}}
    cost, flags = judge.judge(rescue)
    assert cost >= principles.RESCUE_COST and any("to save a grimmsnarl" in f for f in flags)


# -- 2. Protecting one of two in KO range moves a share of the focus to the other ------------------


def _turn_5(valiant_hp=0.42):
    observation = {"turn": 5, "fields": {"PSYCHIC_TERRAIN": 4},
                   "active_pokemon": [{"species": "ironvaliant", "types": TYPES["ironvaliant"], "hp_fraction": valiant_hp},
                                      {"species": "ironhands", "types": TYPES["ironhands"], "hp_fraction": 0.41}],
                   "opponent_active_pokemon": [{"species": "cresselia", "types": TYPES["cresselia"], "hp_fraction": 0.71},
                                               {"species": "indeedeef", "types": TYPES["indeedeef"], "hp_fraction": 1.0}]}
    calc = lookahead.Calculator(observation, TABLE)
    ours = [race._mon("us", e, TABLE) for e in calc._ours]
    sim = race._Sim(calc, race.Board(), ours, [race._mon("them", e, TABLE) for e in calc._theirs], observation)
    return sim


def test_protecting_one_of_two_in_ko_range_shifts_focus_to_the_other():
    sim = _turn_5()
    valiant, hands = sim.ours
    assert sim._in_combined_ko(valiant) and sim._in_combined_ko(hands)
    valiant.protecting = 1.0
    e = FoeExpectation(attacks={"ironvaliant": 0.6})
    shifts = sim._focus_shifts(e)
    assert shifts[id(hands)] > 0 and shifts[id(valiant)] < 0
    assert e.attacks == {"ironvaliant": 0.6}  # the shared read is never changed


def test_no_shift_when_the_partner_is_out_of_ko_range():
    sim = _turn_5(valiant_hp=0.42)
    valiant, hands = sim.ours
    hands.hp = 100.0  # healthy: not in their combined KO range
    if sim._in_combined_ko(hands):
        return  # this board KOs even a full Iron Hands: nothing to show
    valiant.protecting = 1.0
    assert sim._focus_shifts(FoeExpectation(attacks={"ironvaliant": 0.6})) == {}


# -- 3. no screens plan in the draft -----------------------------------------------------------


def test_the_screens_plan_is_not_drafted_toward():
    names = [p["name"].lower() for p in archetypes.load_plans()]
    assert not any("screens" in n or "trick room" in n for n in names)
