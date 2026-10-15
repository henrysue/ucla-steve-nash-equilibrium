"""Regression tests from match 0af0f5a5 (2026-10-12, seat 0, vs yuriko): four hits into immunities."""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, movelist, predict, race, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker
from agent.vgc.style import OpponentStyle

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

CARDS = [
    {"card_id": "vgc-gholdengo", "species": "Gholdengo", "item": "Air Balloon", "ability": "Good as Gold",
     "moves": ["Make It Rain", "Shadow Ball", "Nasty Plot", "Protect"]},
    {"card_id": "vgc-incineroar", "species": "Incineroar", "item": "Safety Goggles", "ability": "Intimidate",
     "moves": ["Fake Out", "Knock Off", "Parting Shot", "Flare Blitz"]},
    {"card_id": "vgc-sylveon", "species": "Sylveon", "item": "Wiki Berry", "ability": "Pixilate",
     "moves": ["Hyper Voice", "Shadow Ball", "Calm Mind", "Protect"]},
    {"card_id": "vgc-archaludon", "species": "Archaludon", "item": "Leftovers", "ability": "Stamina",
     "moves": ["Body Press", "Draco Meteor", "Flash Cannon", "Protect"]},
    {"card_id": "vgc-ting-lu", "species": "Ting-Lu", "item": "Leftovers", "ability": "Vessel of Ruin",
     "moves": ["Ruination", "Spikes", "Earthquake", "Protect"]},
    {"card_id": "vgc-iron-hands", "species": "Iron Hands", "item": "Assault Vest", "ability": "Quark Drive",
     "moves": ["Fake Out", "Close Combat", "Wild Charge", "Drain Punch"]},
    {"card_id": "vgc-whimsicott", "species": "Whimsicott", "item": "Focus Sash", "ability": "Prankster",
     "moves": ["Tailwind", "Moonblast", "Encore", "Light Screen"]},
]
TYPES = {"gholdengo": ["STEEL", "GHOST"], "incineroar": ["FIRE", "DARK"], "sylveon": ["FAIRY"], "tinglu": ["DARK", "GROUND"],
         "ironhands": ["FIGHTING", "ELECTRIC"], "whimsicott": ["GRASS", "FAIRY"], "archaludon": ["STEEL", "DRAGON"]}


def _table():
    sets = CardSets(persist=False)
    sets.remember(CARDS)
    return sets.overlay(meta.table())


def _obs(ours, theirs, hp=None, items=None):
    hp, items = hp or {}, items or {}
    entry = lambda s: {"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0), **({"item": items[s]} if s in items else {})}
    return {"turn": 3, "active_pokemon": [entry(s) for s in ours], "opponent_active_pokemon": [entry(s) for s in theirs]}


def _eq(calc, defender):
    move = movelist.lookup("earthquake")
    return calc.damage({"species": "tinglu"}, defender, {"id": "earthquake", "base_power": move["power"], "type": "GROUND",
                                                         "category": "PHYSICAL"}, spread=True)


def test_an_air_balloon_makes_earthquake_miss_until_it_pops():
    """Turn 3: Earthquake into Gholdengo's Air Balloon; we even read its 0% as a Protect."""
    calc = lookahead.Calculator(_obs(["tinglu"], ["gholdengo"]), _table())
    assert _eq(calc, {"species": "gholdengo"})["immune"]
    assert _eq(calc, {"species": "gholdengo", "item": "Air Balloon"})["immune"]
    assert not _eq(calc, {"species": "gholdengo", "item": ""}).get("immune")  # popped
    assert _eq(calc, {"species": "cresselia"})["immune"]  # Levitate


FOES = [{"target": 1, "side": "opponent", "species": "gholdengo"}, {"target": 2, "side": "opponent", "species": "incineroar"}]


def _rules(action, observation):
    table = _table()
    slots = act.parse_doubles(_as_legal(action))
    return rules.build(slots, observation, MoveTracker(known_moves=table.common_moves), table, lambda line: None), table


def test_encore_into_good_as_gold_is_withheld():
    """Turn 6: Whimsicott's Encore into Gholdengo, whose Good as Gold blocks status moves."""
    # (Not beside Incineroar: Dark types ignore Prankster Encore too.)
    foes = [{"target": 1, "side": "opponent", "species": "gholdengo"}, {"target": 2, "side": "opponent", "species": "sylveon"}]
    action = _doubles("whimsicott", [_move("encore", [1, 2], foes), _move("moonblast", [1, 2], foes)],
                      "tinglu", [_move("ruination", [1, 2], foes)])
    turn_rules, _ = _rules(action, _obs(["whimsicott", "tinglu"], ["gholdengo", "sylveon"]))
    assert not turn_rules.legal({"slot_0": {"option": 0, "target": 1}, "slot_1": {"option": 0, "target": 2}})
    assert turn_rules.legal({"slot_0": {"option": 0, "target": 2}, "slot_1": {"option": 0, "target": 1}})


def test_a_single_target_ground_move_into_a_balloon_is_rejected():
    action = _doubles("tinglu", [_move("highhorsepower", [1, 2], FOES)], "ironhands", [_move("wildcharge", [1, 2], FOES)])
    turn_rules, _ = _rules(action, _obs(["tinglu", "ironhands"], ["gholdengo", "incineroar"]))
    assert not turn_rules.legal({"slot_0": {"option": 0, "target": 1}, "slot_1": {"option": 0, "target": 2}})


def test_their_spread_move_hits_both_of_ours_once():
    """Turn 4: Whimsicott switched into a +2 Make It Rain 'aimed at Sneasler'."""
    observation = _obs(["tinglu", "whimsicott"], ["gholdengo"])
    calc = lookahead.Calculator(observation, _table())
    ours = [race._mon("us", e, calc._table) for e in observation["active_pokemon"]]
    sim = race._Sim(calc, race.Board(), ours, [race._mon("them", observation["opponent_active_pokemon"][0], calc._table)], observation)
    e = predict.FoeExpectation(attacks={"tinglu": 0.4, "whimsicott": 0.4})
    actions = sim.foe_actions(sim.theirs[0], e)
    spread = [a for a in actions if a[2].get("spread")]
    assert len(spread) == 1 and {t.name for t, _ in spread[0][3]} == {"tinglu", "whimsicott"}
    assert abs(spread[0][3][0][1] - 0.8) < 1e-9


def test_a_hit_on_a_switching_foe_lands_on_its_best_switch_in():
    """Turn 1: Close Combat into Incineroar hit the Gholdengo it switched to (immune)."""
    foes = [{"target": 1, "side": "opponent", "species": "incineroar"}, {"target": 2, "side": "opponent", "species": "sylveon"}]
    action = _doubles("ironhands", [_move("closecombat", [1, 2], foes)], "tinglu", [_move("protect", [])])
    observation = _obs(["ironhands", "tinglu"], ["incineroar", "sylveon"])
    turn_rules, table = _rules(action, observation)
    calc = lookahead.Calculator(observation, table)
    expected = {"incineroar": predict.FoeExpectation(switch=0.6), "sylveon": predict.FoeExpectation()}
    row = [{"option": 0, "target": 1}, {"option": 0, "target": None}]
    unknown_bench = race.score(row, turn_rules.slots, turn_rules.per_slot, observation, calc, expected)
    ghost_bench = race.score(row, turn_rules.slots, turn_rules.per_slot, observation, calc, expected, ["gholdengo"])
    assert ghost_bench < unknown_bench


def test_a_switching_opponent_raises_every_foes_switch_chance():
    style = OpponentStyle()
    base = style.switch_rate
    style.resolve({"theirs": {"incineroar": {"left_field": "switched or fainted"}, "sylveon": {"hp_change": -0.3},
                              "gholdengo": {"entered": True}}}, 1)
    assert style.switch_rate > base and "in came gholdengo" in style.notes[-1]
    expected = predict.expectations({"sylveon": {"action": "attack", "target": "tinglu", "confidence": 0.7}}, ["tinglu"])
    before = expected["sylveon"].attacks["tinglu"]
    style.add_switching(expected)
    e = expected["sylveon"]
    assert e.switch == style.switch_rate
    assert abs(e.attacks["tinglu"] - before * (1 - style.switch_rate)) < 1e-9  # the rest scales down
