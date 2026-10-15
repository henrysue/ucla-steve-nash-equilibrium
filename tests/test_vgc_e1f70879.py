"""Regression tests from match e1f70879 (2026-10-14), built from its real boards and cards.

Turn 1 lost Chien-Pao to Archaludon's Body Press while Iron Hands switched out
without acting: both the read and the 70% first-turn table said Archaludon
would Protect, and the opening Fake Out was held. Later: Great Tusk switched
out of a KO, Spore went into an already-asleep Cresselia, Rage Powder was used
into two sleeping foes and twice in a 1-vs-1.
"""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, movelist, predict, principles, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker
from agent.vgc.predict import FoeExpectation
from agent.vgc.style import OpponentStyle

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "ih", "species": "Iron Hands", "ability": "Quark Drive", "item": "Assault Vest", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 252, "spe": 20}, "moves": ["Fake Out", "Close Combat", "Wild Charge", "Drain Punch"]},
    {"card_id": "cp", "species": "Chien-Pao", "ability": "Sword of Ruin", "item": "Focus Sash", "nature": "Jolly",
     "evs": {"hp": 4, "atk": 252, "spe": 252}, "moves": ["Icicle Crash", "Sacred Sword", "Sucker Punch", "Protect"]},
    {"card_id": "gt", "species": "Great Tusk", "ability": "Protosynthesis", "item": "Booster Energy", "nature": "Jolly",
     "evs": {"hp": 252, "atk": 252, "spe": 4}, "moves": ["Headlong Rush", "Close Combat", "Rock Slide", "Protect"]},
    {"card_id": "am", "species": "Amoonguss", "ability": "Regenerator", "item": "Rocky Helmet", "nature": "Bold",
     "evs": {"hp": 244, "def": 172, "spd": 92}, "moves": ["Spore", "Rage Powder", "Pollen Puff", "Sludge Bomb"]},
    {"card_id": "ar", "species": "Archaludon", "ability": "Stamina", "item": "Leftovers", "nature": "Bold",
     "evs": {"hp": 252, "def": 252, "spa": 4}, "moves": ["Body Press", "Draco Meteor", "Flash Cannon", "Protect"]},
    {"card_id": "lt", "species": "Landorus-Therian", "ability": "Intimidate", "item": "Choice Scarf", "nature": "Jolly",
     "evs": {"atk": 252, "def": 4, "spe": 252}, "moves": ["Earthquake", "Rock Slide", "U-turn", "Protect"]},
    {"card_id": "cr", "species": "Cresselia", "ability": "Levitate", "item": "Sitrus Berry", "nature": "Bold",
     "evs": {"hp": 252, "def": 252, "spd": 4}, "moves": ["Icy Wind", "Moonblast", "Helping Hand", "Protect"]},
])
TABLE = SETS.overlay(meta.table())
TYPES = {"ironhands": ["FIGHTING", "ELECTRIC"], "chienpao": ["DARK", "ICE"], "greattusk": ["GROUND", "FIGHTING"],
         "amoonguss": ["GRASS", "POISON"], "archaludon": ["STEEL", "DRAGON"], "landorustherian": ["GROUND", "FLYING"],
         "cresselia": ["PSYCHIC"]}


def _obs(ours, theirs, hp=None, status=None, turn=1):
    hp, status = hp or {}, status or {}
    side = lambda names: [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0), "status": status.get(s, "")}
                          for s in names]  # noqa: E731
    return {"turn": turn, "active_pokemon": side(ours), "opponent_active_pokemon": side(theirs),
            "team": [{"species": s, "hp_fraction": hp.get(s, 1.0)} for s in ("ironhands", "chienpao", "greattusk", "amoonguss")]}


def _foes(*names):
    return [{"target": i, "side": "opponent", "species": s} for i, s in enumerate(names, start=1)]


# -- 1. first-turn Protect is learned, from 35% ------------------------------------------


def test_first_turn_protect_odds_follow_this_opponent():
    style = OpponentStyle()
    assert abs(style.first_turn_rate - 0.35) < 1e-9
    for foe, protected in (("archaludon", False), ("cresselia", False), ("garchomp", False)):
        style._pending = {foe: {"first": True, "threatened": False, "chained": False, "turn": 1}}
        style.resolve({"theirs": {foe: {"revealed": ["protect"] if protected else ["moonblast"]}}})
    assert style.first_turns == 3 and style.first_turn_protects == 0
    assert style.first_turn_rate < 0.25  # three first turns, no Protect: it stops expecting one


# -- 2. the opening Fake Out goes into the foe that threatens a KO -------------------------


def test_turn_1_fakes_out_archaludon_which_threatens_chien_pao():
    """Body Press does 170% to Chien-Pao (4x): that foe is worth a Fake Out, Protect or not."""
    from agent.vgc.strategist import VGCDraftStrategist

    observation = _obs(["ironhands", "chienpao"], ["archaludon", "landorustherian"])
    agent = VGCDraftStrategist(None, log=lambda line: None)
    agent._meta = TABLE
    assert "archaludon" in agent._ko_threats(observation)
    F = _foes("archaludon", "landorustherian")
    action = _doubles("ironhands", [_move("fakeout", [1, 2], F), _move("closecombat", [1, 2], F),
                                    {"type": "switch", "species": "greattusk"}],
                      "chienpao", [_move("iciclecrash", [1, 2], F), _move("protect", [])])
    slots = act.parse_doubles(_as_legal(action))
    tracker = MoveTracker(offense=TABLE.offense, exact_moves=lambda s: (TABLE.card_set(s) or {}).get("moves"),
                          ko_threats=agent._ko_threats)
    per_slot = tracker.begin(slots, observation)
    assert 0 not in per_slot[0].blocked and per_slot[0].targets.get(0) == (1,)


# -- 3. foe status -------------------------------------------------------------------------


def test_spore_into_an_already_asleep_foe_is_withheld():
    observation = _obs(["amoonguss", "greattusk"], ["archaludon", "cresselia"], status={"cresselia": "slp"}, turn=9)
    F = _foes("archaludon", "cresselia")
    action = _doubles("amoonguss", [_move("spore", [1, 2], F), _move("pollenpuff", [1, 2], F)],
                      "greattusk", [_move("headlongrush", [1, 2], F)])
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(), TABLE, lambda line: None)
    spore_on_cresselia = {"slot_0": {"option": 0, "target": 2}, "slot_1": {"option": 0, "target": 1}}
    spore_on_archaludon = {"slot_0": {"option": 0, "target": 1}, "slot_1": {"option": 0, "target": 1}}
    assert not turn_rules.legal(spore_on_cresselia)
    assert turn_rules.legal(spore_on_archaludon)
    both = _obs(["amoonguss", "greattusk"], ["archaludon", "cresselia"], status={"cresselia": "slp", "archaludon": "par"})
    turn_rules = rules.build(slots, both, MoveTracker(), TABLE, lambda line: None)
    assert "already has a status" in turn_rules.per_slot[0].blocked[0]


def test_a_sleeping_foe_mostly_does_nothing():
    observation = _obs(["amoonguss", "greattusk"], ["cresselia", "archaludon"], status={"cresselia": "slp"})
    calc = lookahead.Calculator(observation, TABLE)
    expected = {"cresselia": FoeExpectation(protect=0.3, attacks={"greattusk": 0.6}),
                "archaludon": FoeExpectation(attacks={"amoonguss": 0.8})}
    scaled = lookahead.asleep(calc, expected)
    assert abs(scaled["cresselia"].attacks["greattusk"] - 0.6 * lookahead.SLEEP_AWAKE) < 1e-9
    assert abs(scaled["cresselia"].protect - 0.3 * lookahead.SLEEP_AWAKE) < 1e-9
    assert scaled["archaludon"] is expected["archaludon"]


# -- 4. Rage Powder needs a partner to protect --------------------------------------------


def test_rage_powder_is_worth_nothing_in_a_1v1():
    observation = _obs(["amoonguss"], ["cresselia"], hp={"amoonguss": 0.05, "cresselia": 0.53}, turn=12)
    F = _foes("cresselia")
    action = _doubles("amoonguss", [_move("ragepowder", []), _move("pollenpuff", [1], F), _move("sludgebomb", [1], F)],
                      "greattusk", [{"type": "pass"}])
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(), TABLE, lambda line: None)
    calc = lookahead.Calculator(observation, TABLE)
    expected = predict.expectations({"cresselia": {"action": "attack", "target": "amoonguss", "confidence": 0.75}},
                                    ["amoonguss"])
    best = turn_rules.check(lookahead.best_line(turn_rules, observation, calc, expected))
    assert best["slot_0"]["move_id"] != "ragepowder"


# -- 5. tempo: a healthy Pokemon with a KO stays in ----------------------------------------


def test_turn_3_great_tusk_does_not_switch_out_of_a_ko_on_archaludon():
    observation = _obs(["greattusk", "ironhands"], ["archaludon", "cresselia"],
                       hp={"greattusk": 0.61, "archaludon": 0.46}, turn=3)
    F = _foes("archaludon", "cresselia")
    action = _doubles("greattusk", [_move("headlongrush", [1, 2], F), _move("protect", []),
                                    {"type": "switch", "species": "amoonguss"}],
                      "ironhands", [_move("closecombat", [1, 2], F)])
    slots = act.parse_doubles(_as_legal(action))
    calc = lookahead.Calculator(observation, TABLE)
    expected = predict.expectations({"archaludon": {"action": "attack", "target": "greattusk", "confidence": 0.55}},
                                    ["greattusk", "ironhands"])
    judge = principles.Judge(slots, observation, calc, expected)
    switch = {"type": "doubles_turn", "slot_0": {"type": "switch", "species": "amoonguss"},
              "slot_1": {"type": "move", "move_id": "closecombat", "target": 1}}
    cost, flags = judge.judge(switch)
    assert cost >= principles.TEMPO_COST and any("healthy greattusk that has a KO on archaludon" in f for f in flags)


# -- 6. the read is told what has been wrong -----------------------------------------------


def test_the_read_prompt_carries_the_sash_sleep_and_cornered_facts():
    from agent.vgc import decisions

    text = decisions.predict_decision(["cresselia"], ["amoonguss"], {}).prompt["instructions"]
    assert "Focus Sash only works from full HP" in text
    assert "slp" in text and "in_our_ko_range" in text  # escapes from KO range: learned, not assumed
    assert movelist.STATUS_CONDITION_MOVES >= {"spore", "thunderwave", "willowisp", "yawn"}
