"""The move list's blunder checks, built from match 6a741e0d's real boards."""

from agent.vgc import actions as act
from agent.vgc import meta, movelist

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import POKEMON, FakeProvider, _agent, _move, _state

TABLE = meta.table()
FOES_TA = [
    {"target": 1, "side": "opponent", "species": "tornadus"},
    {"target": 2, "side": "opponent", "species": "archaludon"},
]
FOES_GI = [
    {"target": 1, "side": "opponent", "species": "gyarados"},
    {"target": 2, "side": "opponent", "species": "ironhands"},
]
FOES_GT = [
    {"target": 1, "side": "opponent", "species": "gyarados"},
    {"target": 2, "side": "opponent", "species": "tornadus"},
]
TYPES = {
    "gyarados": ["WATER", "FLYING"], "ironhands": ["FIGHTING", "ELECTRIC"], "tornadus": ["FLYING"],
    "archaludon": ["STEEL", "DRAGON"], "tinglu": ["DARK", "GROUND"], "landorustherian": ["GROUND", "FLYING"],
}


def _obs(ours, theirs):
    return {
        "active_pokemon": [{"species": s, "types": TYPES[s]} for s in ours],
        "opponent_active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": 1.0} for s in theirs],
    }


def _board(action, observation):
    slots = act.parse_doubles(_as_legal(action))
    return slots, movelist.Board(slots, observation, TABLE)


def test_the_move_list_knows_who_earthquake_hits():
    assert movelist.lookup("earthquake")["target"] == "allAdjacent"
    assert movelist.lookup("bleakwindstorm")["target"] == "allAdjacentFoes"
    assert "drops_target_stats" not in movelist.lookup("flashcannon")["tags"]  # 10% is not a reason


def test_earthquake_beside_iron_hands_is_rejected_unless_it_protects():
    """Turn 1, seat 1: Gyarados Earthquake next to Iron Hands (Ground is 2x on it)."""
    action = _doubles("gyarados", [_move("earthquake", []), _move("waterfall", [1, 2], FOES_TA)],
                      "ironhands", [_move("fakeout", [1, 2], FOES_TA), _move("protect", [])])
    slots, board = _board(action, _obs(["gyarados", "ironhands"], ["tornadus", "archaludon"]))
    attack = act.build_doubles(slots, {"slot_0": {"option": 0, "target": None}, "slot_1": {"option": 0, "target": 1}})
    assert "also hits our own ironhands (2x)" in movelist.row_problem(board, attack)
    guarded = act.build_doubles(slots, {"slot_0": {"option": 0, "target": None}, "slot_1": {"option": 1, "target": None}})
    assert movelist.row_problem(board, guarded) is None


def test_earthquake_beside_a_flying_partner_is_fine():
    """Seat 0's Landorus-T Earthquake beside Tornadus: the partner is immune."""
    action = _doubles("landorustherian", [_move("earthquake", [])], "tornadus", [_move("bleakwindstorm", [])])
    slots, board = _board(action, _obs(["landorustherian", "tornadus"], ["gyarados", "tinglu"]))
    payload = act.build_doubles(slots, {"slot_0": {"option": 0, "target": None}, "slot_1": {"option": 0, "target": None}})
    assert movelist.row_problem(board, payload) is None


def test_earthquake_into_two_flying_foes_is_withheld():
    action = _doubles("tinglu", [_move("earthquake", []), _move("ruination", [1, 2], FOES_GT)],
                      "landorustherian", [_move("rockslide", [])])
    slots, board = _board(action, _obs(["tinglu", "landorustherian"], ["gyarados", "tornadus"]))
    blocked = movelist.option_blocks(board)
    assert "cannot affect either foe" in blocked[0][0]


def test_flash_cannon_into_gyarados_is_outclassed_by_draco_meteor():
    """Turn 1, seat 0: Archaludon Flash Cannon (0.5x) with Draco Meteor available."""
    action = _doubles("archaludon", [_move("flashcannon", [1, 2], FOES_GI), _move("dracometeor", [1, 2], FOES_GI)],
                      "tornadus", [_move("tailwind", [])])
    slots, board = _board(action, _obs(["archaludon", "tornadus"], ["gyarados", "ironhands"]))
    blocked = movelist.option_blocks(board)
    assert "outclassed by Draco Meteor" in blocked[0][0] or "resisted by every foe" in blocked[0][0]
    assert 1 not in blocked[0]


def test_a_utility_move_is_never_called_outclassed():
    action = _doubles("ironhands", [_move("fakeout", [1, 2], FOES_TA), _move("closecombat", [1, 2], FOES_TA)],
                      "gyarados", [_move("waterfall", [1, 2], FOES_TA)])
    slots, board = _board(action, _obs(["ironhands", "gyarados"], ["tornadus", "archaludon"]))
    assert 0 not in movelist.option_blocks(board).get(0, {})


def test_thunder_wave_into_ground_types_is_withheld():
    foes = [{"target": 1, "side": "opponent", "species": "tinglu"}, {"target": 2, "side": "opponent", "species": "landorustherian"}]
    action = _doubles("ironhands", [_move("thunderwave", [1, 2], foes), _move("closecombat", [1, 2], foes)],
                      "gyarados", [_move("waterfall", [1, 2], foes)])
    slots, board = _board(action, _obs(["ironhands", "gyarados"], ["tinglu", "landorustherian"]))
    assert "fails on every legal target" in movelist.option_blocks(board)[0][0]


def test_an_attack_into_an_immune_target_is_rejected():
    action = _doubles("tinglu", [_move("highhorsepower", [1, 2], FOES_TA)], "gyarados", [_move("waterfall", [1, 2], FOES_TA)])
    slots, board = _board(action, _obs(["tinglu", "gyarados"], ["tornadus", "archaludon"]))
    into_tornadus = act.build_doubles(slots, {"slot_0": {"option": 0, "target": 1}, "slot_1": {"option": 0, "target": 2}})
    assert "immune tornadus" in movelist.row_problem(board, into_tornadus)
    into_archaludon = act.build_doubles(slots, {"slot_0": {"option": 0, "target": 2}, "slot_1": {"option": 0, "target": 1}})
    assert movelist.row_problem(board, into_archaludon) is None


def test_hostile_moves_at_our_own_partner_are_rejected_but_pollen_puff_is_not():
    entries = FOES_TA + [{"target": -2, "side": "ally", "species": "gyarados"}]
    action = _doubles("ironhands", [_move("taunt", [1, 2, -2], entries), _move("pollenpuff", [1, 2, -2], entries)],
                      "gyarados", [_move("waterfall", [1, 2], FOES_TA)])
    slots, board = _board(action, _obs(["ironhands", "gyarados"], ["tornadus", "archaludon"]))
    taunt = act.build_doubles(slots, {"slot_0": {"option": 0, "target": -2}, "slot_1": {"option": 0, "target": 1}})
    assert "aimed at our own partner" in movelist.row_problem(board, taunt)
    puff = act.build_doubles(slots, {"slot_0": {"option": 1, "target": -2}, "slot_1": {"option": 0, "target": 1}})
    assert movelist.row_problem(board, puff) is None


def test_the_selector_never_earthquakes_beside_an_unprotected_iron_hands():
    action = _doubles("gyarados", [_move("earthquake", []), _move("waterfall", [1, 2], FOES_TA)],
                      "ironhands", [_move("fakeout", [1, 2], FOES_TA), _move("closecombat", [1, 2], FOES_TA)])
    observation = {**_obs(["gyarados", "ironhands"], ["tornadus", "archaludon"]), "turn": 3}
    prediction = {
        "state_summary": "s", "reasoning_summary": "r",
        "foes": [{"species": "tornadus", "action": "attack", "target": "ironhands", "move": "bleakwindstorm", "confidence": 0.7, "why": "w"},
                 {"species": "archaludon", "action": "attack", "target": "gyarados", "move": "electroshot", "confidence": 0.7, "why": "w"}],
    }
    decision = _agent(FakeProvider(prediction)).choose_action(_state(action, observation=observation), POKEMON)
    assert decision.action["slot_0"]["move_id"] != "earthquake"
    fallback = _agent().choose_action(_state(action, observation=observation), POKEMON)
    assert fallback.action["slot_0"]["move_id"] != "earthquake"


def _templated_raw(action):
    return action


# -- match 6a741e0d: lineup and switch-in fixes ---------------------------------------------

SEAT1_ROSTER = ["talonflame", "whimsicott", "ironhands", "gyarados", "fluttermane", "tinglu"]


def _lineup_state(observation=None):
    from test_vgc_teamshape import _lineup

    return _state(_lineup(SEAT1_ROSTER), phase="team_preview",
                  observation=observation or {"opponent_roster": [{"species": s} for s in
                                                                    ["Pelipper", "Archaludon", "Landorus-Therian", "Rillaboom", "Klefki", "Tornadus"]]})


def test_a_weak_tailwind_setter_is_never_forced_into_the_four():
    """Match f14f1652: the setter rule forced a 44.6% Hatterene over Great Tusk. Only a
    setter that is easy to play counts now, and only Tailwind (we do not play Trick Room)."""
    from agent.vgc import pick_priority

    agent = _agent()
    assert pick_priority.easy_strength("Talonflame", agent._meta)[0] < 0
    problems = agent._lineup_problems(SEAT1_ROSTER, ["gyarados", "ironhands", "tinglu", "fluttermane"], ["gyarados", "ironhands"])
    assert not any("Tailwind setter" in p for p in problems)


def test_switch_options_show_the_cost_of_coming_in():
    from agent.vgc import lookahead

    action = _doubles("gyarados", [_move("waterfall", [1, 2], FOES_TA), {"type": "switch", "species": "fluttermane"}],
                      "tinglu", [_move("ruination", [1, 2], FOES_TA)])
    observation = _obs(["gyarados", "tinglu"], ["tornadus", "archaludon"])
    slots = act.parse_doubles(_as_legal(action))
    shown = act.describe_slots(slots, _state(observation=observation),
                               calculator=lookahead.Calculator(observation, TABLE))
    risk = shown[0]["options"][1]["switch_in_risk"]
    assert risk["from"] in ("tornadus", "archaludon") and risk["expected_pct"] > 30


# -- match 48f23a42: re-ask, Stamina, lead clash ------------------------------------------

TYPES.update({"gholdengo": ["STEEL", "GHOST"], "tyranitar": ["ROCK", "DARK"], "dragonite": ["DRAGON", "FLYING"]})
FOES_GA = [
    {"target": 1, "side": "opponent", "species": "gholdengo"},
    {"target": 2, "side": "opponent", "species": "archaludon"},
]


def _turn_one():
    return _doubles(
        "tyranitar", [_move("rockslide", []), _move("knockoff", [1, 2], FOES_GA), _move("protect", [])],
        "landorustherian", [_move("earthquake", []), _move("uturn", [1, 2], FOES_GA), _move("protect", [])],
    )


def _answer(rows, payoffs):
    return {
        "state_summary": "s", "threats": "t", "pruned": "p", "situation": "lead", "opponent_last_action": "none",
        "key_assumption": "k", "reasoning_summary": "r", "our_options": rows,
        "opponent_options": [{"label": "attack", "category": "attack"}], "payoffs": payoffs,
    }


EQ_ROW = {"label": "Rock Slide + EQ", "slot_0": {"option": 0, "target": None}, "slot_1": {"option": 0, "target": None}}
KNOCK_ROW = {"label": "Knock Off + U-turn", "slot_0": {"option": 1, "target": 1}, "slot_1": {"option": 1, "target": 1}}
SAFE_ROWS = [
    {"label": "Rock Slide + Landorus Protect", "slot_0": {"option": 0, "target": None}, "slot_1": {"option": 2, "target": None}},
    {"label": "Knock Off Gholdengo + EQ, Tyranitar... no", "slot_0": {"option": 2, "target": None}, "slot_1": {"option": 0, "target": None}},
]


def test_scale_shot_into_stamina_archaludon_is_rejected():
    action = _doubles("dragonite", [_move("scaleshot", [1, 2], FOES_GA), _move("extremespeed", [1, 2], FOES_GA)],
                      "landorustherian", [_move("protect", [])])
    slots, board = _board(action, _obs(["dragonite", "landorustherian"], ["gholdengo", "archaludon"]))
    payload = act.build_doubles(slots, {"slot_0": {"option": 0, "target": 2}, "slot_1": {"option": 0, "target": None}})
    assert "Stamina" in movelist.row_problem(board, payload)


def test_a_lead_whose_earthquake_hits_its_partner_is_flagged():
    agent = _agent()
    assert "hits lead partner tyranitar" in agent._lead_clash(["tyranitar", "landorustherian"])
    assert agent._lead_clash(["dragonite", "landorustherian"]) is None  # Flying partner: immune


def test_stone_edge_goes_into_gholdengo_rather_than_feeding_stamina_archaludon():
    """No longer a hard rule: the race sees each hit raise Archaludon's Defense
    (weaker physical hits, stronger Body Press) and picks the other foe."""
    from agent.vgc import lookahead, rules
    from agent.vgc.moverules import MoveTracker

    action = _doubles("landorustherian", [_move("stoneedge", [1, 2], FOES_GA)], "tyranitar", [_move("knockoff", [1, 2], FOES_GA)])
    observation = _obs(["landorustherian", "tyranitar"], ["gholdengo", "archaludon"])
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(known_moves=TABLE.common_moves), TABLE, lambda line: None)
    calc = lookahead.Calculator(observation, TABLE)
    payload = turn_rules.check(lookahead.best_line(turn_rules, observation, calc, {}))
    assert payload["slot_0"]["target"] == 1


def test_a_super_effective_target_is_allowed():
    foes = [{"target": 1, "side": "opponent", "species": "gyarados"}, {"target": 2, "side": "opponent", "species": "tinglu"}]
    action = _doubles("ironhands", [_move("wildcharge", [1, 2], foes)], "landorustherian", [_move("protect", [])])
    slots, board = _board(action, _obs(["ironhands", "landorustherian"], ["tinglu", "gyarados"]))
    # Wild Charge: 0x on Ting-Lu (immune) would already be blocked; use Gyarados (2x) vs Ting-Lu.
    good = act.build_doubles(slots, {"slot_0": {"option": 0, "target": 1}, "slot_1": {"option": 0, "target": None}})
    assert movelist.row_problem(board, good) is None
