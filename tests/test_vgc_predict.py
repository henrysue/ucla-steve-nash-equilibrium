"""Move selection: rules -> predict the opponent -> our best legal answer."""

import pytest

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, predict, rules
from agent.vgc.moverules import MoveTracker

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import POKEMON, FakeProvider, _agent, _move, _state

TABLE = meta.table()
FOES = [
    {"target": 1, "side": "opponent", "species": "chienpao"},
    {"target": 2, "side": "opponent", "species": "incineroar"},
]
TYPES = {"chienpao": ["DARK", "ICE"], "incineroar": ["FIRE", "DARK"], "ironhands": ["FIGHTING", "ELECTRIC"],
         "fluttermane": ["GHOST", "FAIRY"]}


def _obs(theirs_hp=None, turn=3):
    theirs_hp = theirs_hp or {}
    return {
        "turn": turn,
        "active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": 1.0} for s in ("ironhands", "fluttermane")],
        "opponent_active_pokemon": [
            {"species": s, "types": TYPES[s], "hp_fraction": theirs_hp.get(s, 1.0)} for s in ("chienpao", "incineroar")
        ],
        "available_moves": [
            [{"id": "closecombat", "type": "FIGHTING", "category": "PHYSICAL", "base_power": 120},
             {"id": "protect", "type": "NORMAL", "category": "STATUS", "base_power": 0}],
            [{"id": "moonblast", "type": "FAIRY", "category": "SPECIAL", "base_power": 95},
             {"id": "protect", "type": "NORMAL", "category": "STATUS", "base_power": 0}],
        ],
    }


ACTION = _doubles(
    "ironhands", [_move("closecombat", [1, 2], FOES), _move("protect", [])],
    "fluttermane", [_move("moonblast", [1, 2], FOES), _move("protect", [])],
)


def _setup(observation, tracker=None):
    slots = act.parse_doubles(_as_legal(ACTION))
    tracker = tracker or MoveTracker(known_moves=TABLE.common_moves)
    turn_rules = rules.build(slots, observation, tracker, TABLE, lambda line: None)
    return slots, tracker, turn_rules, lookahead.Calculator(observation, TABLE)


# -- step 2: the code's guess -------------------------------------------------------------------


def test_a_pokemon_with_protect_that_did_not_protect_last_turn_is_predicted_to_protect():
    _, tracker, _, calc = _setup(_obs())
    guess = predict.code_guess(_obs(), calc, tracker, TABLE, {})
    assert guess["chienpao"]["action"] == "protect"  # Chien-Pao's common set has Protect
    assert "did not Protect last turn" in guess["chienpao"]["why"]
    assert guess["incineroar"]["action"] != "protect"  # Incineroar's common set has no Protect


def test_after_protecting_it_is_predicted_to_attack():
    observation = _obs()
    slots, tracker, _, calc = _setup(observation)
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "closecombat", "target": 1},
                    "slot_1": {"type": "move", "move_id": "protect"}})
    # Next decision: Chien-Pao took no damage from Close Combat, so it protected.
    tracker.begin(slots, _obs())
    assert "chienpao" in tracker.protected_last_turn
    guess = predict.code_guess(_obs(), calc, tracker, TABLE, {})
    assert guess["chienpao"]["action"] == "attack"


def test_a_foe_that_lost_hp_did_not_protect():
    observation = _obs()
    slots, tracker, _, _ = _setup(observation)
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "closecombat", "target": 1},
                    "slot_1": {"type": "move", "move_id": "protect"}})
    tracker.begin(slots, _obs(theirs_hp={"chienpao": 0.4}))
    assert "chienpao" not in tracker.protected_last_turn


def test_a_reported_last_move_wins_over_inference():
    observation = _obs()
    observation["opponent_active_pokemon"][1]["last_move"] = "Protect"
    slots, tracker, _, _ = _setup(observation)
    assert tracker.protected_last_turn == {"incineroar": "reported: last move protect"}


# -- the model's prediction ----------------------------------------------------------------------


def test_the_model_prediction_is_validated():
    rows = {"foes": [{"species": "Chien-Pao", "action": "attack", "target": "ironhands", "move": "Sucker Punch",
                      "confidence": 0.8, "why": "w"},
                     {"species": "Kyogre", "action": "attack", "target": "ironhands", "confidence": 0.9}]}
    found = predict.from_model(rows, ["chienpao", "incineroar"], ["ironhands", "fluttermane"])
    assert set(found) == {"chienpao"} and found["chienpao"]["target"] == "ironhands"
    with pytest.raises(ValueError):
        predict.from_model({"foes": []}, ["chienpao"], ["ironhands"])


def test_expectations_keep_some_weight_off_the_prediction():
    expected = predict.expectations({"chienpao": {"action": "protect", "confidence": 0.8}}, ["ironhands", "fluttermane"])
    assert expected["chienpao"].protect == 0.8
    assert sum(expected["chienpao"].attacks.values()) > 0  # a wrong read is never ignored


# -- step 3: beat the prediction -----------------------------------------------------------------


def _best(prediction, observation=None, tracker=None):
    observation = observation or _obs()
    _, _, turn_rules, calc = _setup(observation, tracker)
    expected = predict.expectations(prediction, ["ironhands", "fluttermane"])
    return turn_rules.check(lookahead.best_line(turn_rules, observation, calc, expected))


def test_we_hit_the_foe_that_is_not_protecting():
    # Close Combat is 2x on both; Chien-Pao predicted to Protect -> hit Incineroar.
    payload = _best({"chienpao": {"action": "protect", "confidence": 0.9},
                     "incineroar": {"action": "attack", "target": "fluttermane", "confidence": 0.7}})
    assert payload["slot_0"] == {"type": "move", "move_id": "closecombat", "target": 2}


def test_flipping_the_prediction_flips_the_target():
    payload = _best({"chienpao": {"action": "attack", "target": "fluttermane", "confidence": 0.9},
                     "incineroar": {"action": "protect", "confidence": 0.9}})
    assert payload["slot_0"]["target"] == 1


def test_a_slow_pokemon_predicted_to_be_ko_d_first_protects():
    observation = _obs()
    observation["active_pokemon"][0]["hp_fraction"] = 0.15  # Iron Hands (base speed 50) nearly down
    payload = _best({"chienpao": {"action": "attack", "target": "ironhands", "confidence": 0.95},
                     "incineroar": {"action": "attack", "target": "ironhands", "confidence": 0.95}}, observation)
    assert payload["slot_0"]["move_id"] == "protect"


def test_a_second_protect_is_never_worth_more_than_a_first():
    """Same board, same read: Protect at 1/3 must not outscore Protect at full odds
    (a Focus Sash bookkeeping bug once made it do so)."""
    from agent.vgc import race

    observation = _obs()
    observation["active_pokemon"][1]["hp_fraction"] = 0.15
    prediction = predict.expectations({"chienpao": {"action": "protect", "confidence": 0.9},
                                       "incineroar": {"action": "attack", "target": "fluttermane", "confidence": 0.9}},
                                      ["ironhands", "fluttermane"])
    values = []
    for chained in (False, True):
        slots, tracker, _, _ = _setup(observation)
        if chained:
            tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "closecombat", "target": 2},
                            "slot_1": {"type": "move", "move_id": "protect"}})
        _, _, turn_rules, calc = _setup(observation, tracker)
        rows = [{"option": 0, "target": 2}, {"option": 1, "target": None}]
        values.append(race.score(rows, turn_rules.slots, turn_rules.per_slot, observation, calc, prediction))
    # Expected-value simulation is not exactly monotone; a real inversion (the
    # Focus Sash bug made it 30 points) must not happen.
    assert values[0] >= values[1] - 10


def test_protect_that_wastes_a_confident_attack_on_a_fast_pokemon_is_worth_it():
    """Race view: Incineroar's attack is wasted and Flutter Mane still moves first
    next turn; attacking now lets that next attack land on our partner instead."""
    observation = _obs()
    observation["active_pokemon"][1]["hp_fraction"] = 0.15
    payload = _best({"chienpao": {"action": "protect", "confidence": 0.9},
                     "incineroar": {"action": "attack", "target": "fluttermane", "confidence": 0.9}}, observation)
    assert payload["slot_1"]["move_id"] == "protect"


def test_the_strategist_plays_the_answer_to_the_models_prediction():
    model_says = {
        "state_summary": "s", "reasoning_summary": "r",
        "foes": [{"species": "chienpao", "action": "protect", "target": "none", "move": "protect", "confidence": 0.9, "why": "w"},
                 {"species": "incineroar", "action": "attack", "target": "fluttermane", "move": "knockoff", "confidence": 0.7, "why": "w"}],
    }
    decision = _agent(FakeProvider(model_says)).choose_action(_state(ACTION, observation=_obs()), POKEMON)
    assert decision.action["slot_0"] == {"type": "move", "move_id": "closecombat", "target": 2}
