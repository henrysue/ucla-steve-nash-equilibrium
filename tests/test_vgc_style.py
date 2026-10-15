"""Opponent style: a book player (Protects when it can) or one who plays around reads."""

import copy

from agent.vgc import meta, predict
from agent.vgc.style import PRIOR_STRONG, OpponentStyle

from test_vgc_strategist import BATTLE_OBSERVATION, POKEMON, FakeProvider, _agent, _doubles_action, _doubles_answer, _state

ODDS = {"archaludon": {"likelihood": 0.42, "why": ["its draft card runs Protect", "first turn on the field"]}}


def _turn(style, protected: bool, turn: int, chained=False):
    style.open(ODDS, {"archaludon": True}, {"archaludon": chained}, turn)
    row = {"revealed": ["protect"]} if protected else {"revealed": ["bodypress"]}
    style.resolve({"theirs": {"archaludon": row}})


def test_a_book_player_is_read_as_basic_and_expected_to_protect():
    style = OpponentStyle()
    for turn in (1, 3, 5):
        _turn(style, True, turn)
    assert style.label == "basic"
    assert style.chance(first_turn=False, threatened=True, chained=False) >= 0.6
    assert style.protected == 3 and style.available == 3


def test_holding_protect_when_it_looks_obvious_reads_as_strong():
    """Match ccb570bb: Great Tusk and Archaludon skipped Protect on turns 1-3."""
    style = OpponentStyle()
    for turn in (1, 2, 3):
        _turn(style, False, turn)
    assert style.label == "strong"
    assert style.chance(first_turn=True, threatened=True, chained=False) < 0.35
    assert "did not Protect" in style.notes[-1]


def test_a_foe_that_left_or_showed_nothing_is_not_evidence():
    style = OpponentStyle()
    style.open(ODDS, {"archaludon": True}, {}, 1)
    style.resolve({"theirs": {"archaludon": {"left_field": "switched or fainted"}}})
    style.open(ODDS, {"archaludon": True}, {}, 2)
    style.resolve({"theirs": {"archaludon": {"hp_change": -0.06}}})  # sand chip only
    assert style.p_strong == PRIOR_STRONG  # no Protect evidence either way
    assert style.switches == 1 and "switched out" in style.notes[-1]  # but the switch is counted


def test_the_code_guess_expects_an_attack_from_a_strong_opponent():
    observation = {"turn": 4,
                   "active_pokemon": [{"species": "incineroar", "types": ["FIRE", "DARK"]}],
                   "opponent_active_pokemon": [{"species": "archaludon", "types": ["STEEL", "DRAGON"]}]}
    from agent.vgc import lookahead
    from agent.vgc.moverules import MoveTracker

    table = meta.table()
    calc = lookahead.Calculator(observation, table)
    strong = OpponentStyle()
    for turn in (1, 2, 3):
        _turn(strong, False, turn)
    odds = strong.open(ODDS, {"archaludon": True}, {}, 4)
    guess = predict.code_guess(observation, calc, MoveTracker(known_moves=table.common_moves), table, odds)
    assert guess["archaludon"]["action"] == "attack"
    basic = OpponentStyle()
    for turn in (1, 2, 3):
        _turn(basic, True, turn)
    odds = basic.open(ODDS, {"archaludon": True}, {}, 4)
    guess = predict.code_guess(observation, calc, MoveTracker(known_moves=table.common_moves), table, odds)
    assert guess["archaludon"]["action"] == "protect"


def test_what_we_learn_is_kept_per_opponent(tmp_path, monkeypatch):
    monkeypatch.setenv("VGC_OPPONENT_STYLES", str(tmp_path / "styles.json"))
    style = OpponentStyle()
    for turn in (1, 2, 3):
        _turn(style, False, turn)
    style.save("agent-x")
    again = OpponentStyle.for_opponent("agent-x")
    assert again.p_strong > PRIOR_STRONG and "met before" in again.notes[0]
    assert OpponentStyle.for_opponent("agent-y").p_strong == PRIOR_STRONG


def test_the_strategist_reads_the_opponent_from_its_protect_choices():
    agent = _agent(FakeProvider(_doubles_answer()))
    agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    later = copy.deepcopy(BATTLE_OBSERVATION)
    later["turn"] = 4
    later["opponent_active_pokemon"][1]["moves"] = ["spore"]  # Amoonguss could Protect, and used Spore
    later["opponent_active_pokemon"][1]["hp_fraction"] = 0.6  # and took our hit
    agent.choose_action(_state(_doubles_action(), observation=later), POKEMON)
    assert agent._style.p_strong > PRIOR_STRONG
    assert "amoonguss" in agent._style.notes[-1]
