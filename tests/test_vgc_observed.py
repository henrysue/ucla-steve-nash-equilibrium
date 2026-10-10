"""What happened last turn, read checks, and the field as the server shows it."""

import copy

from agent.vgc import actions as act
from agent.vgc import observed
from agent.vgc.moverules import MoveTracker

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import (
    BATTLE_OBSERVATION,
    POKEMON,
    FakeProvider,
    _agent,
    _doubles_action,
    _doubles_answer,
    _move,
    _state,
)


def _obs(theirs, ours=None):
    return {
        "active_pokemon": ours or [{"species": "incineroar", "hp_fraction": 0.8}, {"species": "fluttermane", "hp_fraction": 1.0}],
        "opponent_active_pokemon": theirs,
    }


def test_a_foe_that_protected_as_predicted_is_a_hit():
    before = observed.snapshot(_obs([{"species": "amoonguss", "hp_fraction": 1.0}]))
    after = observed.snapshot(_obs([{"species": "amoonguss", "hp_fraction": 1.0, "moves": ["protect"]}]))
    report = observed.last_turn(before, after, {"amoonguss": {"action": "protect"}}, {})
    assert report["theirs"]["amoonguss"]["revealed"] == ["protect"]
    assert report["reads"]["amoonguss"]["result"] == "hit"


def test_a_predicted_attack_into_a_protect_is_a_miss():
    before = observed.snapshot(_obs([{"species": "amoonguss", "hp_fraction": 1.0}]))
    after = observed.snapshot(_obs([{"species": "amoonguss", "hp_fraction": 1.0}]))
    report = observed.last_turn(before, after, {"amoonguss": {"action": "attack", "target": "fluttermane"}},
                                {"amoonguss": "inferred: our attack last turn did no damage"})
    assert report["reads"]["amoonguss"]["result"] == "miss"


def test_an_attack_on_the_predicted_target_is_a_hit_and_a_switch_is_seen():
    before = observed.snapshot(_obs([{"species": "landorustherian", "hp_fraction": 0.5},
                                     {"species": "amoonguss", "hp_fraction": 1.0}]))
    after = observed.snapshot(_obs([{"species": "landorustherian", "hp_fraction": 0.5, "moves": ["rockslide"]},
                                    {"species": "incineroar", "hp_fraction": 1.0}],
                                   ours=[{"species": "incineroar", "hp_fraction": 0.8},
                                         {"species": "fluttermane", "hp_fraction": 0.6}]))
    report = observed.last_turn(before, after, {
        "landorustherian": {"action": "attack", "target": "fluttermane"},
        "amoonguss": {"action": "switch"},
    }, {})
    assert report["reads"]["landorustherian"]["result"] == "hit"
    assert report["reads"]["amoonguss"]["result"] == "hit"
    assert report["theirs"]["incineroar"]["entered"] is True
    assert report["ours"]["fluttermane"]["hp_change"] == -0.4


def test_pp_spent_shows_a_repeated_move():
    before = observed.snapshot(_obs([{"species": "gyarados", "hp_fraction": 1.0,
                                      "moves": [{"id": "waterfall", "current_pp": 15}]}]))
    after = observed.snapshot(_obs([{"species": "gyarados", "hp_fraction": 1.0,
                                     "moves": [{"id": "waterfall", "current_pp": 14}]}]))
    assert observed.last_turn(before, after, None, {})["theirs"]["gyarados"]["used"] == ["waterfall"]


def test_reads_are_tallied_per_predicted_action():
    rows = [{"predicted": "protect", "result": "hit"}, {"predicted": "protect", "result": "miss"},
            {"predicted": "attack -> x", "result": "hit, other target"}, {"predicted": "attack -> x", "result": "unknown"}]
    tally = observed.tally(rows)
    assert tally["protect"] == {"hit": 1, "miss": 1, "unknown": 0, "hit_rate": 0.5}
    assert tally["attack"]["hit"] == 1 and tally["attack"]["hit_rate"] == 1.0


def test_field_conditions_are_read_whatever_their_shape():
    assert observed.conditions({"TAILWIND": 3, "PSYCHIC_TERRAIN": 0}) == {"tailwind", "psychicterrain"}  # 0 = started turn 0
    assert observed.conditions({"REFLECT": None}) == set()
    assert observed.conditions(["SideCondition.LIGHT_SCREEN"]) == {"lightscreen"}
    assert observed.conditions("Field.TRICK_ROOM") == {"trickroom"}
    state = observed.field_state({"side_conditions": {"tailwind": 1}, "fields": {"trick_room": 2}})
    assert state == {"ours": ["tailwind"], "theirs": [], "field": ["trickroom"]}
    assert observed.field_state({}) is None


def test_the_tracker_trusts_the_servers_side_conditions():
    foes = [{"target": 1, "side": "opponent", "species": "gyarados"}]
    slots = act.parse_doubles(_as_legal(_doubles("talonflame", [_move("tailwind", []), _move("bravebird", [1], foes)],
                                                 "incineroar", [_move("knockoff", [1], foes)])))
    base = {"turn": 4, "active_pokemon": [{"species": "talonflame"}, {"species": "incineroar"}],
            "opponent_active_pokemon": [{"species": "gyarados"}]}
    tracker = MoveTracker()
    assert 0 in tracker.begin(slots, {**base, "side_conditions": {"tailwind": 2}})[0].blocked
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "bravebird", "target": 1},
                    "slot_1": {"type": "move", "move_id": "knockoff", "target": 1}})
    assert 0 not in tracker.begin(slots, {**base, "side_conditions": {}})[0].blocked


def test_the_strategist_logs_what_happened_and_checks_its_read():
    agent = _agent(FakeProvider(_doubles_answer()))
    agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    later = copy.deepcopy(BATTLE_OBSERVATION)
    later["turn"] = 4
    later["opponent_active_pokemon"][1]["moves"] = ["protect"]  # Amoonguss protected, as read
    agent.choose_action(_state(_doubles_action(), observation=later), POKEMON)
    assert agent._happened["reads"]["amoonguss"]["result"] == "hit"
