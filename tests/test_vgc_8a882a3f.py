"""Regression tests from match 8a882a3f (2026-10-09, seat 0)."""

from agent.vgc import actions as act
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

FOES = [{"target": 1, "side": "opponent", "species": "archaludon"}, {"target": 2, "side": "opponent", "species": "gyarados"}]
GYARADOS = {"card_id": "vgc-gyarados", "species": "Gyarados", "item": "Life Orb", "ability": "Intimidate",
            "nature": "Adamant", "evs": {"atk": 252, "spe": 252}, "moves": ["Waterfall", "Earthquake", "Taunt", "Protect"]}


def test_team_preview_placeholders_never_overwrite_a_draft_card():
    """Team Preview listed their Pokemon as item "unknown_item", moves []; that
    replaced Gyarados's real card and the agent read it as having no Protect."""
    sets = CardSets(persist=False)
    sets.remember([GYARADOS])
    sets.remember([{"species": "gyarados", "item": "unknown_item", "moves": []},
                   {"species": "archaludon", "item": "unknown_item", "moves": ["unknown"]}])
    assert sets.get("gyarados")["moves"] == GYARADOS["moves"]
    assert sets.get("gyarados")["item"] == "Life Orb"
    assert sets.get("archaludon") is None


def test_a_saved_placeholder_entry_is_ignored_on_load(tmp_path):
    path = tmp_path / "card_sets.json"
    path.write_text('{"archaludon": {"species": "archaludon", "item": "unknown_item", "moves": []}}', encoding="utf-8")
    assert CardSets(path).get("archaludon") is None


def test_a_roster_entry_only_fills_gaps_in_a_card():
    sets = CardSets(persist=False)
    sets.remember([GYARADOS])
    sets.remember([{"species": "gyarados", "item": "Sitrus Berry", "moves": ["Waterfall"]}])
    assert sets.get("gyarados")["item"] == "Life Orb" and len(sets.get("gyarados")["moves"]) == 4


def _talonflame_turn():
    action = _doubles("talonflame", [_move("tailwind", []), _move("bravebird", [1, 2], FOES)],
                      "incineroar", [_move("knockoff", [1, 2], FOES)])
    return act.parse_doubles(_as_legal(action))


def _observation(ours=("talonflame", "incineroar"), turn=2):
    return {"turn": turn, "active_pokemon": [{"species": s} for s in ours],
            "opponent_active_pokemon": [{"species": "archaludon"}, {"species": "gyarados"}]}


def _play(tracker, slots, move, turn):
    tracker.begin(slots, _observation(turn=turn))
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": move, "target": 0 if move == "tailwind" else 2},
                    "slot_1": {"type": "move", "move_id": "knockoff", "target": 1}})


def test_tailwind_is_withheld_while_it_is_still_up():
    """Turns 5, 6 and 8: Tailwind again while Tailwind was up."""
    slots, tracker = _talonflame_turn(), MoveTracker()
    _play(tracker, slots, "tailwind", 2)
    for turn in (3, 4, 5):  # Tailwind covers the turn it is set and the next three
        rules = tracker.begin(slots, _observation(turn=turn))
        assert 0 in rules[0].blocked and "already up" in rules[0].blocked[0]
        tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "bravebird", "target": 2},
                        "slot_1": {"type": "move", "move_id": "knockoff", "target": 1}})
    assert 0 not in tracker.begin(slots, _observation(turn=6))[0].blocked


def test_a_setter_knocked_out_before_moving_did_not_set_it():
    slots, tracker = _talonflame_turn(), MoveTracker()
    _play(tracker, slots, "tailwind", 2)
    other = act.parse_doubles(_as_legal(_doubles("ironvaliant", [_move("tailwind", [])], "incineroar",
                                                 [_move("knockoff", [1, 2], FOES)])))
    rules = tracker.begin(other, _observation(ours=("ironvaliant", "incineroar"), turn=3))
    assert 0 not in rules[0].blocked
