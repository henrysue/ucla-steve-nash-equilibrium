"""Regression tests from match 666b209d (2026-10-14, a win), built from its real cards.

Turn 2: Rillaboom's Fake Out flinched Dragonite, so Scale Shot never reached
Chi-Yu; the tracker logged "Chi-Yu protected" (its card has no Protect) and
the read check credited the Fake Out read with Chi-Yu's Overheat damage. The
log did not keep what the strategist was shown, nor the final board, and the
Specs Chi-Yu the game plan named as their win condition ranked as any foe.
"""

import random

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, observed
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker
from agent.vgc.report import Analysis
from agent.vgc.strategist import VGCDraftStrategist, _ending

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import BATTLE_OBSERVATION, POKEMON, FakeProvider, _doubles_action, _move, _state

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "cy", "species": "Chi-Yu", "ability": "Beads of Ruin", "item": "Choice Specs",
     "moves": ["Heat Wave", "Dark Pulse", "Overheat", "Snarl"]},
    {"card_id": "rb", "species": "Rillaboom", "ability": "Grassy Surge", "item": "Assault Vest",
     "moves": ["Fake Out", "Wood Hammer", "Grassy Glide", "U-turn"]},
    {"card_id": "p2", "species": "Porygon2", "ability": "Download", "item": "Eviolite",
     "moves": ["Trick Room", "Tri Attack", "Ice Beam", "Recover"]},
])
TABLE = SETS.overlay(meta.table())
TYPES = {"ironhands": ["FIGHTING", "ELECTRIC"], "dragonite": ["DRAGON", "FLYING"], "chiyu": ["DARK", "FIRE"],
         "rillaboom": ["GRASS"], "porygon2": ["NORMAL"]}


def _obs(theirs, hp=None, turn=2):
    hp = hp or {}
    return {"turn": turn,
            "active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in ("ironhands", "dragonite")],
            "opponent_active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in theirs]}


def _tracker():
    return MoveTracker(known_moves=TABLE.common_moves, exact_moves=lambda s: (TABLE.card_set(s) or {}).get("moves"))


FOES = [{"target": 1, "side": "opponent", "species": "chiyu"}, {"target": 2, "side": "opponent", "species": "porygon2"}]
SLOTS = act.parse_doubles(_as_legal(_doubles(
    "ironhands", [_move("closecombat", [1, 2], FOES), _move("drainpunch", [1, 2], FOES)],
    "dragonite", [_move("dragonclaw", [1, 2], FOES), _move("protect", [])])))
DRAGON_CLAW_INTO_CHI_YU = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "drainpunch", "target": 2},
                          "slot_1": {"type": "move", "move_id": "dragonclaw", "target": 1}}  # cannot miss


def _two_turns(tracker, theirs):
    """Turn 1 passes (nobody fresh after it), then Dragon Claw into Chi-Yu, which takes nothing."""
    tracker.begin(SLOTS, _obs(theirs, turn=5))
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "drainpunch", "target": 2},
                    "slot_1": {"type": "move", "move_id": "protect"}})
    tracker.begin(SLOTS, _obs(theirs, turn=6))
    tracker.record(DRAGON_CLAW_INTO_CHI_YU)
    tracker.expect({"chiyu": 1.03})
    tracker.begin(SLOTS, _obs(theirs, turn=7))
    return tracker


def test_a_foe_whose_card_has_no_protect_is_never_read_as_protecting():
    assert "chiyu" not in _two_turns(_tracker(), ["chiyu", "porygon2"]).protected_last_turn
    # Without its card, the same board is read as a Protect: the card is what rules it out.
    assert "chiyu" in _two_turns(MoveTracker(), ["chiyu", "porygon2"]).protected_last_turn


def test_a_fresh_rillaboom_with_fake_out_makes_no_damage_no_evidence():
    tracker = _tracker()
    tracker.begin(SLOTS, _obs(["chiyu", "rillaboom"]))
    assert tracker._flinch_risk  # Rillaboom: first turn out, Fake Out on its card
    tracker.record({"type": "doubles_turn"})
    tracker.begin(SLOTS, _obs(["chiyu", "rillaboom"], turn=3))
    assert not tracker._flinch_risk  # its Fake Out turn is over


# -- 2. a Fake Out read is judged only by a Fake Out we saw ---------------------------------


def test_a_fake_out_read_without_a_fake_out_shown_is_unknown():
    ours = {"ironhands": {"hp_change": -0.853}, "dragonite": {"hp_change": -0.096}}
    fainted_before_showing = {"left_field": "fainted"}
    guess = {"action": "fake_out", "target": "ironhands"}
    assert observed.verdict(guess, fainted_before_showing, ours)["result"] == "unknown"
    assert observed.verdict(guess, {"revealed": ["fakeout"]}, ours)["result"].startswith("hit")
    assert observed.verdict(guess, {"revealed": ["woodhammer"]}, ours)["result"] == "miss"


# -- 4. the log keeps what the strategist saw, and the last board ---------------------------


def test_the_final_call_options_are_logged():
    options = [{"option": 1, "value": 122.4, "line": "ironhands: Close Combat -> rillaboom 2; dragonite: Scale Shot -> chiyu 1",
                "breaks_principles": "none"},
               {"option": 2, "value": 121.9, "line": "ironhands: Close Combat -> chiyu 1; dragonite: Scale Shot -> chiyu 1",
                "breaks_principles": ["both attacks into chiyu"]}]
    analysis = Analysis(turn="Turn 2", state_summary="", prediction={}, source="model", code_guess={},
                        lines=[(122.4, "a"), (121.9, "b")], offered=options)
    text = analysis.render()
    assert "Shown to the strategist" in text and "Close Combat -> rillaboom 2" in text
    assert "[breaks: both attacks into chiyu]" in text
    assert analysis.as_dict()["offered"] == options


def test_the_strategist_keeps_the_options_it_showed():
    provider = FakeProvider({"verdict": "play", "play_option": "1", "why": "", "reasoning_summary": ""})
    agent = VGCDraftStrategist(provider, log=[].append, rng=random.Random(1), clock=lambda: 0.0)
    agent._line_flags = [[], []]
    facts = {k: None for k in ("our_actives", "their_actives", "their_threats", "our_threats", "move_order",
                               "their_possible_switch_ins", "game_plan", "recent_turns")}
    agent._review([(100.0, {}), (99.0, {})], {}, "", facts, _state(_doubles_action(), observation=BATTLE_OBSERVATION),
                  POKEMON, 0.0)
    assert [o["option"] for o in agent._offered] == [1, 2]


def test_the_board_at_our_last_decision_is_kept_when_the_end_state_has_none():
    class Final:
        final_result = {"winner_agent_id": "me"}
        raw = {}

    last = observed.snapshot(_obs(["porygon2"], hp={"porygon2": 0.72}, turn=8))
    ending = _ending(Final(), last)
    assert ending["board_at_our_last_decision"]["theirs"]["porygon2"] == 0.72


# -- 5. a Choice Specs / Band attacker is their win condition ---------------------------------


def test_a_specs_chi_yu_is_their_win_condition():
    calc = lookahead.Calculator(_obs(["chiyu", "rillaboom"]), TABLE)
    found = lookahead.win_conditions(calc)
    assert "chiyu" in found and "Choice Specs" in found["chiyu"]
    assert "rillaboom" not in found  # Assault Vest
