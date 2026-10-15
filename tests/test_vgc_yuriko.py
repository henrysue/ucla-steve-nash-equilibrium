"""Regression tests from the three games against Yuriko's agents (2026-10-15).

a7cec307 (win), 91e5f459 and 19b82382 (losses): Intimidate stacks left
Chien-Pao at -2 Atk attacking for 15%; a Booster Energy Iron Valiant moved at
~276 while we had it at 184; Icicle Crash and Draco Meteor (90%) missed three
times while valued as sure hits; the strategist described a Draco Meteor into
Archaludon as "securing the Porygon2 kill"; and the log never named the opponent.
"""

import json
import random

from agent.vgc import actions as act
from agent.vgc import decisions, lookahead, meta, principles, rules
from agent.vgc.cards import CardSets
from agent.vgc.report import MatchLog
from agent.vgc.strategist import VGCDraftStrategist, _line_mismatch

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import BATTLE_OBSERVATION, POKEMON, FakeProvider, _doubles_action, _move, _state

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "iv", "species": "Iron Valiant", "ability": "Quark Drive", "item": "Booster Energy", "nature": "Timid",
     "evs": {"spa": 252, "spd": 4, "spe": 252}, "moves": ["Moonblast", "Close Combat", "Thunderbolt", "Protect"]},
    {"card_id": "cp", "species": "Chien-Pao", "ability": "Sword of Ruin", "item": "Focus Sash", "nature": "Jolly",
     "evs": {"hp": 4, "atk": 252, "spe": 252}, "moves": ["Icicle Crash", "Sacred Sword", "Sucker Punch", "Protect"]},
    {"card_id": "gt", "species": "Great Tusk", "ability": "Protosynthesis", "item": "Booster Energy", "nature": "Jolly",
     "evs": {"hp": 252, "atk": 252, "spe": 4}, "moves": ["Headlong Rush", "Close Combat", "Rock Slide", "Protect"]},
    {"card_id": "p2", "species": "Porygon2", "ability": "Download", "item": "Eviolite", "nature": "Quiet",
     "evs": {"hp": 252, "spa": 252, "spd": 4}, "moves": ["Trick Room", "Tri Attack", "Ice Beam", "Recover"]},
    {"card_id": "ar", "species": "Archaludon", "ability": "Stamina", "item": "Leftovers", "nature": "Bold",
     "evs": {"hp": 252, "def": 252, "spa": 4}, "moves": ["Body Press", "Draco Meteor", "Flash Cannon", "Protect"]},
])
TABLE = SETS.overlay(meta.table())
TYPES = {"ironvaliant": ["FAIRY", "FIGHTING"], "chienpao": ["DARK", "ICE"], "greattusk": ["GROUND", "FIGHTING"],
         "porygon2": ["NORMAL"], "archaludon": ["STEEL", "DRAGON"], "dragonite": ["DRAGON", "FLYING"],
         "salamence": ["DRAGON", "FLYING"], "incineroar": ["FIRE", "DARK"]}


def _obs(ours, theirs, boosts=None, status=None, turn=3):
    boosts, status = boosts or {}, status or {}
    side = lambda names: [{"species": s, "types": TYPES[s], "hp_fraction": 1.0, "boosts": boosts.get(s, {}),
                           "status": status.get(s, "")} for s in names]  # noqa: E731
    return {"turn": turn, "active_pokemon": side(ours), "opponent_active_pokemon": side(theirs)}


# -- 1. stat drops: flagged, and switching to reset them is no tempo loss ---------------------

F = [{"target": 1, "side": "opponent", "species": "porygon2"}, {"target": 2, "side": "opponent", "species": "archaludon"}]


def test_chien_pao_at_minus_two_attack_is_flagged_and_free_to_switch():
    observation = _obs(["chienpao", "incineroar"], ["porygon2", "archaludon"], boosts={"chienpao": {"atk": -2}}, turn=4)
    action = _doubles("chienpao", [_move("sacredsword", [1, 2], F), _move("iciclecrash", [1, 2], F),
                                   {"type": "switch", "species": "salamence"}],
                      "incineroar", [_move("flareblitz", [1, 2], F)])
    slots = act.parse_doubles(_as_legal(action))
    judge = principles.Judge(slots, observation, lookahead.Calculator(observation, TABLE), {})
    hit = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "sacredsword", "target": 2},
           "slot_1": {"type": "move", "move_id": "flareblitz", "target": 2}}
    cost, flags = judge.judge(hit)
    assert cost >= principles.WEAKENED_COST and any("at -2 ATK" in f for f in flags)
    out = {**hit, "slot_0": {"type": "switch", "species": "salamence"}}
    assert not any("healthy chienpao" in f for f in judge.judge(out)[1])  # leaving resets the drop


def test_a_draco_meteor_drop_is_still_flagged_and_body_press_uses_defense():
    assert principles._attacking_stat("dracometeor") == "spa"
    assert principles._attacking_stat("bodypress") == "def"
    assert principles._attacking_stat("sacredsword") == "atk"
    assert principles._attacking_stat("protect") is None


# -- 2. Booster Energy Speed ------------------------------------------------------------------


def test_a_booster_energy_iron_valiant_outspeeds_chien_pao():
    calc = lookahead.Calculator(_obs(["chienpao", "greattusk"], ["porygon2", "ironvaliant"]), TABLE)
    valiant, chienpao, tusk = calc._theirs[1], calc.ours(0), calc.ours(1)
    assert calc.speed(valiant) > calc.speed(chienpao)  # ~276 against 205
    assert calc.speed(tusk) < 130  # Great Tusk's Booster Energy boosts Attack, not Speed


# -- 3. accuracy weights expected damage ------------------------------------------------------


def test_accuracy_reads_fractions_percentages_and_sure_hits():
    assert lookahead.accuracy({"id": "iciclecrash"}) == 0.9
    assert lookahead.accuracy({"accuracy": 0.95}) == 0.95
    assert lookahead.accuracy({"accuracy": 90}) == 0.9
    assert lookahead.accuracy({"id": "aerialace"}) == 1.0  # never misses


# -- 4. the strategist restates the line it sends --------------------------------------------


def test_a_restated_line_naming_the_wrong_target_is_caught():
    F7 = [{"target": 1, "side": "opponent", "species": "porygon2"}, {"target": 2, "side": "opponent", "species": "archaludon"}]
    action = _doubles("salamence", [_move("dracometeor", [1, 2], F7)], "incineroar", [_move("flareblitz", [1, 2], F7)])
    slots = act.parse_doubles(_as_legal(action))
    option_1 = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "dracometeor", "target": 2},
                "slot_1": {"type": "move", "move_id": "flareblitz", "target": 2}}
    wrong = "salamence: Draco Meteor -> porygon2 1; incineroar: Flare Blitz -> archaludon 2"
    assert _line_mismatch(wrong, option_1, slots) == "archaludon"  # the Draco Meteor half names Porygon2
    right = "salamence: Draco Meteor -> archaludon 2; incineroar: Flare Blitz -> archaludon 2"
    assert _line_mismatch(right, option_1, slots) == ""


def test_the_review_schema_asks_for_the_line():
    decision = decisions.review_decision([{"option": 1, "value": 0, "line": "x"}], {})
    assert "chosen_line" in decision.schema["properties"]


def test_a_mismatched_restatement_is_retried():
    provider = FakeProvider(
        {"verdict": "play", "play_option": "1", "why": "", "reasoning_summary": "", "chosen_line": "nothing like it"},
        {"verdict": "play", "play_option": "1", "why": "", "reasoning_summary": "", "chosen_line": ""},
    )
    log = []
    agent = VGCDraftStrategist(provider, log=log.append, rng=random.Random(1), clock=lambda: 0.0)
    agent._line_flags = [[], []]
    facts = {k: None for k in ("our_actives", "their_actives", "their_threats", "our_threats", "move_order",
                               "their_possible_switch_ins", "game_plan", "recent_turns")}
    payload = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "flareblitz", "target": 1},
               "slot_1": {"type": "move", "move_id": "moonblast", "target": 1}}
    agent._turn_rules = rules.TurnRules(act.parse_doubles(_as_legal(_doubles_action())), None, [])
    agent._review([(100.0, payload), (99.0, payload)], {}, "", facts, _state(_doubles_action(), observation=BATTLE_OBSERVATION),
                  POKEMON, 0.0)
    assert any("does not name" in line for line in log)


# -- 6. the log names the opponent ------------------------------------------------------------


def test_the_summary_names_the_opponent(tmp_path):
    from altruagent import DecisionContext

    context = DecisionContext(session_id="s", tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                              agent_id="me", seat_position=0, opponents=("Yuriko",))
    agent = VGCDraftStrategist(FakeProvider(), log=lambda line: None, rng=random.Random(1), clock=lambda: 0.0,
                               log_dir=str(tmp_path))
    agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), context)
    summary = agent.summarize_match(None)
    assert summary["opponent_names"] == ["Yuriko"]
    assert "Yuriko" in sorted(tmp_path.glob("*.md"))[0].read_text(encoding="utf-8")
