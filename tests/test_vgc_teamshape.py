"""Opening-turn Fake Out, a less predictable mix, and role-balanced teams.

Driven by the 2026-10-09 loss (match f547): six drafted, four of them
supports; brought Incineroar, Rillaboom, Amoonguss and Ting-Lu with Iron
Valiant left home; the opening turn timed out and Fake Out was not used;
and 17 of 21 matrix turns across the logs played one row with certainty.
"""

import random

from agent.vgc import actions as act
from agent.vgc.moverules import MoveTracker
from agent.vgc.strategist import VGCDraftStrategist

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import (
    BATTLE_OBSERVATION,
    FOES,
    POKEMON,
    FakeProvider,
    _agent,
    _doubles_action,
    _doubles_answer,
    _move,
    _state,
    _templated,
)

OPENING = {**BATTLE_OBSERVATION, "turn": 1}
OFFENSE = {"incineroar": 115, "rillaboom": 125, "fluttermane": 135}.get


def _opening_rules(action, observation=OPENING):
    slots = act.parse_doubles(_as_legal(action))
    return slots, MoveTracker(offense=OFFENSE).begin(slots, observation)


# -- opening Fake Out ------------------------------------------------------------------------


def test_the_opening_turn_locks_a_lead_into_fake_out():
    _, rules = _opening_rules(_doubles_action())
    assert set(rules[0].blocked) == {1, 2, 3}  # everything but Fake Out (option 0)


def test_with_two_fake_out_leads_the_weaker_one_uses_it_and_the_partner_attacks():
    action = _doubles(
        "incineroar", [_move("fakeout", [1, 2], FOES), _move("flareblitz", [1, 2], FOES)],
        "rillaboom", [_move("fakeout", [1, 2], FOES), _move("woodhammer", [1, 2], FOES)],
    )
    _, rules = _opening_rules(action)
    assert 1 in rules[0].blocked and 0 not in rules[0].blocked  # Incineroar (115) fakes out
    assert set(rules[1].blocked) == {0}  # Rillaboom (125) attacks instead


def test_no_lock_after_the_opening_turn():
    _, rules = _opening_rules(_doubles_action(), observation=BATTLE_OBSERVATION)  # server turn 3
    assert rules[0].blocked == {}


def test_no_lock_into_ghosts_or_a_priority_blocker():
    ghosts = {**OPENING, "opponent_active_pokemon": [
        {"species": "landorustherian", "types": ["GHOST"]}, {"species": "amoonguss", "types": ["GHOST"]},
    ]}
    assert _opening_rules(_doubles_action(), ghosts)[1][0].blocked == {}
    surge = {**OPENING, "opponent_active_pokemon": [
        {"species": "indeedeef", "types": ["PSYCHIC", "NORMAL"]}, {"species": "amoonguss", "types": ["GRASS"]},
    ]}
    assert _opening_rules(_doubles_action(), surge)[1][0].blocked == {}


def test_the_opening_turn_plays_fake_out_whatever_the_prediction():
    provider = FakeProvider(_doubles_answer())
    decision = _agent(provider).choose_action(_state(_doubles_action(), observation=OPENING), POKEMON)
    assert decision.action["slot_0"]["move_id"] == "fakeout"


# -- role-balanced draft and bring -------------------------------------------------------------

F547_ROSTER = ["incineroar", "rillaboom", "amoonguss", "grimmsnarl", "tinglu", "ironvaliant"]


def _draft_state(roster, offered):
    observation = {
        "phase": "draft",
        "available_cards": [{"card_id": c, "species": c} for c in offered],
        "rosters": {"me": [{"species": s} for s in roster], "them": []},
    }
    actions = [_templated(f"draft_pick:{c}", {"type": "draft_pick", "card_id": c}) for c in offered]
    return _state(*actions, phase="draft", observation=observation)


def test_a_third_pure_support_scores_below_an_attacker():
    provider = FakeProvider({"action_id": "draft_pick:Iron Valiant", "reasoning_summary": "x"})
    _agent(provider).choose_action(
        _draft_state(["Incineroar", "Rillaboom", "Amoonguss"], ["Grimmsnarl", "Iron Valiant"]), POKEMON
    )
    cards = provider.prompt()["available_cards"]
    assert cards[0]["species"] == "Iron Valiant"  # best fit first
    grimmsnarl = next(c for c in cards if c["species"] == "Grimmsnarl")
    assert any("pure support" in why for why in grimmsnarl["fit"]["roles"]["why"])


def test_the_last_picks_go_to_attackers_when_short():
    # One attacker (Rillaboom) after four picks: both remaining picks must attack.
    decision = _agent().choose_action(
        _draft_state(["Incineroar", "Rillaboom", "Amoonguss", "Toxapex"], ["Whimsicott", "Chien-Pao"]), POKEMON
    )
    assert decision.action.action_id == "draft_pick:Chien-Pao"


def test_the_board_is_never_emptied():
    decision = _agent().choose_action(
        _draft_state(["Incineroar", "Amoonguss", "Grimmsnarl", "Toxapex"], ["Whimsicott"]), POKEMON
    )
    assert decision.action.action_id == "draft_pick:Whimsicott"


def _lineup(roster):
    return _templated("select_lineup", {"type": "select_lineup", "roster": roster})


def test_the_f547_roster_now_brings_iron_valiant_and_leads_fake_out():
    decision = _agent().choose_action(_state(_lineup(F547_ROSTER), phase="team_preview"), POKEMON)
    bring, leads = decision.action["bring"], decision.action["leads"]
    assert "ironvaliant" in bring
    assert sum(s in ("incineroar", "amoonguss", "grimmsnarl") for s in bring) <= 2
    assert {"incineroar", "rillaboom"} & set(leads)


def _preview_answer(bring, leads):
    return {
        "our_win_conditions": ["w"], "their_win_conditions": ["w"], "their_playstyle": "p",
        "their_likely_sets": [], "our_key_moves": [], "bring": bring, "leads": leads,
        "lead_plan": "l", "early_game": "e", "mid_game": "m", "end_game": "e",
        "preserve": [], "remove_first": [], "avoid": [], "reasoning_summary": "r",
    }


GOOD_BRING = ["incineroar", "rillaboom", "ironvaliant", "tinglu"]


def test_team_preview_is_one_call_that_plans_and_picks_the_lineup():
    provider = FakeProvider(_preview_answer(GOOD_BRING, ["incineroar", "ironvaliant"]))
    agent = _agent(provider)
    decision = agent.choose_action(_state(_lineup(F547_ROSTER), phase="team_preview"), POKEMON)
    assert len(provider.calls) == 1
    assert decision.action == {"type": "select_lineup", "bring": GOOD_BRING, "leads": ["incineroar", "ironvaliant"]}
    assert agent._game_plan["lead_plan"] == "l" and "bring" not in agent._game_plan


def test_a_lineup_that_breaks_the_rules_is_re_asked_once_then_accepted():
    breaking = _preview_answer(GOOD_BRING, ["ironvaliant", "tinglu"])  # Fake Out users brought, none leading
    provider = FakeProvider(breaking, dict(breaking))
    decision = _agent(provider).choose_action(_state(_lineup(F547_ROSTER), phase="team_preview"), POKEMON)
    assert len(provider.calls) == 2
    assert "no Fake Out lead" in provider.calls[1]["messages"][-1]["content"]
    assert decision.action["leads"] == ["ironvaliant", "tinglu"]  # the retry is played as given


# -- synergy-driven draft ----------------------------------------------------------------------


def test_a_first_pick_is_a_win_condition_not_a_support():
    decision = _agent().choose_action(_draft_state([], ["Amoonguss", "Chien-Pao", "Grimmsnarl"]), POKEMON)
    assert decision.action.action_id == "draft_pick:Chien-Pao"


def test_a_listed_partner_is_credited():
    from agent.vgc import draft, meta

    table = meta.table()
    fit = draft.score("Amoonguss", table.roles("Amoonguss"), ["Chien-Pao"], [table.roles("Chien-Pao")], [],
                      None, None, {}, table)
    assert fit["partners"] == ["Chien-Pao"]


def test_a_card_that_stacks_a_team_weakness_is_marked_down():
    from agent.vgc import draft, meta

    table = meta.table()
    roster = ["Incineroar", "Chien-Pao"]  # Fire/Dark and Dark/Ice: both weak to Fighting and Rock
    roles = [table.roles(s) for s in roster]
    stacking = draft.score("Kingambit", table.roles("Kingambit"), roster, roles, [], None, None, {}, table)
    covering = draft.score("Dragonite", table.roles("Dragonite"), roster, roles, [], None, None, {}, table)
    assert "Fighting" in stacking["coverage"]["stacks"]  # Kingambit takes 4x
    assert "Fighting" in covering["coverage"]["covers"]  # Dragonite resists
    assert "Rock" in covering["coverage"]["stacks"]


def test_the_plan_is_carried_to_the_next_pick():
    provider = FakeProvider(
        {"action_id": "draft_pick:Chien-Pao", "team_plan": "Tailwind hyper offense around Chien-Pao.",
         "reasoning_summary": "x"},
        {"action_id": "draft_pick:Dragonite", "team_plan": "same", "reasoning_summary": "x"},
    )
    agent = _agent(provider)
    agent.choose_action(_draft_state([], ["Chien-Pao", "Dragonite"]), POKEMON)
    agent.choose_action(_draft_state(["Chien-Pao"], ["Dragonite", "Amoonguss"]), POKEMON)
    assert provider.prompt(1)["your_team_plan"] == "Tailwind hyper offense around Chien-Pao."


def test_draft_cards_hide_usage_anchors():
    provider = FakeProvider({"action_id": "draft_pick:Chien-Pao", "reasoning_summary": "x"})
    _agent(provider).choose_action(_draft_state([], ["Chien-Pao", "Rillaboom"]), POKEMON)
    for card in provider.prompt()["available_cards"]:
        assert not {"draft_points", "tier", "usage_pct"} & set(card.get("reference", {}))


# -- the dac4104d fixes ------------------------------------------------------------------------

DAC_OPPONENT = ["Chien-Pao", "Archaludon", "Gholdengo", "Lucario", "Baxcalibur", "Iron Bundle"]


def test_the_draft_scores_cards_against_the_opponents_roster():
    from agent.vgc import draft, meta

    table = meta.table()
    grass_value, grass = draft.versus("Rillaboom", DAC_OPPONENT, table)
    ground_value, _ = draft.versus("Great Tusk", DAC_OPPONENT, table)
    # Real moves, not STAB types: U-turn hits Baxcalibur neutrally, so it no longer walls Rillaboom.
    assert {"Archaludon", "Gholdengo", "Lucario"} <= set(grass["walled_by"])
    assert ground_value > grass_value


def test_blank_draft_points_do_not_read_as_weak():
    from agent.vgc import draft, meta

    assert draft.strength("Iron Valiant", meta.table()) >= 1.5


def test_a_null_target_on_a_single_target_move_is_filled_in():
    slots = act.parse_doubles(_as_legal(_doubles(
        "incineroar", [_move("protect", [-1])], "fluttermane", [_move("moonblast", [1, 2], FOES)],
    )))
    payload = act.build_doubles(slots, {"slot_0": {"option": 0, "target": None}, "slot_1": {"option": 0, "target": 1}})
    assert payload["slot_0"] == {"type": "move", "move_id": "protect", "target": -1}


def test_scale_shot_into_archaludon_warns_about_stamina():
    from agent.vgc import meta

    observation = {
        "available_moves": [[{"id": "scaleshot", "type": "DRAGON", "category": "PHYSICAL", "base_power": 25}], []],
        "opponent_active_pokemon": [{"species": "Archaludon", "types": ["STEEL", "DRAGON"]}],
    }
    action = _doubles(
        "dragonite", [_move("scaleshot", [1], [{"target": 1, "side": "opponent", "species": "Archaludon"}])],
        "rillaboom", [{"type": "pass"}],
    )
    shown = act.describe_slots(act.parse_doubles(_as_legal(action)), _state(observation=observation),
                               abilities=meta.table().abilities)
    warning = shown[0]["options"][0]["targets"][0]["ability_warnings"]
    assert any("Stamina" in w and "PER HIT" in w for w in warning)


def test_the_strategist_writes_the_outcome_when_the_match_ends(tmp_path):
    import json as _json

    from altruagent.models import GameState

    agent = VGCDraftStrategist(None, log=[].append, rng=random.Random(1), clock=lambda: 0.0, log_dir=str(tmp_path))
    agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    final = GameState.from_mcp_state(
        {"session_id": "s", "status": "completed", "is_terminal": True, "observation": {}},
        result={"winner_agent_id": "them"},
    )
    agent.on_match_end(final, POKEMON)
    summary = next(tmp_path.glob("*.summary.json"), None)
    text = summary.read_text(encoding="utf-8") if summary else next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert "loss" in text


def test_a_second_form_of_a_drafted_species_is_off_the_board():
    decision = _agent().choose_action(_draft_state(["Urshifu-Rapid-Strike"], ["Urshifu", "Iron Treads"]), POKEMON)
    assert decision.action.action_id == "draft_pick:Iron Treads"
