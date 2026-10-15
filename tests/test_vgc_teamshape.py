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


# Amoonguss's draft card has no Protect; Landorus-T's does.
CARDS = {"amoonguss": ["Spore", "Rage Powder", "Pollen Puff", "Sludge Bomb"],
         "landorustherian": ["Earthquake", "Rock Slide", "U-turn", "Protect"]}.get


def _opening_rules(action, observation=OPENING, exact_moves=CARDS):
    slots = act.parse_doubles(_as_legal(action))
    return slots, MoveTracker(offense=OFFENSE, exact_moves=exact_moves).begin(slots, observation)


def test_the_opening_turn_locks_a_lead_into_fake_out_on_the_foe_that_cannot_protect():
    _, rules = _opening_rules(_doubles_action())
    assert set(rules[0].blocked) == {1, 2, 3}  # everything but Fake Out (option 0)
    assert rules[0].targets == {0: (2,)}  # Amoonguss only: Landorus-T may Protect


def test_the_opening_turn_holds_fake_out_when_every_foe_may_protect():
    """Match 4e60a5a2: Fake Out and Thunder Wave on turn 1 both hit a Protect."""
    _, rules = _opening_rules(_doubles_action(), exact_moves=None)  # no draft cards: nothing is certain
    assert 0 in rules[0].blocked and "may Protect" in rules[0].blocked[0]
    assert not rules[0].targets


def test_all_four_revealed_moves_without_protect_also_count_as_certain():
    revealed = {**OPENING, "opponent_active_pokemon": [
        {"species": "landorustherian", "types": ["GROUND", "FLYING"]},
        {"species": "amoonguss", "types": ["GRASS", "POISON"],
         "revealed_moves": ["spore", "ragepowder", "pollenpuff", "sludgebomb"]},
    ]}
    _, rules = _opening_rules(_doubles_action(), revealed, exact_moves=None)
    assert rules[0].targets == {0: (2,)}


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


def test_fake_out_waits_against_ghosts_or_a_priority_blocker():
    ghosts = {**OPENING, "opponent_active_pokemon": [
        {"species": "landorustherian", "types": ["GHOST"]}, {"species": "amoonguss", "types": ["GHOST"]},
    ]}
    assert set(_opening_rules(_doubles_action(), ghosts)[1][0].blocked) == {0}
    surge = {**OPENING, "opponent_active_pokemon": [
        {"species": "indeedeef", "types": ["PSYCHIC", "NORMAL"]}, {"species": "amoonguss", "types": ["GRASS"]},
    ]}
    assert set(_opening_rules(_doubles_action(), surge)[1][0].blocked) == {0}


def test_a_fake_out_aimed_at_a_foe_that_may_protect_is_rejected():
    from agent.vgc import rules as turn_rules_module

    slots = act.parse_doubles(_as_legal(_doubles_action()))
    tracker = MoveTracker(offense=OFFENSE, exact_moves=CARDS)
    built = turn_rules_module.build(slots, OPENING, tracker, None, lambda line: None)
    flutter = {"option": 0, "target": 1}
    assert not built.legal({"slot_0": {"option": 0, "target": 1}, "slot_1": flutter})
    assert built.legal({"slot_0": {"option": 0, "target": 2}, "slot_1": flutter})


def test_the_opening_turn_fakes_out_a_foe_that_threatens_a_ko():
    """Match e1f70879: Fake Out was held and Archaludon's Body Press took Chien-Pao."""
    _, rules = _opening_rules(_doubles_action(), exact_moves=None)
    assert 0 in rules[0].blocked  # no KO threat known: still held
    slots = act.parse_doubles(_as_legal(_doubles_action()))
    tracker = MoveTracker(offense=OFFENSE, ko_threats=lambda observation: {"landorustherian"})
    rules = tracker.begin(slots, OPENING)
    assert rules[0].targets == {0: (1,)}  # into the threat, Protect or not


def test_the_strategist_opens_with_fake_out_into_a_ko_threat():
    provider = FakeProvider(_doubles_answer())
    decision = _agent(provider).choose_action(_state(_doubles_action(), observation=OPENING), POKEMON)
    agent = _agent()
    assert decision.action["slot_0"] == {"type": "move", "move_id": "fakeout", "target": 1}
    assert "landorustherian" in agent._ko_threats(OPENING)


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
    """Team fit still docks a third pure support (rule 3's tie-break), even though
    a must pick like Grimmsnarl is taken on rule 1 regardless."""
    from agent.vgc import draft, meta

    table = meta.table()
    mine = ["Incineroar", "Rillaboom", "Amoonguss"]
    roles = [table.roles(m) for m in mine]
    grimmsnarl = draft.score("Grimmsnarl", table.roles("Grimmsnarl"), mine, roles, [], None, None, {}, table)
    assert any("pure support" in why for why in grimmsnarl["roles"]["why"])


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


def test_a_must_pick_is_taken_first_highest_win_rate_floor_first():
    """Rule 1: Chien-Pao and Grimmsnarl are both must picks; Grimmsnarl's win-rate
    floor is higher."""
    decision = _agent().choose_action(_draft_state([], ["Amoonguss", "Chien-Pao", "Grimmsnarl"]), POKEMON)
    assert decision.action.action_id == "draft_pick:Grimmsnarl"


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


def test_the_first_picks_proven_partner_is_taken_next():
    """Rule 2: after Dragonite, Amoonguss (55.4% together over ~4.8k games) beats
    the stronger-on-paper Garchomp, which has no proven pairing."""
    agent = _agent()
    decision = agent.choose_action(_draft_state(["Dragonite"], ["Garchomp", "Amoonguss"]), POKEMON)
    assert decision.action.action_id == "draft_pick:Amoonguss"
    assert "partner for our first pick Dragonite" in agent.draft_strategy()["picks"][-1]


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
