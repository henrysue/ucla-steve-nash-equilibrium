"""Archetype-driven drafting, playstyle reads and counters, and the planning sub-agent."""

import json
import random

from agent.vgc import archetypes as ar
from agent.vgc import meta
from agent.vgc.providers import ProviderError
from agent.vgc.strategist import VGCDraftStrategist

from test_vgc_strategist import (
    BATTLE_OBSERVATION,
    POKEMON,
    FakeProvider,
    _agent,
    _doubles_action,
    _doubles_answer,
    _state,
)
from test_vgc_teamshape import DAC_OPPONENT, F547_ROSTER, GOOD_BRING, _draft_state, _lineup, _preview_answer

TABLE = meta.table()


def _styles(roster):
    return {k: v["confidence"] for k, v in ar.infer_playstyle(roster, TABLE).items()}


# -- plans and matching ------------------------------------------------------------------------


def test_the_users_teams_are_loaded():
    names = [p["name"] for p in ar.load_plans()]
    assert names[0] == "Charizard Y sun"
    for sample in ("Worlds Floette hyper offense", "Tailwind + Icy Wind speed control",
                   "Incineroar cycling core", "Milotic stat-drop punish", "Psychic Terrain Trick Room"):
        assert sample in names


def test_every_plan_is_legal_in_the_champions_table():
    import csv

    names = [r["Pokemon"] for r in csv.DictReader(open("data/vgc_reg_mc_draft_doubles.csv", encoding="utf-8-sig"))]
    for plan in ar.load_plans():
        for member in plan["core"] + plan["members"] + plan["flex"]:
            assert any(ar.matches(member, n, TABLE) for n in names), (plan["name"], member)


def test_a_form_qualifier_is_respected():
    assert not ar.matches("Charizard Y", "Charizard-Mega-X", TABLE)
    assert not ar.matches("Charizard Y", "Charizard", TABLE)


def test_plan_members_match_the_servers_form_names():
    assert ar.matches("Charizard Y", "Charizard-Mega-Y", TABLE)
    assert ar.matches("Basculegion", "Basculegion-F", TABLE)
    assert not ar.matches("Urshifu-Rapid-Strike", "Urshifu", TABLE)  # a different form, not a substitute


def test_a_plan_can_declare_roles_the_table_cannot():
    assert "attacker" in ar.declared_roles("Charizard-Mega-Y", TABLE)


# -- playstyle reads ---------------------------------------------------------------------------


def test_playstyles_are_read_from_setters_and_shape():
    assert max(_styles(["Farigiraf", "Ursaluna", "Iron Hands"]).items(), key=lambda kv: kv[1])[0] == "trick_room"
    assert max(_styles(["Tornadus", "Chien-Pao", "Flutter Mane"]).items(), key=lambda kv: kv[1])[0] == "tailwind"
    assert _styles(["Miraidon", "Iron Hands", "Iron Bundle"])["electric_terrain"] == 1.0
    assert _styles(["Incineroar", "Amoonguss", "Landorus-Therian"])["cycling"] >= 0.6
    assert _styles(["Milotic", "Kingambit", "Annihilape"])["stat_punish"] == 1.0


def test_generic_members_do_not_imply_a_weather_team():
    styles = _styles(["Incineroar", "Amoonguss", "Landorus-Therian"])
    assert "rain" not in styles and "sand" not in styles


def test_counters_answer_the_style_not_just_the_types():
    vs_room = ar.infer_playstyle(["Farigiraf", "Ursaluna", "Iron Hands"], TABLE)
    assert ar.counter_bonus("Tornadus", vs_room, TABLE)[0] > ar.counter_bonus("Ursaluna", vs_room, TABLE)[0]
    vs_cycle = ar.infer_playstyle(["Incineroar", "Amoonguss", "Landorus-Therian"], TABLE)
    bonus, how = ar.counter_bonus("Milotic", vs_cycle, TABLE)
    assert bonus > 1.0 and any("stat punish vs their cycling" in h for h in how)


# -- the plan the draft targets ---------------------------------------------------------------


def test_two_members_commit_the_draft_to_a_plan():
    plan, _, status = ar.choose_plan(["Whimsicott", "Raichu-Mega-Y"], [], ["Ninetales-Alola", "Gholdengo"], {}, TABLE)
    assert plan["name"] == "Tailwind + Icy Wind speed control" and status["committed"]
    assert "Ninetales-Alola" in status["on_board_now"]


def test_a_plan_whose_core_the_opponent_took_is_dropped():
    plan, _, _ = ar.choose_plan([], ["Milotic"], ["Kingambit", "Annihilape"], {}, TABLE)
    assert plan is None or plan["name"] != "Milotic stat-drop punish"


def test_their_playstyle_steers_which_plan_we_build():
    room = ar.infer_playstyle(["Farigiraf", "Hatterene", "Indeedee-F"], TABLE)
    board = ["Milotic", "Kingambit", "Whimsicott", "Floette-Mega", "Charizard-Mega-Y", "Incineroar"]
    plan, _, _ = ar.choose_plan([], ["Farigiraf", "Hatterene", "Indeedee-F"], board, room, TABLE)
    assert "trick_room" not in plan.get("loses_to", [])


def test_the_draft_takes_the_plan_member_over_a_higher_usage_card():
    # Holding Tornadus + Flutter Mane (Tailwind plan): Iron Bundle (member) beats Amoonguss (higher usage).
    decision = _agent().choose_action(
        _draft_state(["Tornadus", "Flutter Mane"], ["Amoonguss", "Iron Bundle"]), POKEMON
    )
    assert decision.action.action_id == "draft_pick:Iron Bundle"


def test_the_draft_prompt_carries_target_and_their_style():
    provider = FakeProvider({"action_id": "draft_pick:Incineroar", "reasoning_summary": "x"})
    state = _draft_state(["Garchomp"], ["Whimsicott", "Incineroar"])
    state.raw["observation"]["rosters"]["them"] = [{"species": "Farigiraf"}, {"species": "Ursaluna"}]
    _agent(provider).choose_action(state, POKEMON)
    prompt = provider.prompt()
    assert "trick_room" in prompt["opponent_playstyle"]
    assert prompt["draft_target"]["name"]
    incineroar = next(c for c in prompt["available_cards"] if c["species"] == "Incineroar")
    assert incineroar["fit"]["versus"]["answers_their_style"]  # Fake Out on their setter


# -- team preview (plan + lineup in one call) --------------------------------------------


def _preview(observation=None):
    return _state(_lineup(F547_ROSTER), phase="team_preview",
                  observation=observation or {"opponent_roster": [{"species": s} for s in DAC_OPPONENT]})


def test_the_preview_call_gets_both_teams_and_the_plan_is_logged(tmp_path):
    provider = FakeProvider(_preview_answer(GOOD_BRING, ["incineroar", "ironvaliant"]))
    agent = VGCDraftStrategist(provider, log=[].append, rng=random.Random(1), clock=lambda: 0.0, log_dir=str(tmp_path))
    agent.choose_action(_preview(), POKEMON)
    prompt = provider.prompt()
    assert {t["species"] for t in prompt["their_team"]} == set(DAC_OPPONENT)
    assert "vs_their_team" in prompt["our_team"][0]
    records = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert any(r["kind"] == "game_plan" for r in records)


def test_a_failed_preview_call_still_brings_a_legal_four():
    agent = VGCDraftStrategist(FakeProvider(ProviderError("down"), ProviderError("down")), log=[].append,
                               rng=random.Random(1), clock=lambda: 0.0)
    decision = agent.choose_action(_preview(), POKEMON)
    assert decision.action["type"] == "select_lineup" and len(decision.action["bring"]) == 4


def test_the_preview_call_gets_the_drafted_moves_and_items():
    provider = FakeProvider({"action_id": "draft_pick:c1", "reasoning_summary": "x"},
                            _preview_answer(GOOD_BRING, ["incineroar", "ironvaliant"]))
    agent = _agent(provider)
    draft = _state(
        {"action_id": "draft_pick:c1", "label": "x", "input": {"action_id": "draft_pick:c1", "action": {"type": "draft_pick", "card_id": "c1"}}},
        phase="draft",
        observation={"available_cards": [{"card_id": "c1", "species": "Iron Valiant", "item": "Booster Energy",
                                          "moves": ["closecombat", "moonblast", "spiritbreak", "protect"]}],
                     "rosters": {"me": [], "them": []}},
    )
    agent.choose_action(draft, POKEMON)
    agent.choose_action(_preview(), POKEMON)
    valiant = next(t for t in provider.prompt(1)["our_team"] if t["species"] == "ironvaliant")
    assert valiant["moves"] == ["closecombat", "moonblast", "spiritbreak", "protect"]
    assert valiant["item"] == "Booster Energy"


def test_the_core_piece_goes_first():
    decision = _agent().choose_action(_draft_state([], ["Sneasler", "Charizard-Mega-Y", "Whimsicott"]), POKEMON)
    assert decision.action.action_id == "draft_pick:Charizard-Mega-Y"


def test_a_plan_with_no_core_left_on_the_board_is_not_targeted():
    plan, _, _ = ar.choose_plan([], [], ["Sneasler", "Basculegion", "Sylveon"], {}, TABLE)
    assert plan is None or plan["name"] != "Charizard Y sun"


def test_the_opening_plan_varies_between_matches():
    board = ["Charizard-Mega-Y", "Farigiraf", "Indeedee-F", "Tornadus", "Whimsicott", "Incineroar", "Milotic"]
    chosen = {ar.choose_plan([], [], board, {}, TABLE, rng=random.Random(seed))[0]["name"] for seed in range(20)}
    assert len(chosen) >= 2


# -- format detection (Reg I vs Champions Reg M-C) -------------------------------------------

CHAMPIONS_POOL = ["Rillaboom", "Sneasler", "Incineroar", "Salamence-Mega", "Kingambit", "Basculegion",
                  "Charizard-Mega-Y", "Floette-Mega", "Pelipper", "Archaludon", "Whimsicott", "Garchomp"]


def test_a_champions_pool_switches_to_the_champions_table():
    reg_i = meta.load("data/vgc_2025_reg_i_draft_doubles.csv")
    chosen = meta.best_table(CHAMPIONS_POOL, reg_i)
    assert chosen.source.endswith("vgc_reg_mc_draft_doubles.csv")


def test_a_reg_i_pool_keeps_the_reg_i_table():
    reg_i = meta.load("data/vgc_2025_reg_i_draft_doubles.csv")
    pool = ["Flutter Mane", "Chien-Pao", "Urshifu-Rapid-Strike", "Iron Hands", "Amoonguss", "Incineroar"]
    assert meta.best_table(pool, reg_i) is reg_i


def test_the_strategist_switches_tables_at_the_first_draft_pick():
    log = []
    agent = _agent(log=log)
    agent._meta = meta.load("data/vgc_2025_reg_i_draft_doubles.csv")
    agent.choose_action(_draft_state([], CHAMPIONS_POOL), POKEMON)
    assert agent._meta.source.endswith("vgc_reg_mc_draft_doubles.csv")
    assert any("reference table switched" in line for line in log)


def test_megas_count_as_the_one_per_battle_slot_in_champions():
    champions = meta.load("data/vgc_reg_mc_draft_doubles.csv")
    assert champions.is_restricted("Charizard-Mega-Y") and not champions.is_restricted("Incineroar")
    assert champions.restricted_in(["Charizard-Mega-Y", "Floette-Mega", "Sneasler"]) == ["Charizard-Mega-Y", "Floette-Mega"]


def test_a_rival_weather_setter_on_our_own_weather_team_is_penalised():
    champions = meta.load("data/vgc_reg_mc_draft_doubles.csv")
    sun = next(p for p in ar.load_plans() if p["name"] == "Charizard Y sun")
    penalty, why = ar.field_clash("Pelipper", ["Charizard-Mega-Y"], sun, champions)
    assert penalty < 0 and "rain" in why
    assert ar.field_clash("Torkoal", ["Charizard-Mega-Y"], sun, champions) == (0.0, None)
