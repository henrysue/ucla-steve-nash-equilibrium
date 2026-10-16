"""PokéChamp-style one-step lookahead: damage, turns to KO, tool lines, context."""

from agent.vgc import lookahead, meta

from test_vgc_strategist import (
    BATTLE_OBSERVATION,
    POKEMON,
    FakeProvider,
    _agent,
    _doubles_action,
    _doubles_answer,
    _state,
)
from test_vgc_teamshape import F547_ROSTER, GOOD_BRING, _lineup, _preview_answer

TABLE = meta.table()
DAC = {
    "active_pokemon": [{"species": "Dragonite", "hp_fraction": 1.0}, {"species": "Rillaboom", "hp_fraction": 1.0}],
    "opponent_active_pokemon": [
        {"species": "Archaludon", "types": ["STEEL", "DRAGON"], "hp_fraction": 0.47},
        {"species": "Lucario", "types": ["FIGHTING", "STEEL"], "hp_fraction": 0.91},
    ],
}


def test_wood_hammer_into_archaludon_is_shown_as_the_chip_it_is():
    calc = lookahead.Calculator(DAC, TABLE)
    hit = calc.damage(calc.ours(1), calc.foe("Archaludon"), {"type": "GRASS", "base_power": 120, "category": "PHYSICAL"})
    assert hit["expected_pct"] < 20 and hit["turns_to_ko"] >= 3


def test_a_super_effective_stab_hit_on_a_weakened_target_is_a_likely_ko():
    calc = lookahead.Calculator(DAC, TABLE)
    hit = calc.damage({"species": "Great Tusk"}, calc.foe("Archaludon"), {"type": "GROUND", "base_power": 120, "category": "PHYSICAL"})
    assert hit["ko_this_turn_likely"] and hit["turns_to_ko"] == 1


def test_multi_hit_moves_count_their_hits():
    calc = lookahead.Calculator(DAC, TABLE)
    single = calc.damage(calc.ours(0), calc.foe("Lucario"), {"type": "DRAGON", "base_power": 25, "category": "PHYSICAL"})
    multi = calc.damage(calc.ours(0), calc.foe("Lucario"), {"id": "scaleshot", "type": "DRAGON", "base_power": 25, "category": "PHYSICAL"})
    assert multi["expected_pct"] > 2.5 * single["expected_pct"]


def test_immunity_is_zero_damage():
    calc = lookahead.Calculator(DAC, TABLE)
    hit = calc.damage({"species": "Incineroar"}, {"species": "Gengar", "types": ["GHOST", "POISON"]},
                      {"type": "NORMAL", "base_power": 40, "category": "PHYSICAL"})
    assert hit["immune"] and hit["expected_pct"] == 0


def test_boosts_and_assault_vest_move_the_numbers():
    calc = lookahead.Calculator(DAC, TABLE)
    move = {"type": "FIRE", "base_power": 90, "category": "SPECIAL"}
    plain = calc.damage({"species": "Chi-Yu"}, {"species": "Rillaboom", "item": "Leftovers"}, move)
    vest = calc.damage({"species": "Chi-Yu"}, {"species": "Rillaboom", "item": "Assault Vest"}, move)
    boosted = calc.damage({"species": "Chi-Yu", "boosts": {"spa": 2}}, {"species": "Rillaboom", "item": "Leftovers"}, move)
    assert vest["expected_pct"] < plain["expected_pct"] < boosted["expected_pct"]


def test_the_prediction_prompt_carries_both_sides_threats_and_the_code_guess():
    provider = FakeProvider(_doubles_answer())
    _agent(provider).choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    prompt = provider.prompt()
    assert prompt["their_threats"] and prompt["our_threats"]
    assert prompt["code_guess"]["amoonguss"]["action"] in ("protect", "attack", "fake_out", "support")  # support: Rage Powder


def test_the_game_plan_reaches_every_turn_and_history_accumulates():
    plan = {**_preview_answer(GOOD_BRING, ["incineroar", "ironvaliant"]), "lead_plan": "Fake Out + Iron Valiant pressure"}
    provider = FakeProvider(plan, _doubles_answer(), _doubles_answer())
    agent = _agent(provider)
    agent.choose_action(_state(_lineup(F547_ROSTER), phase="team_preview"), POKEMON)
    agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    agent.choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    turn_two = provider.prompt(2)
    assert turn_two["game_plan"]["lead_plan"].startswith("Fake Out + Iron Valiant")
    assert turn_two["recent_turns"][0]["turn"] == "Turn 1"
    assert "slot 0:" in turn_two["recent_turns"][0]["we_did"]


def test_team_preview_sees_each_pokemons_matchup_with_their_roster():
    from test_vgc_teamshape import DAC_OPPONENT

    provider = FakeProvider(_preview_answer(GOOD_BRING, ["incineroar", "ironvaliant"]))
    observation = {"opponent_roster": [{"species": s} for s in DAC_OPPONENT]}
    _agent(provider).choose_action(_state(_lineup(F547_ROSTER), phase="team_preview", observation=observation), POKEMON)
    vs = {entry["species"]: entry["vs_their_team"] for entry in provider.prompt()["our_team"]}
    assert vs["ironvaliant"]["score"] > vs["rillaboom"]["score"]
    assert "Archaludon" in vs["rillaboom"]["walled_by"]


def test_an_immune_target_does_not_break_the_tool_lines():
    calc = lookahead.Calculator(DAC, TABLE)
    hit = calc.damage(calc.ours(0), {"species": "Tornadus", "types": ["FLYING"]}, {"type": "GROUND", "base_power": 100, "category": "PHYSICAL"})
    assert hit["immune"] and hit["ko_this_turn_likely"] is False
