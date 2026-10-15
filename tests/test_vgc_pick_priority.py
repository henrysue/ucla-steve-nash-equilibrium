"""Draft pick priority: must picks, the first pick's partners, then raw strength.

Numbers come from data/draft_stats.json (Regulation I ladder, ~112k battles).
"""

from agent.vgc import meta, pick_priority as pp

from test_vgc_strategist import (
    BATTLE_OBSERVATION,
    POKEMON,
    FakeProvider,
    _agent,
    _doubles_action,
    _doubles_answer,
    _lineup_action,
    _lineup_answer,
    _state,
)

TABLE = meta.table()


def test_must_picks_need_a_high_win_rate_with_enough_games():
    for species in ("Grimmsnarl", "Chien-Pao", "Iron Hands", "Chi-Yu"):
        assert pp.is_must_pick(species), species
    # 52.5% but from ~5k games: the 90% floor (51.4%) falls short.
    assert pp.entry("Iron Treads")["win_rate"] > 52 and not pp.is_must_pick("Iron Treads")
    assert not pp.is_must_pick("Incineroar")
    assert not pp.is_must_pick("Garchomp")  # no ladder data


def test_proven_partners_win_together_beyond_either_alone():
    win, games = pp.complements("Chi-Yu", "Iron Hands")
    assert win >= 55 and games > 2000
    assert pp.complements("Dragonite", "Amoonguss")
    assert pp.complements("Amoonguss", "Dragonite")  # either direction
    assert pp.complements("Chi-Yu", "Gholdengo") is None


def test_strength_favours_bulk_typing_and_abilities():
    dragonite, why = pp.strength("Dragonite", TABLE, {"ability": "Multiscale"})
    assert dragonite > pp.strength("Torkoal", TABLE)[0]
    assert any("Ground" in w for w in why) and "Multiscale" in why
    flutter, why = pp.strength("Flutter Mane", TABLE)
    assert any("Normal/Fighting/Dragon" in w for w in why)


def test_the_rules_apply_in_order():
    # 1. A must pick beats a partner of our first pick.
    pick = pp.choose(["Amoonguss", "Grimmsnarl"], ["Dragonite"], TABLE)
    assert (pick.species, pick.rule) == ("Grimmsnarl", "must pick")
    # 2. With no must pick, the first pick's proven partner.
    pick = pp.choose(["Garchomp", "Amoonguss"], ["Dragonite"], TABLE)
    assert (pick.species, pick.rule) == ("Amoonguss", "partner")
    # 3. Otherwise the strongest card.
    pick = pp.choose(["Torkoal", "Dragonite"], [], TABLE)
    assert (pick.species, pick.rule) == ("Dragonite", "strongest")


def test_the_priority_table_lists_must_picks_first_with_partners():
    rows = pp.priority_table(["Chi-Yu", "Iron Hands", "Torkoal", "Garchomp"], TABLE)
    assert [r["must_pick"] for r in rows][:2] == [True, True]
    chiyu = next(r for r in rows if r["species"] == "Chi-Yu")
    assert "Iron Hands" in chiyu["partners"]


def _drafted_agent(provider):
    agent = _agent(provider)
    agent._draft_reasons = [
        {"pick": 1, "species": "Chi-Yu", "rule": "must pick", "reason": "must pick: 52.1% win rate"},
        {"pick": 4, "species": "Iron Hands", "rule": "must pick", "reason": "must pick: 52.3% win rate"},
    ]
    return agent


def test_team_preview_is_given_the_draft_reasoning():
    provider = FakeProvider(_lineup_answer())
    _drafted_agent(provider).choose_action(
        _state(_lineup_action(), phase="team_preview", observation={"opponent_roster": [{"species": "Torkoal"}]}), POKEMON)
    strategy = provider.prompt()["our_draft_strategy"]
    assert "must picks: Chi-Yu, Iron Hands" in strategy["summary"]
    assert "Chi-Yu + Iron Hands" in strategy["summary"]  # a proven pair
    assert len(strategy["picks"]) == 2


def test_every_battle_turn_is_given_the_draft_strategy():
    provider = FakeProvider(_doubles_answer())
    _drafted_agent(provider).choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    assert "Chi-Yu + Iron Hands" in provider.prompt()["our_draft_strategy"]
