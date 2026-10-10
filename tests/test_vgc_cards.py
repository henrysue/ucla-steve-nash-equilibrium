"""Draft cards are exact sets: remembered from every board, used in place of guesses."""

from agent.vgc import lookahead, meta, protect
from agent.vgc.cards import CardSets

from test_vgc_strategist import DRAFT_OBSERVATION, POKEMON, FakeProvider, _agent, _draft_actions, _state

TABLE = meta.table()
CHI_YU = {
    "card_id": "vgc-chi-yu", "species": "Chi-Yu", "item": "Choice Scarf", "ability": "Beads of Ruin",
    "nature": "Timid", "level": 50, "evs": {"hp": 4, "spa": 252, "spe": 252}, "ivs": {},
    "moves": ["Heat Wave", "Dark Pulse", "Snarl", "Overheat"],
}
URSHIFU = {
    "card_id": "vgc-urshifu-rapid-strike", "species": "Urshifu-Rapid-Strike", "item": "Mystic Water",
    "ability": "Unseen Fist", "nature": "Adamant", "level": 50, "evs": {"hp": 4, "atk": 252, "spe": 252},
    "ivs": {}, "moves": ["Surging Strikes", "Close Combat", "Aqua Jet", "Protect"],
}


def _sets(*cards, path=None):
    sets = CardSets(path, persist=path is not None)
    sets.remember(list(cards))
    return sets


def test_a_board_is_remembered_and_saved_for_later_matches(tmp_path):
    path = tmp_path / "card_sets.json"
    assert _sets(CHI_YU, path=path).get("chiyu")["item"] == "Choice Scarf"
    later = CardSets(path)
    assert later.get("Chi-Yu")["moves"] == CHI_YU["moves"]
    assert not later.seen_this_match("Chi-Yu")


def test_this_matchs_board_beats_a_saved_set(tmp_path):
    path = tmp_path / "card_sets.json"
    _sets(CHI_YU, path=path)
    sets = CardSets(path)
    sets.remember([{**CHI_YU, "item": "Choice Specs"}])
    assert sets.get("chiyu")["item"] == "Choice Specs"


def test_the_overlay_answers_from_the_card_and_falls_back_to_the_table():
    table = _sets(CHI_YU).overlay(TABLE)
    assert table.common_moves("Chi-Yu") == CHI_YU["moves"]
    assert table.common_item("Chi-Yu") == "Choice Scarf"
    assert table.abilities("Chi-Yu") == ["Beads of Ruin"]
    assert table.common_moves("Incineroar") == TABLE.common_moves("Incineroar")
    assert table.card_set("Incineroar") is None and TABLE.card_set("Chi-Yu") is None
    assert table.base_stats("Chi-Yu") == TABLE.base_stats("Chi-Yu")


def _calc(table, ours=("urshifurapidstrike",), theirs=("chiyu",)):
    return lookahead.Calculator({
        "active_pokemon": [{"species": s} for s in ours],
        "opponent_active_pokemon": [{"species": s} for s in theirs],
    }, table)


def test_a_carded_foe_threatens_with_its_real_moves_only():
    table = _sets(CHI_YU, URSHIFU).overlay(TABLE)
    calc = _calc(table)
    moves = {m["name"] for m in calc.likely_attacks(calc.foe("chiyu"))}
    assert moves == {"Heat Wave", "Dark Pulse", "Snarl", "Overheat"}
    assert calc.threats()["chiyu"]["urshifurapidstrike"]["move"] == "Overheat"


def test_a_known_defender_has_one_spread_so_only_the_roll_varies():
    calc = _calc(_sets(CHI_YU, URSHIFU).overlay(TABLE))
    hit = calc.best_hit(calc.foe("chiyu"), {"species": "urshifurapidstrike"})
    low, high = hit["range_pct"]
    assert abs(low / high - lookahead.ROLL_LOW / lookahead.ROLL_HIGH) < 0.01


def test_speed_comes_from_the_cards_evs_nature_and_choice_scarf():
    calc = _calc(_sets(CHI_YU).overlay(TABLE))
    # Base 100, 252 EVs, Timid: 167; x1.5 Choice Scarf.
    assert calc.speed({"species": "chiyu"}) == 167 * 1.5


def test_the_servers_base_stats_cover_a_species_the_table_lacks():
    entry = {"species": "notarealmon", "types": ["FIRE"],
             "base_stats": {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80}}
    calc = lookahead.Calculator({"active_pokemon": [entry], "opponent_active_pokemon": []}, TABLE)
    assert calc.base_stats(entry)["spe"] == 80 and calc.speed(entry)


def test_a_card_without_protect_is_never_expected_to_protect():
    table = _sets(CHI_YU, URSHIFU).overlay(TABLE)
    observation = {"active_pokemon": [], "opponent_active_pokemon": [{"species": "chiyu"}, {"species": "urshifurapidstrike"}]}
    odds = protect.odds(observation, first_turn={}, common_moves=table.common_moves, card_set=table.card_set)
    assert odds["chiyu"]["likelihood"] == 0.0
    assert odds["urshifurapidstrike"]["likelihood"] > 0


def test_the_draft_board_feeds_the_agents_tables():
    agent = _agent(FakeProvider({"action_id": "draft_pick:c1", "reasoning_summary": "Best card."}))
    agent.choose_action(_state(*_draft_actions("c1", "c2"), phase="draft", observation=DRAFT_OBSERVATION), POKEMON)
    # Both cards on the board are known, including the one we passed on.
    assert agent._meta.common_moves("Pyukumuku") == ["recover"]
    assert agent._meta.common_item("Rillaboom") == "Miracle Seed"
