"""Regression tests from match 07a3cb1f (2026-10-09, seat 0, our first win)."""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, movelist, observed, race
from agent.vgc.cards import CardSets

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

CARDS = [
    {"card_id": "vgc-rillaboom", "species": "Rillaboom", "item": "Assault Vest", "ability": "Grassy Surge", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 116, "spe": 148}, "moves": ["Fake Out", "Wood Hammer", "Grassy Glide", "U-turn"]},
    {"card_id": "vgc-grimmsnarl", "species": "Grimmsnarl", "item": "Light Clay", "ability": "Prankster", "nature": "Careful",
     "evs": {"hp": 252, "spd": 252}, "moves": ["Light Screen", "Reflect", "Spirit Break", "Thunder Wave"]},
    {"card_id": "vgc-indeedee-f", "species": "Indeedee-F", "item": "Rocky Helmet", "ability": "Psychic Surge", "nature": "Bold",
     "evs": {"hp": 252, "def": 252}, "moves": ["Follow Me", "Helping Hand", "Dazzling Gleam", "Psychic"]},
    {"card_id": "vgc-basculegion", "species": "Basculegion", "item": "Life Orb", "ability": "Adaptability", "nature": "Adamant",
     "evs": {"atk": 252, "spe": 252}, "moves": ["Wave Crash", "Aqua Jet", "Liquidation", "Protect"]},
    {"card_id": "vgc-chi-yu", "species": "Chi-Yu", "item": "Choice Specs", "ability": "Beads of Ruin", "nature": "Timid",
     "evs": {"spa": 252, "spe": 252}, "moves": ["Heat Wave", "Dark Pulse", "Overheat", "Snarl"]},
]
FOES = [{"target": 1, "side": "opponent", "species": "basculegion"}, {"target": 2, "side": "opponent", "species": "indeedeef"}]


def _table():
    sets = CardSets(persist=False)
    sets.remember(CARDS)
    return sets.overlay(meta.table())


def _board(action, observation):
    slots = act.parse_doubles(_as_legal(action))
    return movelist.Board(slots, observation, _table())


def test_a_lead_terrain_started_on_turn_zero_is_read():
    """No terrain ever showed: the field maps each effect to its start turn, and 0 read as off."""
    assert observed.field_state({"fields": {"PSYCHIC_TERRAIN": 0}})["field"] == ["psychicterrain"]


def test_psychic_terrain_blocks_fake_out_on_grounded_foes():
    action = _doubles("rillaboom", [_move("fakeout", [1, 2], FOES), _move("woodhammer", [1, 2], FOES)],
                      "grimmsnarl", [_move("spiritbreak", [1, 2], FOES)])
    observation = {"turn": 1, "fields": {"psychic_terrain": 0},
                   "active_pokemon": [{"species": "rillaboom"}, {"species": "grimmsnarl"}],
                   "opponent_active_pokemon": [{"species": "basculegion", "types": ["WATER", "GHOST"]},
                                               {"species": "indeedeef", "types": ["PSYCHIC", "NORMAL"]}]}
    blocks = movelist.option_blocks(_board(action, observation))
    assert "Psychic Terrain" in blocks[0][0]
    assert 1 not in blocks[0]


def test_prankster_status_moves_fail_on_dark_types():
    foes = [{"target": 1, "side": "opponent", "species": "chiyu"}]
    action = _doubles("grimmsnarl", [_move("thunderwave", [1], foes), _move("spiritbreak", [1], foes)],
                      "rillaboom", [_move("woodhammer", [1], foes)])
    observation = {"turn": 4, "active_pokemon": [{"species": "grimmsnarl"}, {"species": "rillaboom"}],
                   "opponent_active_pokemon": [{"species": "chiyu", "types": ["DARK", "FIRE"]}]}
    assert "Prankster" in movelist.option_blocks(_board(action, observation))[0][0]


def test_follow_me_is_not_an_attack_and_our_recoil_is_not_their_hit():
    """Turn 1: Indeedee used Follow Me and Rillaboom lost HP to its own Wood Hammer;
    the read "Indeedee attacks Rillaboom" was logged as a hit."""
    before = observed.snapshot({"active_pokemon": [{"species": "rillaboom", "hp_fraction": 1.0}],
                                "opponent_active_pokemon": [{"species": "indeedeef", "hp_fraction": 1.0}]})
    after = observed.snapshot({"active_pokemon": [{"species": "rillaboom", "hp_fraction": 0.67}],
                               "opponent_active_pokemon": [{"species": "indeedeef", "hp_fraction": 0.45, "moves": ["followme"]}]})
    report = observed.last_turn(before, after, {"indeedeef": {"action": "attack", "target": "rillaboom"}}, {},
                                our_moves={"rillaboom": "woodhammer"})
    assert report["reads"]["indeedeef"]["result"] == "miss"


def test_a_predicted_fake_out_that_was_close_combat_is_a_miss():
    before = observed.snapshot({"active_pokemon": [{"species": "grimmsnarl", "hp_fraction": 1.0}],
                                "opponent_active_pokemon": [{"species": "ironhands", "hp_fraction": 1.0}]})
    after = observed.snapshot({"active_pokemon": [],
                               "opponent_active_pokemon": [{"species": "ironhands", "hp_fraction": 1.0, "moves": ["closecombat"]}]})
    report = observed.last_turn(before, after, {"ironhands": {"action": "fake_out", "target": "grimmsnarl"}}, {})
    assert report["reads"]["ironhands"]["result"] == "miss"


def test_recoil_and_rocky_helmet_cost_the_attacker():
    observation = {"active_pokemon": [{"species": "rillaboom", "hp_fraction": 1.0}],
                   "opponent_active_pokemon": [{"species": "indeedeef", "types": ["PSYCHIC", "NORMAL"], "hp_fraction": 1.0}]}
    calc = lookahead.Calculator(observation, _table())
    sim = race._Sim(calc, race.Board(), [], [], observation)
    user = race.Mon("us", {"species": "rillaboom"}, 100.0)
    target = race.Mon("them", {"species": "indeedeef"}, 100.0)
    sim._self_damage(user, target, {"id": "woodhammer"}, dealt=45.0, chance=1.0)
    assert user.hp < 100 - 100 / 6 - 10  # a third of 45% of Indeedee's HP, plus Rocky Helmet
    assert user.alive == 1.0


def test_follow_up_turns_never_use_fake_out():
    observation = {"active_pokemon": [{"species": "rillaboom", "hp_fraction": 1.0}],
                   "opponent_active_pokemon": [{"species": "chiyu", "types": ["DARK", "FIRE"], "hp_fraction": 0.05}]}
    calc = lookahead.Calculator(observation, _table())
    sim = race._Sim(calc, race.Board(), [], [], observation)
    assert all(_id != "fakeout" for _id in (m["id"] for m in sim._attacks(race.Mon("us", {"species": "rillaboom"}, 100.0))))
