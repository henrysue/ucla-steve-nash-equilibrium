"""Regression tests from match f14f1652 (2026-10-15), a loss to Ace Trainer Claude, and the
simpler game plan that followed: easy-to-play, strong Pokemon with few weaknesses; no
Trick Room.

The draft took Grimmsnarl (a support) first on ladder win rate, then Great Tusk and
Garchomp (both Ground, both weak to Fairy like Iron Hands) - benched at Team Preview -
and a 44.6% Hatterene that the setter rule forced into the four; it set Trick Room at 24%.
The endgame put three Wild Charges into a Flutter Mane that kept Protecting.
"""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, pick_priority as pp, principles, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker
from agent.vgc.predict import FoeExpectation

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "gs", "species": "Grimmsnarl", "ability": "Prankster", "item": "Light Clay",
     "moves": ["Light Screen", "Reflect", "Spirit Break", "Thunder Wave"]},
    {"card_id": "ht", "species": "Hatterene", "ability": "Magic Bounce", "item": "Safety Goggles",
     "moves": ["Trick Room", "Dazzling Gleam", "Psychic", "Mystical Fire"]},
    {"card_id": "ih", "species": "Iron Hands", "ability": "Quark Drive", "item": "Assault Vest", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 252, "spe": 20}, "moves": ["Fake Out", "Close Combat", "Wild Charge", "Drain Punch"]},
    {"card_id": "gt", "species": "Great Tusk", "ability": "Protosynthesis", "item": "Booster Energy",
     "moves": ["Headlong Rush", "Close Combat", "Rock Slide", "Protect"]},
    {"card_id": "gc", "species": "Garchomp", "ability": "Rough Skin", "item": "Focus Sash",
     "moves": ["Earthquake", "Rock Slide", "Iron Head", "Protect"]},
    {"card_id": "cr", "species": "Cresselia", "ability": "Levitate", "item": "Sitrus Berry",
     "moves": ["Icy Wind", "Moonblast", "Helping Hand", "Protect"]},
    {"card_id": "th", "species": "Thundurus", "ability": "Defiant", "item": "Life Orb",
     "moves": ["Thunderbolt", "Sludge Bomb", "Taunt", "Protect"]},
    {"card_id": "ac", "species": "Arcanine", "ability": "Intimidate", "item": "Safety Goggles",
     "moves": ["Flare Blitz", "Extreme Speed", "Snarl", "Protect"]},
    {"card_id": "lt", "species": "Landorus-Therian", "ability": "Intimidate", "item": "Choice Scarf",
     "moves": ["Earthquake", "Rock Slide", "U-turn", "Protect"]},
    {"card_id": "fm", "species": "Flutter Mane", "ability": "Protosynthesis", "item": "Booster Energy",
     "moves": ["Moonblast", "Shadow Ball", "Icy Wind", "Protect"]},
])
TABLE = SETS.overlay(meta.table())


def _choose(board, mine=(), theirs=()):
    return pp.choose(list(board), list(mine), TABLE, cards={b: TABLE.card_set(b) for b in board}, theirs=list(theirs))


# -- must picks: easy to play, strong, few weaknesses -----------------------------------------


def test_iron_hands_not_grimmsnarl_is_the_first_pick():
    pick = _choose(["Grimmsnarl", "Iron Hands", "Hatterene", "Flutter Mane"])
    assert (pick.species, pick.rule) == ("Iron Hands", "must pick")


def test_a_support_and_a_trick_room_setter_are_marked_down():
    grimmsnarl, why = pp.easy_strength("Grimmsnarl", TABLE, TABLE.card_set("Grimmsnarl"))
    assert any("a support" in w for w in why) and grimmsnarl < pp.strength("Grimmsnarl", TABLE, TABLE.card_set("Grimmsnarl"))[0]
    hatterene, why = pp.easy_strength("Hatterene", TABLE, TABLE.card_set("Hatterene"))
    assert any("Trick Room setter" in w for w in why) and hatterene < 0
    assert pp.easy_strength("Iron Hands", TABLE, TABLE.card_set("Iron Hands"))[0] >= pp.MUST_EASY_AT


# -- draft for the bring, speed control, Intimidate ----------------------------------------------


def test_a_third_pokemon_weak_to_their_attack_type_is_marked_down():
    """Iron Hands and Great Tusk are weak to Fairy, which Flutter Mane carries: Garchomp would be the third."""
    score, why = pp.draft_adjustment("Garchomp", ["Iron Hands", "Great Tusk"], ["Flutter Mane", "Salamence"], TABLE)
    assert score <= -pp.STACKED_PENALTY and any("weak to Fairy" in w for w in why)


def test_speed_control_is_marked_up_when_we_have_none():
    score, why = pp.draft_adjustment("Cresselia", ["Iron Hands", "Thundurus"], [], TABLE)
    assert score >= pp.SPEED_CONTROL_BONUS and any("speed control" in w for w in why)
    assert pp.draft_adjustment("Cresselia", ["Iron Hands"], [], TABLE)[0] < pp.SPEED_CONTROL_BONUS  # too early


def test_physical_attackers_are_marked_down_into_two_intimidates():
    score, why = pp.draft_adjustment("Great Tusk", [], ["Arcanine", "Landorus-Therian"], TABLE)
    assert score < 0 and any("Intimidates" in w for w in why)
    assert pp.draft_adjustment("Thundurus", [], ["Arcanine", "Landorus-Therian"], TABLE)[0] > 0


def test_no_trick_room_plans_are_drafted_toward():
    from agent.vgc import archetypes

    assert not any("trick room" in p["name"].lower() for p in archetypes.load_plans())


# -- the battle: we never set Trick Room, except to end theirs -----------------------------------


def _hatterene_slots():
    F = [{"target": 1, "side": "opponent", "species": "fluttermane"}]
    return act.parse_doubles(_as_legal(_doubles("hatterene", [_move("trickroom", []), _move("psychic", [1], F)],
                                                "ironhands", [_move("wildcharge", [1], F)])))


def _obs(field=None):
    return {"turn": 3, "fields": field or {},
            "active_pokemon": [{"species": "hatterene", "types": ["PSYCHIC", "FAIRY"], "hp_fraction": 1.0},
                               {"species": "ironhands", "types": ["FIGHTING", "ELECTRIC"], "hp_fraction": 1.0}],
            "opponent_active_pokemon": [{"species": "fluttermane", "types": ["GHOST", "FAIRY"], "hp_fraction": 1.0}]}


def test_we_do_not_set_trick_room():
    turn_rules = rules.build(_hatterene_slots(), _obs(), MoveTracker(), TABLE, lambda line: None)
    assert "we do not play Trick Room" in turn_rules.per_slot[0].blocked[0]


def test_trick_room_is_allowed_to_end_theirs():
    turn_rules = rules.build(_hatterene_slots(), _obs({"TRICK_ROOM": 3}), MoveTracker(), TABLE, lambda line: None)
    assert 0 not in turn_rules.per_slot[0].blocked


# -- the sure-KO principle stands down when the foe is likely to escape ----------------------------


def test_no_missed_ko_cost_for_a_foe_likely_to_escape():
    observation = {"turn": 9,
                   "active_pokemon": [{"species": "ironhands", "types": ["FIGHTING", "ELECTRIC"], "hp_fraction": 0.62}],
                   "opponent_active_pokemon": [{"species": "fluttermane", "types": ["GHOST", "FAIRY"], "hp_fraction": 0.2},
                                               {"species": "cresselia", "types": ["PSYCHIC"], "hp_fraction": 1.0}]}
    F = [{"target": 1, "side": "opponent", "species": "fluttermane"}, {"target": 2, "side": "opponent", "species": "cresselia"}]
    slots = act.parse_doubles(_as_legal(_doubles("ironhands", [_move("wildcharge", [1, 2], F)], "hatterene", [{"type": "pass"}])))
    calc = lookahead.Calculator(observation, TABLE)
    cresselia = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "wildcharge", "target": 2}, "slot_1": {"type": "pass"}}
    assert principles.Judge(slots, observation, calc, {})._kos(slots[0]) == {"fluttermane"}
    assert principles.Judge(slots, observation, calc, {}).judge(cresselia)[0] >= principles.MISSED_KO
    escaping = {"fluttermane": FoeExpectation(protect=0.3, switch=0.3)}
    assert principles.Judge(slots, observation, calc, escaping).judge(cresselia)[0] < principles.MISSED_KO
