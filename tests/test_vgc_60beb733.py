"""Regression tests from match 60beb733 (2026-10-15, our agent against itself), seat 0's loss.

Grimmsnarl set Reflect, Thunder Wave and Light Screen in three turns while our
Archaludon slot used Draco Meteor into an Assault Vest, then switched to
Dragonite and straight back. Both screens halved every hit after. Turn 6, an
82% Dragonite (Multiscale gone) came in to a faster Flutter Mane's Moonblast
and fell without acting. The board log showed absolute HP for our side and
bench Pokemon we never brought.
"""

from agent.vgc import actions as act
from agent.vgc import decisions, lookahead, meta, predict, principles, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker
from agent.vgc.strategist import _board_detail

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "ar", "species": "Archaludon", "ability": "Stamina", "item": "Leftovers", "nature": "Bold",
     "evs": {"hp": 252, "def": 252, "spa": 4}, "moves": ["Body Press", "Draco Meteor", "Flash Cannon", "Protect"]},
    {"card_id": "dn", "species": "Dragonite", "ability": "Multiscale", "item": "Loaded Dice", "nature": "Adamant",
     "evs": {"hp": 252, "atk": 252, "spe": 4}, "moves": ["Extreme Speed", "Scale Shot", "Tailwind", "Protect"]},
    {"card_id": "fm", "species": "Flutter Mane", "ability": "Protosynthesis", "item": "Booster Energy", "nature": "Timid",
     "evs": {"spa": 252, "spd": 4, "spe": 252}, "moves": ["Moonblast", "Shadow Ball", "Icy Wind", "Protect"]},
    {"card_id": "gs", "species": "Grimmsnarl", "ability": "Prankster", "item": "Light Clay", "nature": "Careful",
     "evs": {"hp": 252, "atk": 4, "spd": 252}, "moves": ["Light Screen", "Reflect", "Spirit Break", "Thunder Wave"]},
    {"card_id": "ih", "species": "Iron Hands", "ability": "Quark Drive", "item": "Assault Vest", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 252, "spe": 20}, "moves": ["Fake Out", "Close Combat", "Wild Charge", "Drain Punch"]},
    {"card_id": "rb", "species": "Rillaboom", "ability": "Grassy Surge", "item": "Assault Vest", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 116, "def": 4, "spd": 4, "spe": 148}, "moves": ["Fake Out", "Wood Hammer", "Grassy Glide", "U-turn"]},
    {"card_id": "th", "species": "Thundurus", "ability": "Defiant", "item": "Life Orb", "nature": "Timid",
     "evs": {"spa": 252, "spd": 4, "spe": 252}, "moves": ["Thunderbolt", "Sludge Bomb", "Taunt", "Protect"]},
    {"card_id": "cy", "species": "Chi-Yu", "ability": "Beads of Ruin", "item": "Choice Specs",
     "moves": ["Heat Wave", "Dark Pulse", "Overheat", "Snarl"]},
])
TABLE = SETS.overlay(meta.table())
TYPES = {"archaludon": ["STEEL", "DRAGON"], "dragonite": ["DRAGON", "FLYING"], "fluttermane": ["GHOST", "FAIRY"],
         "grimmsnarl": ["DARK", "FAIRY"], "ironhands": ["FIGHTING", "ELECTRIC"], "rillaboom": ["GRASS"],
         "thundurus": ["ELECTRIC", "FLYING"], "chiyu": ["DARK", "FIRE"]}


def _obs(ours, theirs, hp=None, turn=3, team=None, **extra):
    hp = hp or {}
    return {"turn": turn, **extra,
            "active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in ours],
            "opponent_active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0),
                                         "item": "unknown_item"} for s in theirs],
            "team": team or []}


# -- 1. tempo: switching straight back out costs a turn and more -------------------------------


def test_turn_3_dragonite_switching_straight_back_out_is_flagged():
    observation = _obs(["rillaboom", "dragonite"], ["ironhands", "grimmsnarl"], hp={"rillaboom": 0.86, "dragonite": 0.82,
                                                                                     "ironhands": 0.7, "grimmsnarl": 0.46})
    F = [{"target": 1, "side": "opponent", "species": "ironhands"}, {"target": 2, "side": "opponent", "species": "grimmsnarl"}]
    slots = act.parse_doubles(_as_legal(_doubles(
        "rillaboom", [_move("woodhammer", [1, 2], F)],
        "dragonite", [_move("scaleshot", [1, 2], F), {"type": "switch", "species": "archaludon"}])))
    calc = lookahead.Calculator(observation, TABLE)
    expected = predict.expectations({"ironhands": {"action": "attack", "target": "rillaboom", "confidence": 0.65},
                                     "grimmsnarl": {"action": "attack", "target": "dragonite", "confidence": 0.6}},
                                    ["rillaboom", "dragonite"])
    back = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "woodhammer", "target": 2},
            "slot_1": {"type": "switch", "species": "archaludon"}}
    fresh = principles.Judge(slots, observation, calc, expected, just_in={"dragonite"}).judge(back)
    settled = principles.Judge(slots, observation, calc, expected).judge(back)
    assert any("a turn after it came in" in f for f in fresh[1])
    assert fresh[0] >= principles.TEMPO_SWITCH + principles.TEMPO_PING_PONG
    assert settled == (0.0, [])  # one switch into a resisted hit is a real pivot; twice running is not


def test_the_tracker_knows_who_just_came_in():
    """Not the turn-1 leads (match b0e756ca); a Pokemon that switched in after a turn, until it has had one."""
    F = [{"target": 1, "side": "opponent", "species": "ironhands"}]
    slots = act.parse_doubles(_as_legal(_doubles("rillaboom", [_move("woodhammer", [1], F)],
                                                 "dragonite", [_move("scaleshot", [1], F)])))
    tracker = MoveTracker()
    tracker.begin(slots, _obs(["rillaboom", "dragonite"], ["ironhands"]))
    assert tracker.just_in() == set()  # leads on turn 1
    tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "woodhammer", "target": 1},
                    "slot_1": {"type": "switch", "species": "archaludon"}})
    swapped = act.parse_doubles(_as_legal(_doubles("rillaboom", [_move("woodhammer", [1], F)],
                                                   "archaludon", [_move("flashcannon", [1], F)])))
    tracker.begin(swapped, _obs(["rillaboom", "archaludon"], ["ironhands"]))
    assert tracker.just_in() == {"archaludon"}


# -- 2. a Prankster screen setter is their win condition ---------------------------------------


def test_grimmsnarl_the_screen_setter_is_a_win_condition():
    calc = lookahead.Calculator(_obs(["rillaboom", "archaludon"], ["ironhands", "grimmsnarl"], turn=1), TABLE)
    found = lookahead.win_conditions(calc)
    assert "grimmsnarl" in found and "Prankster screens" in found["grimmsnarl"]
    assert "ironhands" not in found


# -- 3. the Choice item check reads through "unknown_item" -------------------------------------


def test_a_specs_chi_yu_showing_unknown_item_is_still_flagged():
    calc = lookahead.Calculator(_obs(["rillaboom", "archaludon"], ["chiyu", "ironhands"], turn=1), TABLE)
    assert "Choice Specs" in lookahead.win_conditions(calc)["chiyu"]


# -- 4. the board log: HP as a fraction, the bench as the four we brought ------------------------


def test_the_board_log_uses_fractions_and_only_the_brought_bench():
    observation = {
        "active_pokemon": [{"species": "archaludon", "current_hp": 59, "max_hp": 197}],
        "opponent_active_pokemon": [{"species": "fluttermane", "hp_fraction": 0.6}],
        "team": [{"species": s, "current_hp": hp, "max_hp": mx} for s, hp, mx in (
            ("archaludon", 59, 197), ("dragonite", 163, 198), ("thundurus", 154, 154), ("rillaboom", 0, 205),
            ("greattusk", 222, 222), ("milotic", 202, 202))],
    }
    board = _board_detail(observation, ["archaludon", "rillaboom", "dragonite", "thundurus"])
    assert board["ours"]["archaludon"]["hp"] == 0.299 and board["ours"]["archaludon"]["hp_exact"] == "59/197"
    assert board["our_bench_hp"] == {"dragonite": 0.823, "thundurus": 1.0, "rillaboom": 0.0}  # no Great Tusk, Milotic


# -- 5. Protect only protects its user ----------------------------------------------------------


def test_the_final_call_knows_protect_shields_only_its_user():
    text = decisions.review_decision([{"option": 1, "value": 0, "line": "x"}], {}).prompt["instructions"]
    assert "Protect only shields the Pokémon that uses it, never its partner" in text


# -- 6. replacements: current HP, both foes, and moving before falling ---------------------------


def test_turn_6_thundurus_replaces_rillaboom_not_an_82_percent_dragonite():
    team = [{"species": "dragonite", "current_hp": 163, "max_hp": 198}, {"species": "thundurus", "current_hp": 154, "max_hp": 154}]
    observation = _obs(["archaludon"], ["fluttermane", "ironhands"], hp={"archaludon": 0.3, "fluttermane": 0.6,
                                                                          "ironhands": 0.83}, turn=6, team=team)
    calc = lookahead.Calculator(observation, TABLE)
    dragonite = lookahead.replacement_value(calc, observation, "dragonite")
    thundurus = lookahead.replacement_value(calc, observation, "thundurus")
    assert thundurus > dragonite
    action = _doubles("rillaboom", [{"type": "switch", "species": "dragonite"}, {"type": "switch", "species": "thundurus"}],
                      "archaludon", [{"type": "pass"}])
    parsed = act.parse_doubles(_as_legal(action))
    slots = [act.Slot(0, "rillaboom", True, parsed[0].options), parsed[1]]  # Rillaboom fainted: a forced switch
    turn_rules = rules.TurnRules(slots, None, rules.build(slots, observation, MoveTracker(), TABLE, lambda line: None).per_slot)
    best = turn_rules.check(lookahead.best_line(turn_rules, observation, calc, {}))
    assert best["slot_0"] == {"type": "switch", "species": "thundurus"}
