"""Regression tests from matches 744c9e10 (a loss) and 41d776ed (a win), 2026-10-15.

744c9e10: turn 3, Baxcalibur's Ice Shard (priority) took a 27% Rillaboom
while the sim had it Icicle Spear after our faster Chien-Pao; Sucker Punch
would have KO'd it first. Turn 5, a Choice Scarf Landorus used Protect and was
locked into a second, 1-in-3 Protect. 41d776ed: "Dondozo protected" was
inferred from a Scale Shot (90%) that may have missed, and turn 12 could not
be replayed: the log had no boosts or exact HP.
"""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, predict, race, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker
from agent.vgc.report import Analysis
from agent.vgc.strategist import _board_detail

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

SETS = CardSets(persist=False)
SETS.remember([
    {"card_id": "cp", "species": "Chien-Pao", "ability": "Sword of Ruin", "item": "Focus Sash", "nature": "Jolly",
     "evs": {"hp": 4, "atk": 252, "spe": 252}, "moves": ["Icicle Crash", "Sacred Sword", "Sucker Punch", "Protect"]},
    {"card_id": "rb", "species": "Rillaboom", "ability": "Grassy Surge", "item": "Assault Vest", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 116, "def": 4, "spd": 4, "spe": 148}, "moves": ["Fake Out", "Wood Hammer", "Grassy Glide", "U-turn"]},
    {"card_id": "ih", "species": "Iron Hands", "ability": "Quark Drive", "item": "Assault Vest", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 252, "spe": 20}, "moves": ["Fake Out", "Close Combat", "Wild Charge", "Drain Punch"]},
    {"card_id": "bx", "species": "Baxcalibur", "ability": "Thermal Exchange", "item": "Loaded Dice", "nature": "Jolly",
     "evs": {"hp": 4, "atk": 252, "spe": 252}, "moves": ["Icicle Spear", "Glaive Rush", "Ice Shard", "Protect"]},
    {"card_id": "lt", "species": "Landorus-Therian", "ability": "Intimidate", "item": "Choice Scarf", "nature": "Jolly",
     "evs": {"atk": 252, "def": 4, "spe": 252}, "moves": ["Earthquake", "Rock Slide", "U-turn", "Protect"]},
    {"card_id": "fg", "species": "Farigiraf", "ability": "Armor Tail", "moves": ["Trick Room", "Psychic", "Dazzling Gleam", "Helping Hand"]},
    {"card_id": "p2", "species": "Porygon2", "ability": "Download", "item": "Eviolite", "moves": ["Trick Room", "Tri Attack", "Ice Beam", "Recover"]},
])
TABLE = SETS.overlay(meta.table())
TYPES = {"chienpao": ["DARK", "ICE"], "rillaboom": ["GRASS"], "ironhands": ["FIGHTING", "ELECTRIC"],
         "baxcalibur": ["DRAGON", "ICE"], "landorustherian": ["GROUND", "FLYING"], "farigiraf": ["NORMAL", "PSYCHIC"],
         "porygon2": ["NORMAL"], "dragonite": ["DRAGON", "FLYING"], "dondozo": ["WATER"]}


def _obs(ours, theirs, hp=None, turn=3, **extra):
    hp = hp or {}
    return {"turn": turn, **extra,
            "active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in ours],
            "opponent_active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in theirs]}


# -- 1. a Choice item holder never Protects --------------------------------------------------


def test_turn_5_a_choice_scarf_landorus_never_protects():
    F = [{"target": 1, "side": "opponent", "species": "ironhands"}, {"target": 2, "side": "opponent", "species": "porygon2"}]
    action = _doubles("landorustherian", [_move("earthquake", []), _move("rockslide", []), _move("uturn", [1, 2], F),
                                          _move("protect", [])],
                      "farigiraf", [_move("psychic", [1, 2], F), _move("trickroom", [])])
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, _obs(["landorustherian", "farigiraf"], ["ironhands", "porygon2"], turn=5),
                             MoveTracker(), TABLE, lambda line: None)
    assert "lock landorustherian into it" in turn_rules.per_slot[0].blocked[3]
    assert {1, 2} - set(turn_rules.per_slot[0].blocked) == {1, 2}  # Rock Slide and U-turn stay


def test_a_choice_locked_holder_keeps_the_only_move_it_has():
    """Already locked into Protect, the server offers nothing else: it must still send it."""
    action = _doubles("landorustherian", [_move("protect", [])], "farigiraf", [_move("trickroom", [])])
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, _obs(["landorustherian", "farigiraf"], ["ironhands", "porygon2"], turn=6),
                             MoveTracker(), TABLE, lambda line: None)
    assert 0 not in turn_rules.per_slot[0].blocked


# -- 2. a priority KO beats a bigger, slower hit ---------------------------------------------

F3 = [{"target": 1, "side": "opponent", "species": "ironhands"}, {"target": 2, "side": "opponent", "species": "baxcalibur"}]
TURN_3 = _doubles(
    "chienpao", [_move("iciclecrash", [1, 2], F3), _move("sacredsword", [1, 2], F3), _move("suckerpunch", [1, 2], F3),
                 _move("protect", [])],
    "rillaboom", [_move("woodhammer", [1, 2], F3), _move("grassyglide", [1, 2], F3), _move("uturn", [1, 2], F3)])
TURN_3_OBS = _obs(["chienpao", "rillaboom"], ["ironhands", "baxcalibur"],
                  hp={"chienpao": 0.12, "rillaboom": 0.27, "ironhands": 0.64, "baxcalibur": 0.06}, fields={"GRASSY_TERRAIN": 3})
READ = {"ironhands": {"action": "attack", "target": "chienpao", "confidence": 0.65},
        "baxcalibur": {"action": "attack", "target": "rillaboom", "confidence": 0.6}}


def test_baxcalibur_finishes_rillaboom_with_ice_shard():
    calc = lookahead.Calculator(TURN_3_OBS, TABLE)
    sim = race._Sim(calc, race.Board(), [race._mon("us", e, TABLE) for e in calc._ours],
                    [race._mon("them", e, TABLE) for e in calc._theirs], TURN_3_OBS)
    assert sim._best_move(sim.theirs[1], [sim.ours[1]])["id"] == "iceshard"


def test_turn_3_sucker_punch_takes_baxcalibur_before_its_ice_shard():
    slots = act.parse_doubles(_as_legal(TURN_3))
    turn_rules = rules.build(slots, TURN_3_OBS, MoveTracker(known_moves=TABLE.common_moves), TABLE, lambda line: None)
    calc = lookahead.Calculator(TURN_3_OBS, TABLE)
    expected = predict.expectations(READ, ["chienpao", "rillaboom"])
    best = turn_rules.check(lookahead.best_line(turn_rules, TURN_3_OBS, calc, expected))
    assert best["slot_0"] == {"type": "move", "move_id": "suckerpunch", "target": 2}


# -- 3. no Protect read from a move that can miss ---------------------------------------------


def test_no_damage_from_a_scale_shot_is_no_protect():
    F = [{"target": 1, "side": "opponent", "species": "dondozo"}, {"target": 2, "side": "opponent", "species": "porygon2"}]
    slots = act.parse_doubles(_as_legal(_doubles(
        "dragonite", [_move("scaleshot", [1, 2], F), _move("dragonclaw", [1, 2], F)],
        "farigiraf", [_move("psychic", [1, 2], F)])))
    observation = _obs(["dragonite", "farigiraf"], ["dondozo", "porygon2"], turn=8)

    def run(move_id):
        tracker = MoveTracker()
        tracker.begin(slots, observation)
        tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "dragonclaw", "target": 2},
                        "slot_1": {"type": "move", "move_id": "psychic", "target": 2}})
        tracker.begin(slots, observation)
        tracker.record({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": move_id, "target": 1},
                        "slot_1": {"type": "move", "move_id": "psychic", "target": 2}})
        tracker.expect({"dondozo": 0.38})
        tracker.begin(slots, observation)  # Dondozo took nothing
        return tracker.protected_last_turn

    assert "dondozo" not in run("scaleshot")  # 90%: it may have missed
    assert "dondozo" in run("dragonclaw")  # cannot miss: no damage means a Protect


# -- 4. the exact board is logged at every decision --------------------------------------------


def test_the_log_keeps_exact_hp_boosts_and_status():
    observation = {
        "active_pokemon": [{"species": "dragonite", "current_hp": 160, "max_hp": 232, "hp_fraction": 0.69,
                            "boosts": {"spe": 1, "def": -1}},
                           {"species": "greattusk", "hp_fraction": 0.43}],
        "opponent_active_pokemon": [{"species": "dondozo", "hp_fraction": 0.5, "status": "brn", "item": "Leftovers"}],
        "team": [{"species": "dragonite", "hp_fraction": 0.69}, {"species": "lucario", "hp_fraction": 0.0}],
        "fields": {"GRASSY_TERRAIN": 5},
    }
    board = _board_detail(observation)
    assert board["ours"]["dragonite"] == {"hp": 0.69, "hp_exact": "160/232", "boosts": {"spe": 1, "def": -1}}
    assert board["theirs"]["dondozo"] == {"hp": 0.5, "status": "brn", "item": "Leftovers"}
    assert board["our_bench_hp"] == {"lucario": 0.0}
    text = Analysis(turn="Turn 12", state_summary="", prediction={}, source="model", code_guess={}, board=board).render()
    assert "**Board:**" in text and '"hp_exact":"160/232"' in text
