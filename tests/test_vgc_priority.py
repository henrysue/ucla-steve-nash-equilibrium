"""Priority and turn order: Fake Out, Extreme Speed, Sucker Punch, guards, pivots."""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, predict, race, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import BATTLE_OBSERVATION, POKEMON, FakeProvider, _agent, _doubles_action, _doubles_answer, _move, _state

CARDS = [
    {"card_id": "vgc-incineroar", "species": "Incineroar", "item": "Safety Goggles", "ability": "Intimidate", "nature": "Careful",
     "evs": {"hp": 244, "def": 12, "spd": 252}, "moves": ["Fake Out", "Knock Off", "Parting Shot", "Flare Blitz"]},
    {"card_id": "vgc-dragonite", "species": "Dragonite", "item": "Loaded Dice", "ability": "Multiscale", "nature": "Adamant",
     "evs": {"hp": 252, "atk": 252, "spe": 4}, "moves": ["Extreme Speed", "Scale Shot", "Tailwind", "Protect"]},
    {"card_id": "vgc-landorus-therian", "species": "Landorus-Therian", "item": "Choice Scarf", "ability": "Intimidate",
     "nature": "Jolly", "evs": {"atk": 252, "spe": 252}, "moves": ["Earthquake", "Rock Slide", "U-turn", "Protect"]},
    {"card_id": "vgc-rillaboom", "species": "Rillaboom", "item": "Assault Vest", "ability": "Grassy Surge", "nature": "Adamant",
     "evs": {"hp": 236, "atk": 116, "spe": 148}, "moves": ["Fake Out", "Wood Hammer", "Grassy Glide", "U-turn"]},
    {"card_id": "vgc-chien-pao", "species": "Chien-Pao", "item": "Focus Sash", "ability": "Sword of Ruin", "nature": "Jolly",
     "evs": {"atk": 252, "spe": 252}, "moves": ["Icicle Crash", "Sacred Sword", "Sucker Punch", "Protect"]},
    {"card_id": "vgc-ursaluna", "species": "Ursaluna", "item": "Flame Orb", "ability": "Guts", "nature": "Adamant",
     "evs": {"hp": 252, "atk": 252}, "moves": ["Facade", "Headlong Rush", "Earthquake", "Protect"]},
    {"card_id": "vgc-grimmsnarl", "species": "Grimmsnarl", "item": "Light Clay", "ability": "Prankster", "nature": "Careful",
     "evs": {"hp": 252, "spd": 252}, "moves": ["Light Screen", "Reflect", "Spirit Break", "Thunder Wave"]},
]
TYPES = {"incineroar": ["FIRE", "DARK"], "dragonite": ["DRAGON", "FLYING"], "landorustherian": ["GROUND", "FLYING"],
         "rillaboom": ["GRASS"], "chienpao": ["DARK", "ICE"], "ursaluna": ["GROUND", "NORMAL"], "grimmsnarl": ["DARK", "FAIRY"]}


def _table():
    sets = CardSets(persist=False)
    sets.remember(CARDS)
    return sets.overlay(meta.table())


def _obs(ours, theirs, extra=None):
    return {"turn": 1, "active_pokemon": [{"species": s, "types": TYPES[s]} for s in ours],
            "opponent_active_pokemon": [{"species": s, "types": TYPES[s]} for s in theirs], **(extra or {})}


def _setup(action, observation, prediction, ours):
    table = _table()
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(known_moves=table.common_moves), table, lambda line: None)
    return turn_rules, lookahead.Calculator(observation, table), predict.expectations(prediction, ours)


def _score(turn_rules, calc, observation, expected, slot_0, slot_1, bench=None):
    return race.score([slot_0, slot_1], turn_rules.slots, turn_rules.per_slot, observation, calc, expected, bench)


FOES = [{"target": 1, "side": "opponent", "species": "incineroar"}, {"target": 2, "side": "opponent", "species": "ursaluna"}]


def test_fake_out_plus3_flinches_extreme_speed_plus2_so_protect_answers_it():
    """A confident read of Fake Out into Dragonite: Extreme Speed (+2) still moves
    after Fake Out (+3) and is flinched; Protect (+4) is the answer."""
    action = _doubles("dragonite", [_move("extremespeed", [1, 2], FOES), _move("protect", [])],
                      "chienpao", [_move("icicleCrash".lower(), [1, 2], FOES)])
    observation = _obs(["dragonite", "chienpao"], ["incineroar", "ursaluna"])
    read = {"incineroar": {"action": "fake_out", "target": "dragonite", "confidence": 0.9},
            "ursaluna": {"action": "attack", "target": "chienpao", "confidence": 0.6}}
    turn_rules, calc, expected = _setup(action, observation, read, ["dragonite", "chienpao"])
    icicle = {"option": 0, "target": 2}
    flinched_es = _score(turn_rules, calc, observation, expected, {"option": 0, "target": 2}, icicle)
    protected = _score(turn_rules, calc, observation, expected, {"option": 1, "target": None}, icicle)
    no_fake_out = predict.expectations({"incineroar": {"action": "attack", "target": "dragonite", "confidence": 0.9},
                                        "ursaluna": read["ursaluna"]}, ["dragonite", "chienpao"])
    free_es = _score(turn_rules, calc, observation, no_fake_out, {"option": 0, "target": 2}, icicle)
    assert protected > flinched_es
    assert free_es > flinched_es  # the flinch is what costs Extreme Speed


def _sim(observation, table=None):
    calc = lookahead.Calculator(observation, table or _table())
    ours = [race._mon("us", e, calc._table) for e in observation["active_pokemon"]]
    theirs = [race._mon("them", e, calc._table) for e in observation["opponent_active_pokemon"]]
    return race._Sim(calc, race.Board(), ours, theirs, observation)


FAKE_OUT = {"id": "fakeout", "base_power": 40, "type": "NORMAL", "category": "PHYSICAL", "spread": False, "priority": 3}


def test_inner_focus_and_covert_cloak_are_not_flinched():
    observation = _obs(["dragonite"], ["incineroar"])
    sim = _sim(observation)
    target = sim.ours[0]
    target.entry["item"] = "Covert Cloak"
    sim._hit("fakeout", sim.theirs[0], FAKE_OUT, [(target, 1.0)], 3)
    assert target.flinch == 0.0
    target.entry["item"] = "Loaded Dice"
    sim._hit("fakeout", sim.theirs[0], FAKE_OUT, [(target, 1.0)], 3)
    assert target.flinch > 0.9


def test_quick_guard_blocks_fake_out():
    sim = _sim(_obs(["dragonite"], ["incineroar"]))
    sim.board.guard["us"]["quick"] = 1.0
    sim._hit("fakeout", sim.theirs[0], FAKE_OUT, [(sim.ours[0], 1.0)], 3)
    assert sim.ours[0].flinch == 0.0 and sim.ours[0].hp == 100.0


def test_sucker_punch_fails_into_a_target_that_does_not_attack():
    sim = _sim(_obs(["chienpao"], ["ursaluna"]))
    foe = sim.theirs[0]
    sucker = {"id": "suckerpunch", "base_power": 70, "type": "DARK", "category": "PHYSICAL", "spread": False, "priority": 1}
    foe.attacking = 0.0  # it is protecting / setting up / switching
    sim._hit("attack", sim.ours[0], sucker, [(foe, 1.0)], 1)
    assert foe.hp == 100.0
    foe.attacking = 1.0
    sim._hit("attack", sim.ours[0], sucker, [(foe, 1.0)], 1)
    assert foe.hp < 100.0


def test_grassy_glide_has_priority_only_in_grassy_terrain():
    glide = {"id": "grassyglide", "type": "GRASS", "category": "PHYSICAL", "priority": 0}
    plain = _sim(_obs(["rillaboom"], ["incineroar"]))
    grassy = _sim(_obs(["rillaboom"], ["incineroar"], {"fields": {"GRASSY_TERRAIN": 1}}))
    assert plain.priority(plain.ours[0], glide) == 0
    assert grassy.priority(grassy.ours[0], glide) == 1


def test_a_faster_u_turn_user_escapes_the_hit_aimed_at_it():
    """Scarf Landorus U-turns first; Ursaluna's attack aimed at its slot (Facade, its
    best move into Landorus) lands on whoever came in, Dragonite, instead."""
    observation = _obs(["landorustherian", "chienpao"], ["incineroar", "ursaluna"])
    sim = _sim(observation)
    lando, ursaluna = sim.ours[0], sim.theirs[1]
    dragonite = race._mon("us", {"species": "dragonite"}, sim.calc._table)
    dragonite.presence, lando.pivot_to = 0.0, dragonite
    sim.ours.append(dragonite)
    uturn = {"id": "uturn", "base_power": 70, "type": "BUG", "category": "PHYSICAL", "spread": False, "priority": 0}
    rush = sim._best_move(ursaluna, [lando])
    sim.resolve([("attack", lando, uturn, [(sim.theirs[0], 1.0)], 0),
                 ("attack", ursaluna, rush, [(lando, 1.0)], 0)])
    assert lando.presence < 0.01 and dragonite.presence > 0.99
    assert lando.hp > 99  # it was gone before the hit
    assert dragonite.hp < 95  # the switch-in took it


def test_the_model_sees_who_moves_first():
    provider = FakeProvider(_doubles_answer())
    _agent(provider).choose_action(_state(_doubles_action(), observation=BATTLE_OBSERVATION), POKEMON)
    order = provider.prompt()["move_order"]
    assert "Fake Out +3" in " ".join(order["priority_moves"].get("incineroar", []))
    assert order["speed_order_within_a_bracket"] and "bracket" in order["rule"]
