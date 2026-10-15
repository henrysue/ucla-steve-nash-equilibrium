"""The two-turn damage race that ranks our joint actions."""

from agent.vgc import actions as act
from agent.vgc import lookahead, meta, predict, race, rules
from agent.vgc.cards import CardSets
from agent.vgc.moverules import MoveTracker

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import _move

TABLE = meta.table()
TYPES = {"incineroar": ["FIRE", "DARK"], "talonflame": ["FIRE", "FLYING"], "archaludon": ["STEEL", "DRAGON"],
         "gyarados": ["WATER", "FLYING"], "amoonguss": ["GRASS", "POISON"], "landorustherian": ["GROUND", "FLYING"]}
FOES = [{"target": 1, "side": "opponent", "species": "archaludon"}, {"target": 2, "side": "opponent", "species": "gyarados"}]


def _obs(ours, theirs, hp=None, turn=1):
    hp = hp or {}
    return {"turn": turn,
            "active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in ours],
            "opponent_active_pokemon": [{"species": s, "types": TYPES[s], "hp_fraction": hp.get(s, 1.0)} for s in theirs]}


def _setup(action, observation, prediction, ours):
    slots = act.parse_doubles(_as_legal(action))
    turn_rules = rules.build(slots, observation, MoveTracker(known_moves=TABLE.common_moves), TABLE, lambda line: None)
    calc = lookahead.Calculator(observation, TABLE)
    return turn_rules, calc, predict.expectations(prediction, ours)


def _score(turn_rules, calc, observation, expected, slot_0, slot_1):
    return race.score([slot_0, slot_1], turn_rules.slots, turn_rules.per_slot, observation, calc, expected)


TURN_1 = _doubles(
    "incineroar", [_move("fakeout", [1, 2], FOES), _move("knockoff", [1, 2], FOES), _move("flareblitz", [1, 2], FOES)],
    "talonflame", [_move("bravebird", [1, 2], FOES), _move("tailwind", []), _move("protect", [])],
)
READ = {"archaludon": {"action": "attack", "target": "talonflame", "confidence": 0.55},
        "gyarados": {"action": "attack", "target": "talonflame", "confidence": 0.55}}


def test_turn_one_of_8a882a3f_attacks_instead_of_protecting():
    """Played: Fake Out + Protect on a full-HP Talonflame, scored 204 vs 145 for
    Tailwind by the one-turn view; the hits just came the turn after."""
    observation = _obs(["incineroar", "talonflame"], ["archaludon", "gyarados"])
    turn_rules, calc, expected = _setup(TURN_1, observation, READ, ["incineroar", "talonflame"])
    payload = turn_rules.check(lookahead.best_line(turn_rules, observation, calc, expected))
    # Opening turn: both foes may Protect, so Fake Out waits (match 4e60a5a2); Incineroar attacks.
    assert payload["slot_0"]["move_id"] in ("knockoff", "flareblitz")
    assert payload["slot_1"]["move_id"] != "protect"


def test_gale_wings_gives_a_full_hp_talonflames_flying_moves_priority():
    sets = CardSets(persist=False)
    sets.remember([{"card_id": "vgc-talonflame", "species": "Talonflame", "ability": "Gale Wings",
                    "moves": ["Brave Bird", "Tailwind", "Protect", "Taunt"]}])
    observation = _obs(["incineroar", "talonflame"], ["archaludon", "gyarados"])
    sim = race._Sim(lookahead.Calculator(observation, sets.overlay(TABLE)), race.Board(), [], [], observation)
    bravebird = {"type": "FLYING", "category": "PHYSICAL", "priority": 0}
    assert sim.priority(race.Mon("us", {"species": "talonflame"}, 100.0), bravebird) == 1
    assert sim.priority(race.Mon("us", {"species": "talonflame"}, 60.0), bravebird) == 0
    # Without the card the table offers Flame Body too, so no priority is assumed.
    plain = race._Sim(lookahead.Calculator(observation, TABLE), race.Board(), [], [], observation)
    assert plain.priority(race.Mon("us", {"species": "talonflame"}, 100.0), bravebird) == 0


def test_spore_into_a_grass_type_does_nothing():
    foes = [{"target": 1, "side": "opponent", "species": "amoonguss"}, {"target": 2, "side": "opponent", "species": "landorustherian"}]
    action = _doubles("amoonguss", [_move("spore", [1, 2], foes)], "incineroar", [_move("flareblitz", [1, 2], foes)])
    observation = _obs(["amoonguss", "incineroar"], ["amoonguss", "landorustherian"])
    turn_rules, calc, expected = _setup(action, observation, {}, ["amoonguss", "incineroar"])
    flare = {"option": 0, "target": 2}
    on_grass = _score(turn_rules, calc, observation, expected, {"option": 0, "target": 1}, flare)
    on_lando = _score(turn_rules, calc, observation, expected, {"option": 0, "target": 2}, flare)
    assert on_lando > on_grass


def test_ko_chance_follows_the_damage_range():
    assert race._ko_chance(40, 60, 50) == 0.5
    assert race._ko_chance(10, 20, 50) == 0.0
    assert race._ko_chance(60, 70, 50) == 1.0
