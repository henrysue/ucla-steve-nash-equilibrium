"""Protect likelihood per opposing active, and targeting around it."""

from agent.vgc import actions as act
from agent.vgc import protect

from test_vgc_moverules import _as_legal, _doubles
from test_vgc_strategist import BATTLE_OBSERVATION, FOES, POKEMON, FakeProvider, _agent, _doubles_answer, _move, _state


def _foes(*entries):
    return {**BATTLE_OBSERVATION, "opponent_active_pokemon": list(entries)}


COMMON = {"chienpao": ["Protect", "Sucker Punch"], "incineroar": ["Fake Out", "Knock Off", "Flare Blitz", "Parting Shot"]}.get


def _odds(observation, first=True, rate=protect.FIRST_TURN):
    found = protect.odds(observation, first_turn={"chienpao": first, "incineroar": first},
                         common_moves=lambda n: COMMON(protect._id(n)) or [], first_turn_rate=rate)
    return {k: v["likelihood"] for k, v in found.items()}


def test_a_first_turn_protect_follows_what_this_opponent_has_shown():
    observation = _foes({"species": "Chien-Pao", "types": ["DARK", "ICE"], "hp_fraction": 1.0})
    observation["available_moves"] = [[{"id": "closecombat", "type": "FIGHTING", "base_power": 120}], []]
    assert _odds(observation)["chienpao"] == protect.FIRST_TURN < protect.LIKELY  # unknown opponent: 35%
    assert _odds(observation, rate=0.8)["chienpao"] >= protect.LIKELY  # one that always Protects first


def test_a_pokemon_without_protect_is_rarely_expected_to():
    assert _odds(_foes({"species": "Incineroar", "types": ["FIRE", "DARK"]}))["incineroar"] <= 0.05


def test_protecting_last_turn_makes_a_repeat_unlikely():
    entry = {"species": "Chien-Pao", "types": ["DARK", "ICE"], "last_move": "Protect"}
    assert _odds(_foes(entry), first=False)["chienpao"] < 0.15


def _slots():
    action = _doubles(
        "incineroar", [_move("flareblitz", [1, 2], FOES), _move("tailwind", [])],
        "fluttermane", [_move("moonblast", [1, 2], FOES)],
    )
    return act.parse_doubles(_as_legal(action))


OBS = {
    **BATTLE_OBSERVATION,
    "available_moves": [
        [{"id": "flareblitz", "type": "FIRE", "category": "PHYSICAL", "base_power": 120},
         {"id": "tailwind", "type": "FLYING", "category": "STATUS", "base_power": 0}],
        [{"id": "moonblast", "type": "FAIRY", "category": "SPECIAL", "base_power": 95}],
    ],
    "opponent_active_pokemon": [
        {"species": "landorustherian", "types": ["GROUND", "FLYING"], "hp_fraction": 1.0},
        {"species": "amoonguss", "types": ["GRASS", "POISON"], "hp_fraction": 1.0},
    ],
}


def test_the_prompt_shows_protect_likelihood_for_each_foe():
    provider = FakeProvider(_doubles_answer())
    _agent(provider).choose_action(_state(_as_legal_raw(), observation=BATTLE_OBSERVATION), POKEMON)
    assert set(provider.prompt()["protect_odds"]) == {"landorustherian", "amoonguss"}


def _as_legal_raw():
    from test_vgc_strategist import _doubles_action

    return _doubles_action()
