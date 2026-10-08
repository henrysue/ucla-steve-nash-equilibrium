"""Type effectiveness, the deterministic fallback, and the meta reference table."""

import pytest

from altruagent.models import GameState, LegalAction
from agent.vgc import actions as act
from agent.vgc import meta
from agent.vgc.typechart import effectiveness, is_immune


# -- fixtures ------------------------------------------------------------------------------


def _action(template):
    return LegalAction.from_dict({
        "action_id": template["type"],
        "label": template["type"],
        "input": {"action_id": template["type"], "action": template},
    })


def _state(observation):
    return GameState.from_mcp_state({
        "session_id": "s", "status": "in_progress", "phase": "moving",
        "is_current_actor": True, "is_terminal": False, "state_version": 1,
        "observation": observation,
    })


def _template(slot_0_options, slot_1_options=None):
    return {
        "type": "doubles_turn",
        "slots": [
            {"slot": 0, "active": "a", "force_switch": False, "options": slot_0_options},
            {"slot": 1, "active": "b", "force_switch": False, "options": slot_1_options or [{"type": "pass"}]},
        ],
    }


def _move(move_id, targets, sides=None):
    option = {"type": "move", "move_id": move_id, "targets": list(targets)}
    if sides:
        option["target_options"] = sides
    return option


FLYING_FOE = {"species": "landorustherian", "types": ["GROUND", "FLYING"], "hp_fraction": 1.0}
GRASS_FOE = {"species": "amoonguss", "types": ["GRASS", "POISON"], "hp_fraction": 1.0}


def _observation(moves_slot_0, foes=None, allies=None):
    return {
        "is_doubles": True,
        "available_moves": [moves_slot_0, []],
        "active_pokemon": allies or [{"species": "incineroar", "types": ["FIRE", "DARK"]}],
        "opponent_active_pokemon": foes or [FLYING_FOE, GRASS_FOE],
    }


# -- the type chart ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "move_type,defender,expected",
    [
        ("ELECTRIC", ["GROUND"], 0.0),
        ("GROUND", ["FLYING"], 0.0),
        ("NORMAL", ["GHOST"], 0.0),
        ("DRAGON", ["FAIRY"], 0.0),
        ("POISON", ["STEEL"], 0.0),
        ("PSYCHIC", ["DARK"], 0.0),
        ("FIRE", ["GRASS", "POISON"], 2.0),
        ("ICE", ["GROUND", "FLYING"], 4.0),  # Landorus-T's famous 4x
        ("ROCK", ["FIRE", "FLYING"], 4.0),
        ("WATER", ["WATER", "GRASS"], 0.25),
        ("FIGHTING", ["NORMAL"], 2.0),
        ("FAIRY", ["DRAGON", "FLYING"], 2.0),
        ("GRASS", ["WATER", "GROUND"], 4.0),
        ("STEEL", ["FAIRY"], 2.0),
        ("NORMAL", ["WATER"], 1.0),
    ],
)
def test_type_effectiveness(move_type, defender, expected):
    assert effectiveness(move_type, defender) == pytest.approx(expected)


def test_effectiveness_is_case_insensitive():
    assert effectiveness("fire", ["grass"]) == 2.0


@pytest.mark.parametrize("move_type,defender", [(None, ["FIRE"]), ("FIRE", None), ("MYSTERY", ["FIRE"]), ("FIRE", [])])
def test_unknown_types_are_neutral_never_pruned(move_type, defender):
    """A type we cannot read must never silently remove a legal move."""
    assert effectiveness(move_type, defender) == 1.0
    assert is_immune(move_type, defender) is False


def test_is_immune_matches_zero_effectiveness():
    assert is_immune("GROUND", ["FLYING"]) is True
    assert is_immune("GROUND", ["FIRE"]) is False


# -- the deterministic fallback ------------------------------------------------------------


def test_the_fallback_never_attacks_into_an_immunity():
    """Earthquake into a Flying foe is strictly dominated (3.2)."""
    slots = act.parse_doubles(_action(_template([
        _move("earthquake", [1], [{"target": 1, "side": "opponent", "species": "landorustherian"}]),
        _move("knockoff", [1], [{"target": 1, "side": "opponent", "species": "landorustherian"}]),
    ])))
    observation = _observation(
        [
            {"id": "earthquake", "type": "GROUND", "base_power": 100, "accuracy": 1.0},
            {"id": "knockoff", "type": "DARK", "base_power": 65, "accuracy": 1.0},
        ],
        foes=[FLYING_FOE],
    )
    payload = act.heuristic_doubles(slots, _state(observation))
    assert payload["slot_0"]["move_id"] == "knockoff"


def test_the_fallback_prefers_the_super_effective_target():
    slots = act.parse_doubles(_action(_template([
        _move("flareblitz", [1, 2], [
            {"target": 1, "side": "opponent", "species": "landorustherian"},
            {"target": 2, "side": "opponent", "species": "amoonguss"},
        ]),
    ])))
    observation = _observation([{"id": "flareblitz", "type": "FIRE", "base_power": 120, "accuracy": 1.0}])
    payload = act.heuristic_doubles(slots, _state(observation))
    assert payload["slot_0"]["target"] == 2  # Fire into Grass/Poison, not Ground/Flying


def test_the_fallback_does_not_attack_its_own_ally():
    slots = act.parse_doubles(_action(_template([
        _move("flareblitz", [-2, 1], [
            {"target": -2, "side": "ally", "species": "fluttermane"},
            {"target": 1, "side": "opponent", "species": "amoonguss"},
        ]),
    ])))
    observation = _observation(
        [{"id": "flareblitz", "type": "FIRE", "base_power": 120, "accuracy": 1.0}], foes=[GRASS_FOE]
    )
    payload = act.heuristic_doubles(slots, _state(observation))
    assert payload["slot_0"]["target"] == 1


def test_the_fallback_finishes_a_weakened_foe():
    slots = act.parse_doubles(_action(_template([
        _move("knockoff", [1, 2], [
            {"target": 1, "side": "opponent", "species": "healthy"},
            {"target": 2, "side": "opponent", "species": "chipped"},
        ]),
    ])))
    observation = _observation(
        [{"id": "knockoff", "type": "DARK", "base_power": 65, "accuracy": 1.0}],
        foes=[
            {"species": "healthy", "types": ["NORMAL"], "hp_fraction": 1.0},
            {"species": "chipped", "types": ["NORMAL"], "hp_fraction": 0.12},
        ],
    )
    payload = act.heuristic_doubles(slots, _state(observation))
    assert payload["slot_0"]["target"] == 2


def test_the_fallback_prefers_damage_over_a_status_move():
    slots = act.parse_doubles(_action(_template([
        _move("protect", []),
        _move("knockoff", [1], [{"target": 1, "side": "opponent", "species": "amoonguss"}]),
    ])))
    observation = _observation(
        [
            {"id": "protect", "type": "NORMAL", "base_power": 0, "accuracy": 1.0},
            {"id": "knockoff", "type": "DARK", "base_power": 65, "accuracy": 1.0},
        ],
        foes=[GRASS_FOE],
    )
    payload = act.heuristic_doubles(slots, _state(observation))
    assert payload["slot_0"]["move_id"] == "knockoff"


def test_the_fallback_switches_when_switching_is_all_there_is():
    slots = act.parse_doubles(_action(_template(
        [{"type": "switch", "species": "rillaboom"}],
        [{"type": "switch", "species": "ursaluna"}],
    )))
    payload = act.heuristic_doubles(slots, _state(_observation([])))
    assert payload["slot_0"] == {"type": "switch", "species": "rillaboom"}
    assert payload["slot_1"] == {"type": "switch", "species": "ursaluna"}


def test_the_fallback_passes_when_pass_is_the_only_option():
    slots = act.parse_doubles(_action(_template([{"type": "pass"}], [{"type": "pass"}])))
    payload = act.heuristic_doubles(slots, _state(_observation([])))
    assert payload == {"type": "doubles_turn", "slot_0": {"type": "pass"}, "slot_1": {"type": "pass"}}


def test_the_fallback_does_not_switch_both_slots_into_the_same_pokemon():
    slots = act.parse_doubles(_action(_template(
        [{"type": "switch", "species": "rillaboom"}],
        [{"type": "switch", "species": "rillaboom"}, {"type": "switch", "species": "ursaluna"}],
    )))
    payload = act.heuristic_doubles(slots, _state(_observation([])))
    assert payload["slot_1"]["species"] == "ursaluna"


def test_a_move_with_no_targets_omits_the_target_key():
    slots = act.parse_doubles(_action(_template([_move("tailwind", [])])))
    observation = _observation([{"id": "tailwind", "type": "FLYING", "base_power": 0, "accuracy": 1.0}])
    payload = act.heuristic_doubles(slots, _state(observation))
    assert payload["slot_0"] == {"type": "move", "move_id": "tailwind"}


# -- describing options to the model -------------------------------------------------------


def test_options_are_described_with_named_targets_and_multipliers():
    slots = act.parse_doubles(_action(_template([
        _move("earthquake", [1, 2], [
            {"target": 1, "side": "opponent", "species": "landorustherian"},
            {"target": 2, "side": "opponent", "species": "amoonguss"},
        ]),
    ])))
    observation = _observation([{"id": "earthquake", "type": "GROUND", "base_power": 100, "accuracy": 1.0}])
    described = act.describe_slots(slots, _state(observation))

    option = described[0]["options"][0]
    assert option["move_type"] == "GROUND"
    assert option["base_power"] == 100
    targets = {entry["target"]: entry for entry in option["targets"]}
    assert targets[1]["immune"] is True  # Ground into Flying
    assert targets[1]["type_multiplier"] == 0.0
    assert "OPPONENT landorustherian" in targets[1]["is"]
    assert "immune" not in targets[2]


def test_the_option_kind_is_never_mistaken_for_the_move_type():
    """A regression guard: the option's own 'type' is 'move', not an element."""
    slots = act.parse_doubles(_action(_template([_move("knockoff", [1])])))
    described = act.describe_slots(slots, _state(_observation([])))
    option = described[0]["options"][0]
    assert option["type"] == "move"
    assert "move_type" not in option  # nothing in the observation to read it from


def test_support_move_families_are_labelled():
    slots = act.parse_doubles(_action(_template([
        _move("protect", []), _move("tailwind", []), _move("ragepowder", []),
    ])))
    described = act.describe_slots(slots, _state(_observation([])))
    families = [option.get("family") for option in described[0]["options"]]
    assert families == ["protection", "speed_control", "redirection"]


# -- template validation -------------------------------------------------------------------


@pytest.mark.parametrize("template", [
    {"type": "doubles_turn", "slots": [{"slot": 0, "options": [{"type": "pass"}]}]},
    {"type": "doubles_turn", "slots": "nope"},
    {"type": "doubles_turn", "slots": [
        {"slot": 0, "options": []}, {"slot": 1, "options": [{"type": "pass"}]}]},
    {"type": "doubles_turn", "slots": [
        {"slot": 0, "options": [{"type": "nonsense"}]}, {"slot": 1, "options": [{"type": "pass"}]}]},
    {"type": "doubles_turn", "slots": [
        {"slot": 0, "options": [{"type": "pass"}]}, {"slot": 0, "options": [{"type": "pass"}]}]},
])
def test_unreadable_doubles_templates_are_refused(template):
    with pytest.raises(act.TemplateError):
        act.parse_doubles(_action(template))


def test_a_lineup_template_needs_four_distinct_species():
    with pytest.raises(act.TemplateError):
        act.parse_lineup(_action({"type": "select_lineup", "roster": ["a", "b", "a"]}))


def test_a_lineup_roster_is_deduplicated():
    roster = act.parse_lineup(
        _action({"type": "select_lineup", "roster": ["a", "b", "c", "d", "a"]})
    )
    assert roster == ["a", "b", "c", "d"]


def test_the_wrong_template_type_is_refused():
    with pytest.raises(act.TemplateError, match="select_lineup"):
        act.parse_lineup(_action({"type": "doubles_turn", "slots": []}))


# -- the meta reference table --------------------------------------------------------------


def test_the_reference_table_loads():
    assert len(meta.table()) > 400


def test_the_loaded_table_is_the_tournament_format():
    """gen9vgc2025regi — Reg I, which is what GAMES.md says is played."""
    assert meta.DEFAULT_CSV.name == "vgc_2025_reg_i_draft_doubles.csv"
    table = meta.table()
    assert table.columns["usage_pct"] == "Usage % (Pikalytics, 1760+)"
    assert table.columns["cores"] == "2025 Worlds Top 8 Teams Featuring It"
    # Megas are not legal in Reg I and are absent from the table.
    assert table.brief("salamencemega") is None


@pytest.mark.parametrize("csv_name,usage_column", [
    ("vgc_2025_reg_i_draft_doubles.csv", "Usage % (Pikalytics, 1760+)"),
    ("vgc_reg_mc_draft_doubles.csv", "Usage % (Sept 2026 ladder)"),
])
def test_renamed_columns_resolve_through_aliases(monkeypatch, csv_name, usage_column):
    """The table is replaced when the format rotates and headers move with it.

    Both real files must load: the usage column is named after whatever
    measured it, and Reg MC's Mega columns became Reg I's restricted ones.
    """
    path = meta.DATA_DIR / csv_name
    if not path.exists():
        pytest.skip(f"{csv_name} is not present")
    meta.table.cache_clear()
    monkeypatch.setenv("VGC_META_CSV", str(path))
    try:
        table = meta.table()
        assert len(table) > 300
        assert table.columns["usage_pct"] == usage_column
        brief = table.brief("incineroar")
        assert brief["species"] == "Incineroar"
        assert "role" in brief and "countered_by" in brief
    finally:
        meta.table.cache_clear()


def test_an_absent_column_drops_out_rather_than_raising():
    """A future table missing a column must not break the briefs."""
    table = meta.MetaTable([
        {"Pokemon": "Incineroar", "Primary Type": "Fire", "Doubles Role": "pivot"},
    ])
    brief = table.brief("incineroar")
    assert brief == {"species": "Incineroar", "type1": "Fire", "role": "pivot"}
    assert table.draft_points("incineroar") is None
    assert table.is_restricted("incineroar") is False


@pytest.mark.parametrize("name,expected", [
    ("rillaboom", "Rillaboom"),
    ("Rillaboom", "Rillaboom"),
    ("calyrexshadow", "Calyrex-Shadow"),
    ("urshifurapidstrike", "Urshifu-Rapid-Strike"),
    ("zamazentacrowned", "Zamazenta-Crowned"),
    ("ogerponhearthflame", "Ogerpon-Hearthflame"),
    ("arcaninehisui", "Arcanine-Hisui"),
    ("indeedeef", "Indeedee-F"),
    ("chiyu", "Chi-Yu"),
    ("tinglu", "Ting-Lu"),
    ("hooh", "Ho-Oh"),
])
def test_species_ids_match_showdown_naming(name, expected):
    assert meta.table().brief(name)["species"] == expected


def test_an_unlisted_species_is_simply_absent():
    assert meta.table().brief("notapokemon") is None
    assert meta.table().draft_points("notapokemon") is None
    assert meta.table().briefs(["notapokemon"]) == {}


def test_draft_points_are_read_as_integers():
    assert meta.table().draft_points("incineroar") == 20
    assert meta.table().draft_points("rillaboom") == 17


def test_a_blank_draft_points_cell_is_none_not_zero():
    """The table leaves points blank below the usage cutoff."""
    table = meta.table()
    assert table.row("landorustherian") is not None
    assert table.draft_points("landorustherian") is None


def test_restricted_legendaries_are_flagged():
    table = meta.table()
    assert table.is_restricted("calyrexshadow") is True
    assert table.is_restricted("miraidon") is True
    assert table.is_restricted("incineroar") is False
    assert table.is_restricted("notapokemon") is False


def test_restricted_in_filters_a_roster_in_order():
    roster = ["Incineroar", "Calyrex-Shadow", "Rillaboom", "Miraidon"]
    assert meta.table().restricted_in(roster) == ["Calyrex-Shadow", "Miraidon"]


def test_a_brief_shows_restricted_only_when_it_is_true():
    """"No" is true of almost every row, so it would be pure prompt noise."""
    table = meta.table()
    assert table.brief("calyrexshadow")["restricted"] is True
    assert "restricted" not in table.brief("incineroar")


def test_a_required_item_is_carried_into_the_brief():
    assert meta.table().brief("zaciancrowned")["required_item"] == "Rusted Sword"
    assert "required_item" not in meta.table().brief("incineroar")


def test_a_battle_brief_carries_more_than_a_draft_brief():
    table = meta.table()
    assert set(table.brief("rillaboom", battle=True)) > {"species", "role"}
    assert "spread_moves" in table.brief("rillaboom", battle=True)
    assert "spread_moves" not in table.brief("rillaboom")


def test_matchup_notes_cross_reference_both_directions():
    notes = meta.table().matchup_notes(["Incineroar"], ["Rillaboom"])
    assert notes["our_pokemon_that_check_theirs"]["Rillaboom"] == ["Incineroar"]


def test_matchup_notes_find_differently_spelled_forms():
    """The table's row is 'Landorus-Therian'; the prose just says 'Landorus'."""
    notes = meta.table().matchup_notes(["Landorus-Therian"], ["Incineroar"])
    assert notes["our_pokemon_that_check_theirs"]["Incineroar"] == ["Landorus-Therian"]


def test_partner_notes_link_a_roster_together():
    assert meta.table().partner_notes(["Incineroar", "Rillaboom"]) == {
        "Incineroar": ["Rillaboom"],
        "Rillaboom": ["Incineroar"],
    }


def test_a_pokemon_is_never_its_own_partner():
    for partners in meta.table().partner_notes(["Rillaboom"]).values():
        assert "Rillaboom" not in partners


def test_a_missing_table_file_yields_an_empty_table_not_an_error(monkeypatch, tmp_path):
    meta.table.cache_clear()
    monkeypatch.setenv("VGC_META_CSV", str(tmp_path / "nope.csv"))
    try:
        table = meta.table()
        assert len(table) == 0
        assert table.brief("rillaboom") is None
        assert table.matchup_notes(["a"], ["b"]) == {
            "our_pokemon_that_check_theirs": {},
            "their_pokemon_that_check_ours": {},
        }
    finally:
        meta.table.cache_clear()
