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


# -- the whole chart, checked against an independent reference ----------------
#
# The chart is the foundation under every damage estimate and every piece of
# counterplay the agent gives, so a single transposed or mistyped entry would
# quietly hand the opponent an advantage on many turns. These tests verify all
# 18x18 ordered (attacker, defender) pairs against a reference encoded in a
# DIFFERENT SHAPE than agent/vgc/typechart.py: per ATTACKING type, the lists of
# defenders it is super-effective / not-very-effective / no-effect INTO.
# A copy error in the source cannot be mirrored here, and transposition
# (confusing attacker with defender) is caught by the directionality tests
# below. Source: canonical Pokemon type chart, Generation 6 onward.

_GEN6_TYPES = (
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING", "POISON",
    "GROUND", "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST", "DRAGON", "DARK",
    "STEEL", "FAIRY",
)

# attacker -> (super_effective_into, not_very_effective_into, no_effect_into)
_GEN6_REF = {
    "NORMAL":   ([], ["ROCK", "STEEL"], ["GHOST"]),
    "FIRE":     (["GRASS", "ICE", "BUG", "STEEL"], ["FIRE", "WATER", "ROCK", "DRAGON"], []),
    "WATER":    (["FIRE", "GROUND", "ROCK"], ["WATER", "GRASS", "DRAGON"], []),
    "ELECTRIC": (["WATER", "FLYING"], ["ELECTRIC", "GRASS", "DRAGON"], ["GROUND"]),
    "GRASS":    (["WATER", "GROUND", "ROCK"],
                 ["FIRE", "GRASS", "POISON", "FLYING", "BUG", "DRAGON", "STEEL"], []),
    "ICE":      (["GRASS", "GROUND", "FLYING", "DRAGON"], ["FIRE", "WATER", "ICE", "STEEL"], []),
    "FIGHTING": (["NORMAL", "ICE", "ROCK", "DARK", "STEEL"],
                 ["POISON", "FLYING", "PSYCHIC", "BUG", "FAIRY"], ["GHOST"]),
    "POISON":   (["GRASS", "FAIRY"], ["POISON", "GROUND", "ROCK", "GHOST"], ["STEEL"]),
    "GROUND":   (["FIRE", "ELECTRIC", "POISON", "ROCK", "STEEL"], ["GRASS", "BUG"], ["FLYING"]),
    "FLYING":   (["GRASS", "FIGHTING", "BUG"], ["ELECTRIC", "ROCK", "STEEL"], []),
    "PSYCHIC":  (["FIGHTING", "POISON"], ["PSYCHIC", "STEEL"], ["DARK"]),
    "BUG":      (["GRASS", "PSYCHIC", "DARK"],
                 ["FIRE", "FIGHTING", "POISON", "FLYING", "GHOST", "STEEL", "FAIRY"], []),
    "ROCK":     (["FIRE", "ICE", "FLYING", "BUG"], ["FIGHTING", "GROUND", "STEEL"], []),
    "GHOST":    (["PSYCHIC", "GHOST"], ["DARK"], ["NORMAL"]),
    "DRAGON":   (["DRAGON"], ["STEEL"], ["FAIRY"]),
    "DARK":     (["PSYCHIC", "GHOST"], ["FIGHTING", "DARK", "FAIRY"], []),
    "STEEL":    (["ICE", "ROCK", "FAIRY"], ["FIRE", "WATER", "ELECTRIC", "STEEL"], []),
    "FAIRY":    (["FIGHTING", "DRAGON", "DARK"], ["FIRE", "POISON", "STEEL"], []),
}


def _ref_mult(attacker, defender):
    se, nve, no = _GEN6_REF[attacker]
    if defender in no:
        return 0.0
    if defender in se:
        return 2.0
    if defender in nve:
        return 0.5
    return 1.0


def test_reference_chart_is_internally_consistent():
    """Guard the guard: no defender may appear in two relation lists."""
    for attacker, (se, nve, no) in _GEN6_REF.items():
        combined = se + nve + no
        assert len(combined) == len(set(combined)), f"{attacker} lists overlap"
    assert set(_GEN6_REF) == set(_GEN6_TYPES)


@pytest.mark.parametrize("attacker", _GEN6_TYPES)
@pytest.mark.parametrize("defender", _GEN6_TYPES)
def test_every_ordered_pair_matches_the_canonical_chart(attacker, defender):
    """All 324 (attacker -> defender) single-type matchups, value and direction."""
    assert effectiveness(attacker, [defender]) == pytest.approx(_ref_mult(attacker, defender))


@pytest.mark.parametrize(
    "attacker,defender,atk_into_def,def_into_atk",
    [
        # Pairs where attacker->defender and defender->attacker DIFFER: the
        # whole point of a directional chart. If the table were transposed,
        # these would be swapped.
        ("WATER", "FIRE", 2.0, 0.5),
        ("GRASS", "WATER", 2.0, 0.5),
        ("FIGHTING", "ROCK", 2.0, 0.5),
        ("GROUND", "FLYING", 0.0, 1.0),   # Ground cannot touch Flying; Flying is neutral into Ground
        ("ELECTRIC", "GROUND", 0.0, 2.0), # Electric cannot touch Ground; Ground is SE into Electric
        ("FAIRY", "DRAGON", 2.0, 0.0),    # Fairy wrecks Dragon; Dragon cannot touch Fairy
        ("PSYCHIC", "DARK", 0.0, 2.0),    # Psychic cannot touch Dark; Dark is SE into Psychic
        ("STEEL", "FAIRY", 2.0, 0.5),
        ("FIRE", "STEEL", 2.0, 0.5),
    ],
)
def test_effectiveness_is_directional(attacker, defender, atk_into_def, def_into_atk):
    """attacker->defender must not equal defender->attacker for asymmetric pairs."""
    assert effectiveness(attacker, [defender]) == pytest.approx(atk_into_def)
    assert effectiveness(defender, [attacker]) == pytest.approx(def_into_atk)
    assert atk_into_def != def_into_atk  # these pairs are chosen to be asymmetric


def test_ghost_normal_mutual_immunity_is_the_one_symmetric_zero():
    """Ghost<->Normal is the rare pair that is 0x in BOTH directions."""
    assert effectiveness("GHOST", ["NORMAL"]) == 0.0
    assert effectiveness("NORMAL", ["GHOST"]) == 0.0


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
    # Nothing in the observation: the move list supplies the element type.
    assert option["move_type"] == "DARK"


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


def test_species_types_reads_both_types_and_is_empty_when_unlisted():
    table = meta.table()
    assert [t.upper() for t in table.species_types("Incineroar")] == ["FIRE", "DARK"]
    assert table.species_types("Rillaboom") and "GRASS" in [t.upper() for t in table.species_types("Rillaboom")]
    assert table.species_types("NotARealSpecies") == []


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
