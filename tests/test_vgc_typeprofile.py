"""The deterministic opponent type-profile (agent/vgc/typeprofile.py).

A wrong profile would point the agent at the opponent's STRENGTH, so these
pin the arithmetic (correct and directional), the sharing thresholds, and
safe degradation on empty/untyped/unknown input.
"""

from agent.vgc import typeprofile


def _profile(team):
    return typeprofile.opponent_profile(team)


def test_shared_weakness_counts_and_direction():
    team = {
        "Incineroar": ["FIRE", "DARK"],
        "Chi-Yu": ["DARK", "FIRE"],
        "Raging Bolt": ["ELECTRIC", "DRAGON"],
    }
    profile = _profile(team)
    ground = next(w for w in profile["shared_weaknesses"] if w["type"] == "Ground")
    assert ground["hits"] == 3 and ground["of"] == 3
    # Direction: ATTACK Ground INTO their types. Electric is resisted by
    # Raging Bolt (0.25x), so it is not a *shared* weakness.
    assert all(w["type"] != "Electric" for w in profile["shared_weaknesses"])


def test_four_times_weakness_shows_in_max_multiplier():
    profile = _profile({"Ferrothorn": ["GRASS", "STEEL"], "Scizor": ["BUG", "STEEL"]})
    fire = next(w for w in profile["shared_weaknesses"] if w["type"] == "Fire")
    assert fire["hits"] == 2
    assert fire["max_multiplier"] == 4.0  # Fire 2x Grass * 2x Steel on Ferrothorn


def test_a_single_members_weakness_is_not_shared():
    # Water hits Incineroar 2x but Rillaboom (Grass) resists Water, so Water
    # is a weakness of only one revealed mon -> not reported as shared.
    profile = _profile({"Incineroar": ["FIRE", "DARK"], "Rillaboom": ["GRASS"]})
    assert all(w["type"] != "Water" for w in profile["shared_weaknesses"])


def test_stab_coverage_is_the_union_of_their_types():
    profile = _profile({"Incineroar": ["FIRE", "DARK"], "Raging Bolt": ["ELECTRIC", "DRAGON"]})
    assert set(profile["their_stab_coverage"]) == {"Fire", "Dark", "Electric", "Dragon"}


def test_weaknesses_are_sorted_most_shared_first():
    team = {
        "Incineroar": ["FIRE", "DARK"], "Chi-Yu": ["DARK", "FIRE"],
        "Raging Bolt": ["ELECTRIC", "DRAGON"], "Flutter Mane": ["GHOST", "FAIRY"],
    }
    hits = [w["hits"] for w in _profile(team)["shared_weaknesses"]]
    assert hits == sorted(hits, reverse=True)


def test_basis_warns_it_is_a_prior():
    profile = _profile({"Incineroar": ["FIRE", "DARK"], "Chi-Yu": ["DARK", "FIRE"]})
    assert "Tera" in profile["basis"] and "prior" in profile["basis"].lower()


def test_empty_team_is_none():
    assert typeprofile.opponent_profile({}) is None


def test_untyped_or_unknown_types_degrade_to_none():
    assert typeprofile.opponent_profile({"X": [], "Y": None}) is None
    assert typeprofile.opponent_profile({"X": ["MYSTERY"]}) is None
