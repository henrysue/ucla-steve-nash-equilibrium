"""The game-theory core: equilibrium, exploitation, and the cheese criteria."""

import random

import pytest

from agent.vgc import nash
from agent.vgc.opponent import (
    MIN_OBSERVATIONS,
    OpponentModel,
    check_cheese,
    entropy,
    fold_to_categories,
    unfold_to_columns,
)


def _verify_minimax(matrix):
    """An equilibrium's defining property: both sides' guarantees equal the value."""
    solution = nash.solve_zero_sum(matrix)
    rows, cols = len(matrix), len(matrix[0])
    our_guarantee = min(
        sum(matrix[i][j] * solution.row_mix[i] for i in range(rows)) for j in range(cols)
    )
    their_guarantee = max(
        sum(matrix[i][j] * solution.col_mix[j] for j in range(cols)) for i in range(rows)
    )
    assert our_guarantee == pytest.approx(solution.value, abs=1e-6)
    assert their_guarantee == pytest.approx(solution.value, abs=1e-6)
    assert sum(solution.row_mix) == pytest.approx(1.0)
    assert sum(solution.col_mix) == pytest.approx(1.0)
    return solution


# -- solving ---------------------------------------------------------------------------


def test_rock_paper_scissors_is_uniform_with_value_zero():
    solution = _verify_minimax([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    assert solution.value == pytest.approx(0.0)
    assert solution.row_mix == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert solution.col_mix == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_a_dominated_row_gets_no_weight():
    # Row 1 beats row 0 against either column, so the mix is pure.
    solution = _verify_minimax([[1, 2], [3, 4]])
    assert solution.value == pytest.approx(3.0)
    assert solution.row_mix == pytest.approx([0.0, 1.0])


def test_saddle_point_is_found_as_a_pure_strategy():
    # Row 1 / column 1 is a saddle: the row min and the column max coincide.
    solution = _verify_minimax([[0.9, 0.2], [0.8, 0.6]])
    assert solution.value == pytest.approx(0.6)
    assert solution.row_mix == pytest.approx([0.0, 1.0])


@pytest.mark.parametrize(
    "matrix",
    [
        [[0.5]],
        [[3, -1, 2], [-2, 4, 0]],
        [[3, -2], [-1, 4], [2, 0]],
        [[0.33, 0.61, 0.66], [0.52, 0.58, 0.55], [0.64, 0.49, 0.47]],
        [[0.5, 0.5], [0.5, 0.5]],
    ],
)
def test_minimax_holds_on_fixed_matrices(matrix):
    _verify_minimax(matrix)


def test_minimax_holds_on_random_matrices():
    """Non-square shapes are where an orientation bug hides."""
    rng = random.Random(20260207)
    for _ in range(200):
        rows, cols = rng.randint(1, 6), rng.randint(1, 6)
        _verify_minimax([[round(rng.random(), 3) for _ in range(cols)] for _ in range(rows)])


def test_equilibrium_value_beats_every_pure_row_guarantee():
    matrix = [[0.33, 0.61, 0.66], [0.52, 0.58, 0.55], [0.64, 0.49, 0.47]]
    solution = nash.solve_zero_sum(matrix)
    assert solution.value >= max(min(row) for row in matrix)


@pytest.mark.parametrize("matrix", [[], [[]], [[1, 2], [3]], [[1, "x"]], [[True, 1]]])
def test_malformed_matrices_are_rejected(matrix):
    with pytest.raises(ValueError):
        nash.solve_zero_sum(matrix)


# -- best response and exploitation ----------------------------------------------------


def test_best_response_picks_the_highest_expected_value_row():
    matrix = [[1.0, 0.0], [0.0, 1.0]]
    assert nash.best_response(matrix, [1.0, 0.0]) == [1.0, 0.0]
    assert nash.best_response(matrix, [0.0, 1.0]) == [0.0, 1.0]


def test_safe_exploit_moves_toward_a_predictable_opponent():
    # Matching pennies: equilibrium is 50/50 and worth 0.5, but an opponent
    # locked onto column 0 can be punished by committing to row 0.
    matrix = [[1.0, 0.0], [0.0, 1.0]]
    equilibrium = nash.solve_zero_sum(matrix)
    assert equilibrium.value == pytest.approx(0.5)

    exploit = nash.safe_exploit(matrix, [1.0, 0.0], equilibrium, risk_budget=0.5)
    assert exploit.epsilon > 0
    assert exploit.value_vs_prediction > equilibrium.value


def test_safe_exploit_respects_the_risk_budget():
    """A tight budget must keep the worst case near the game value."""
    matrix = [[1.0, 0.0], [0.0, 1.0]]
    equilibrium = nash.solve_zero_sum(matrix)
    exploit = nash.safe_exploit(matrix, [1.0, 0.0], equilibrium, risk_budget=0.05)
    assert exploit.worst_case >= equilibrium.value - 0.05 - 1e-9
    assert exploit.epsilon <= 0.1


def test_safe_exploit_never_returns_something_worse_than_equilibrium():
    matrix = [[0.4, 0.7], [0.6, 0.3]]
    equilibrium = nash.solve_zero_sum(matrix)
    # Predicting the equilibrium mix itself gives nothing to exploit.
    exploit = nash.safe_exploit(matrix, equilibrium.col_mix, equilibrium)
    assert exploit.value_vs_prediction >= equilibrium.value - 1e-9
    assert exploit.worst_case >= equilibrium.value - 0.10 - 1e-9


def test_sample_respects_the_mix():
    rng = random.Random(7)
    draws = [nash.sample([0.0, 1.0, 0.0], rng) for _ in range(50)]
    assert set(draws) == {1}

    counts = [0, 0]
    for _ in range(4000):
        counts[nash.sample([0.25, 0.75], rng)] += 1
    assert 0.70 < counts[1] / 4000 < 0.80


# -- the opponent model ----------------------------------------------------------------


def test_posterior_starts_at_the_equilibrium_prior():
    model = OpponentModel()
    prior = {"protect": 0.6, "attack": 0.4}
    assert model.posterior("lead", prior) == pytest.approx({"protect": 0.6, "attack": 0.4})


def test_posterior_moves_toward_observed_behaviour():
    model = OpponentModel()
    prior = {"protect": 0.5, "attack": 0.5}
    for _ in range(8):
        model.observe("threatened", "protect")
    posterior = model.posterior("threatened", prior)
    assert posterior["protect"] > 0.7
    assert posterior["protect"] + posterior["attack"] == pytest.approx(1.0)
    assert model.observations("threatened") == 8


def test_observations_are_bucketed_by_situation():
    model = OpponentModel()
    model.observe("lead", "protect")
    assert model.observations("lead") == 1
    assert model.observations("threatened") == 0


@pytest.mark.parametrize(
    "situation,category",
    [("nonsense", "protect"), ("lead", "nonsense")],
)
def test_unknown_situations_and_categories_are_rejected(situation, category):
    with pytest.raises(ValueError):
        OpponentModel().observe(situation, category)


def test_entropy_is_zero_for_a_certainty_and_maximal_for_uniform():
    assert entropy([1.0]) == pytest.approx(0.0)
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == pytest.approx(2.0)


# -- folding between columns and categories --------------------------------------------


def test_fold_sums_column_mass_by_category():
    folded = fold_to_categories([0.2, 0.3, 0.5], ["protect", "protect", "attack"])
    assert folded == pytest.approx({"protect": 0.5, "attack": 0.5})


def test_unfold_splits_category_mass_across_its_columns():
    columns = unfold_to_columns({"protect": 0.6, "attack": 0.4}, ["protect", "protect", "attack"])
    assert columns == pytest.approx([0.3, 0.3, 0.4])


def test_unfold_drops_categories_absent_from_this_turn():
    """A tendency we cannot act on this turn must not move the prediction."""
    columns = unfold_to_columns({"protect": 0.9, "tera": 0.1}, ["protect", "attack"])
    assert sum(columns) == pytest.approx(1.0)
    assert columns[0] > columns[1]


def test_fold_then_unfold_round_trips_a_single_column_per_category():
    categories = ["protect", "attack", "switch"]
    mix = [0.2, 0.5, 0.3]
    assert unfold_to_columns(fold_to_categories(mix, categories), categories) == pytest.approx(mix)


# -- the cheese criteria ---------------------------------------------------------------


def _check(**overrides):
    defaults = dict(
        situation="threatened",
        observations=MIN_OBSERVATIONS,
        posterior={"protect": 0.9, "attack": 0.1},
        equilibrium_prior={"protect": 0.5, "attack": 0.5},
        game_value=0.50,
        exploit_value=0.62,
        exploit_worst_case=0.45,
    )
    return check_cheese(**{**defaults, **overrides})


def test_all_four_criteria_met_permits_exploitation():
    result = _check()
    assert result.exploit is True
    assert result.verdict == "EXPLOIT"
    assert result.reasons == ()


def test_too_few_observations_blocks_exploitation():
    result = _check(observations=MIN_OBSERVATIONS - 1)
    assert result.exploit is False
    assert any("sample too small" in reason for reason in result.reasons)


def test_an_unpredictable_opponent_blocks_exploitation():
    result = _check(
        posterior={"protect": 0.5, "attack": 0.5}, equilibrium_prior={"protect": 0.5, "attack": 0.5}
    )
    assert result.exploit is False
    assert any("not predictable enough" in reason for reason in result.reasons)


def test_a_concentrated_posterior_qualifies_even_below_the_modal_threshold():
    """6.3(2) accepts either a dominant mode or entropy well below equilibrium's."""
    result = _check(
        posterior={"a": 0.6, "b": 0.4},
        equilibrium_prior={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
    )
    assert result.exploit is True


def test_insufficient_gain_blocks_exploitation():
    result = _check(exploit_value=0.52)  # +0.02, under the 0.05 floor
    assert result.exploit is False
    assert any("gain" in reason for reason in result.reasons)


def test_excessive_worst_case_risk_blocks_exploitation():
    result = _check(exploit_worst_case=0.30)  # 0.20 below the game value
    assert result.exploit is False
    assert any("worst case" in reason for reason in result.reasons)


def test_a_widened_risk_budget_can_permit_a_riskier_deviation():
    assert _check(exploit_worst_case=0.38).exploit is False
    assert _check(exploit_worst_case=0.38, risk_budget=0.20).exploit is True
