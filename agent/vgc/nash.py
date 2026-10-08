"""Zero-sum matrix games: equilibrium, best response, bounded exploitation.

Sections 5, 6.3 and 6.4 of ``vgc-draft-strategist.md``. The spec's reference
code (Section 10) solves the equilibrium LP with ``scipy.optimize.linprog``;
this module solves the same program with a self-contained simplex tableau so
the agent adds no dependency to the starter's three. The matrices a turn
produces are at most a few rows by a few columns, where solver speed is
irrelevant and an exact rational-ish pivot is as good as scipy.

One simplex solve yields *both* sides' equilibrium mixes: the column
player's from the primal variables, ours from the dual values that the final
objective row carries under the slack columns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

EPS = 1e-9

Matrix = list[list[float]]


@dataclass(frozen=True)
class Equilibrium:
    """Solution of one zero-sum matrix game, from the row player's side."""

    row_mix: list[float]
    col_mix: list[float]
    value: float


def solve_zero_sum(matrix: Matrix) -> Equilibrium:
    """Mixed-strategy equilibrium of a zero-sum game.

    ``matrix[i][j]`` is the row player's payoff — our win probability — when
    we play row ``i`` and the opponent plays column ``j``. ``value`` is the
    game value we secure against any opponent response.
    """
    rows, cols = _shape(matrix)

    # Shift every payoff to >= 1 so the LP's optimum is positive and the
    # standard simplex needs no Phase I. Undone at the end.
    shift = 1.0 - min(min(row) for row in matrix)
    shifted = [[value + shift for value in row] for row in matrix]

    # max 1'w  s.t.  A'w <= 1, w >= 0 — one variable per opponent column, one
    # constraint per our row. Its optimum is 1 / (value + shift); w scaled by
    # that value is the opponent's mix, and the duals give ours.
    weights, duals, total = _simplex(shifted, [1.0] * cols, [1.0] * rows)
    if total <= EPS:  # pragma: no cover - impossible once every payoff is >= 1
        raise ValueError("degenerate payoff matrix: the equilibrium LP has no positive optimum")

    shifted_value = 1.0 / total
    return Equilibrium(
        row_mix=_normalized([value * shifted_value for value in duals], rows),
        col_mix=_normalized([weight * shifted_value for weight in weights], cols),
        value=shifted_value - shift,
    )


def expected_value(matrix: Matrix, row_mix: list[float], col_mix: list[float]) -> float:
    """Our expected payoff when both sides play the given mixes."""
    return sum(
        weight * sum(matrix[i][j] * col_mix[j] for j in range(len(col_mix)))
        for i, weight in enumerate(row_mix)
    )


def worst_case(matrix: Matrix, row_mix: list[float]) -> float:
    """Our payoff if the opponent best-responds to ``row_mix``."""
    return min(
        sum(matrix[i][j] * weight for i, weight in enumerate(row_mix))
        for j in range(len(matrix[0]))
    )


def best_response(matrix: Matrix, col_mix: list[float]) -> list[float]:
    """The pure row that maximizes our payoff against a predicted mix."""
    scores = [sum(matrix[i][j] * col_mix[j] for j in range(len(col_mix))) for i in range(len(matrix))]
    best = max(range(len(scores)), key=lambda i: scores[i])
    return [1.0 if i == best else 0.0 for i in range(len(scores))]


@dataclass(frozen=True)
class Exploitation:
    """A bounded deviation from equilibrium toward a best response (6.4)."""

    epsilon: float
    mix: list[float]
    value_vs_prediction: float
    value_vs_equilibrium: float
    worst_case: float


def safe_exploit(
    matrix: Matrix,
    predicted_col_mix: list[float],
    equilibrium: Equilibrium,
    *,
    risk_budget: float = 0.10,
    steps: int = 20,
) -> Exploitation:
    """Blend toward the best response while bounding the worst case (6.4).

    Scans ``epsilon`` in ``x = (1 - eps) * x* + eps * BR`` and keeps the
    largest expected value against the prediction whose worst case stays
    within ``risk_budget`` of the game value. ``epsilon = 0`` always
    qualifies, so this never returns something riskier than equilibrium.
    """
    response = best_response(matrix, predicted_col_mix)
    floor = equilibrium.value - risk_budget
    best = _exploitation(matrix, 0.0, equilibrium, predicted_col_mix)

    for step in range(1, steps + 1):
        epsilon = step / steps
        candidate = _exploitation(matrix, epsilon, equilibrium, predicted_col_mix, response)
        if candidate.worst_case >= floor - EPS and candidate.value_vs_prediction > best.value_vs_prediction + EPS:
            best = candidate
    return best


def _exploitation(
    matrix: Matrix,
    epsilon: float,
    equilibrium: Equilibrium,
    predicted_col_mix: list[float],
    response: list[float] | None = None,
) -> Exploitation:
    if epsilon == 0.0:
        mix = list(equilibrium.row_mix)
    else:
        assert response is not None
        mix = [(1 - epsilon) * eq + epsilon * br for eq, br in zip(equilibrium.row_mix, response)]
    return Exploitation(
        epsilon=epsilon,
        mix=mix,
        value_vs_prediction=expected_value(matrix, mix, predicted_col_mix),
        value_vs_equilibrium=expected_value(matrix, mix, equilibrium.col_mix),
        worst_case=worst_case(matrix, mix),
    )


def sample(mix: list[float], rng: random.Random) -> int:
    """Draw one row index from a mixed strategy."""
    threshold = rng.random() * sum(mix)
    cumulative = 0.0
    for index, weight in enumerate(mix):
        cumulative += weight
        if cumulative >= threshold:
            return index
    return len(mix) - 1  # pragma: no cover - only reachable on float drift


# -- simplex ------------------------------------------------------------------


def _simplex(
    constraints: Matrix, objective: list[float], limits: list[float]
) -> tuple[list[float], list[float], float]:
    """Solve ``max objective'w s.t. constraints @ w <= limits, w >= 0``.

    Requires ``limits >= 0`` (so the slack basis is feasible) and a bounded
    optimum, both guaranteed by the caller's all-positive shifted payoffs.
    Returns the primal solution, the dual values, and the optimum.

    Bland's rule picks the pivots: it is slower than Dantzig's but cannot
    cycle, and at these sizes the extra iterations cost nothing.
    """
    rows = len(constraints)  # one constraint per our row
    cols = len(constraints[0])  # one variable per opponent column

    # [ A | I | b ] over rows, then the objective row [ -c | 0 | 0 ].
    tableau = [
        list(constraints[i]) + [1.0 if k == i else 0.0 for k in range(rows)] + [limits[i]]
        for i in range(rows)
    ]
    tableau.append([-value for value in objective] + [0.0] * (rows + 1))
    basis = [cols + i for i in range(rows)]

    while True:
        entering = next((j for j in range(cols + rows) if tableau[-1][j] < -EPS), None)
        if entering is None:
            break

        leaving, best_ratio = None, None
        for i in range(rows):
            if tableau[i][entering] <= EPS:
                continue
            ratio = tableau[i][-1] / tableau[i][entering]
            # Bland's rule: on a tie, the row whose basic variable has the
            # smallest index. This is what rules out cycling.
            if best_ratio is None or ratio < best_ratio - EPS or (ratio < best_ratio + EPS and basis[i] < basis[leaving]):
                leaving, best_ratio = i, ratio
        if leaving is None:  # pragma: no cover - all-positive payoffs bound every column
            raise ValueError("unbounded equilibrium LP")

        _pivot(tableau, leaving, entering)
        basis[leaving] = entering

    primal = [0.0] * cols
    for i, variable in enumerate(basis):
        if variable < cols:
            primal[variable] = tableau[i][-1]
    duals = [tableau[-1][cols + i] for i in range(rows)]
    return primal, duals, tableau[-1][-1]


def _pivot(tableau: Matrix, row: int, column: int) -> None:
    divisor = tableau[row][column]
    tableau[row] = [value / divisor for value in tableau[row]]
    for index, other in enumerate(tableau):
        if index == row:
            continue
        factor = other[column]
        if factor:
            tableau[index] = [value - factor * pivot for value, pivot in zip(other, tableau[row])]


# -- validation ---------------------------------------------------------------


def _shape(matrix: Matrix) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("payoff matrix must have at least one row and one column")
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise ValueError("payoff matrix rows must all have the same length")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for row in matrix for value in row):
        raise ValueError("payoff matrix must contain only numbers")
    return len(matrix), cols


def _normalized(mix: list[float], size: int) -> list[float]:
    """Clamp float drift: no negative weights, and the mix sums to 1."""
    clamped = [max(0.0, value) for value in mix]
    total = sum(clamped)
    if total <= EPS:  # pragma: no cover - a solved LP always leaves mass somewhere
        return [1.0 / size] * size
    return [value / total for value in clamped]
