"""Opponent modelling and the decision to deviate from equilibrium.

Sections 6.2 and 6.3 of ``vgc-draft-strategist.md``. Observations are
bucketed by *situation type* so they are comparable, and each bucket holds a
Dirichlet posterior over action categories whose prior is the opponent's
equilibrium mix. That is what makes the agent start at equilibrium and move
toward exploitation only as evidence accumulates.

The model lives for one match: the runtime calls ``create_agent()`` in a
fresh process per match, so nothing leaks between games of a set. Within a
match it accumulates across every turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Situation types (6.2). A turn is classified into exactly one so that
# observations of "what they do when threatened" are not diluted by
# observations of "what they do on turn one".
SITUATIONS = (
    "lead",  # turn-one actions
    "threatened",  # one of their actives is facing a likely KO
    "lacks_speed_control",  # they have no Tailwind/Trick Room and we do
    "after_protect",  # they protected last turn
    "even",  # none of the above
)

# Action categories the opponent's joint action is reduced to. Coarse on
# purpose: four observations of "protect" is evidence, four observations of
# an exact move pair never happens inside one match.
CATEGORIES = (
    "protect",  # protection on one or both slots
    "attack",  # straightforward damage
    "double_target",  # both actives onto one of ours
    "switch",  # repositioning
    "speed_control",  # Tailwind, Trick Room, Icy Wind, Thunder Wave...
    "status",  # setup, redirection, status moves
    "tera",  # Terastallized this turn
)

PRIOR_WEIGHT = 2.5  # pseudo-counts of equilibrium prior, i.e. "2 to 3 observations' worth"
MIN_OBSERVATIONS = 4  # 6.3 (1) sample size
PREDICTABILITY = 0.65  # 6.3 (2) modal posterior probability
MIN_GAIN = 0.05  # 6.3 (3) win-probability gain over the game value
RISK_BUDGET = 0.10  # 6.3 (4) worst case allowed below the game value


@dataclass
class OpponentModel:
    """Counts of the opponent's observed action categories per situation."""

    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def observe(self, situation: str, category: str) -> None:
        if situation not in SITUATIONS:
            raise ValueError(f"unknown situation type {situation!r}; expected one of {SITUATIONS}")
        if category not in CATEGORIES:
            raise ValueError(f"unknown action category {category!r}; expected one of {CATEGORIES}")
        bucket = self.counts.setdefault(situation, {})
        bucket[category] = bucket.get(category, 0) + 1

    def observations(self, situation: str) -> int:
        return sum(self.counts.get(situation, {}).values())

    def posterior(self, situation: str, prior: dict[str, float]) -> dict[str, float]:
        """Dirichlet posterior mean over categories, centred on ``prior``.

        ``prior`` is the opponent's equilibrium mix folded onto categories,
        so with no observations the posterior *is* the equilibrium
        prediction.
        """
        observed = self.counts.get(situation, {})
        categories = sorted(set(prior) | set(observed))
        alpha = {
            category: PRIOR_WEIGHT * max(0.0, prior.get(category, 0.0)) + observed.get(category, 0)
            for category in categories
        }
        total = sum(alpha.values())
        if total <= 0:  # pragma: no cover - prior is a probability mix, so this needs an empty one
            return {category: 1.0 / len(categories) for category in categories}
        return {category: value / total for category, value in alpha.items()}


def entropy(distribution: dict[str, float] | list[float]) -> float:
    """Shannon entropy in bits; lower means more predictable."""
    weights = distribution.values() if isinstance(distribution, dict) else distribution
    return -sum(weight * math.log2(weight) for weight in weights if weight > 0)


@dataclass(frozen=True)
class CheeseCheck:
    """Whether the four criteria in 6.3 all hold, and why."""

    exploit: bool
    situation: str
    observations: int
    modal_category: str
    modal_probability: float
    posterior_entropy: float
    equilibrium_entropy: float
    gain: float
    worst_case: float
    reasons: tuple[str, ...]

    @property
    def verdict(self) -> str:
        return "EXPLOIT" if self.exploit else "EQUILIBRIUM"


def check_cheese(
    *,
    situation: str,
    observations: int,
    posterior: dict[str, float],
    equilibrium_prior: dict[str, float],
    game_value: float,
    exploit_value: float,
    exploit_worst_case: float,
    risk_budget: float = RISK_BUDGET,
    min_gain: float = MIN_GAIN,
) -> CheeseCheck:
    """Apply all four cheese criteria. Every one must hold to deviate.

    ``risk_budget`` is widened or narrowed by series context upstream (6.3):
    take more risk when behind or when one read wins outright, less when
    ahead or when the opponent has shown they adapt.
    """
    modal_category, modal_probability = ("", 0.0)
    if posterior:
        modal_category, modal_probability = max(posterior.items(), key=lambda item: item[1])

    posterior_entropy = entropy(posterior)
    equilibrium_entropy = entropy(equilibrium_prior)
    gain = exploit_value - game_value

    reasons: list[str] = []
    if observations < MIN_OBSERVATIONS:
        reasons.append(f"sample too small ({observations} < {MIN_OBSERVATIONS} in '{situation}')")
    # Either criterion in 6.3 (2): a dominant modal action, or a posterior
    # that is simply far more concentrated than equilibrium would be.
    if modal_probability < PREDICTABILITY and posterior_entropy >= equilibrium_entropy:
        reasons.append(
            f"not predictable enough (modal {modal_category or 'n/a'} {modal_probability:.2f} < {PREDICTABILITY:.2f}, "
            f"entropy {posterior_entropy:.2f} not below equilibrium's {equilibrium_entropy:.2f})"
        )
    if gain < min_gain:
        reasons.append(f"gain {gain:+.3f} below the {min_gain:.2f} threshold")
    if exploit_worst_case < game_value - risk_budget:
        reasons.append(
            f"worst case {exploit_worst_case:.3f} risks more than {risk_budget:.2f} below the game value {game_value:.3f}"
        )

    return CheeseCheck(
        exploit=not reasons,
        situation=situation,
        observations=observations,
        modal_category=modal_category,
        modal_probability=modal_probability,
        posterior_entropy=posterior_entropy,
        equilibrium_entropy=equilibrium_entropy,
        gain=gain,
        worst_case=exploit_worst_case,
        reasons=tuple(reasons),
    )


# -- folding between columns and categories -----------------------------------


def fold_to_categories(col_mix: list[float], categories: list[str]) -> dict[str, float]:
    """Sum an equilibrium mix over this turn's columns into category mass.

    The equilibrium is a mix over the concrete opponent lines we enumerated;
    the opponent model is keyed by category. This is the bridge that lets
    ``y*`` serve as the posterior's prior (6.2).
    """
    folded: dict[str, float] = {}
    for weight, category in zip(col_mix, categories):
        folded[category] = folded.get(category, 0.0) + weight
    return folded


def unfold_to_columns(posterior: dict[str, float], categories: list[str]) -> list[float]:
    """Spread category posteriors back over this turn's columns.

    Mass for a category is split evenly between the columns carrying it;
    categories with no column here are dropped and the rest renormalized, so
    a tendency we cannot act on this turn simply does not move the
    prediction.
    """
    share = {category: categories.count(category) for category in set(categories)}
    weights = [posterior.get(category, 0.0) / share[category] for category in categories]
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(categories)] * len(categories)
    return [weight / total for weight in weights]
