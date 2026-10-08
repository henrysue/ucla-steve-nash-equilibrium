"""A Pokémon VGC Draft Doubles strategist, built to ``vgc-draft-strategist.md``.

Each turn is modelled as a simultaneous-move matrix game: candidate lines
for both sides, payoffs as win probabilities, a mixed-strategy Nash
equilibrium solved exactly, and a bounded deviation into exploitation only
once the opponent has proven predictable enough to punish.

    from agent.vgc import create_strategist

Modules, mapped to the spec's sections:

- ``nash``       — equilibrium, best response, safe exploitation (5, 6.4, 10)
- ``opponent``   — situation types, Dirichlet posteriors, cheese criteria (6.2, 6.3)
- ``actions``    — template parsing, action pruning, payload building (3.2)
- ``typechart``  — type effectiveness behind damage and immunity pruning (3.3)
- ``meta``       — the draft reference table: roles, counters, partners (7)
- ``decisions``  — prompts and strict schemas for the three decision points
- ``report``     — payoff matrix, Mermaid graphs, Section 9 output (4, 9)
- ``strategist`` — the per-turn workflow (8)
- ``providers``  — Claude or OpenAI, on a hard per-request deadline
"""

from __future__ import annotations

from .nash import Equilibrium, solve_zero_sum
from .opponent import OpponentModel, check_cheese
from .strategist import VGCDraftStrategist


def create_strategist(**kwargs) -> VGCDraftStrategist:
    """Build a strategist for one match, reading config from the environment.

    ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` selects the model; with
    neither set the strategist still plays every phase deterministically.
    """
    from dotenv import load_dotenv

    from altruagent.console import print_line

    from .providers import provider_from_env

    load_dotenv()  # no-op if already loaded; never overrides real env vars
    log = kwargs.get("log", print_line)
    return VGCDraftStrategist(provider_from_env(log=log), **kwargs)


__all__ = [
    "Equilibrium",
    "OpponentModel",
    "VGCDraftStrategist",
    "check_cheese",
    "create_strategist",
    "solve_zero_sum",
]
