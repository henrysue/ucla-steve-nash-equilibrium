"""A Pokémon VGC Draft Doubles strategist, built to ``vgc-draft-strategist.md``.

Each turn is modelled as a simultaneous-move matrix game: candidate lines
for both sides, payoffs as win probabilities, a mixed-strategy Nash
equilibrium solved exactly, and a bounded deviation into exploitation only
once the opponent has proven predictable enough to punish.

    from agent.vgc import create_strategist

Modules:

- ``strategist`` - the three decisions: draft pick, Team Preview, battle turn (rules -> predict -> answer)
- ``decisions``  - prompts and strict schemas for those three model calls
- ``draft``      - one score per draft card (plan, roles, coverage, matchup)
- ``archetypes`` - team plans (data/team_plans.json) and opponent playstyle
- ``rules``      - the one rules layer: what is withheld, what is a blunder
- ``moverules``  - per-match move history (Fake Out, Protect chains)
- ``movelist``   - move data (data/moves.json) and the blunder checks
- ``lookahead``  - damage estimates, the threat table, and our best answer to the prediction
- ``protect``    - each foe's chance to Protect this turn
- ``abilities``  - target abilities that punish a move
- ``actions``    - template parsing, option descriptions, payload building
- ``predict``    - each foe's most likely action (code's guess, refined by the model)
- ``meta``       - the reference tables (Reg I and Champions Reg M-C)
- ``typechart``  - type effectiveness
- ``report``     - the match log
- ``providers``  - Claude or OpenAI, on a hard per-request deadline
"""

from __future__ import annotations

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
    "VGCDraftStrategist",
    "create_strategist",
]
