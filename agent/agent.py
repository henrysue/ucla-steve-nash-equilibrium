"""Your agent's decision logic: the VGC Draft Doubles strategist.

``create_agent()`` is what the runtime looks for at startup. It is called
once per match, in that match's own process, and whatever it returns plays
every turn of that one match — so each match gets a fresh strategist, and
nothing it remembers (move history, the match plan) leaks into the next game.

    python -m agent --check-tournament
    python -m agent --match          # your test matches
    python -m agent --tournament     # your tournament games

The strategist drafts toward a team plan, writes a match plan at Team
Preview, and treats each battle turn as a simultaneous-move matrix game:
code removes moves that cannot work, the model estimates win probabilities,
and the mixed-strategy Nash equilibrium is solved exactly. See HANDOFF.md
and ``agent/vgc/`` for the pieces.

Configuration (environment, or the starter's gitignored ``.env``):

- ``ANTHROPIC_API_KEY`` — play with Claude (``claude-sonnet-5`` by default;
  ``VGC_MODEL=claude-opus-5`` for stronger payoff estimates, ~2.5x the cost)
- ``OPENAI_API_KEY`` — play with OpenAI instead; used only if no Anthropic key
- ``VGC_MODEL`` — override the model id for whichever provider is chosen
- ``VGC_LOG_DIR`` — write a per-match audit log there (markdown + JSONL)
- ``VGC_META_CSV`` — override the draft reference table's path

With no model key set the strategist still plays a complete, legal match
from its type-aware heuristics — weaker, but it never forfeits.

Werewolf and Red Alert are not played by this agent: it is a Pokémon
specialist, and the starter's ``examples/llm_agent.py`` covers all three
games if you want a generalist instead.
"""

from agent.vgc import create_strategist


def create_agent():
    return create_strategist()
