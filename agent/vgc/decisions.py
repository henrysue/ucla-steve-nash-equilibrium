"""The three model decisions, each a prompt plus a strict schema.

- draft pick: choose one card (9 s budget);
- team preview: one planning call that writes the match plan AND picks the
  bring/leads (78 s budget);
- battle turn: a prediction of each opposing Pokemon's action; code then
  plays our best legal answer to it (40 s budget, thinking off).

The model never writes a GameAPI payload: draft and preview answers are
re-checked against the server's options, and battle moves are chosen by code
under the hard rules (rules.py). Schemas are
strict - all properties required, nothing extra - because both providers'
structured-output modes demand it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from altruagent import GameState

from .providers import NO_THINKING

STATE_CHAR_LIMIT = 6_000

# Keys of the state payload the model does not need: options are supplied
# separately and the rest is transport bookkeeping.
_OMITTED = frozenset({
    "legal_actions", "next_actions", "new_messages", "session_id", "runtime_adapter", "legal_action_count",
})

SYSTEM_PROMPT = """\
You are a professional Pokémon VGC doubles player competing in a tournament. Make the moves a \
smart, strategic top player would: play toward a win condition, value positioning and \
preserving key Pokémon over flashy damage, and prefer lines a strong opponent would find \
hard to punish. Both sides draft six, bring four, lead two, and choose actions SIMULTANEOUSLY \
each turn, so reading the opponent is the game: in battle you predict what they will do, and \
code plays the move that beats it.

Use only the options, targets and Pokémon you are given; never invent any. The state and the \
facts on each option (type multipliers, damage estimates, Protect odds, warnings) are \
authoritative; anything about the opponent you have not been shown is an inference. Options \
that cannot work have already been removed. Be brief: one short sentence per text field, a \
few words per label. reasoning_summary is a private note for our own log.\
"""


@dataclass(frozen=True)
class Decision:
    """One request to the model: what it sees, what it must answer with."""

    kind: str
    prompt: dict
    schema: dict
    effort: str

    def messages(self, state: GameState, seat: object, game_type: object) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "game_type": game_type,
                        "your_seat": seat,
                        "phase": state.phase,
                        "decision": self.kind,
                        **self.prompt,
                        "state": _state_json(state),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]

    def with_prompt(self, extra: dict) -> "Decision":
        return Decision(self.kind, {**self.prompt, **extra}, self.schema, self.effort)


# -- draft --------------------------------------------------------------------


def draft_decision(action_ids: list[str], cards: list[dict]) -> Decision:
    return Decision(
        kind="draft_pick",
        prompt={
            "instructions": (
                "Pick one card. Build a team that is strong, balanced and counters this opponent. "
                "Each card's 'fit' scores it against YOUR roster and THEIR picks: plan (the team "
                "plan you are building toward), roles (what the team still needs: attackers, "
                "speed control, one Fake Out), coverage (team weaknesses it covers or stacks), "
                "versus (their Pokémon it hits super-effectively, is walled by, or counters, "
                "including their playstyle) and strength. Cards are listed best fit first. Keep "
                "team_plan (one sentence) consistent from pick to pick. restricted:true marks a "
                "restricted legendary or a Mega: only one is worth bringing. Answer immediately."
            ),
            "available_cards": cards,
            "legal_action_ids": action_ids,
        },
        schema=object_schema({"action_id": {"type": "string", "enum": action_ids}, "team_plan": {"type": "string"}}),
        effort="low",
    )


# -- team preview: plan + lineup in one call ------------------------------------------------


def team_preview_decision(roster: list[str], ours: list[dict], theirs: list[dict], rules: list[str]) -> Decision:
    named = {"type": "array", "items": {"type": "string"}}
    species = {"type": "array", "items": {"type": "string", "enum": roster}}
    return Decision(
        kind="team_preview",
        prompt={
            "instructions": (
                "Both teams of six are known. Write the plan the battle agent will follow all "
                "match, and choose our four and our two leads. our_team has our real moves and "
                "items and each Pokémon's matchup with their team; in their_team, an entry with "
                "moves/item is its exact draft card, one with likely_moves/likely_item is a usage "
                "guess. Give: our and their win conditions (2-3 each, concrete); their "
                "playstyle; their likely sets (flag Fake Out, speed control, redirection, spread "
                "moves); our key moves; bring (4 different species from our roster) and leads (2 "
                "of those 4); then lead_plan, early/mid/end game, preserve, remove_first, avoid. "
                "The lineup must follow lineup_rules."
            ),
            "our_team": ours,
            "their_team": theirs,
            "lineup_rules": rules,
        },
        schema=object_schema({
            "our_win_conditions": named,
            "their_win_conditions": named,
            "their_playstyle": {"type": "string"},
            "their_likely_sets": {
                "type": "array",
                "items": object_schema(
                    {"species": {"type": "string"}, "moves": named, "item": {"type": "string"}}, reasoning=False
                ),
            },
            "our_key_moves": {
                "type": "array",
                "items": object_schema(
                    {"species": {"type": "string"}, "moves": named, "use": {"type": "string"}}, reasoning=False
                ),
            },
            "bring": species,
            "leads": species,
            "lead_plan": {"type": "string"},
            "early_game": {"type": "string"},
            "mid_game": {"type": "string"},
            "end_game": {"type": "string"},
            "preserve": named,
            "remove_first": named,
            "avoid": named,
        }),
        effort="low",
    )


# -- battle turn: predict the opponent --------------------------------------------


def predict_decision(foes: list[str], ours: list[str], facts: dict) -> Decision:
    """Step 2 of move selection: the model predicts each foe's action.

    Our move is then chosen by code as the best legal answer to that
    prediction (lookahead.best_line), so the model's whole job here is to
    read the opponent - the rock in rock-paper-scissors we have to beat.
    """
    from .predict import ACTIONS

    row = object_schema({
        "species": {"type": "string", "enum": foes},
        "action": {"type": "string", "enum": list(ACTIONS)},
        "target": {"type": "string", "enum": [*ours, "none"]},
        "move": {"type": "string"},
        "confidence": {"type": "number"},
        "why": {"type": "string"},
    }, reasoning=False)
    return Decision(
        kind="doubles_turn",
        prompt={
            "instructions": (
                "Predict what EACH opposing active Pokémon will most likely do this turn: protect, "
                "attack (name which of ours it targets and the likely move), fake_out, switch, "
                "speed_control (Tailwind, Trick Room, Icy Wind...), setup, or support. Give a "
                "confidence from 0 to 1 that this is what it does. code_guess is a first read by "
                "simple rules (a Pokémon that carries Protect and did not Protect last turn will "
                "probably Protect; otherwise it attacks whichever of ours it hurts most); agree or "
                "overrule it using the board: who is threatened (our_threats), who threatens us "
                "(their_threats), who protected last turn, first turns and Fake Out, HP, speed, "
                "the game plan and recent turns. opponent_style says how THIS opponent has used "
                "Protect so far: basic Protects whenever it can (expect it), strong holds Protect "
                "when it looks obvious and attacks instead (do not over-predict Protect), unsure "
                "means little evidence; protect_odds already blends this in. Our move is chosen by "
                "code to beat your prediction, so predict the opponent honestly, not what you wish "
                "they would do."
            ),
            **facts,
        },
        schema=object_schema({
            "state_summary": {"type": "string"},
            "foes": {"type": "array", "items": row},
        }),
        effort=NO_THINKING,
    )


# -- schema helpers -----------------------------------------------------------


def object_schema(properties: dict, *, reasoning: bool = True) -> dict:
    """A strict object schema: every property required, nothing extra."""
    if reasoning:
        properties = {**properties, "reasoning_summary": {"type": "string"}}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _state_json(state: GameState) -> str:
    data = {key: value for key, value in state.raw.items() if key not in _OMITTED}
    observation = data.get("observation")
    if isinstance(observation, dict):
        # Options and their facts are supplied separately; drop the bulky copies.
        data["observation"] = {k: v for k, v in observation.items() if k not in ("available_moves", "available_cards")}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) > STATE_CHAR_LIMIT:
        text = text[:STATE_CHAR_LIMIT] + "...(truncated)"
    return text
