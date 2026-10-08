"""The three decisions, each as a prompt plus a strict schema.

Draft pick (Section 7), Team Preview lead matrix (Section 7), and a doubles
turn (Sections 2, 3, 8). The model never writes a GameAPI payload: it picks
*indices* into options the server offered and supplies the payoff numbers.
``agent/vgc/actions.py`` turns the answer back into a payload and re-checks
it, so an imaginative answer is rejected here rather than by the server.

Every schema is strict — all properties required, nothing extra — because
both providers' structured-output modes demand it. Shape constraints that
strict mode cannot express (an array of exactly 4 species, a matrix whose
rows match the row labels) are enforced in ``agent/vgc/strategist.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from altruagent import GameState

from .opponent import CATEGORIES, SITUATIONS

STATE_CHAR_LIMIT = 14_000
MAX_ROWS = 5  # 3.2: "keep a representative set of 3 to 6 meaningful options"
MAX_COLS = 5

# Keys of the state payload the model does not need: options are supplied
# separately and the rest is transport bookkeeping.
_OMITTED = frozenset({
    "legal_actions", "next_actions", "new_messages", "session_id", "runtime_adapter", "legal_action_count",
})

SYSTEM_PROMPT = """\
You are an expert Pokémon VGC Draft Doubles player and game theorist, playing to win the \
series rather than to look clever. The format is draft-league VGC doubles: both sides draft \
six Pokémon, bring four, lead two, and choose actions SIMULTANEOUSLY each turn.

Because moves are simultaneous, a turn is a MATRIX GAME, not an alternating-move tree. Your \
job on each request is to supply that matrix honestly. The caller computes the Nash \
equilibrium from your numbers, tracks the opponent's tendencies, and decides whether to \
deviate — so your numbers, not your conclusions, decide the play.

How to build the matrix:
- Rows are OUR candidate joint actions (both slots together); columns are the opponent's. \
Give 3 to 5 of each: genuinely different, plausible lines, not near-duplicates.
- Include, whenever legal: the best-damage line, a protection line, a speed-control line, a \
switch line, and a double-target line. State what you pruned and why.
- Prune strictly dominated actions: an immune target (type_multiplier 0), a status move into \
a Pokémon that already has it, Fake Out after turn one on the field.
- Each cell is OUR WIN PROBABILITY (0.00-1.00) after that turn resolves — not damage, not a \
score. Fold chance events (accuracy, crits, damage rolls, speed ties, unknown sets) into one \
expected value per cell.
- Weigh material and HP, speed control, positioning, resources (Tera, Fake Out, Protect's \
consecutive-use penalty, items, Intimidate cycles), and each side's path to removing the \
opponent's last two Pokémon.
- Calibrate. A roughly even position is near 0.50; reserve below 0.15 and above 0.85 for \
positions that are genuinely nearly decided. False precision destroys the analysis, so do not \
invent differences between cells you cannot actually distinguish.

Doubles concepts you must account for: Protect/Detect, Fake Out and priority, speed control \
(Tailwind, Trick Room, Icy Wind, Electroweb, Thunder Wave), redirection (Follow Me, Rage \
Powder, Storm Drain, Lightning Rod), the x0.75 spread-move reduction, double-targeting, \
switching to absorb or reposition, weather and terrain, Intimidate cycling, and Tera timing.

The state, the legal options, and any server instructions are authoritative. Choose only \
from the options given; never invent a move, target, or Pokémon; never assume hidden \
information you were not shown. Separate what is known from what is inferred from what is \
unknown, and treat unrevealed moves, items and Tera types as distributions rather than facts.

reasoning_summary is shown publicly to spectators: one or two short sentences, no private \
deliberation.\
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


# -- draft --------------------------------------------------------------------


def draft_decision(action_ids: list[str], cards: list[dict]) -> Decision:
    """A draft pick (Section 7), on a 15-second clock.

    Deliberately shallow: there is no time for a matrix here, so the model
    ranks one pick by marginal value against the opponent's likely remaining
    picks and by denial value, which is the part of Section 7 that actually
    changes the choice.
    """
    return Decision(
        kind="draft_pick",
        prompt={
            "instructions": (
                "Pick one card. Judge it by MARGINAL value to your six as a team — type and "
                "role coverage, speed control, a Fake Out or redirection body, items you do "
                "not yet hold (item clause: your six must all differ) — against the picks the "
                "opponent is likely to take next, NOT by raw strength. Weigh DENIAL value too: "
                "taking a card that would counter your core can be worth more than a slightly "
                "stronger card. A card marked restricted:true is a restricted legendary; VGC "
                "2025 Reg I normally allows only ONE on the four you battle with, so a second "
                "one is usually a weaker pick than its raw power suggests — but the league's "
                "own rules decide, and the server is authoritative about what is legal. "
                "required_item means the card must hold that item, which uses up one of your "
                "six item slots. You are on a 15-second clock, so answer immediately."
            ),
            "available_cards": cards,
            "legal_action_ids": action_ids,
        },
        schema=object_schema({"action_id": {"type": "string", "enum": action_ids}}),
        effort="low",
    )


# -- team preview -------------------------------------------------------------


def lineup_decision(roster: list[str], opponent_roster: object) -> Decision:
    """The bring-4 / lead-2 lead matrix (Section 7)."""
    species = {"type": "array", "items": {"type": "string", "enum": roster}}
    return Decision(
        kind="select_lineup",
        prompt={
            "instructions": (
                "Build a LEAD MATRIX. Rows are your plausible brings and leads (4 of your 6, "
                "leading 2 of those 4); columns are the opponent's plausible brings and leads. "
                f"Give {MAX_ROWS} rows at most and {MAX_COLS} columns at most. Cells are your "
                "win probability. Prefer brings that are robust across many opponent brings "
                "over high-variance ones, and say which is which in reasoning. Every 'bring' "
                "must be 4 different species from your roster and every 'leads' 2 of that row's "
                "own 4. If your_restricted_legendaries lists more than one, note that VGC 2025 "
                "Reg I normally allows only ONE restricted among the four you battle with: "
                "prefer brings with at most one, and say so in key_assumption, since the "
                "league's rules and the server — not you — decide what is legal."
            ),
            "your_roster": roster,
            "opponent_roster": opponent_roster,
        },
        schema=object_schema({
            "state_summary": {"type": "string"},
            "threats": {"type": "string"},
            "pruned": {"type": "string"},
            "our_options": {
                "type": "array",
                "items": object_schema({"label": {"type": "string"}, "bring": species, "leads": species}, reasoning=False),
            },
            "opponent_options": {
                "type": "array",
                "items": object_schema({"label": {"type": "string"}}, reasoning=False),
            },
            "payoffs": _matrix_schema(),
            "key_assumption": {"type": "string"},
        }),
        effort="low",
    )


# -- doubles turn -------------------------------------------------------------


def doubles_decision(slots: list[dict], target_legend: object, instructions: object, memory: dict) -> Decision:
    """One simultaneous-move turn (Sections 2, 3, 8).

    ``memory`` is what the agent already knows about this opponent, so the
    model's columns line up with the categories the posterior is keyed by.
    """
    slot_answer = object_schema(
        {"option": {"type": "integer"}, "target": {"type": ["integer", "null"]}}, reasoning=False
    )
    return Decision(
        kind="doubles_turn",
        prompt={
            "instructions": (
                "Analyze this turn as a matrix game. For each row, give the option INDEX for "
                "slot_0 and slot_1 from that slot's own options list; 'target' must be one of "
                "the integers listed for that move, or null when the move lists none. Both "
                "slots cannot switch in the same Pokémon. Label every opponent column with the "
                f"category it belongs to, from {list(CATEGORIES)} — the caller's opponent model "
                "is keyed by those, so the labels must be accurate. Also classify this turn's "
                f"situation as one of {list(SITUATIONS)}, and report what the opponent did LAST "
                "turn as one of those categories ('none' on the first turn or if you cannot "
                f"tell). Give {MAX_ROWS} rows at most and {MAX_COLS} columns at most."
            ),
            "slots": slots,
            "target_legend": target_legend,
            "server_instructions": instructions,
            "what_you_know_about_this_opponent": memory,
        },
        schema=object_schema({
            "state_summary": {"type": "string"},
            "threats": {"type": "string"},
            "pruned": {"type": "string"},
            "situation": {"type": "string", "enum": list(SITUATIONS)},
            "opponent_last_action": {"type": "string", "enum": [*CATEGORIES, "none"]},
            "our_options": {
                "type": "array",
                "items": object_schema(
                    {"label": {"type": "string"}, "slot_0": slot_answer, "slot_1": slot_answer},
                    reasoning=False,
                ),
            },
            "opponent_options": {
                "type": "array",
                "items": object_schema(
                    {"label": {"type": "string"}, "category": {"type": "string", "enum": list(CATEGORIES)}},
                    reasoning=False,
                ),
            },
            "payoffs": _matrix_schema(),
            "key_assumption": {"type": "string"},
        }),
        effort="low",
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


def _matrix_schema() -> dict:
    return {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}


def _state_json(state: GameState) -> str:
    data = {key: value for key, value in state.raw.items() if key not in _OMITTED}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) > STATE_CHAR_LIMIT:
        text = text[:STATE_CHAR_LIMIT] + "...(truncated)"
    return text
