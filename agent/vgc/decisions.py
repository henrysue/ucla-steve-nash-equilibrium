"""The three model decisions, each a prompt plus a strict schema.

- draft pick: choose one card (9 s budget);
- team preview: one planning call that writes the match plan AND picks the
  bring/leads (78 s budget);
- battle turn, two calls sharing a 40 s budget (thinking off): a read of
  each opposing Pokemon's action; code ranks our legal answers to it; then
  the model plays the code's pick or vetoes it for another ranked line.

The model never writes a GameAPI payload: draft and preview answers are
re-checked against the server's options, and battle moves are ranked by code
under the hard rules (rules.py) - a veto only picks among those. Schemas are
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
few words per label. reasoning_summary is our reasoning for our decisions.\
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
                "guess. our_draft_strategy says why each of ours was drafted: must picks (the "
                "highest win rates in the format) and proven pairs (partners that win together); "
                "build the four and the leads around those pairs unless the matchup says otherwise, "
                "and make the game plan play to them. Give: our and their win conditions (2-3 each, concrete); their "
                "playstyle; their likely sets (flag strong pairs, Fake Out, speed control, redirection, spread "
                "moves); our key moves; bring (strong combos that counter their potential picks) and leads (2 "
                "of those 4); then lead_plan, early/mid/end game, preserve, remove_first, avoid. "
                "Our game plan is simple: strong, self-sufficient attackers with few weaknesses, Fake "
                "Out and Intimidate support, Tailwind or Icy Wind for speed - never Trick Room. "
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
    }, reasoning=True)
    return Decision(
        kind="doubles_turn",
        prompt={
            "instructions": (
                "Predict what EACH opposing active Pokémon will most likely do this turn: protect, "
                "attack (name which of ours it targets and the likely move), fake_out, switch, "
                "speed_control (Tailwind, Trick Room, Icy Wind...), setup, or support. Give a "
                "confidence from 0 to 1 that this is what it does. protect_odds is one simple table: "
                "on a Pokémon's first turn out, how often THIS opponent has Protected then (it starts "
                "at 35%), 0 right after it Protected (nobody tries a 1-in-3 repeat), 35% when we "
                "threaten it super-effectively, 25% otherwise, 0 if it has no Protect. Most turns a "
                "Pokémon attacks: do not predict Protect without a concrete reason. A Pokémon in our "
                "KO range (in_our_ko_range) attacks to trade against some opponents and escapes "
                "against others (Protect if it can, else switch out): ko_range_escape is how often "
                "THIS opponent's did so this match - follow it, not a general rule. Opponents double-"
                "target: their_focus_target names the one of ours their two can KO together, and focus_fire "
                "how often THIS opponent has hit only one of ours on turns both attacked - predict both foes "
                "onto that target accordingly. A foe with Rage Powder or Follow Me whose partner is in our "
                "KO range often redirects to save it: predict action support with move 'Rage Powder'. "
                "Focus Sash only works from "
                "full HP: a Pokémon below 100% has no Sash left. A foe whose status is slp is "
                "asleep: it acts only if it wakes (about 1 turn in 3), so it is unlikely to attack, "
                "Protect or use Fake Out; Spore and other status moves fail on a foe that already has "
                "a status. code_guess is a first read by simple rules (Protect when "
                "protect_odds is 50%+; otherwise it attacks whichever of ours it hurts most); agree or "
                "overrule it using the board: who is threatened (our_threats), who threatens us "
                "(their_threats), who protected last turn, first turns and Fake Out, HP, speed, "
                "the game plan, our_draft_strategy (what our team was built to do), move_order "
                "(who acts first: priority bracket, then Speed; Fake Out (+3) only flinches a target "
                "that has not moved yet and goes before Extreme Speed (+2); a Protect (+4) or Quick "
                "Guard stops it; Sucker Punch fails into a non-attack) and recent "
                "turns. opponent_switch_rate says how "
                "often this opponent switches out: a foe we threaten hard may switch to one of "
                "their_possible_switch_ins that resists or is immune to our attack (Incineroar to a "
                "Ghost against Fighting moves), so predict switch when that fits. Our move is chosen by "
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


# -- battle turn: play or veto the code's move --------------------------------------


def review_decision(options: list[dict], facts: dict) -> Decision:
    """Step 4: the model makes the final call on the code's move.

    ``options`` are the code's ranked legal lines, best first, each already
    checked against the hard rules; option "1" is the code's pick. The model
    plays it or vetoes it for another listed line - it never writes a move.
    """
    numbers = [str(o["option"]) for o in options]
    return Decision(
        kind="doubles_turn_review",
        prompt={
            "instructions": (
                "You make the final call on our move. Decide on PRINCIPLES FIRST, reads second. "
                "In order: (1) Protect the plan: never trade away a Pokémon the game plan preserves or "
                "our speed-control setter for nothing, and do not switch it into a big hit. (2) Do your "
                "job: a support was brought for one thing (Prankster screens before their attacks, "
                "Tailwind, Fake Out that lands, redirection); we never set Trick Room except to end theirs; "
                "early in the game that usually beats a small attack. (3) Take a sure KO and remove "
                "their win condition before chipping. (4) Stay robust: prefer a line that is good "
                "whether they Protect or attack; do not put both attacks into a foe that may Protect "
                "when the other foe is a target. (5) Keep tempo: switching a healthy Pokémon hands them "
                "a free turn, and Draco Meteor / Overheat weaken with every use. Protect only shields "
                "the Pokémon that uses it, never its partner. A switch happens before any attack: the Pokémon "
                "switching in takes every hit aimed at that slot this turn and the one leaving takes none, so "
                "switching a threatened Pokémon out into a resist saves it. A pivot move marked 'no one to "
                "switch in' cannot switch: the user stays and takes the hits - and a spread move (marked 'hits both foes') "
                "still hits a foe that is not protecting when the other one Protects. One of OURS whose status is slp is asleep: "
                "it almost surely does nothing this turn (it wakes about 1 turn in 3) - not its attack, "
                "not its Protect - so never keep it in to make a KO, and switching it out costs no "
                "tempo. Only then (6) lean on "
                "your read (your_read). Options are our legal lines ranked by code (value on a damage "
                "scale, already priced for these principles); breaks_principles names what each "
                "one gets wrong. Option 1 is the code's pick. Verdict 'play' sends option 1; 'veto' "
                "sends the option you name in play_option. Veto when option 1 breaks a principle that "
                "another option keeps, or for a concrete reason the numbers miss. Every option is legal. "
                "On 'play', play_option is \"1\". chosen_line: copy the line of the option you send, "
                "word for word, and check that its moves and targets are what your reasoning says."
            ),
            "options": options,
            **facts,
        },
        schema=object_schema({
            "verdict": {"type": "string", "enum": ["play", "veto"]},
            "play_option": {"type": "string", "enum": numbers},
            "chosen_line": {"type": "string"},
            "why": {"type": "string"},
        }),
        effort=NO_THINKING,
    )


def veto_lock_in_decision(veto: dict, facts: dict) -> Decision:
    """Step 4b: a veto is locked in only after a second, deliberate look.

    ``veto`` holds the code's pick, the vetoed-for line, the value gap, what
    each breaks, the strategist's own reason, and the code's checks of that
    reason against the strategist's own read. Withdrawing plays the code's pick.
    """
    return Decision(
        kind="doubles_turn_veto_lock_in",
        prompt={
            "instructions": (
                "You are about to VETO the code's pick. Before it is sent, lock it in: think it through "
                "and confirm only if you are sure. Check, in order: (1) What does the code's pick do this "
                "turn, and what does your line do differently? (2) Does your reason still hold under YOUR "
                "OWN read? Read every line of checks: each is a fact the code found that may contradict "
                "your reason (a redirection read pulls single-target attacks, a Protect read wastes an attack, "
                "a spread move breaks a Focus Sash before a second hit). (3) The code's pick is worth "
                "value_gap more on the damage scale, already priced for every principle: a veto needs a "
                "principle the code's pick breaks and yours keeps, or a concrete fact the numbers miss. "
                "Name it. If any check contradicts your reason, or you cannot name that fact, set lock_in "
                "false: the code's pick is played. lock_in true sends your line."
            ),
            "veto": veto,
            **facts,
        },
        schema=object_schema({
            "lock_in": {"type": "boolean"},
            "why": {"type": "string"},
        }),
        effort="low",  # this one thinks
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
