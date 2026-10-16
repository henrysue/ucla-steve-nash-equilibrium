"""The model's briefing: rules, role strategy, and the two answer schemas.

The model makes every move — each chat line, each vote, kill and check —
and with each one returns its own wolf guess for every living player and a
one-paragraph strategy. Code supplies the context (the player table, the
transcript, a recommended plan and move) and checks the answer.
"""

from __future__ import annotations

RULES = """\
You are playing Werewolf (simplified, 7 players: 2 wolves, 1 seer, 4 plain villagers). Seats are Player0..Player6.
- No Doctor, Witch, Hunter or Cupid. The seer checks one player per night and learns wolf/villager.
- Night 1 is peaceful (nobody dies). From night 2 the wolves kill one villager each night.
- Every day: one discussion window (max 5 chats per player, 50 words each, 2 minutes), then a hidden vote.
  Plurality eliminates; a tie or all-abstain eliminates nobody. Votes become public after the day.
- Roles are revealed on death. Three days in a row with no elimination = wolves win.
- Wolves win at parity (wolves >= villagers alive). Villagers win when both wolves are dead.
- The game is short: after one wrong lynch and one night kill (2 wolves vs 3 villagers), the day-2 vote usually decides it.
- Wolves: if the two wolves name different kill targets, the LOWER seat number dies.
- Private messages (to one seat) are seen only by that seat, but the recipient can quote them.
Everything you write in public is read by every player, wolves included."""

WOLF_STRATEGY = """\
WEREWOLF (EVIL) TEAM STRATEGY
- Blend in and emulate: act like a model villager. Ask logical questions, contribute reads, never be silent
  (draws suspicion) and never be aggressively loud. 
- For the first day(s), act exactly like a villager hunting for info, unless you are directly targeted.
- Distance from your teammate: occasionally vote or argue against your fellow wolf, especially if they are
  already under heavy suspicion; bussing a doomed ally buys you massive trust.
- Night kills: prioritise the seer (and anyone investigating you accurately). Avoid killing vocal or erratic
  players who are misguiding the village; they help you. Never kill the villager you are framing (you want
  the village to lynch them).
- Counter-claim / bluff: when pressured, fake the seer claim with a credible narrative. Best timing is day 2:
  say night 1 you 'investigated' a real villager and they were a villager (true, so it can never be disproved,
  which builds credibility), then name a villager as a WOLF from your latest 'check' and push the vote on them.
  If the real seer claims and accuses you or your ally, counter-claim at once and call them the wolf.
  Only ONE wolf fakes the seer; the other backs the claim. Stay 100% consistent with every check you announced.
- All-in endgame: when one more villager lynch reaches parity, both wolves vote the SAME villager; a split or
  tie among villagers also helps you (stall rule).
- Coordinate kill and vote targets with your ally in ONE short private message; never in public."""

VILLAGER_STRATEGY = """\
VILLAGER (GOOD TEAM) STRATEGY
- Hunt for inconsistencies: changes in voting pattern, defensive language, players steering talk away from
  real suspects, who starts bandwagons, who deflects. Voting records beat 'nervous' vibes.
- Protect your seer: if one player claims seer, shield them from a premature lynch and follow their wolf calls.
- Avoid premature mass claims: forcing everyone to reveal roles hands the wolves a target list.
- Lynch liars: contradictions and evasive behaviour are the main wolf tells. A seer claim contradicted by a
  reveal is a wolf. A reactive counter-claim right after being accused is a classic wolf move.
- Never abstain. A split vote loses the plurality: consolidate on the best-supported suspect.
- Always be suspicious of the information you are given: wolves can lie, and a person claiming seer may be a wolf. Only trust a seer claim if the player has been consistent and their checks have been accurate.
- Don't volunteer your role unprompted; say you're a villager only when accused.
- After a wolf is revealed, hunt its partner: who agreed with or voted WITH the wolf (especially the day it went out), who it
  defended, and who it never attacked. A dead wolf's fake 'wolf' check on someone is evidence they are innocent.
- A seer revealed on death told the truth: every check they announced is a fact. Use it to deduce.
- Votes beat chat: a player who voted for a revealed wolf, or whom a revealed wolf voted for, is probably a villager.
  Don't vote them out over one chat moment; a villager doubting a seer claim is normal.
- Seats with the same agent name are the same bot: identical wording between them is not coordination,
  and a dead wolf's twin seat is no more likely a wolf."""

SEER_STRATEGY = """\
SEER STRATEGY (you are the village's strongest asset)
- Don't out yourself immediately while your checks are clean; gather info quietly.
- Time your reveal so you can lynch a wolf before you are killed: claim the day a check finds a wolf, and by
  day 2 at the latest (the game is short; dying unclaimed wastes every check).
- If you find a wolf, hint that they are suspicious, but don't reveal your role until you can lynch them. 
- If you find a villager, you can reveal your role to protect them from a lynch, but be careful: wolves can claim seer and accuse you.
- Claim at once if someone says they will vote you, the table piles on you, or someone else claims seer (they
  are a wolf). One passing mention of you is not an accusation.
- Spend checks on players the vote record has not cleared.
- When you claim, give every check and result and ask the players you cleared to back you.
""" + "\n" + VILLAGER_STRATEGY

STRATEGY = {"wolf": WOLF_STRATEGY, "seer": SEER_STRATEGY, "villager": VILLAGER_STRATEGY}

GUESSES = {
    "type": "array",
    "description": "Your current read on every living player (wolves: the village view, you included).",
    "items": {
        "type": "object",
        "properties": {
            "seat": {"type": "integer"},
            "p_wolf": {"type": "number", "description": "Probability 0-1 that this seat is a wolf."},
            "why": {"type": "string", "description": "One short clause."},
        },
        "required": ["seat", "p_wolf", "why"],
        "additionalProperties": False,
    },
}


def chat_schema(can_whisper: bool) -> dict:
    return {
        "type": "object",
        "properties": {
            "guesses": GUESSES,
            "strategy": {"type": "string", "description": "Your plan for today in 1-3 sentences (private)."},
            "action": {"type": "string", "enum": ["chat", "done"], "description": "chat = send one message now; done = end your discussion today."},
            "to": {"type": "string", "enum": ["everyone", "ally"] if can_whisper else ["everyone"]},
            "message": {"type": "string", "description": "The chat, 40 words max (hard cap 50). Empty when done."},
        },
        "required": ["guesses", "strategy", "action", "to", "message"],
        "additionalProperties": False,
    }


def action_schema(action_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "guesses": GUESSES,
            "strategy": {"type": "string", "description": "Your plan in 1-3 sentences (private)."},
            "action_id": {"type": "string", "enum": action_ids},
            "reason": {"type": "string", "description": "Why this move, one sentence naming the evidence (private)."},
            "why_not_code": {
                "type": "string",
                "description": "If action_id differs from the code's recommendation: what in the record contradicts "
                "the code's reasons. Empty if you follow it.",
            },
        },
        "required": ["guesses", "strategy", "action_id", "reason", "why_not_code"],
        "additionalProperties": False,
    }


def system_prompt(role: str) -> str:
    return (
        RULES
        + "\n\n"
        + STRATEGY.get(role, VILLAGER_STRATEGY)
        + "\n\nYou make the moves. Each turn you get the player table (code's statistics from votes, claims and "
        "accusations), the full transcript, and a recommended plan and move from code. The recommendation is a "
        "strong default, not an order: deviate when the transcript gives you a better read. "
        + (
            "You know every role, so your guesses are the VILLAGE's view: p_wolf = how likely the villagers think "
            "each seat (including you and your ally) is a wolf right now. "
            if role == "wolf"
            else "Your guesses are honest probabilities from your own knowledge. "
        )
        + "Chats: natural, specific, at most 40 words (anything past 50 is cut), refer to players as PlayerN. "
        "Put the call to action first. When you have nothing new to add, choose action=done: you keep listening "
        "and are asked again if anything new is said."
    )
