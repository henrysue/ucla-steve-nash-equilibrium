"""The Werewolf agent: Claude makes every move, code briefs and checks it.

Per decision (each chat while the discussion window is open, each vote,
kill or seer check) the model gets the player table, the transcript, our
private knowledge and the code's plan, and answers with:

- ``guesses`` — P(wolf) and a reason for every other living player,
- ``strategy`` — its plan, private,
- the move — a chat line (public or to our ally), "done", or an action id.

Code keeps the guard rails: the action must be legal, a chat is cut to 50
words, a wolf's public line may not out the team, a villager never abstains,
and when the model is slow, absent or wrong the code's own plan is played
(``plan.py``), so the agent never stalls the table.

Configuration (environment or the starter's ``.env``):

- ``ANTHROPIC_API_KEY`` — needed for the model; without it, code plays alone
- ``WEREWOLF_MODEL`` — model id (default ``claude-sonnet-5-5``)
- ``WEREWOLF_EFFORT`` — thinking effort (default ``medium``)
- ``WEREWOLF_FAKE_CLAIM_DAY`` — the day a wolf fakes the seer (default 2)
- ``WEREWOLF_LOG_DIR`` (or ``VGC_LOG_DIR``) — per-match markdown log with the
  player table and the model's guesses at every decision
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path

from altruagent import TERMINATE_MESSAGING, WAIT, SendMessage

from .ledger import ABSTAIN, SEER, WOLF, Ledger
from .plan import Memory, choose_investigation, choose_kill, choose_vote, day_plan, kill_briefing
from .prompts import action_schema, chat_schema, system_prompt

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_WORDS = 50
MAX_CHATS = 5
# Below this many seconds left in the window there is no time for a model
# call: send a staged line verbatim or end the discussion.
MIN_SECONDS_FOR_MODEL = 25
CHAT_TIMEOUT = 45.0
ACTION_TIMEOUT = 60.0  # the inactivity timeout is 120 s
# With nothing to say we keep listening (WAIT) rather than end the discussion,
# so a late claim can still be answered (438990c2: the seer had terminated
# before the fake claim and could not counter it). Below this many seconds
# left we end it.
LISTEN_UNTIL_SECONDS = 12

# A wolf's public chat must never say what it is.
_WOLF_LEAK = re.compile(
    r"\b(my (fellow )?wolf|fellow wolf|wolf (partner|team|buddy|ally|mate)|we wolves|us wolves|as a wolf|"
    r"i'?m a wolf|i am a wolf|my ally|my partner|my teammate|kill .{0,12}tonight|"
    # A hint of private coordination (b4bdb262: "I was told not to claim early").
    r"i was told|we agreed|as agreed|our plan|told me privately|in private|privately)\b",
    re.IGNORECASE,
)


def _log(message: str) -> None:
    try:
        from altruagent.console import print_line

        print_line(message)
    except Exception:  # noqa: BLE001 - logging must never cost a move
        print(message, flush=True)


def make_provider():
    """Claude via the VGC provider (structured outputs, hard timeout), or None."""
    try:
        from agent.vgc.providers import AnthropicProvider, anthropic_credentials
    except Exception as exc:  # noqa: BLE001
        _log(f"[werewolf] model unavailable, playing code only: {exc}")
        return None
    if not anthropic_credentials():
        _log("[werewolf] no ANTHROPIC_API_KEY: playing the code plan only.")
        return None
    try:
        return AnthropicProvider(model=os.environ.get("WEREWOLF_MODEL") or DEFAULT_MODEL)
    except Exception as exc:  # noqa: BLE001
        _log(f"[werewolf] Claude is configured but unusable: {exc}")
        return None


def _trim(text: str) -> str:
    """At most 50 words, cut at a sentence end when one is close enough, so a
    long line loses its tail sentence instead of stopping mid-word."""
    words = " ".join(str(text or "").split()).split(" ")
    if len(words) <= MAX_WORDS:
        return " ".join(words)[:1000].strip()
    cut = " ".join(words[:MAX_WORDS])
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if end > 0 and len(cut[: end + 1].split()) >= 15:
        cut = cut[: end + 1]
    return cut[:1000].strip()


class WerewolfAgent:
    """One seat for one match."""

    def __init__(self, provider=None, *, use_model: bool = True, log_dir: str | None = None) -> None:
        self.provider = provider if provider is not None or not use_model else make_provider()
        self.effort = os.environ.get("WEREWOLF_EFFORT") or "medium"
        self.ledger = Ledger()
        self.memory = Memory()
        self.guesses: dict[int, tuple[float, str]] = {}
        self.strategy = ""
        self.chats: dict[int, int] = {}  # day -> chats sent (the quota counts private ones too)
        self.public_chats: dict[int, int] = {}  # day -> public chats sent
        self.last_seen_seq: dict[int, int] = {}  # day -> highest seq seen when we last chose
        self._log_dir = log_dir if log_dir is not None else (os.environ.get("WEREWOLF_LOG_DIR") or os.environ.get("VGC_LOG_DIR"))
        self._log_path: Path | None = None

    # -- the runtime's hooks --------------------------------------------------

    def choose_message(self, state, context):
        ledger = self._update(state, context)
        day = ledger.day
        if ledger.me in (state.terminated_messaging or []):
            return TERMINATE_MESSAGING
        sent = self.chats.get(day, 0)
        if sent >= MAX_CHATS:
            return TERMINATE_MESSAGING
        seconds_left = state.raw.get("messaging_seconds_left")
        listening = seconds_left is not None and seconds_left > LISTEN_UNTIL_SECONDS
        idle = WAIT if listening else TERMINATE_MESSAGING
        plan = day_plan(ledger, self.memory)
        done_keys = self.memory.sent.setdefault(day, [])
        pending = [t for t in plan.lines if t.key not in done_keys]

        # The private plan to our ally goes first, verbatim: fast and exact.
        whisper = next((t for t in pending if t.recipients), None)
        if whisper is not None:
            return self._send(day, whisper.fallback, whisper.recipients, key=whisper.key, plan=plan)

        newest = max((m.index for m in state.new_messages if m.sender != ledger.me), default=-1)
        news = newest > self.last_seen_seq.get(day, -1)
        self.last_seen_seq[day] = max(newest, self.last_seen_seq.get(day, -1))
        staged = [t for t in pending if not t.recipients]
        if self.public_chats.get(day, 0) and not news and not staged:
            return idle  # nothing new to answer, nothing staged: keep listening
        if sent >= plan.budget + 1 and not ledger.accusers_of(ledger.me) and not staged:
            return idle

        if self.provider is not None and (seconds_left is None or seconds_left >= MIN_SECONDS_FOR_MODEL):
            timeout = CHAT_TIMEOUT if seconds_left is None else min(CHAT_TIMEOUT, seconds_left - 10)
            answer = self._ask_chat(plan, pending, sent, seconds_left, timeout)
            if answer is not None:
                if answer["action"] == "done" or not answer["message"].strip():
                    return idle  # done talking, but still listening for anything new
                to_ally = answer["to"] == "ally" and self._living_ally() is not None
                text = _trim(answer["message"])
                if not to_ally and ledger.role == WOLF and _WOLF_LEAK.search(text):
                    _log(f"[werewolf] blocked a line that outs us: {text!r}")
                else:
                    return self._send(day, text, [self._living_ally()] if to_ally else None, plan=plan)

        # Code fallback: the next staged line, then one generic line a day.
        line = next((t for t in pending if not t.recipients), None)
        if line is not None:
            return self._send(day, line.fallback, None, key=line.key, plan=plan)
        if not self.public_chats.get(day, 0):
            return self._send(day, self._generic_line(), None, plan=plan)
        return idle

    def choose_action(self, state, context):
        ledger = self._update(state, context)
        legal = list(state.legal_actions)
        if not legal:
            return None
        ids = [a.action_id for a in legal]
        label = (legal[0].label or "").lower()
        kind = "kill" if label.startswith("kill") else "investigate" if label.startswith("investigate") else "vote"
        recommended = self._recommend(kind, ids)
        choice = recommended
        reason = "code plan"
        if self.provider is not None:
            answer = self._ask_action(kind, legal, recommended)
            if answer is not None:
                picked = str(answer["action_id"])
                if picked == ABSTAIN and kind == "vote" and ledger.role != WOLF and len(ids) > 1:
                    reason = "model abstained; villagers never abstain, code plan"
                elif picked in ids:
                    choice, reason = picked, answer.get("reason", "")
                    if picked != recommended and answer.get("why_not_code"):
                        reason += f" | overrode code's {recommended}: {answer['why_not_code']}"
        self._record(f"{kind.upper()} -> {'abstain' if choice == ABSTAIN else 'Player' + choice} ({reason})")
        return next(a for a in legal if a.action_id == choice)

    def on_match_end(self, final_state, context) -> None:
        self._update(final_state, context)
        result = getattr(final_state, "final_result", None) or {}
        self._record(f"GAME OVER: winner={self.ledger.winner} our_return={result.get('your_return')}", table=True)

    # -- internals ------------------------------------------------------------

    def _update(self, state, context) -> Ledger:
        if self.ledger.me is None and getattr(context, "seat_position", None) is not None:
            self.ledger.me = int(context.seat_position)
        self.ledger.update(state)
        if self._log_path is None and self._log_dir:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            sid = str(getattr(context, "session_id", "") or "match")[:8]
            self._log_path = Path(self._log_dir) / f"{stamp}-werewolf-{sid}-seat{self.ledger.me}.md"
        return self.ledger

    def _living_ally(self) -> int | None:
        if self.ledger.role != WOLF:
            return None
        return next((a for a in sorted(self.ledger.allies) if self.ledger.players[a].alive), None)

    def _recommend(self, kind: str, ids: list[str]) -> str:
        if kind == "kill":
            pick = choose_kill(self.ledger, self.memory, ids, self._village_view())
        elif kind == "investigate":
            pick = choose_investigation(self.ledger, ids)
        else:
            pick = choose_vote(self.ledger, self.memory)
        pick = str(pick) if pick is not None else None
        if pick in ids:
            return pick
        return next((i for i in ids if i != ABSTAIN), ids[0])

    def _village_view(self) -> dict[int, float] | None:
        """A wolf's model guesses are the village's view of each seat."""
        if self.ledger.role != WOLF or not self.guesses:
            return None
        return {s: p for s, (p, _why) in self.guesses.items()}

    def _send(self, day: int, text: str, recipients, *, key: str | None = None, plan=None):
        text = _trim(text)
        self.chats[day] = self.chats.get(day, 0) + 1
        if not recipients:
            self.public_chats[day] = self.public_chats.get(day, 0) + 1
        done = self.memory.sent.setdefault(day, [])
        if key:
            done.append(key)
        if plan is not None:  # a model line that covers a staged one counts as it
            for talk in plan.lines:
                if talk.key not in done and talk.must_mention and not talk.recipients and not recipients and all(
                    token.lower() in text.lower() for token in talk.must_mention
                ):
                    done.append(talk.key)
        where = f"to Player{recipients[0]}" if recipients else "public"
        self._record(f"CHAT ({where}): {text}", table=False)
        return SendMessage(text, recipients=recipients)

    def _generic_line(self) -> str:
        ledger = self.ledger
        top = choose_vote(ledger, self.memory)
        if ledger.day == 1:
            return (
                f"No hard info yet, so I'm reading behaviour. Please don't mass-claim roles. "
                f"Player{top} feels off to me so far; what does everyone else think?"
            )
        return f"Look at the vote record before we decide. Right now Player{top} fits worst for me. Let's not split the vote."

    # -- the model ------------------------------------------------------------

    def _briefing(self, plan) -> str:
        ledger = self.ledger
        lines = [f"You are Player{ledger.me}. Your role: {(ledger.role or '?').upper()}."]
        if ledger.role == WOLF:
            allies = ", ".join(f"Player{a}{'' if ledger.players[a].alive else ' (dead)'}" for a in sorted(ledger.allies))
            lines.append(f"Your fellow wolf: {allies or 'none'}.")
            if self.memory.ally_kill is not None or self.memory.ally_vote is not None:
                lines.append(f"Your ally privately proposed: kill Player{self.memory.ally_kill}, vote Player{self.memory.ally_vote}.")
            if self.memory.fake_checks:
                story = "; ".join(f"night {n}: Player{s} = {r}" for n, (s, r) in sorted(self.memory.fake_checks.items()))
                state = "ALREADY CLAIMED - stay consistent" if self.memory.fake_claim_day else "drafted, not yet said"
                lines.append(f"Your fake seer story ({state}): {story}.")
        if ledger.role == SEER:
            checks = ", ".join(f"Player{s} is {r}" for s, r in sorted(ledger.investigations.items()))
            lines.append(f"Your true checks: {checks or 'none yet'}.")
        alive = ledger.living()
        lines.append(f"Day {ledger.day}, phase {ledger.sub_phase}. Alive ({len(alive)}): {', '.join(f'Player{s}' for s in alive)}. Wolves left: {ledger.wolves_left()}.")
        if plan is not None:
            lines.append(f"\nCODE PLAN - posture '{plan.posture}':")
            lines += [f"- {n}" for n in plan.notes]
        lines.append("\nPLAYER TABLE (code statistics; P(wolf) from votes/claims/accusations):")
        lines.append(ledger.render_table())
        if self.guesses:
            prior = ", ".join(f"Player{s} {p:.0%}" for s, (p, _w) in sorted(self.guesses.items()))
            lines.append(f"\nYour previous guesses: {prior}. Your previous strategy: {self.strategy}")
        transcript = ledger.transcript()
        lines.append("\nTRANSCRIPT (all days):\n" + (transcript or "(nothing said yet)"))
        return "\n".join(lines)

    def _ask(self, prompt: str, schema: dict, timeout: float) -> dict | None:
        if timeout < 5:
            return None
        started = time.monotonic()
        try:
            answer = self.provider.complete_structured(
                [{"role": "system", "content": system_prompt(self.ledger.role or "villager")}, {"role": "user", "content": prompt}],
                "werewolf",
                schema,
                timeout=timeout,
                effort=self.effort,
            )
        except Exception as exc:  # noqa: BLE001 - a failed call falls back to code
            _log(f"[werewolf] model call failed after {time.monotonic() - started:.0f}s, using code plan: {exc}")
            return None
        self.strategy = str(answer.get("strategy") or "")
        guesses = {}
        for g in answer.get("guesses") or []:
            try:
                guesses[int(g["seat"])] = (max(0.0, min(1.0, float(g["p_wolf"]))), str(g.get("why") or ""))
            except (KeyError, TypeError, ValueError):
                continue
        if guesses:
            self.guesses = guesses
        return answer

    def _ask_chat(self, plan, pending, sent: int, seconds_left, timeout: float) -> dict | None:
        staged = "\n".join(f"- {t.goal} Suggested wording: \"{t.fallback}\"" for t in pending if not t.recipients)
        prompt = (
            self._briefing(plan)
            + f"\n\nDISCUSSION WINDOW: you have sent {sent} of {MAX_CHATS} chats today"
            + (f", about {int(seconds_left)}s left" if seconds_left is not None else "")
            + f". Code suggests about {plan.budget} chats today for this posture."
            + (f"\nStaged lines from the plan (send in order unless you have a better move):\n{staged}" if staged else "")
            + "\n\nDecide: send ONE message now (action=chat) or end your discussion (action=done). "
            "Reply to anything new that matters, especially accusations of you or claims."
        )
        answer = self._ask(prompt, chat_schema(self._living_ally() is not None), timeout)
        if answer is not None:
            self._record(f"MODEL ({plan.posture}): {answer.get('strategy', '')}", table=False)
        return answer

    def _ask_action(self, kind: str, legal, recommended: str) -> dict | None:
        options = "\n".join(f"- action_id {a.action_id}: {a.label}" for a in legal)
        what = {
            "vote": "Cast your day vote (hidden until everyone has voted).",
            "kill": "Choose tonight's kill. If you and your ally name different targets, the LOWER seat dies.",
            "investigate": "Choose who to investigate tonight.",
        }[kind]
        ledger = self.ledger
        plan = day_plan(ledger, self.memory) if kind == "vote" else None
        extra: list[str] = []
        if kind == "vote" and recommended != ABSTAIN:
            why = ledger.explain(int(recommended))
            if why:
                extra.append(f"Code's reasons for Player{recommended}: " + "; ".join(why[:8]) + ".")
            said = ledger.stated_vote(ledger.me)
            if said is not None:
                extra.append(
                    f"You said in public today that you would vote Player{said}. Breaking that costs trust: switch only "
                    "for a reason that was argued in today's discussion. A seer claim that came only after the claimant "
                    "was accused, too late for anyone to test it, is a wolf tell, not a reason to switch."
                )
        if kind == "kill":
            extra += kill_briefing(ledger, self.memory, [a.action_id for a in legal], self._village_view())
        prompt = (
            self._briefing(plan)
            + f"\n\n{what}\nLegal actions:\n{options}\nCode recommends action_id {recommended}."
            + ("\n" + "\n".join(f"- {line}" for line in extra) if extra else "")
            + "\nRe-read today's discussion before deciding. If you pick something other than the code's "
            "recommendation, why_not_code must say what in the record contradicts the code's reasons. "
            "Your reason must name the evidence (never a placeholder)."
        )
        return self._ask(prompt, action_schema([a.action_id for a in legal]), ACTION_TIMEOUT)

    # -- the match log --------------------------------------------------------

    def _record(self, event: str, *, table: bool = True) -> None:
        _log(f"[werewolf P{self.ledger.me} D{self.ledger.day}] {event}")
        if self._log_path is None:
            return
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n## Day {self.ledger.day} ({self.ledger.sub_phase}) - {event}\n\n")
                if self.guesses:
                    fh.write("Model guesses: " + "; ".join(
                        f"P{s} {p:.0%} ({why})" for s, (p, why) in sorted(self.guesses.items())) + "\n\n")
                if self.strategy:
                    fh.write(f"Model strategy: {self.strategy}\n\n")
                if table:
                    fh.write(self.ledger.render_table() + "\n")
        except OSError:
            pass


def create_agent() -> WerewolfAgent:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001
        pass
    return WerewolfAgent()
