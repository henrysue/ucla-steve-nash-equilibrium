"""The match log: an auditable record of how every decision was made.

Sections 4 and 9 of ``vgc-draft-strategist.md`` for the shape of the
analysis, plus the audit trail around it. The platform gives an agent one
public output channel — ``play_action``'s short ``reasoning_summary`` — which
can hold neither a matrix nor a diagram, so the full reasoning goes to the
operator instead:

- always to the match log (stdout), so it is visible while a game runs;
- and, when ``VGC_LOG_DIR`` is set, to one markdown file per match plus a
  JSONL sibling, so a finished match can be read back decision by decision
  or aggregated across matches.

Each record carries what was decided *and why*: the model's own reasoning,
the opponent prediction, our ranked answers to it, the action
actually sent, and the cost in
attempts and seconds. Fallbacks are recorded too — "the model timed out" is
the single most important thing an audit can tell you about a bad move.

One privacy rule runs through this file: ``model_reasoning`` is PRIVATE
deliberation. It is written to the local log only, and never to the
``reasoning_summary`` the platform shows spectators.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REASONING_LOG_LIMIT = 4_000  # per decision, in the log; prompts are unaffected


@dataclass
class Analysis:
    """One battle turn: the opponent prediction and our best answers to it."""

    turn: str
    state_summary: str
    prediction: dict  # foe -> {action, target, move, confidence, why}
    source: str  # "model" or "code"
    code_guess: dict
    lines: list = field(default_factory=list)  # [(value, description)], best first
    played: int = 0  # index into lines; not 0 when the model vetoed the code's pick
    review: dict | None = None  # the model's final call: {verdict, play_option, why}
    offered: list | None = None  # the options shown at the final call, as the strategist saw them
    board: dict | None = None  # every active's exact HP, boosts, status and item, and our bench HP

    def render(self) -> str:
        parts = [f"### {self.turn} Analysis", "", f"**State summary:** {self.state_summary or '(not supplied)'}", ""]
        if self.board:
            parts += [f"**Board:** `{json.dumps(self.board, separators=(',', ':'), default=str)}`", ""]
        parts.append(f"**Predicted opponent ({self.source}):**")
        for foe, guess in self.prediction.items():
            target = f" -> {guess['target']}" if guess.get("target") else ""
            move = f" ({guess['move']})" if guess.get("move") else ""
            parts.append(f"- {foe}: {guess.get('action')}{target}{move}, confidence {guess.get('confidence')}: {guess.get('why', '')}")
        if self.source == "model" and self.code_guess:
            parts.append("- code guess: " + "; ".join(f"{foe} {g.get('action')}" for foe, g in self.code_guess.items()))
        parts += ["", "**Our best answers** (value on the damage scale):"]
        for index, (value, description) in enumerate(self.lines):
            mark = " **<- played**" if index == self.played else " (code's pick, vetoed)" if index == 0 else ""
            parts.append(f"- {value:.1f} {description}{mark}")
        if self.offered:
            parts += ["", "**Shown to the strategist:**"]
            for option in self.offered:
                breaks = option.get("breaks_principles")
                note = f" [breaks: {'; '.join(breaks)}]" if isinstance(breaks, list) and breaks else ""
                parts.append(f"- option {option.get('option')}: {option.get('line')} ({option.get('value')}){note}")
        if self.review:
            withdrawn = self.review.get("withdrawn")
            call = (f"veto option {withdrawn}, withdrawn at lock-in: played option 1" if withdrawn
                    else f"{self.review.get('verdict')} option {self.review.get('play_option')}")
            parts += ["", f"**Strategist's call:** {call}: {self.review.get('why', '')}"]
            lock = self.review.get("lock_in")
            if isinstance(lock, dict):
                parts.append(f"**Lock-in:** {'locked in' if lock.get('locked') else 'withdrawn'}: {lock.get('why', '')}")
        return "\n".join(parts)

    def as_dict(self) -> dict:
        return {
            "state_summary": self.state_summary,
            "prediction": self.prediction,
            "prediction_source": self.source,
            "code_guess": self.code_guess,
            "lines": [{"value": value, "line": description} for value, description in self.lines],
            "played": self.played,
            "review": self.review,
            "offered": self.offered,
            "board": self.board,
        }


@dataclass
class Decision:
    """One audited decision: the analysis, the cost, and what was sent."""

    turn: str
    phase: str
    kind: str  # draft_pick / select_lineup / doubles_turn
    payload: Any
    public_summary: str
    model: str | None = None
    model_reasoning: str | None = None  # PRIVATE: local log only
    attempts: int = 0
    seconds: float = 0.0
    fallback_reason: str | None = None
    analysis: Analysis | None = None
    move_state: dict | None = None  # the move tracker's view (doubles turns only)
    last_turn: dict | None = None  # observed.last_turn: what happened since the previous decision
    board: list | None = None  # draft picks: the cards on offer (JSONL only), for the card-pool CSV

    def render(self) -> str:
        cost = f"attempts {self.attempts}, {self.seconds:.1f}s" if self.attempts else "no model call"
        head = [
            f"## {self.turn} - {self.kind}",
            "",
            f"- Phase: `{self.phase}` | Model: `{self.model or 'none'}` | {cost}",
        ]
        if self.fallback_reason:
            head.append(f"- **FALLBACK**: {self.fallback_reason}")
        head += [
            f"- Sent: `{_compact_json(self.payload)}`",
            f"- Note (private; the platform was sent a neutral line): {self.public_summary or '(none)'}",
        ]
        if self.move_state:
            head.append(f"- Move rules: `{_compact_json(self.move_state)}`")
        if self.last_turn:
            reads = self.last_turn.get("reads") or {}
            if reads:
                head.append("- Last turn's reads: " + "; ".join(
                    f"{foe} {r['predicted']}: **{r['result']}**" for foe, r in reads.items()))
            seen = {k: v for k, v in self.last_turn.items() if k != "reads"}
            head.append(f"- What happened: `{_compact_json(seen)}`")
        parts = ["\n".join(head)]
        if self.model_reasoning:
            parts += [
                "",
                "<details><summary>Model reasoning (private - local log only)</summary>",
                "",
                self.model_reasoning[:REASONING_LOG_LIMIT],
                "",
                "</details>",
            ]
        if self.analysis is not None:
            parts += ["", self.analysis.render()]
        return "\n".join(parts)

    def as_dict(self) -> dict:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "turn": self.turn,
            "phase": self.phase,
            "kind": self.kind,
            "payload": _jsonable(self.payload),
            "public_summary": self.public_summary,
            "model": self.model,
            "model_reasoning": (self.model_reasoning or None) and self.model_reasoning[:REASONING_LOG_LIMIT],
            "attempts": self.attempts,
            "seconds": round(self.seconds, 2),
            "fallback_reason": self.fallback_reason,
            "analysis": None if self.analysis is None else self.analysis.as_dict(),
            "move_state": self.move_state,
            "last_turn": self.last_turn,
            "board": self.board,
        }


class MatchLog:
    """Writes each decision to the console and, optionally, to disk.

    The files are named on the first decision, once the match's identity is
    known. Both are append-only: a crash mid-match still leaves everything
    decided up to that point.
    """

    def __init__(self, log, directory: str | None = None) -> None:
        self._log = log
        raw = directory if directory is not None else os.environ.get("VGC_LOG_DIR")
        self._directory = Path(raw).expanduser() if raw else None
        self._stem: Path | None = None
        self._disabled = False
        self.decisions = 0
        # A compact, append-only record of each decision, kept so the match
        # can be summarized at the end without re-reading the JSONL. One small
        # dict per turn; this is cheap and never touched on the hot path
        # beyond an append.
        self._trail: list[dict] = []
        self._identity: dict = {}
        self._summarized = False

    def name_match(self, session_id: object, seat: object, game_type: object) -> None:
        """Fix this match's file names. Safe to call every turn; acts once."""
        self._identity = {
            "session_id": str(session_id or ""),
            "seat": seat,
            "game_type": str(game_type or ""),
        }
        if self._directory is None or self._stem is not None:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        slug = _slug(f"{stamp}-{game_type or 'match'}-{session_id or 'session'}-seat{seat}")
        self._stem = self._directory / slug

    def write(self, decision: Decision) -> None:
        self.decisions += 1
        self._trail.append(_trail_entry(decision))
        rendered = decision.render()
        for line in rendered.splitlines():
            self._log(f"[vgc] {line}")
        if self._stem is None or self._disabled:
            return
        try:
            self._stem.parent.mkdir(parents=True, exist_ok=True)
            with self._stem.with_suffix(".md").open("a", encoding="utf-8") as handle:
                handle.write(rendered + "\n\n")
            with self._stem.with_suffix(".jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(decision.as_dict(), ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            # Say it once: a full disk should not spam every remaining turn.
            self._disabled = True
            self._log(f"[vgc] match log disabled, could not write {self._stem}: {exc}")

    def summarize(self, *, outcome: str | None = None, opponent: list | None = None,
                  archetype: str | None = None, note: str | None = None, ending: dict | None = None,
                  style: dict | None = None, opponent_names: list | None = None) -> dict:
        """Emit a compact end-of-match summary for offline, no-time-pressure review.

        The per-turn log already holds every move and its reasoning; this adds
        the skimmable layer on top — what we faced, how each turn resolved
        (MODEL / FALLBACK), how many fallbacks we were forced
        into, and the outcome when it is known — so a human can spot mistakes
        and decide which archetype/counterplay to PRELOAD into the playbook.

        Safe to call more than once (acts once) and when logging is disabled
        (it still returns the summary dict and logs a one-line digest).
        ``outcome`` is only recorded when the caller actually knows it; it is
        never guessed.
        """
        from .observed import tally

        fallbacks = sum(1 for e in self._trail if e["result"] == "FALLBACK")
        summary = {
            "match": self._identity,
            "outcome": outcome,  # "win" / "loss" / None (unknown)
            "opponent_names": list(opponent_names) if opponent_names else [],  # who we played
            "opponent_species": list(opponent) if opponent else [],
            "opponent_archetype": archetype,
            "decisions": len(self._trail),
            "fallbacks": fallbacks,
            "reads": tally([read for e in self._trail for read in e.get("reads", [])]),
            "ending": ending,  # the server's final_result and the last board: how it ended
            "opponent_style": style,  # basic (book Protects) or strong (plays around reads), and why
            "turns": self._trail,
            "note": note,
        }
        digest = (
            f"match summary: {len(self._trail)} decisions, {fallbacks} fallback(s)"
            + (f", outcome={outcome}" if outcome else "")
            + (f", vs {archetype}" if archetype else "")
        )
        if not self._summarized:
            self._log(f"[vgc] {digest}")
            self._summarized = True
        if self._stem is None or self._disabled:
            return summary
        try:
            self._stem.parent.mkdir(parents=True, exist_ok=True)
            with self._stem.with_suffix(".md").open("a", encoding="utf-8") as handle:
                handle.write(_render_summary(summary) + "\n")
            with self._stem.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, ensure_ascii=False, default=str, indent=2) + "\n")
        except OSError as exc:
            self._disabled = True
            self._log(f"[vgc] match summary not written to {self._stem}: {exc}")
        return summary

    @property
    def path(self) -> Path | None:
        return self._stem.with_suffix(".md") if self._stem else None


def _trail_entry(decision: Decision) -> dict:
    """One compact row for the match summary, drawn from a Decision."""
    if decision.fallback_reason:
        result = "FALLBACK"
    elif decision.analysis is not None:
        result = "MODEL"
    else:
        result = "-"
    line_value = None
    if decision.analysis is not None:
        lines, played = decision.analysis.lines, decision.analysis.played
        line_value = lines[played][0] if 0 <= played < len(lines) else None
        if decision.analysis.review and decision.analysis.review.get("verdict") == "veto":
            result = "VETO"
    return {
        "turn": decision.turn,
        "kind": decision.kind,
        "result": result,
        "line_value": line_value,
        "seconds": round(decision.seconds, 2),
        "summary": decision.public_summary or "",
        "fallback_reason": decision.fallback_reason,
        "reads": list(((decision.last_turn or {}).get("reads") or {}).values()),
    }


def _render_summary(summary: dict) -> str:
    lines = [
        "## Match summary",
        "",
        f"- Outcome: **{summary['outcome'] or 'unknown'}**"
        + (f" | vs archetype: {summary['opponent_archetype']}" if summary.get("opponent_archetype") else ""),
        f"- Opponent: {', '.join(summary.get('opponent_names') or []) or '(name not given)'} - "
        f"{', '.join(summary['opponent_species']) or '(not recorded)'}",
        f"- Decisions: {summary['decisions']} | Fallbacks: {summary['fallbacks']}",
    ]
    if summary.get("note"):
        lines.append(f"- Note: {summary['note']}")
    if summary.get("opponent_style"):
        st = summary["opponent_style"]
        lines.append(f"- Opponent style: **{st.get('label')}** (P strong {st.get('p_strong')}; protected when available "
                     f"{st.get('protected_when_available')})")
        lines += [f"  - {n}" for n in st.get("notes") or []]
    if summary.get("ending"):
        lines.append(f"- How it ended: `{_compact_json(summary['ending'])}`")
    for action, row in (summary.get("reads") or {}).items():
        rate = "n/a" if row["hit_rate"] is None else f"{row['hit_rate']:.0%}"
        lines.append(f"- Reads, predicted {action}: {row['hit']} hit / {row['miss']} miss / {row['unknown']} unknown (hit rate {rate})")
    lines += ["", "| Turn | Kind | Result | Best line value | s | Public summary |",
              "|---|---|---|---|---|---|"]
    for e in summary["turns"]:
        gv = "" if e["line_value"] is None else f"{e['line_value']:.1f}"
        text = (e["summary"] or e.get("fallback_reason") or "").replace("|", "/")
        lines.append(f"| {e['turn']} | {e['kind']} | {e['result']} | {gv} | {e['seconds']:.1f} | {text} |")
    return "\n".join(lines)


def _slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^A-Za-z0-9.-]", "-", text)).strip("-").lower() or "match"


def _jsonable(value: Any) -> Any:
    """A payload as JSON, including a LegalAction (draft picks are objects)."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    action_id = getattr(value, "action_id", None)
    return action_id if isinstance(action_id, str) else str(value)


def _compact_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"), default=str)
