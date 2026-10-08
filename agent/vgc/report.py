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
the candidate lines, the payoff matrix, the equilibrium, the opponent-model
posterior and the cheese verdict, the action actually sent, and the cost in
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

from .nash import Equilibrium, Exploitation
from .opponent import CheeseCheck

MIXED_WEIGHT = 0.10  # a row this heavy counts as genuinely part of the mix
REASONING_LOG_LIMIT = 4_000  # per decision, in the log; prompts are unaffected


@dataclass
class Analysis:
    """One matrix decision's reasoning, ready to render."""

    turn: str
    state_summary: str
    threats: str
    row_labels: list[str]
    col_labels: list[str]
    matrix: list[list[float]]
    equilibrium: Equilibrium
    chosen_row: int
    recommendation: str  # "EQUILIBRIUM" or "EXPLOIT"
    slot_descriptions: list[str] = field(default_factory=list)
    cheese: CheeseCheck | None = None
    exploitation: Exploitation | None = None
    predicted_col_mix: list[float] | None = None
    key_assumption: str = ""
    pruned: str = ""

    def render(self) -> str:
        parts = [
            f"### {self.turn} Analysis",
            "",
            f"**State summary:** {self.state_summary or '(not supplied)'}",
            f"**Threats & win conditions:** {self.threats or '(not supplied)'}",
        ]
        if self.pruned:
            parts.append(f"**Pruned:** {self.pruned}")
        parts += ["", "**Payoff matrix** (our win probability):", "", self._matrix_table()]

        tree = self._mermaid()
        if tree:
            parts += ["", "**Game tree:**", "", tree]

        parts += [
            "",
            "**Nash equilibrium**",
            f"- Our mix: {self._mix(self.equilibrium.row_mix, self.row_labels)}",
            f"- Opponent mix: {self._mix(self.equilibrium.col_mix, self.col_labels)}",
            f"- Game value: {self.equilibrium.value:.3f}",
        ]

        if self.cheese is not None:
            posterior = (
                self._mix(self.predicted_col_mix, self.col_labels)
                if self.predicted_col_mix is not None
                else "(none)"
            )
            met = "Yes" if self.cheese.exploit else f"No - {'; '.join(self.cheese.reasons)}"
            parts += [
                "",
                "**Opponent model check**",
                f"- Situation type: {self.cheese.situation}",
                f"- Observations: {self.cheese.observations}, modal: {self.cheese.modal_category or 'n/a'} "
                f"{self.cheese.modal_probability:.2f} (entropy {self.cheese.posterior_entropy:.2f} vs "
                f"equilibrium {self.cheese.equilibrium_entropy:.2f})",
                f"- Predicted opponent mix: {posterior}",
                f"- Cheese criteria met? {met}",
            ]

        parts += ["", f"**Recommendation: {self.recommendation}**"]
        for index, description in enumerate(self.slot_descriptions):
            parts.append(f"- Slot {index}: {description}")
        parts.append(f"- Playing: {self._label(self.row_labels, self.chosen_row)}")
        if self.exploitation is not None:
            parts.append(
                f"- Expected value vs prediction: {self.exploitation.value_vs_prediction:.3f} | "
                f"vs equilibrium: {self.exploitation.value_vs_equilibrium:.3f} | "
                f"worst case: {self.exploitation.worst_case:.3f} | epsilon: {self.exploitation.epsilon:.2f}"
            )
        if self.key_assumption:
            parts.append(f"- Key assumption that would change this: {self.key_assumption}")
        return "\n".join(parts)

    def as_dict(self) -> dict:
        return {
            "row_labels": self.row_labels,
            "col_labels": self.col_labels,
            "matrix": self.matrix,
            "equilibrium": {
                "our_mix": self.equilibrium.row_mix,
                "opponent_mix": self.equilibrium.col_mix,
                "game_value": self.equilibrium.value,
            },
            "predicted_opponent_mix": self.predicted_col_mix,
            "chosen_row": self.chosen_row,
            "recommendation": self.recommendation,
            "state_summary": self.state_summary,
            "threats": self.threats,
            "pruned": self.pruned,
            "key_assumption": self.key_assumption,
            "cheese": None if self.cheese is None else {
                "exploit": self.cheese.exploit,
                "situation": self.cheese.situation,
                "observations": self.cheese.observations,
                "modal_category": self.cheese.modal_category,
                "modal_probability": self.cheese.modal_probability,
                "posterior_entropy": self.cheese.posterior_entropy,
                "equilibrium_entropy": self.cheese.equilibrium_entropy,
                "reasons": list(self.cheese.reasons),
            },
            "exploitation": None if self.exploitation is None else {
                "epsilon": self.exploitation.epsilon,
                "value_vs_prediction": self.exploitation.value_vs_prediction,
                "value_vs_equilibrium": self.exploitation.value_vs_equilibrium,
                "worst_case": self.exploitation.worst_case,
            },
        }

    # -- pieces ---------------------------------------------------------------

    def _matrix_table(self) -> str:
        header = "| You \\ Opp | " + " | ".join(self._label(self.col_labels, j) for j in range(len(self.col_labels))) + " |"
        divider = "|---" * (len(self.col_labels) + 1) + "|"
        rows = [
            f"| {self._label(self.row_labels, i)}{' **<-**' if i == self.chosen_row else ''} | "
            + " | ".join(f"{value:.2f}" for value in row)
            + " |"
            for i, row in enumerate(self.matrix)
        ]
        return "\n".join([header, divider, *rows])

    def _mermaid(self) -> str:
        """A focused tree: the chosen line against each opponent reply (4.4).

        Only drawn when the decision is actually close — a genuinely mixed
        equilibrium or two rows within a hair of each other. A sprawling
        tree is worse than no tree.
        """
        if not self._is_close():
            return ""
        lines = ["```mermaid", "graph TD", f'    S{{{{"{self.turn}: joint decision<br/>see payoff matrix"}}}}']
        chosen = self._label(self.row_labels, self.chosen_row)
        for j, weight in enumerate(self.equilibrium.col_mix):
            payoff = self.matrix[self.chosen_row][j]
            lines.append(
                f'    S -->|"We: {_escape(chosen)}<br/>Opp: {_escape(self._label(self.col_labels, j))} '
                f'(y* {weight:.0%})"| T{j}(["Win prob {payoff:.2f}"])'
            )
        lines += ["    style S stroke-width:2px", "```"]
        return "\n".join(lines)

    def _is_close(self) -> bool:
        mixed = sum(1 for weight in self.equilibrium.row_mix if weight >= MIXED_WEIGHT)
        if mixed >= 2:
            return True
        guarantees = sorted((min(row) for row in self.matrix), reverse=True)
        return len(guarantees) >= 2 and guarantees[0] - guarantees[1] < 0.05

    @staticmethod
    def _label(labels: list[str], index: int) -> str:
        if 0 <= index < len(labels) and labels[index]:
            return labels[index]
        return f"option {index}"

    def _mix(self, mix: list[float], labels: list[str]) -> str:
        shown = [
            f"{self._label(labels, index)} {weight:.0%}"
            for index, weight in enumerate(mix)
            if weight >= 0.005
        ]
        return ", ".join(shown) or "(degenerate)"


def _escape(text: str) -> str:
    """Keep a label safe inside a Mermaid edge string."""
    return text.replace('"', "'").replace("|", "/").replace("\n", " ")


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
            f"- Public summary: {self.public_summary or '(none)'}",
        ]
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
                  archetype: str | None = None, note: str | None = None) -> dict:
        """Emit a compact end-of-match summary for offline, no-time-pressure review.

        The per-turn log already holds every move and its reasoning; this adds
        the skimmable layer on top — what we faced, how each turn resolved
        (EQUILIBRIUM / EXPLOIT / FALLBACK), how many fallbacks we were forced
        into, and the outcome when it is known — so a human can spot mistakes
        and decide which archetype/counterplay to PRELOAD into the playbook.

        Safe to call more than once (acts once) and when logging is disabled
        (it still returns the summary dict and logs a one-line digest).
        ``outcome`` is only recorded when the caller actually knows it; it is
        never guessed.
        """
        fallbacks = sum(1 for e in self._trail if e["result"] == "FALLBACK")
        exploits = sum(1 for e in self._trail if e["result"] == "EXPLOIT")
        summary = {
            "match": self._identity,
            "outcome": outcome,  # "win" / "loss" / None (unknown)
            "opponent_species": list(opponent) if opponent else [],
            "opponent_archetype": archetype,
            "decisions": len(self._trail),
            "fallbacks": fallbacks,
            "exploits": exploits,
            "turns": self._trail,
            "note": note,
        }
        digest = (
            f"match summary: {len(self._trail)} decisions, {fallbacks} fallback(s), "
            f"{exploits} exploit(s)"
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
        result = decision.analysis.recommendation  # "EQUILIBRIUM" or "EXPLOIT"
    else:
        result = "-"
    game_value = None
    if decision.analysis is not None:
        game_value = round(decision.analysis.equilibrium.value, 3)
    return {
        "turn": decision.turn,
        "kind": decision.kind,
        "result": result,
        "game_value": game_value,
        "seconds": round(decision.seconds, 2),
        "summary": decision.public_summary or "",
        "fallback_reason": decision.fallback_reason,
    }


def _render_summary(summary: dict) -> str:
    lines = [
        "## Match summary",
        "",
        f"- Outcome: **{summary['outcome'] or 'unknown'}**"
        + (f" | vs archetype: {summary['opponent_archetype']}" if summary.get("opponent_archetype") else ""),
        f"- Opponent: {', '.join(summary['opponent_species']) or '(not recorded)'}",
        f"- Decisions: {summary['decisions']} | Fallbacks: {summary['fallbacks']} | Exploits: {summary['exploits']}",
    ]
    if summary.get("note"):
        lines.append(f"- Note: {summary['note']}")
    lines += ["", "| Turn | Kind | Result | Game value | s | Public summary |",
              "|---|---|---|---|---|---|"]
    for e in summary["turns"]:
        gv = "" if e["game_value"] is None else f"{e['game_value']:.3f}"
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
