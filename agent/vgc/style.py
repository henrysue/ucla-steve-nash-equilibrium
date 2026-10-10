"""How this opponent uses Protect: by the book, or around our reads.

Two kinds of opponent, and which one we face changes every Protect read:

- Basic: follows the book. It Protects as soon as Protect is available (more
  so on its first turn out or when threatened) and almost never twice in a
  row, since a second Protect works 1 time in 3.
- Strong: knows we expect that. It holds Protect when the Protect looks
  obvious and attacks instead, saving it for when we do not expect it.

Every Protect opportunity we see resolved (a foe that carries Protect, could
use it at full odds, and whose action we saw: it Protected, or it attacked or
took our hit) updates P(strong) by Bayes' rule. The blended chance it
Protects feeds the code's guess and the model's prompt, with a running log of
what this opponent did. Per opponent agent, what we learn is kept in
``data/opponent_styles.json`` (``VGC_OPPONENT_STYLES``; empty = off), so a
rematch starts from it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .meta import DATA_DIR, species_id

DEFAULT_PATH = DATA_DIR / "opponent_styles.json"
PRIOR_STRONG = 0.35  # most opponents play the book
# Chance of Protect when it is available at full odds.
BASIC_AVAILABLE, BASIC_FIRST_TURN, BASIC_THREATENED, BASIC_CAP = 0.60, 0.10, 0.10, 0.80
STRONG_AVAILABLE = 0.25
# ...and right after a Protect (1 time in 3 it works): both rarely try.
BASIC_CHAINED, STRONG_CHAINED = 0.15, 0.08
LABEL_AT = 0.65  # P(strong) above this reads "strong", below 1 - this "basic"
MAX_NOTES = 12


def _path() -> Path | None:
    value = os.environ.get("VGC_OPPONENT_STYLES")
    if value is None:
        return DEFAULT_PATH
    return Path(value) if value.strip() else None


class OpponentStyle:
    """P(strong) for one opponent, updated from its Protect decisions."""

    def __init__(self, prior: float = PRIOR_STRONG) -> None:
        self.p_strong = prior
        self.prior = prior
        self.available = 0  # opportunities resolved
        self.protected = 0  # ...of which it Protected
        self.chained_tries = 0  # Protects right after a Protect
        self.notes: list[str] = []
        self._pending: dict[str, dict] = {}

    # -- per-opponent memory ----------------------------------------------------------

    @classmethod
    def for_opponent(cls, opponent_id: str | None) -> "OpponentStyle":
        """Start from what we learned about this opponent before, if anything."""
        record = _load().get(str(opponent_id)) if opponent_id else None
        if not isinstance(record, dict) or not isinstance(record.get("p_strong"), (int, float)):
            return cls()
        # Lean toward the old read, but not all the way: they may have changed.
        style = cls(prior=0.5 * PRIOR_STRONG + 0.5 * float(record["p_strong"]))
        style.notes.append(f"met before ({record.get('games', 1)} game(s)): P(strong) was {record['p_strong']:.2f}")
        return style

    def save(self, opponent_id: str | None) -> None:
        path = _path()
        if not path or not opponent_id or not self.available:
            return
        data = _load()
        old = data.get(str(opponent_id)) or {}
        data[str(opponent_id)] = {
            "p_strong": round(self.p_strong, 3),
            "games": int(old.get("games", 0)) + 1,
            "protect_opportunities": int(old.get("protect_opportunities", 0)) + self.available,
            "protected": int(old.get("protected", 0)) + self.protected,
            "label": self.label,
        }
        try:
            path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        except OSError:
            pass  # a convenience: never cost a move over it

    # -- reading ----------------------------------------------------------------------

    @property
    def label(self) -> str:
        if self.p_strong >= LABEL_AT:
            return "strong"
        if self.p_strong <= 1 - LABEL_AT:
            return "basic"
        return "unsure"

    @staticmethod
    def book(first_turn: bool, threatened: bool, chained: bool) -> float:
        """What a by-the-book opponent does: Protect when available."""
        if chained:
            return BASIC_CHAINED
        return min(BASIC_CAP, BASIC_AVAILABLE + BASIC_FIRST_TURN * first_turn + BASIC_THREATENED * threatened)

    def chance(self, first_turn: bool, threatened: bool, chained: bool) -> float:
        """Blended chance this foe Protects now."""
        basic = self.book(first_turn, threatened, chained)
        strong = STRONG_CHAINED if chained else STRONG_AVAILABLE
        return (1 - self.p_strong) * basic + self.p_strong * strong

    def open(self, odds: dict[str, dict], carries: dict[str, bool], chained: dict[str, bool], turn: int) -> dict[str, dict]:
        """This turn's Protect odds, re-read through the opponent's style.

        ``odds`` is protect.odds's by-the-book read (likelihood + why).
        Returns a copy with ``likelihood`` blended by style and ``style_p``
        set, and remembers each foe that could Protect so its choice can be
        scored next turn.
        """
        self._pending = {}
        found = {}
        for foe, row in odds.items():
            row = dict(row)
            why = list(row.get("why") or [])
            if carries.get(foe):
                first = "first turn on the field" in why
                threatened = any("threaten" in w or "below half" in w for w in why)
                chain = bool(chained.get(foe))
                p = self.chance(first, threatened, chain)
                row["likelihood"] = round(p, 2)
                row["style_p"] = round(p, 2)
                why.append(f"opponent reads {self.label} (P strong {self.p_strong:.2f})")
                self._pending[species_id(foe)] = {"first": first, "threatened": threatened, "chained": chain, "turn": turn}
            row["why"] = why
            found[foe] = row
        return found

    def resolve(self, report: dict | None) -> None:
        """Score last turn's opportunities against what each foe did."""
        if not report or not self._pending:
            self._pending = {}
            return
        theirs = report.get("theirs") or {}
        for foe, spot in self._pending.items():
            row = theirs.get(foe)
            if row is None:
                continue
            outcome = _protected(row)
            if outcome is None:
                continue
            basic = self.book(spot["first"], spot["threatened"], spot["chained"])
            strong = STRONG_CHAINED if spot["chained"] else STRONG_AVAILABLE
            like_basic, like_strong = (basic, strong) if outcome else (1 - basic, 1 - strong)
            before = self.p_strong
            self.p_strong = like_strong * before / (like_strong * before + like_basic * (1 - before))
            self.available += not spot["chained"]
            self.protected += bool(outcome) and not spot["chained"]
            self.chained_tries += bool(outcome) and spot["chained"]
            setting = "right after a Protect" if spot["chained"] else ", ".join(
                [w for w, on in (("first turn out", spot["first"]), ("threatened", spot["threatened"])) if on]) or "Protect available"
            did = "Protected" if outcome else "did not Protect"
            self.notes.append(f"T{spot['turn']} {foe} ({setting}; the book says {basic:.0%}): {did} -> P(strong) {before:.2f}->{self.p_strong:.2f}")
            self.notes = self.notes[-MAX_NOTES:]
        self._pending = {}

    def summary(self) -> dict:
        """For the model's prompt and the log."""
        rate = f"{self.protected}/{self.available}" if self.available else "no data yet"
        advice = {
            "basic": "It plays the book: expect Protect whenever Protect is available and it is threatened or on its first turn out; never twice in a row. Attack the other foe or set up into the Protect.",
            "strong": "It plays around reads: when Protect looks obvious, expect it to attack instead (and Protect when it does not look obvious). Do not give up damage or tempo just because Protect is 'available'.",
            "unsure": "Not enough evidence yet: weigh the book read against the board.",
        }[self.label]
        return {"label": self.label, "p_strong": round(self.p_strong, 2), "protected_when_available": rate,
                "chained_protect_tries": self.chained_tries, "advice": advice, "notes": self.notes[-6:]}


def _protected(row: dict) -> bool | None:
    """Did it Protect last turn? None when we cannot tell (it left, or no evidence)."""
    if row.get("left_field") or row.get("entered"):
        return None
    # A move the server showed beats our HP-based inference.
    moves = set(row.get("revealed") or []) | set(row.get("used") or [])
    if moves & {"protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "burningbulwark", "silktrap", "obstruct"}:
        return True
    if moves:
        return False  # it used another move
    if row.get("protected"):
        return True
    if (row.get("hp_change") or 0) < -0.07:  # beyond weather or Life Orb chip
        return False  # our hit landed on it
    return None


def _load() -> dict:
    path = _path()
    if not path:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
