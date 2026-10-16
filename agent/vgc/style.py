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
# First turn out: how often THIS opponent Protects, from protect.FIRST_TURN
# weighted as this many seen first turns (match e1f70879: 6 of 16, not 70%).
FIRST_TURN_PRIOR_WEIGHT = 4.0
# Switching: (switches + SWITCH_PRIOR_HITS) / (turns on field + SWITCH_PRIOR_TURNS) -> 10% to start.
SWITCH_PRIOR_HITS, SWITCH_PRIOR_TURNS, SWITCH_CAP = 0.5, 5.0, 0.5
LABEL_AT = 0.65  # P(strong) above this reads "strong", below 1 - this "basic"
MAX_NOTES = 12
# A foe in our KO range: does it attack anyway, or escape (Protect if it can,
# else switch)? Ace Trainer Claude escaped every time (match b0e756ca: four
# finishing hits wasted); earlier opponents mostly attacked. Learned per match.
KO_ESCAPE_PRIOR, KO_ESCAPE_WEIGHT = 0.3, 3.0
# Do their two attackers double-target one of ours? Seen from our HP: one took
# a hit and the other none, with both foes acting. Match 37a1ee8a: focused one
# at a time. Learned per match.
FOCUS_PRIOR, FOCUS_WEIGHT, FOCUS_HIT = 0.5, 3.0, 0.05


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
        self.switches = 0  # times one of theirs left the field by choice
        self.ko_spots = 0  # foes we had in KO range whose next move we saw
        self.focus_turns = 0  # turns both foes attacked and we saw where
        self.focused = 0  # ...of which they hit only one of ours
        self.ko_escapes = 0  # ...of which it Protected or switched out
        self._ko_pending: set[str] = set()
        self.first_turns = 0  # first turns out we saw resolved (Protect available)
        self.first_turn_protects = 0  # ...of which it Protected
        self.stays = 0  # foe-turns it stayed in (or left by choice): the switch chances it had

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

    @property
    def first_turn_rate(self) -> float:
        """Chance one of theirs Protects on its first turn out: protect.FIRST_TURN,
        moved toward what this opponent has done this match."""
        from .protect import FIRST_TURN

        weight = FIRST_TURN_PRIOR_WEIGHT
        return (self.first_turn_protects + FIRST_TURN * weight) / (self.first_turns + weight)

    @property
    def switch_rate(self) -> float:
        """Chance one of theirs switches out on a given turn, from what we have seen."""
        return min(SWITCH_CAP, (self.switches + SWITCH_PRIOR_HITS) / (self.stays + SWITCH_PRIOR_TURNS))

    def _count_switches(self, report: dict, turn: int) -> None:
        theirs = report.get("theirs") or {}
        came_in = [foe for foe, row in theirs.items() if row.get("entered")]
        for foe, row in theirs.items():
            if row.get("entered"):
                continue
            self.stays += 1
            if row.get("left_field") == "switched or fainted" and not row.get("revealed"):
                self.switches += 1
                into = f" (in came {', '.join(came_in)})" if came_in else ""
                self.notes.append(f"T{turn} {foe} switched out{into} -> switch rate {self.switch_rate:.0%}")
                self.notes = self.notes[-MAX_NOTES:]

    def add_switching(self, expected: dict) -> None:
        """Give every foe at least this opponent's switch rate (predict.FoeExpectation map)."""
        rate = self.switch_rate
        for e in expected.values():
            if e.switch >= rate:
                continue
            keep = (1 - rate) / max(1e-9, 1 - e.switch)
            e.protect *= keep
            e.attacks = {k: v * keep for k, v in e.attacks.items()}
            e.fake_out = {k: v * keep for k, v in e.fake_out.items()}
            e.switch = rate

    def watch_ko(self, foes) -> None:
        """Foe ids we have a likely KO on this turn: scored at the next resolve."""
        self._ko_pending = {species_id(f) for f in foes or ()}

    @property
    def ko_escape_rate(self) -> float:
        """Chance one of theirs in our KO range Protects or switches out instead of attacking."""
        return (self.ko_escapes + KO_ESCAPE_PRIOR * KO_ESCAPE_WEIGHT) / (self.ko_spots + KO_ESCAPE_WEIGHT)

    def _score_ko_escapes(self, report: dict | None, turn: int) -> None:
        pending, self._ko_pending = self._ko_pending, set()
        theirs = (report or {}).get("theirs") or {}
        for foe in pending:
            row = theirs.get(foe)
            if row is None or row.get("left_field") == "fainted":
                continue  # unseen, or we got it
            if row.get("left_field") == "switched or fainted" and not row.get("revealed"):
                escaped = True
            else:
                outcome = _protected(row)
                if outcome is None:
                    continue
                escaped = bool(outcome)
            self.ko_spots += 1
            self.ko_escapes += escaped
            self.notes.append(f"T{turn} {foe} in our KO range: {'escaped' if escaped else 'stayed and fought'}"
                              f" -> escape rate {self.ko_escape_rate:.0%}")
            self.notes = self.notes[-MAX_NOTES:]

    @property
    def focus_rate(self) -> float:
        """Chance their attackers double-target one of ours on a turn they both attack."""
        return (self.focused + FOCUS_PRIOR * FOCUS_WEIGHT) / (self.focus_turns + FOCUS_WEIGHT)

    def _score_focus(self, report: dict | None, turn: int) -> None:
        theirs = (report or {}).get("theirs") or {}
        ours = (report or {}).get("ours") or {}
        acting = [row for row in theirs.values() if not row.get("entered") and not row.get("left_field")
                  and _protected(row) is not True]
        stayed = {m: row for m, row in ours.items() if not row.get("entered") and row.get("left_field") != "switched"}
        if len(acting) < 2 or len(stayed) < 2:
            return
        hurt = [m for m, row in stayed.items() if row.get("left_field") == "fainted" or (row.get("hp_change") or 0) < -FOCUS_HIT]
        if not hurt:
            return
        self.focus_turns += 1
        self.focused += len(hurt) == 1
        if len(hurt) == 1:
            self.notes.append(f"T{turn} both foes acted and only {hurt[0]} was hit: focus fire -> rate {self.focus_rate:.0%}")
            self.notes = self.notes[-MAX_NOTES:]

    def resolve(self, report: dict | None, turn: int = 0) -> None:
        """Score last turn's opportunities against what each foe did."""
        self._score_ko_escapes(report, turn)
        self._score_focus(report, turn)
        if report:
            self._count_switches(report, turn)
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
            if spot["first"] and not spot["chained"]:
                self.first_turns += 1
                self.first_turn_protects += bool(outcome)
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
                "switch_rate": f"{self.switch_rate:.0%} ({self.switches} switch(es) in {self.stays} foe-turns)",
                "ko_range_escape": f"{self.ko_escape_rate:.0%} ({self.ko_escapes} of {self.ko_spots} in our KO range "
                                   f"Protected or switched)",
                "focus_fire": f"{self.focus_rate:.0%} ({self.focused} of {self.focus_turns} turns they hit only one of ours)",
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
