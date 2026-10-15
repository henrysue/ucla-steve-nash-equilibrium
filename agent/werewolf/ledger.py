"""The player table: everything said, voted, claimed and revealed, per seat.

One ``Ledger`` per match. It is rebuilt from the public record on every read
(``observation`` is the source of truth for roles, deaths and votes) and
accumulates the discussion transcript itself, because ``new_messages`` is
reset at every phase change and is empty during the vote.

For every seat it keeps:

- what they said (each chat, by day; private messages to us are flagged),
- how they voted each day (``None`` = abstain),
- the role they claimed, and any seer "checks" they announced,
- who they accused and defended in chat,
- their track record: accusations of players later revealed as wolves
  (``correct_calls``) or villagers (``wrong_calls``),

and from that a wolf-suspicion score turned into a probability. The score is
the *public* read: it uses only what every player can see, plus our own
private knowledge layered on top in ``wolf_probability`` (a seer's checks, a
wolf's ally). For a wolf, the public score is the village's heat on each
seat, which is what the wolf plan needs.

Parsing chat is heuristic (regexes over ``PlayerN`` mentions). It does not
need to be perfect: the model reads the raw transcript too, and the scores
only rank players.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

TOTAL_WOLVES = 2
ABSTAIN = "7"

WOLF, SEER, VILLAGER = "wolf", "seer", "villager"

_PLAYER = re.compile(r"\bP(?:layer)?\s?(\d)\b", re.IGNORECASE)
_ROLE = re.compile(r"Your role:\s*(\w+)")
_ALLIES = re.compile(r"Your fellow wolves:\s*(.+)")
_INVESTIGATIONS = re.compile(r"Your investigations:\s*(.+)")
_CHECK = re.compile(r"Player(\d+) is (wolf|villager)")
_HEADER = re.compile(r"Day (\d+) \| phase (\w+) \| alive (\d+)/7")
_LIVING = re.compile(r"Living:\s*(.*)")
_DEAD = re.compile(r"day (\d+): Player(\d+) was (\w+) \(([^)]*)\)")
_VOTE_LINE = re.compile(r"Day (\d+) vote:\s*(.*)")
_VOTE = re.compile(r"Player(\d+) -> (?:Player(\d+)|abstain)")
_GAME_OVER = re.compile(r"Game over\W+(WOLVES|VILLAGERS) win")

# Chat heuristics. A seer claim is first person; a check is "PlayerN is wolf",
# "checked PlayerN: villager" and similar, only counted from a seer claimant.
_SEER_CLAIM = re.compile(
    # First person only: "PlayerN's seer claim is fake" must not read as a claim.
    r"\b(i'?m|i am)\s+(the\s+|a\s+)?(real\s+|true\s+|actual\s+)?seer\b|\bas (the )?seer,?\s+i\b|"
    r"\bseer here\b|\bi\s+(?:will\s+|'ll\s+)?claim(?:ing)?\s+seer\b",
    re.IGNORECASE,
)
_VILLAGER_CLAIM = re.compile(r"\b(i'?m|i am)\s+(just\s+)?(a\s+)?(plain\s+|simple\s+)?villager\b", re.IGNORECASE)
_CHECK_CLAIMS = (
    re.compile(
        # "Player3 is wolf", "Player3: villager", and the bare list "Checks: Player0 villager, Player4 villager"
        r"\bP(?:layer)?\s?(\d)\b\s*(?:(?:is|=|was|:|-|came back(?: as)?|checks? (?:out )?as|reads? as)\s*)?(?:a\s+)?(?:\*+)?"
        r"(wolf|werewolf|villager|clean|town|innocent)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:checked|investigated|looked at|inspected)\s+P(?:layer)?\s?(\d)\b[^.;!?]{0,40}?\b(wolf|werewolf|villager|clean|town|innocent)",
        re.IGNORECASE,
    ),
)
_ACCUSE_WORDS = re.compile(
    r"\b(wolf|wolves|werewolf|sus|suspicious|suspect|lying|liar|lie|vote|voting|lynch|eliminate|scum|evil|fake|"
    r"deflect\w*|dodg\w*|off|shady|bus\w*)\b",
    re.IGNORECASE,
)
_DEFEND_WORDS = re.compile(
    r"\b(trust|clean|villager|innocent|town|confirmed|believe|vouch|protect|cleared|not (a )?wolf|isn'?t (a )?wolf)\b",
    re.IGNORECASE,
)
_NEGATED_ACCUSE = re.compile(
    r"\b(not|isn'?t|don'?t|doesn'?t|no reason to|wouldn'?t)\b[^.;!?]{0,20}\b(wolf|sus|suspicious|vote|lynch)",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[^.;!?]+[.;!?]?")
_STATED_VOTE = re.compile(
    r"\bI(?:'?m| am|'?ll| will)?\s+(?:still\s+|now\s+|be\s+)?vot(?:e|ing)\s+(?:for\s+)?P(?:layer)?\s?(\d)\b",
    re.IGNORECASE,
)
# "alime588 (Player0, Self-hosted)" -> "alime588"; older names end "— Seat N (…)".
_NAME_SUFFIX = re.compile(r"\s*(?:—\s*Seat\s*\d+\s*)?\([^)]*\)\s*$")


def seat_label(seat: int) -> str:
    return f"Player{seat}"


def _canon_result(word: str) -> str:
    return WOLF if word.lower() in ("wolf", "werewolf") else VILLAGER


@dataclass
class Statement:
    day: int
    seq: int
    text: str
    private: bool = False


@dataclass
class PlayerRecord:
    seat: int
    name: str = ""
    alive: bool = True
    revealed_role: str | None = None  # truth, from a death
    death: tuple[int, str] | None = None  # (day, cause)
    claimed_role: str | None = None
    claim_day: int | None = None
    claim_seq: int | None = None
    claimed_checks: dict[int, tuple[str, int]] = field(default_factory=dict)  # seat -> (result, day said)
    statements: list[Statement] = field(default_factory=list)
    votes: dict[int, int | None] = field(default_factory=dict)  # day -> target seat, None = abstain
    accused: dict[int, list[int]] = field(default_factory=dict)  # target -> days
    defended: dict[int, list[int]] = field(default_factory=dict)
    correct_calls: set[int] = field(default_factory=set)  # accused/voted, later revealed wolf
    wrong_calls: set[int] = field(default_factory=set)  # accused/voted, later revealed villager

    def said_on(self, day: int, *, public_only: bool = True) -> list[Statement]:
        return [s for s in self.statements if s.day == day and not (public_only and s.private)]

    def pointed_at(self, target: int) -> bool:
        """Accused in chat or voted for ``target`` at any point."""
        return target in self.accused or target in self.votes.values()


class Ledger:
    """The table for one match, from one seat's point of view."""

    def __init__(self, me: int | None = None) -> None:
        self.me = me
        self.players: dict[int, PlayerRecord] = {seat: PlayerRecord(seat) for seat in range(7)}
        self.role: str | None = None
        self.allies: set[int] = set()
        self.investigations: dict[int, str] = {}  # our own seer checks (truth)
        self.day = 0
        self.sub_phase: str | None = None
        self.vote_results: dict[int, int | None] = {}  # day -> eliminated seat (None = nobody)
        self.winner: str | None = None
        self._seen_seqs: set[int] = set()
        self.private_from_allies: list[Statement] = []

    # -- ingest ---------------------------------------------------------------

    def update(self, state) -> None:
        """Fold in one ``GameState``: observation, seat names, new messages."""
        raw = getattr(state, "raw", None) or {}
        if self.me is None and raw.get("your_position") is not None:
            self.me = int(raw["your_position"])
        for entry in raw.get("players") or []:
            try:
                self.players[int(entry["position"])].name = str(entry.get("name") or "")
            except (KeyError, TypeError, ValueError):
                continue
        self.parse_observation(getattr(state, "observation", "") or "")
        for message in getattr(state, "new_messages", None) or []:
            self.add_message(message)

    def parse_observation(self, text: str) -> None:
        if m := _ROLE.search(text):
            self.role = m.group(1).lower()
        if m := _ALLIES.search(text):
            self.allies = {int(s) for s in re.findall(r"Player(\d+)", m.group(1))}
        if m := _INVESTIGATIONS.search(text):
            self.investigations = {int(s): r for s, r in _CHECK.findall(m.group(1))}
        if m := _HEADER.search(text):
            self.day, self.sub_phase = int(m.group(1)), m.group(2)
        if m := _LIVING.search(text):
            living = {int(s) for s in re.findall(r"Player(\d+)", m.group(1))}
            if living:
                for seat, record in self.players.items():
                    record.alive = seat in living
        for day, seat, role, cause in _DEAD.findall(text):
            record = self.players[int(seat)]
            record.alive = False
            record.revealed_role = role.lower()
            record.death = (int(day), cause)
        for line in text.splitlines():
            if m := _VOTE_LINE.match(line.strip()):
                day = int(m.group(1))
                for voter, target in _VOTE.findall(m.group(2)):
                    self.players[int(voter)].votes[day] = int(target) if target else None
                eliminated = re.search(r"Player(\d+) eliminated", m.group(2))
                self.vote_results[day] = int(eliminated.group(1)) if eliminated else None
        if m := _GAME_OVER.search(text):
            self.winner = WOLF if m.group(1) == "WOLVES" else VILLAGER
        self._score_track_records()

    def add_message(self, message) -> None:
        seq = int(getattr(message, "index", 0))
        if seq in self._seen_seqs or getattr(message, "type", "chat") != "chat":
            return
        self._seen_seqs.add(seq)
        sender = int(getattr(message, "sender", -1))
        if sender not in self.players:
            return
        text = str(getattr(message, "content", "") or "")
        private = bool(getattr(message, "recipients", None))
        statement = Statement(self.day, seq, text, private)
        self.players[sender].statements.append(statement)
        if private:
            if sender in self.allies:
                self.private_from_allies.append(statement)
            return  # a private word is not a public claim or accusation
        self._read_chat(sender, statement)

    def _read_chat(self, sender: int, statement: Statement) -> None:
        record = self.players[sender]
        text = statement.text
        if _SEER_CLAIM.search(text) and record.claimed_role != SEER:
            record.claimed_role, record.claim_day, record.claim_seq = SEER, statement.day, statement.seq
        elif _VILLAGER_CLAIM.search(text) and record.claimed_role is None:
            record.claimed_role, record.claim_day, record.claim_seq = VILLAGER, statement.day, statement.seq

        checked: set[int] = set()
        if record.claimed_role == SEER:
            for pattern in _CHECK_CLAIMS:
                for seat, result in pattern.findall(text):
                    seat = int(seat)
                    if seat != sender and seat in self.players:
                        record.claimed_checks[seat] = (_canon_result(result), statement.day)
                        checked.add(seat)

        for sentence in _SENTENCE.findall(text):
            targets = {int(s) for s in _PLAYER.findall(sentence)} - {sender} - checked
            targets &= set(self.players)
            if not targets:
                continue
            if _NEGATED_ACCUSE.search(sentence) or (_DEFEND_WORDS.search(sentence) and not _ACCUSE_WORDS.search(sentence)):
                bucket = record.defended
            elif _ACCUSE_WORDS.search(sentence):
                bucket = record.accused
            else:
                continue
            for target in targets:
                days = bucket.setdefault(target, [])
                if statement.day not in days:
                    days.append(statement.day)
        for seat, (result, day) in record.claimed_checks.items():
            bucket = record.accused if result == WOLF else record.defended
            days = bucket.setdefault(seat, [])
            if day not in days:
                days.append(day)

    def _score_track_records(self) -> None:
        for record in self.players.values():
            record.correct_calls.clear()
            record.wrong_calls.clear()
            for target in set(record.accused) | {t for t in record.votes.values() if t is not None}:
                revealed = self.players[target].revealed_role
                if revealed == WOLF:
                    record.correct_calls.add(target)
                elif revealed in (VILLAGER, SEER):
                    record.wrong_calls.add(target)

    # -- queries --------------------------------------------------------------

    def living(self) -> list[int]:
        return [seat for seat, r in self.players.items() if r.alive]

    def living_others(self) -> list[int]:
        return [seat for seat in self.living() if seat != self.me]

    def wolves_left(self) -> int:
        return TOTAL_WOLVES - sum(1 for r in self.players.values() if r.revealed_role == WOLF)

    def seer_claimants(self, *, living_only: bool = True) -> list[int]:
        return sorted(
            (seat for seat, r in self.players.items() if r.claimed_role == SEER and (r.alive or not living_only)),
            key=lambda seat: self.players[seat].claim_seq or 0,
        )

    def real_seer_dead(self) -> bool:
        return any(r.revealed_role == SEER for r in self.players.values())

    def caught_lying(self, seat: int) -> bool:
        """A seer claim the public record has proved false."""
        record = self.players[seat]
        if record.claimed_role != SEER:
            return False
        if record.revealed_role is not None:
            return record.revealed_role != SEER
        if self.real_seer_dead():
            return True
        for target, (result, _day) in record.claimed_checks.items():
            truth = self.players[target].revealed_role
            if truth is not None and (truth == WOLF) != (result == WOLF):
                return True
        return False

    def confirmed_seer(self, seat: int) -> bool:
        """A seer claim whose wolf call was proved right by a reveal."""
        record = self.players[seat]
        return record.claimed_role == SEER and any(
            result == WOLF and self.players[t].revealed_role == WOLF for t, (result, _d) in record.claimed_checks.items()
        )

    def proven_seer(self, seat: int) -> bool:
        """A seer claim the public record proves: revealed as seer on death, or
        a wolf call proved right by a reveal."""
        record = self.players[seat]
        return record.claimed_role == SEER and (record.revealed_role == SEER or self.confirmed_seer(seat))

    def proven_roles(self) -> dict[int, str]:
        """Roles everyone can be sure of: deaths, plus every check announced by
        a proven seer (a seer revealed on death told the truth about all of
        them). When the proven wolves account for every wolf left, everyone
        else is a villager."""
        known = {seat: r.revealed_role for seat, r in self.players.items() if r.revealed_role}
        for seat, record in self.players.items():
            if self.proven_seer(seat):
                for target, (result, _day) in record.claimed_checks.items():
                    known.setdefault(target, result)
                for other in self.seer_claimants():
                    if other != seat:
                        known.setdefault(other, WOLF)
        living_wolves = [s for s in self.living() if known.get(s) == WOLF]
        if living_wolves and len(living_wolves) >= self.wolves_left():
            for seat in self.living():
                known.setdefault(seat, VILLAGER)
        return known

    def owner(self, seat: int) -> str:
        """The agent behind a seat: its display name without the seat suffix."""
        return _NAME_SUFFIX.sub("", self.players[seat].name or "").strip()

    def same_agent_groups(self) -> list[list[int]]:
        """Seats played by the same agent (same display name). Their wording
        matches because it is the same bot, not because they coordinate."""
        groups: dict[str, list[int]] = {}
        for seat in self.players:
            name = self.owner(seat)
            if name:
                groups.setdefault(name, []).append(seat)
        return [seats for seats in groups.values() if len(seats) > 1]

    def stated_vote(self, seat: int, day: int | None = None) -> int | None:
        """The vote ``seat`` last announced in public on ``day`` ("I'm voting
        Player6", "I'll vote Player6"), if any."""
        day = self.day if day is None else day
        stated = None
        for statement in self.players[seat].said_on(day):
            for m in _STATED_VOTE.finditer(statement.text):
                target = int(m.group(1))
                if target in self.players and target != seat:
                    stated = target
        return stated

    def heat_today(self, seat: int, day: int | None = None) -> int:
        """How many distinct living players accused ``seat`` in chat today."""
        day = self.day if day is None else day
        return sum(
            1
            for other, r in self.players.items()
            if other != seat and r.alive and day in r.accused.get(seat, [])
        )

    def accusers_of(self, seat: int, day: int | None = None) -> list[int]:
        day = self.day if day is None else day
        return [o for o, r in self.players.items() if o != seat and r.alive and day in r.accused.get(seat, [])]

    # -- suspicion ------------------------------------------------------------

    def suspicion(self, seat: int, notes: list[str] | None = None) -> float:
        """Public wolf-suspicion score for ``seat`` (higher = more wolf-like).

        Built only from what every player can see, so for a wolf it doubles
        as the village's heat on that seat.
        """
        record = self.players[seat]
        score = 0.0

        def add(points: float, why: str) -> None:
            nonlocal score
            score += points
            if notes is not None and points:
                notes.append(f"{points:+.1f} {why}")

        # Votes: wolves bandwagon onto villagers and rarely onto each other.
        for day, target in sorted(record.votes.items()):
            if target is None:
                add(0.6, f"abstained D{day}")
                continue
            truth = self.players[target].revealed_role
            if truth == WOLF:
                add(-1.2, f"voted wolf P{target} D{day}")
            elif truth in (VILLAGER, SEER):
                add(0.8 + (0.3 if self.vote_results.get(day) == target else 0.0), f"voted villager P{target} D{day}")

        # Chat: accusing a revealed wolf is a correct call; defending one is not.
        for target in record.accused:
            truth = self.players[target].revealed_role
            if truth == WOLF:
                add(-1.5, f"called wolf P{target}")
            elif truth in (VILLAGER, SEER):
                add(0.4, f"accused villager P{target}")
        for target in record.defended:
            truth = self.players[target].revealed_role
            if truth == WOLF:
                add(1.5, f"defended wolf P{target}")

        # Claims.
        if record.claimed_role == SEER:
            claimants = self.seer_claimants()
            if self.caught_lying(seat):
                add(10.0, "fake seer (contradicted)")
            elif self.confirmed_seer(seat):
                add(-10.0, "seer call proved right")
            elif len(claimants) == 1:
                add(-2.5, "only seer claim")
            elif claimants and claimants[0] != seat:
                add(2.0, "counter-claimed seer")
            else:
                add(0.5, "contested seer claim")

        # Seer claims about this seat: weighted by how believable the claimant is.
        for other, r in self.players.items():
            if other == seat or r.claimed_role != SEER or seat not in r.claimed_checks:
                continue
            result = r.claimed_checks[seat][0]
            if self.caught_lying(other):
                add(-2.0 if result == WOLF else 1.0, f"P{other} (fake seer) called {result}")
                continue
            trusted = self.confirmed_seer(other) or len(self.seer_claimants()) == 1
            weight = 4.0 if trusted else 1.5
            add(weight if result == WOLF else -weight * 0.6, f"seer-claim P{other} says {result}")

        # Partners vote with their wolf: on the day the village voted a wolf out,
        # standing with that wolf's vote is the strongest partner tell there is.
        for other, r in self.players.items():
            if other == seat or r.revealed_role != WOLF:
                continue
            for day, target in r.votes.items():
                if target is None or target == seat or record.votes.get(day) != target:
                    continue
                if self.vote_results.get(day) == other:
                    add(1.5, f"voted with wolf P{other} on P{target} the day P{other} went out (D{day})")
                else:
                    add(0.6, f"voted with wolf P{other} on P{target} D{day}")

        # Wolves kill whoever threatens them: the night-killed player's suspects.
        for other, r in self.players.items():
            if r.death and r.death[1].startswith("killed") and seat in r.accused:
                add(0.8, f"accused by night-kill P{other}")
            if r.revealed_role == WOLF and other != seat:
                if seat in r.defended:
                    add(1.0, f"defended by dead wolf P{other}")
                if seat in r.accused or seat in r.votes.values():
                    add(-0.4, f"attacked by dead wolf P{other}")

        # Accusations from players with a good record carry weight.
        for other, r in self.players.items():
            if other == seat or not r.alive or seat not in r.accused:
                continue
            if len(r.correct_calls) > len(r.wrong_calls):
                add(0.9, f"accused by reliable P{other}")
            else:
                add(0.25, f"accused by P{other}")

        # Silence on a finished day they were alive for.
        for day in range(1, self.day):
            alive_that_day = record.death is None or record.death[0] > day or (
                record.death[0] == day and record.death[1] != "killed at night"
            )
            if alive_that_day and not record.said_on(day) and any(r.said_on(day) for r in self.players.values()):
                add(0.4, f"silent D{day}")
        return score

    def explain(self, seat: int) -> list[str]:
        """The reasons behind ``seat``'s suspicion score, one line each."""
        notes: list[str] = []
        self.suspicion(seat, notes)
        return notes

    def wolf_probability(self, *, public: bool = False) -> dict[int, float]:
        """P(wolf) for every living seat other than us.

        ``public=True`` is the village's view (public evidence only). Otherwise
        our private knowledge pins what it can: a wolf knows every role, a
        seer knows its checks.
        """
        if not public and self.role == WOLF:
            return {s: (1.0 if s in self.allies else 0.0) for s in self.living_others()}
        known = {s: (1.0 if r == WOLF else 0.0) for s, r in self.proven_roles().items()}
        if not public:
            for seat, result in self.investigations.items():
                known[seat] = 1.0 if result == WOLF else 0.0
            if self.role == SEER:  # we are the seer, so any other seer claim is a wolf's
                for seat in self.seer_claimants():
                    if seat != self.me:
                        known[seat] = 1.0
        return self._spread(self.living_others(), known, self.wolves_left())

    def _spread(self, seats: list[int], known: dict[int, float], wolves_left: int, extra=None) -> dict[int, float]:
        """Share the unknown wolves over the unpinned ``seats`` by suspicion.
        A forced deduction (as many wolves as candidates) is a certainty."""
        candidates = [s for s in seats if s not in known]
        wolves = wolves_left - sum(1 for s in seats if known.get(s) == 1.0)
        wolves = max(0, min(wolves, len(candidates)))
        weights = {
            s: math.exp(max(-8.0, min(8.0, self.suspicion(s) + (extra(s) if extra else 0.0)))) for s in candidates
        }
        total = sum(weights.values()) or 1.0
        cap = 1.0 if wolves >= len(candidates) else 0.97
        probs = {s: min(cap, wolves * w / total) for s, w in weights.items()}
        return {**probs, **{s: p for s, p in known.items() if s in seats}}

    def village_heat(self) -> dict[int, float]:
        """The village's P(wolf) for every living seat, us included: what a
        villager with only public information would think. For a wolf."""
        known = {s: (1.0 if r == WOLF else 0.0) for s, r in self.proven_roles().items()}
        return self._spread(self.living(), known, self.wolves_left(), extra=lambda s: 0.6 * self.heat_today(s))

    # -- rendering ------------------------------------------------------------

    def render_table(self) -> str:
        """The player table as markdown: status, claim, votes, what they said,
        who they accused, and our wolf read. Then the wolf-callers record."""
        probs = self.wolf_probability()
        heat = self.village_heat() if self.role == WOLF else {}
        header = "| Seat | Name | Status | Claim | Votes | Accused | Last said | P(wolf) |"
        if heat:
            header += " Village heat |"
        lines = [header, "|" + "---|" * (header.count("|") - 1)]
        for seat, r in self.players.items():
            if seat == self.me:
                status = f"**me** ({self.role or '?'})" + ("" if r.alive else ", dead")
            elif r.alive:
                status = "alive"
            else:
                status = f"dead D{r.death[0]} {r.death[1]}: **{r.revealed_role}**" if r.death else "dead"
            claim = r.claimed_role or ""
            if r.claimed_checks:
                claim += " (" + ", ".join(f"P{t}={res}" for t, (res, _d) in sorted(r.claimed_checks.items())) + ")"
            if r.claimed_role == SEER and self.caught_lying(seat):
                claim += " LIAR"
            votes = ", ".join(f"D{d}->{'abstain' if t is None else f'P{t}'}" for d, t in sorted(r.votes.items()))
            accused = ", ".join(f"P{t}" for t in sorted(r.accused))
            said = r.statements[-1].text if r.statements and not r.statements[-1].private else ""
            said = (said[:70] + "...") if len(said) > 70 else said
            said = said.replace("|", "/")
            if seat == self.me:
                p = ""
            elif seat in probs:
                p = f"{probs[seat]:.0%}"
            elif r.revealed_role:
                p = "100%" if r.revealed_role == WOLF else "0%"
            else:
                p = ""
            row = f"| P{seat} | {r.name or '-'} | {status} | {claim} | {votes} | {accused} | {said} | {p} |"
            if heat:
                row += f" {heat.get(seat, 0):.0%} |" if r.alive else " |"
            lines.append(row)
        lines.append("")
        for group in self.same_agent_groups():
            seats = ", ".join(f"P{s}" for s in group)
            lines.append(
                f"Same agent: {seats} are all '{self.owner(group[0])}'. Matching wording or style between them is the "
                "same bot talking, not evidence that they coordinate."
            )
        proven = {s: r for s, r in self.proven_roles().items() if self.players[s].alive}
        if proven:
            lines.append("Proven by the public record: " + ", ".join(f"P{s} {r}" for s, r in sorted(proven.items())))
        lines.append(self.render_callers())
        return "\n".join(lines)

    def render_callers(self) -> str:
        """Who pointed out each revealed wolf, and who got it wrong."""
        wolves = [s for s, r in self.players.items() if r.revealed_role == WOLF]
        if not wolves:
            return "Revealed wolves: none yet."
        out = []
        for wolf in wolves:
            callers = [s for s, r in self.players.items() if s != wolf and r.pointed_at(wolf)]
            out.append(f"Revealed wolf P{wolf}: called correctly by " + (", ".join(f"P{s}" for s in callers) or "nobody"))
        record = [
            f"P{s} {len(r.correct_calls)} right / {len(r.wrong_calls)} wrong"
            for s, r in self.players.items()
            if r.correct_calls or r.wrong_calls
        ]
        if record:
            out.append("Track record: " + "; ".join(record))
        return "\n".join(out)

    def transcript(self, *, days: int | None = None, include_private: bool = True) -> str:
        """Every chat we've seen, in order, as ``D2 P3: text`` lines."""
        rows = []
        for seat, r in self.players.items():
            for s in r.statements:
                if s.private and not include_private:
                    continue
                if days is not None and s.day < self.day - days + 1:
                    continue
                tag = " (private to me)" if s.private and seat != self.me else (" (private)" if s.private else "")
                rows.append((s.seq, f"D{s.day} P{seat}{tag}: {s.text}"))
        return "\n".join(text for _seq, text in sorted(rows))
