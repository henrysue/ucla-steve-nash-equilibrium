"""What to do today: posture, scripted lines, and the vote / kill / check.

Everything here is deterministic and reads only the ``Ledger`` and our own
``Memory``, so the agent can always fall back to it when the model is slow
or absent. The model's job (``agent.py``) is to say these things well.

Postures
--------

Wolf (two wolves, one seer, four villagers; parity wins for us):

- ``blend`` — days before ``FAKE_CLAIM_DAY``, and any day nothing forces our
  hand: talk like a model villager, ask questions, share a mild read on a
  villager, vote with the table's bandwagon when it is on a villager.
- ``defend`` — one player accuses us: stay calm, answer, and question the
  accuser's own record.
- ``fake_seer`` — claim seer. Proactively on ``FAKE_CLAIM_DAY`` (day 2): the
  night-1 "check" is a real villager reported as villager (true, so it can
  never be contradicted, and it buys credibility), then the latest night's
  "check" names a villager as WOLF. Reactively, any day: when the real seer
  accuses us or our ally, or two or more players pile on us. Only one wolf
  claims; the other backs the claim.
- ``distance`` — our ally is going down anyway: agree they are suspicious
  and vote with the table, so their reveal doesn't drag us with them.
- ``all_in`` — one mis-lynch from parity: both wolves vote the same villager
  (and a split or tie still helps: three days with no elimination is a wolf
  win).

Villager: hunt inconsistencies in votes and claims, protect a lone seer
claim, never call for mass role claims, lynch proven liars, never abstain,
and consolidate on the bandwagon when it is close to our top read (a split
village loses a plurality vote).

Seer: hide while every check is clean; claim the moment a check finds a
wolf, from day 2 regardless (this game is short — dying unclaimed wastes
the checks), or when accused or counter-claimed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .ledger import ABSTAIN, SEER, VILLAGER, WOLF, Ledger

FAKE_CLAIM_DAY = int(os.environ.get("WEREWOLF_FAKE_CLAIM_DAY", "2") or 2)
MAX_CHATS = 5

_ALLY_KILL = re.compile(r"\bkill\s+P(?:layer)?\s?(\d)", re.IGNORECASE)
_ALLY_VOTE = re.compile(r"\bvote\s+P(?:layer)?\s?(\d)", re.IGNORECASE)
_ALLY_CLAIMING = re.compile(r"\bI'?(?:m| am|'ll| will)\s+claim(?:ing)?\s+seer", re.IGNORECASE)


@dataclass
class Talk:
    """One line we intend to send today.

    ``goal`` tells the model what the line must achieve; ``must_mention``
    are tokens the final text has to contain (else ``fallback`` is sent);
    ``recipients`` is ``[seat]`` for a private message.
    """

    key: str
    goal: str
    fallback: str
    must_mention: tuple[str, ...] = ()
    recipients: list[int] | None = None
    scripted: bool = True  # sent verbatim from the fallback, no model


@dataclass
class DayPlan:
    posture: str
    notes: list[str]
    lines: list[Talk]
    budget: int = 3


@dataclass
class Memory:
    """What a seat remembers across days beyond the public record."""

    fake_claim_day: int | None = None  # the day we actually claimed (seen in our own chat)
    fake_checks: dict[int, tuple[int, str]] = field(default_factory=dict)  # night -> (seat, result), our story
    proposed_day: int | None = None  # the day ``fake_checks`` was last drafted
    frame: int | None = None  # the villager our fake seer calls wolf
    ally_kill: int | None = None
    ally_vote: int | None = None
    ally_claiming_day: int | None = None
    sent: dict[int, list[str]] = field(default_factory=dict)  # day -> keys of lines sent

    def read_ally_messages(self, ledger: Ledger) -> None:
        for statement in ledger.private_from_allies:
            if statement.day != ledger.day:
                continue
            if m := _ALLY_KILL.search(statement.text):
                self.ally_kill = int(m.group(1))
            if m := _ALLY_VOTE.search(statement.text):
                self.ally_vote = int(m.group(1))
            if _ALLY_CLAIMING.search(statement.text):
                self.ally_claiming_day = statement.day


def _p(seat: int) -> str:
    return f"Player{seat}"


# -- shared reads ---------------------------------------------------------------


def wolves_alive(ledger: Ledger) -> list[int]:
    if ledger.role != WOLF:
        return []
    return [s for s in sorted({ledger.me, *ledger.allies}) if s is not None and ledger.players[s].alive]


def non_wolves_alive(ledger: Ledger) -> list[int]:
    wolves = set(wolves_alive(ledger))
    return [s for s in ledger.living() if s not in wolves]


def bandwagon(ledger: Ledger, *, exclude: set[int] = frozenset(), minimum: int = 2) -> int | None:
    """The living seat most players accused today, if at least ``minimum`` did."""
    counts = {s: ledger.heat_today(s) for s in ledger.living() if s not in exclude}
    if not counts:
        return None
    best = max(counts, key=lambda s: (counts[s], -s))
    return best if counts[best] >= minimum else None


def _real_seer_claimants(ledger: Ledger) -> list[int]:
    """Seer claimants a wolf knows are not wolves (the real seer, or a bluffer)."""
    wolves = {ledger.me, *ledger.allies}
    return [s for s in ledger.seer_claimants() if s not in wolves]


def _seer_called(ledger: Ledger, claimant: int, target: int) -> bool:
    check = ledger.players[claimant].claimed_checks.get(target)
    return bool(check and check[0] == WOLF)


# -- wolf -----------------------------------------------------------------------


def i_should_claim(ledger: Ledger, memory: Memory) -> bool:
    """Of the two wolves, the one the village trusts more (lower heat) claims;
    ties go to the lower seat. An ally who already said it is claiming wins."""
    if memory.fake_claim_day is not None:
        return True
    allies = [a for a in ledger.allies if ledger.players[a].alive]
    if any(ledger.players[a].claimed_role == SEER for a in allies):
        return False
    if memory.ally_claiming_day == ledger.day:
        return False
    if not allies:
        return True
    heat = ledger.village_heat()
    me = (round(heat.get(ledger.me, 0), 3), ledger.me)
    return all(me < (round(heat.get(a, 0), 3), a) for a in allies)


def _pick_cleared(ledger: Ledger, avoid: set[int]) -> int | None:
    """A villager to 'check' as villager: true, so it can never be disproved.
    Prefer someone who isn't accusing us and the village already trusts."""
    wolves = {ledger.me, *ledger.allies}
    pool = [s for s in ledger.living() if s not in wolves and s not in avoid]
    if not pool:
        pool = [s for s, r in ledger.players.items() if s not in wolves and s not in avoid]
    if not pool:
        return None
    heat = ledger.village_heat()
    return min(pool, key=lambda s: (any(ledger.players[s].pointed_at(w) for w in wolves), heat.get(s, 0), s))


def _pick_frame(ledger: Ledger, accusers: list[int] = ()) -> int | None:
    """A villager to call WOLF: the real seer if they've claimed, else whoever
    is accusing us, else the villager the village already suspects most."""
    wolves = {ledger.me, *ledger.allies}
    living = [s for s in ledger.living() if s not in wolves]
    if not living:
        return None
    for claimant in _real_seer_claimants(ledger):
        if claimant in living:
            return claimant
    for accuser in accusers:
        if accuser in living:
            return accuser
    heat = ledger.village_heat()
    return max(living, key=lambda s: (heat.get(s, 0), -s))


def start_fake_claim(ledger: Ledger, memory: Memory, *, frame: int | None, wolf_call: bool) -> None:
    """Invent our seer history up to last night.

    A draft until our own chat shows the claim (``sync_claim``). Night 1 is a
    clean check on a real villager (credibility). If
    ``wolf_call``, the latest night names ``frame`` as wolf; nights in
    between are clean checks on other villagers.
    """
    day = ledger.day
    if memory.proposed_day == day:
        return  # one story per day, however often we re-plan
    memory.proposed_day = day
    memory.fake_checks.clear()
    used: set[int] = set() if frame is None else {frame}
    if wolf_call and frame is not None and day == 1:
        memory.fake_checks[1] = (frame, WOLF)
    else:
        cleared = _pick_cleared(ledger, used)
        if cleared is not None:
            memory.fake_checks[1] = (cleared, VILLAGER)
            used.add(cleared)
        for night in range(2, day + 1):
            if night == day and wolf_call and frame is not None:
                memory.fake_checks[night] = (frame, WOLF)
            else:
                clean = _pick_cleared(ledger, used)
                if clean is not None:
                    memory.fake_checks[night] = (clean, VILLAGER)
                    used.add(clean)
    memory.frame = frame if wolf_call else None


def sync_claim(ledger: Ledger, memory: Memory) -> bool:
    """Whether we (a wolf) have claimed seer in public. Our own chat is the
    truth: whatever checks we actually announced become the story, and the
    villager we called wolf becomes the frame."""
    me = ledger.players[ledger.me]
    if me.claimed_role != SEER:
        return False
    if memory.fake_claim_day is None:
        memory.fake_claim_day = me.claim_day or ledger.day
    told = {seat for seat, _ in memory.fake_checks.values()}
    for seat, (result, day) in sorted(me.claimed_checks.items(), key=lambda item: item[1][1]):
        if seat not in told:
            night = max([n for n in range(1, day + 1) if n not in memory.fake_checks], default=day)
            memory.fake_checks[night] = (seat, result)
        if result == WOLF and ledger.players[seat].alive:
            memory.frame = seat
    memory.fake_checks = {n: c for n, c in memory.fake_checks.items() if c[0] in me.claimed_checks}
    return True


def continue_fake_claim(ledger: Ledger, memory: Memory) -> None:
    """A day after our claim: add last night's 'check'. Keep pushing a living
    frame; otherwise call the villager the village suspects most."""
    night = ledger.day
    if night in memory.fake_checks:
        return
    frame = memory.frame if memory.frame is not None and ledger.players[memory.frame].alive else None
    if frame is None:
        frame = _pick_frame(ledger)
        memory.frame = frame
    if frame is not None:
        memory.fake_checks[night] = (frame, WOLF)


def _claim_lines(ledger: Ledger, memory: Memory, opener: str) -> list[Talk]:
    """The claim, split in two so the clean check lands before the wolf call."""
    said = ledger.players[ledger.me].claimed_checks
    clean = [(n, s) for n, (s, r) in sorted(memory.fake_checks.items()) if r == VILLAGER and s not in said]
    wolf = [(n, s) for n, (s, r) in sorted(memory.fake_checks.items()) if r == WOLF and s not in said]
    lines: list[Talk] = []
    claimed = ledger.players[ledger.me].claimed_role == SEER
    if clean or not claimed:
        checks = "; ".join(f"night {n} I checked {_p(s)}: villager" for n, s in clean)
        text = f"{opener} I'm the seer. {checks.capitalize() if checks else ''}".strip()
        if clean:
            text += f". {_p(clean[0][1])} is clean, leave them alone."
        lines.append(Talk("claim", "Claim seer and report the clean check(s), credibly.", text, ("seer",)))
    for night, seat in wolf:
        lines.append(
            Talk(
                f"wolfcall{night}",
                f"Report the night {night} check: {_p(seat)} is a WOLF; push the table to vote {_p(seat)} today.",
                f"Night {night} I checked {_p(seat)}: WOLF. Everyone vote {_p(seat)} today.",
                (_p(seat), "wolf"),
            )
        )
    return lines


def wolf_plan(ledger: Ledger, memory: Memory) -> DayPlan:
    memory.read_ally_messages(ledger)
    me, day = ledger.me, ledger.day
    allies = [a for a in ledger.allies if ledger.players[a].alive]
    villagers = non_wolves_alive(ledger)
    wolves = wolves_alive(ledger)
    accusers = ledger.accusers_of(me)
    seer_on_me = [c for c in _real_seer_claimants(ledger) if _seer_called(ledger, c, me)]
    seer_on_ally = [c for c in _real_seer_claimants(ledger) for a in allies if _seer_called(ledger, c, a)]
    heat = ledger.village_heat()
    notes = [
        "You are a WOLF. Never reveal it in public, never name your ally as a wolf-mate, never say 'we wolves'.",
        "Sound like a model villager: logical questions, short reads, not silent and not shouty.",
    ]
    lines: list[Talk] = []
    posture = "blend"
    ally_claimed = any(ledger.players[a].claimed_role == SEER for a in allies) or memory.ally_claiming_day == day
    # Targeted directly, we claim ourselves; a planned claim goes to the wolf the village trusts more.
    must_claim_ok = not ledger.real_seer_dead() and not ally_claimed
    claim_ok = not ledger.real_seer_dead() and i_should_claim(ledger, memory)

    if sync_claim(ledger, memory):
        posture = "fake_seer"
        continue_fake_claim(ledger, memory)
        lines += _claim_lines(ledger, memory, "Still here.")
        notes.append("You claimed SEER earlier. Stay consistent with every check you reported; defend the claim.")
    elif seer_on_me and must_claim_ok:
        posture = "fake_seer"
        start_fake_claim(ledger, memory, frame=seer_on_me[0], wolf_call=True)
        lines += _claim_lines(ledger, memory, f"{_p(seer_on_me[0])} is lying.")
        notes.append(f"Counter-claim: {_p(seer_on_me[0])} falsely called you wolf. You are the real seer and they are a wolf.")
    elif len(accusers) >= 2 and must_claim_ok:
        posture = "fake_seer"
        start_fake_claim(ledger, memory, frame=accusers[0], wolf_call=day >= 2)
        lines += _claim_lines(ledger, memory, "I have to come out.")
        notes.append("You are under heavy pressure: claim seer to survive and turn the table on your accuser.")
    elif day >= FAKE_CLAIM_DAY and claim_ok and len(villagers) > len(wolves):
        posture = "fake_seer"
        start_fake_claim(ledger, memory, frame=_pick_frame(ledger, accusers), wolf_call=True)
        lines += _claim_lines(ledger, memory, "Time to come out.")
        notes.append("Planned fake claim: the clean check is true (credibility); the wolf call frames a villager.")
    elif accusers:
        posture = "defend"
        notes.append(
            f"{', '.join(_p(a) for a in accusers)} accused you. Answer calmly with your reasoning, then ask why "
            "they are pushing you and not the votes/inconsistencies on the record."
        )

    if seer_on_ally and posture != "fake_seer":
        notes.append(
            f"{_p(seer_on_ally[0])} claimed seer and called your ally a wolf. Your ally will counter-claim; "
            "back them quietly unless the table has clearly decided, then distance yourself."
        )

    # Distance from a doomed ally (only if we aren't claiming to defend them).
    for ally in allies:
        doomed = ledger.heat_today(ally) * 2 >= len(villagers) or (seer_on_ally and posture != "fake_seer")
        if doomed and posture != "fake_seer":
            posture = "distance"
            notes.append(f"Your ally {_p(ally)} is going down. Distance yourself: agree {_p(ally)} looks suspicious and vote with the table.")

    # All-in: one more mis-lynch reaches parity.
    if wolves and len(villagers) - len(wolves) <= 1:
        notes.append("ENDGAME: one villager lynch wins. Both wolves vote the same villager; a tie also helps (stall rule).")
        if posture == "blend":
            posture = "all_in"

    vote = wolf_vote(ledger, memory)
    if allies and memory.sent.get(day, []).count("dm") == 0:
        kill = choose_kill(ledger, memory, [str(s) for s in ledger.living() if s not in wolves])
        claiming = " I'm claiming seer, back me." if posture == "fake_seer" else " Don't claim."
        dm = f"Kill {_p(kill)} tonight. Vote {_p(vote)} today.{claiming}" if kill is not None else f"Vote {_p(vote)} today."
        lines.insert(0, Talk("dm", "Private plan to your ally.", dm, recipients=[allies[0]]))
    if posture in ("blend", "defend", "all_in", "distance"):
        notes.append(f"Push the vote toward {_p(vote)} if it fits the conversation (it is a villager), without leading too hard.")
    budget = {"fake_seer": 5, "blend": 3, "defend": 4, "distance": 3, "all_in": 4}[posture]
    return DayPlan(posture, notes, lines, budget)


def wolf_vote(ledger: Ledger, memory: Memory) -> int:
    """Our vote as a wolf (a seat; never abstain unless nothing else is legal)."""
    wolves = set(wolves_alive(ledger))
    villagers = [s for s in ledger.living() if s not in wolves]
    if not villagers:
        return int(ABSTAIN)
    heat = ledger.village_heat()
    if memory.fake_claim_day is not None and memory.frame in villagers:
        return memory.frame  # consistency with our seer claim
    if memory.ally_vote in villagers and len(villagers) - len(wolves) <= 1:
        return memory.ally_vote  # all-in: one target for both wolves
    if len(villagers) - len(wolves) <= 1:
        return max(villagers, key=lambda s: (heat.get(s, 0), -s))
    allies = [a for a in ledger.allies if ledger.players[a].alive]
    wagon = bandwagon(ledger)
    if wagon in allies:
        if ledger.heat_today(wagon) * 2 >= len(villagers):
            return wagon  # bus a doomed ally
        return max(villagers, key=lambda s: (heat.get(s, 0), -s))
    if wagon == ledger.me:
        accusers = [a for a in ledger.accusers_of(ledger.me) if a in villagers]
        if accusers:
            return accusers[0]
        return max(villagers, key=lambda s: (heat.get(s, 0), -s))
    if wagon in villagers:
        return wagon  # blend in: ride the table's wagon onto a villager
    if memory.ally_vote in villagers:
        return memory.ally_vote
    return max(villagers, key=lambda s: (heat.get(s, 0), -s))


def _kill_options(legal: list[str]) -> list[int]:
    return [int(a) for a in legal if a != ABSTAIN]


def endgame_frame(ledger: Ledger, legal: list[str], village_view: dict[int, float] | None = None) -> int | None:
    """In the endgame (after tonight's kill the next vote decides, or nearly),
    the villager the village suspects most: keep them alive and frame them
    tomorrow. ``village_view`` is the model's read of the village (a wolf's
    guesses), else the code's village heat. None outside the endgame."""
    options = _kill_options(legal)
    wolves = wolves_alive(ledger)
    if len(options) < 2 or (len(options) - 1) - len(wolves) > 1:
        return None
    view = village_view or ledger.village_heat()
    return max(options, key=lambda s: (view.get(s, 0.0), -s))


def lone_accusers(ledger: Ledger) -> dict[int, int]:
    """Villager seat -> the living wolf only they have accused. Killing a
    wolf's only accuser points straight at that wolf."""
    out: dict[int, int] = {}
    for wolf in wolves_alive(ledger):
        accusers = [s for s, r in ledger.players.items() if r.alive and s != wolf and wolf in r.accused]
        if len(accusers) == 1:
            out[accusers[0]] = wolf
    return out


def choose_kill(
    ledger: Ledger, memory: Memory, legal: list[str], village_view: dict[int, float] | None = None
) -> int | None:
    """Night kill: the real seer first, then whoever is hunting us accurately,
    then the discussion leaders. Keep alive vocal players who mislead the
    village, our frame (we want the village to lynch them), and in the endgame
    the villager the village suspects most (b4bdb262: killing the village's
    top suspect, who had accused us, left nobody else to blame)."""
    options = _kill_options(legal)
    if not options:
        return None
    frame = endgame_frame(ledger, legal, village_view)
    if memory.ally_kill in options and memory.ally_kill != frame and not (
        memory.fake_claim_day is not None and memory.ally_kill == memory.frame
    ):
        return memory.ally_kill
    wolves = {ledger.me, *ledger.allies}
    real_seers = [c for c in _real_seer_claimants(ledger) if c in options]
    lone = lone_accusers(ledger)

    def value(seat: int) -> float:
        r = ledger.players[seat]
        score = 0.0
        if seat in real_seers:
            # If we are in a live seer war, killing the real seer reveals 'seer'
            # and proves our claim fake; hit their strongest supporter instead.
            score += 20 if memory.fake_claim_day is not None else 100
        if memory.fake_claim_day is not None and real_seers:
            if any(seat in ledger.players[c].defended or c in r.defended for c in real_seers):
                score += 40
        score += 40 * len(r.correct_calls)
        score += 15 * sum(1 for w in wolves if r.pointed_at(w))
        score += 2 * len([s for s in r.statements if not s.private])
        if len(r.wrong_calls) > len(r.correct_calls) and not any(r.pointed_at(w) for w in wolves):
            score -= 20  # a useful misleader
        if seat == memory.frame and memory.fake_claim_day is not None:
            score -= 50
        if seat in lone and seat not in real_seers:
            score -= 25  # the "killed my only accuser" tell
        if seat == frame:
            score -= 200
        return score

    return max(options, key=lambda s: (value(s), -s))


def kill_briefing(
    ledger: Ledger, memory: Memory, legal: list[str], village_view: dict[int, float] | None = None
) -> list[str]:
    """What the code knows about tonight's kill, for the model."""
    notes: list[str] = []
    options = _kill_options(legal)
    frame = endgame_frame(ledger, legal, village_view)
    if frame is not None:
        wolves = wolves_alive(ledger)
        notes.append(
            f"ENDGAME: after tonight it is {len(wolves)} wolf/wolves vs {len(options) - 1} villagers and the next vote "
            f"decides. Keep Player{frame} ALIVE: the village already suspects them most, so you frame them tomorrow. "
            "Kill one of the others, preferably whoever is most likely to vote against you."
        )
    for seat, wolf in lone_accusers(ledger).items():
        if seat in options:
            notes.append(
                f"Player{seat} is the only player who accused Player{wolf}. Killing them points straight at Player{wolf}."
            )
    for seat in _real_seer_claimants(ledger):
        if seat in options:
            notes.append(f"Player{seat} claimed seer (not one of us): the usual priority kill.")
    if memory.ally_kill is not None:
        notes.append(f"Your ally proposed killing Player{memory.ally_kill}; different picks kill the lower seat.")
    return notes


# -- seer -----------------------------------------------------------------------


def choose_investigation(ledger: Ledger, legal: list[str]) -> int | None:
    """Check the most suspicious unchecked player; ties to the most vocal."""
    options = [int(a) for a in legal if a != ABSTAIN]
    unchecked = [s for s in options if s not in ledger.investigations] or options
    if not unchecked:
        return None
    probs = ledger.wolf_probability()
    return max(
        unchecked,
        key=lambda s: (round(probs.get(s, 0), 3), len(ledger.players[s].statements), -s),
    )


def in_danger(ledger: Ledger) -> bool:
    """Someone said they will vote us, or two or more players accuse us today:
    claim now rather than be voted out unclaimed. One loose accusation is not danger."""
    me = ledger.me
    if any(ledger.stated_vote(s) == me for s in ledger.living_others()):
        return True
    return len(ledger.accusers_of(me)) >= 2


def seer_plan(ledger: Ledger, memory: Memory) -> DayPlan:
    me, day = ledger.me, ledger.day
    found = [s for s, r in ledger.investigations.items() if r == WOLF and ledger.players[s].alive]
    clean = [s for s, r in ledger.investigations.items() if r == VILLAGER]
    fakes = [c for c in ledger.seer_claimants() if c != me]
    danger = in_danger(ledger)
    notes = [
        "You are the SEER, the village's strongest asset. Your checks are true.",
        "Never call for everyone to claim roles; that hands the wolves a target list.",
    ]
    lines: list[Talk] = []
    claimed = ledger.players[me].claimed_role == SEER
    claim = claimed or found or fakes or danger or day >= FAKE_CLAIM_DAY
    if not claim:
        notes.append("Hide your role today: talk like a thoughtful villager and steer gently away from your clean checks.")
        return DayPlan("seer_hidden", notes, lines, 3)

    unsaid = {s: r for s, r in ledger.investigations.items() if s not in ledger.players[me].claimed_checks}
    if not claimed or unsaid:
        results = "; ".join(f"{_p(s)} {r.upper() if r == WOLF else r}" for s, r in sorted(ledger.investigations.items()))
        lines.append(Talk(f"claim{day}", "Claim seer (or restate) with every check.", f"I'm the seer. My checks: {results}.", ("seer",)))
    for wolf in found:
        lines.append(
            Talk(
                f"push{wolf}",
                f"Push the whole table to vote {_p(wolf)}, a checked wolf.",
                f"{_p(wolf)} is a wolf, I checked them. Everyone vote {_p(wolf)}, don't split.",
                (_p(wolf),),
            )
        )
    for fake in fakes:
        notes.append(f"{_p(fake)} counter-claimed seer: they are lying, so they are a wolf. Say so and ask the table to compare our records.")
    if clean:
        notes.append(f"Your clean checks ({', '.join(_p(s) for s in clean)}) are villagers: ask them to back you.")
    return DayPlan("seer_claim", notes, lines, 5)


def seer_vote(ledger: Ledger) -> int:
    found = [s for s, r in ledger.investigations.items() if r == WOLF and ledger.players[s].alive and s != ledger.me]
    if found:
        return found[0]
    fakes = [c for c in ledger.seer_claimants() if c != ledger.me]
    if fakes:
        return fakes[0]
    return villager_vote(ledger)


# -- villager -------------------------------------------------------------------


def villager_plan(ledger: Ledger, memory: Memory) -> DayPlan:
    me = ledger.me
    notes = [
        "You are a VILLAGER. Track votes and claims, not vibes: who voted revealed villagers, who started wagons, who deflects.",
        "Protect a lone seer claim; don't push everyone to claim roles; treat contradictions as wolf tells.",
        "Push for an elimination today: abstaining and split votes help the wolves (three days without one is a wolf win).",
    ]
    lines: list[Talk] = []
    liars = [s for s in ledger.living_others() if ledger.caught_lying(s)]
    for liar in liars:
        lines.append(
            Talk(
                f"liar{liar}",
                f"{_p(liar)} is a proven fake seer (their claim is contradicted by a reveal). Lay out the proof; vote {_p(liar)}.",
                f"{_p(liar)}'s seer claim is proven false by the reveals. Fake seer = wolf. Vote {_p(liar)}.",
                (_p(liar),),
            )
        )
    claimants = ledger.seer_claimants()
    if len(claimants) == 1 and claimants[0] != me:
        notes.append(f"{_p(claimants[0])} is the only seer claim: back them and follow their wolf calls.")
    elif len(claimants) >= 2:
        notes.append(
            f"Seer claims from {', '.join(_p(c) for c in claimants)}: one is a wolf. The first claimant is usually real; "
            "a reactive counter-claim right after being accused is a wolf move. Check whose 'clean' calls fit the votes."
        )
    if ledger.accusers_of(me):
        notes.append("You are being accused: answer with your vote record and reasoning. You may say you're a plain villager.")
    target = villager_vote(ledger)
    notes.append(f"Your current top read is {_p(target)}; make the case from the record and ask others to consolidate.")
    return DayPlan("villager", notes, lines, 3)


def villager_vote(ledger: Ledger) -> int:
    probs = ledger.wolf_probability()
    candidates = [s for s in ledger.living_others() if probs.get(s, 0) > 0] or ledger.living_others()
    if not candidates:
        return int(ABSTAIN)
    top = max(candidates, key=lambda s: (probs.get(s, 0), -s))
    wagon = bandwagon(ledger, exclude={ledger.me})
    if wagon in candidates and probs.get(wagon, 0) >= 0.7 * probs.get(top, 0):
        return wagon  # consolidate: a split village loses a plurality vote
    return top


# -- dispatch -------------------------------------------------------------------


def day_plan(ledger: Ledger, memory: Memory) -> DayPlan:
    if ledger.role == WOLF:
        return wolf_plan(ledger, memory)
    if ledger.role == SEER:
        return seer_plan(ledger, memory)
    return villager_plan(ledger, memory)


def choose_vote(ledger: Ledger, memory: Memory) -> int:
    if ledger.role == WOLF:
        return wolf_vote(ledger, memory)
    if ledger.role == SEER:
        return seer_vote(ledger)
    return villager_vote(ledger)
