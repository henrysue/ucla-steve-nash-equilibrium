"""Step 2 of move selection: what each opposing Pokemon will most likely do.

The code makes a first guess per foe, with its reasons:

- on its first turn with Fake Out -> Fake Out on whichever of ours it fears
  most (match 286bbc2e: a fresh Sneasler's Fake Out was read as Protect);
- its Protect odds (protect.py) are high -> it Protects; that is only on its
  first turn out (never right after a Protect: a repeat works 1 time in 3);
- otherwise -> it attacks whichever of ours its STAB hits hardest.

The model sees that guess and the board and gives its own prediction per
foe; the model's answer is used when it is valid, the code's guess when the
model is slow or wrong. Either way it becomes ``Expectation``: per foe, the
chance it Protects, switches, or attacks each of ours and for how much, which
step 3 (lookahead.best_line) plays against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .moverules import PROTECT_CHAIN

ACTIONS = ("protect", "attack", "fake_out", "switch", "speed_control", "setup", "support")
CODE_CONFIDENCE = 0.6  # the code's guess, when the model gives none
ATTACK_SHARE_OF_REST = 0.7  # what's left after the predicted action: mostly attacks


def _id(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


@dataclass
class FoeExpectation:
    protect: float = 0.0
    switch: float = 0.0
    attacks: dict[str, float] = field(default_factory=dict)  # our species id -> chance it is attacked
    fake_out: dict[str, float] = field(default_factory=dict)  # our species id -> chance of that attack being Fake Out
    redirect: float = 0.0  # chance it uses Rage Powder / Follow Me to cover its partner


REDIRECTION = frozenset({"ragepowder", "followme"})
REDIRECT_CONFIDENCE = 0.5


def focus_target(threats: dict, ours_hp: dict[str, float]) -> tuple[str | None, str]:
    """The one of ours their two actives KO together (and neither alone), or one
    either KOs alone, with why: who an opponent double-targets. ``threats`` is
    foe -> our species -> hit; ``ours_hp`` our species id -> HP fraction."""
    best, best_margin, why = None, 0.0, ""
    for ours_name in {o for row in threats.values() for o in row}:
        hp = 100 * ours_hp.get(_id(ours_name), 1.0)
        hits = sorted((row[ours_name]["expected_pct"] for row in threats.values() if ours_name in row), reverse=True)
        if not hits or hp <= 0:
            continue
        together = sum(hits[:2])
        if together >= hp and together / hp > best_margin:
            best, best_margin = _id(ours_name), together / hp
            why = (f"either KOs {ours_name} alone" if hits[0] >= hp
                   else f"together they KO {ours_name} ({hits[0]:.0f}% + {hits[1]:.0f}% vs {hp:.0f}% HP)")
    return best, why


def code_guess(observation: dict, calc, tracker, table, protect_odds: dict[str, dict],
               in_range=frozenset()) -> dict[str, dict]:
    """foe species id -> {"action", "target", "confidence", "why"}.

    ``in_range``: their actives one of our attacks likely KOs this turn - a
    partner with Rage Powder / Follow Me covers them."""
    from .protect import LIKELY

    threats = calc.threats()  # foe name -> {our name: hit}
    ours_hp = {_id(e.get("species") or e.get("name")): _hp_of(e) for e in observation.get("active_pokemon") or []
               if isinstance(e, dict)}
    focus, focus_why = focus_target(threats, ours_hp)
    actives = [_id(e.get("species") or e.get("name")) for e in observation.get("opponent_active_pokemon") or []
               if isinstance(e, dict)]
    first_turn = tracker.opponent_first_turn()
    protected = tracker.protected_last_turn
    guesses = {}
    for entry in observation.get("opponent_active_pokemon") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("species") or entry.get("name")
        foe = _id(name)
        if not foe:
            continue
        on_us = threats.get(name) or {}
        target = max(on_us, key=lambda ours: on_us[ours]["expected_pct"]) if on_us else None
        if focus and any(_id(o) == focus for o in on_us):
            target = next(o for o in on_us if _id(o) == focus)  # focus fire: both go for the KO
        moves = {_id(m) for m in table.common_moves(name)}
        partner = next((a for a in actives if a != foe), None)
        chance = float((protect_odds.get(foe) or {}).get("likelihood") or 0.0)
        if str(entry.get("status") or "").lower().split(".")[-1] in ("slp", "sleep"):
            guess = {"action": "attack", "target": target,
                     "why": "asleep: it acts only if it wakes (about 1 turn in 3)"}
        elif first_turn.get(foe) and "fakeout" in moves:
            guess = {"action": "fake_out", "target": target, "why": "first turn on the field with Fake Out"}
        elif chance >= LIKELY:
            guess = {"action": "protect", "target": None, "confidence": round(chance, 2),
                     "why": "; ".join((protect_odds.get(foe) or {}).get("why") or []) or "likely to Protect"}
        elif moves & REDIRECTION and partner in {_id(f) for f in in_range or ()}:
            guess = {"action": "support", "target": None, "move": "Rage Powder" if "ragepowder" in moves else "Follow Me",
                     "confidence": REDIRECT_CONFIDENCE, "why": f"redirects to cover {partner}, which we can KO"}
        else:
            reason = "protected last turn, so it attacks" if foe in protected else f"Protect unlikely ({chance:.0%})"
            aim = f"focus fire: {focus_why}" if focus and _id(target) == focus else f"its STAB hits {target} hardest"
            guess = {"action": "attack", "target": target, "why": f"{reason}; {aim}"}
        guess.setdefault("confidence", CODE_CONFIDENCE)
        guesses[foe] = guess
    return guesses


def _hp_of(entry: dict) -> float:
    value = entry.get("hp_fraction")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    current, maximum = entry.get("current_hp"), entry.get("max_hp")
    if isinstance(current, (int, float)) and isinstance(maximum, (int, float)) and maximum:
        return current / maximum
    return 1.0


def expectations(prediction: dict[str, dict], ours: list[str]) -> dict[str, FoeExpectation]:
    """Turn a per-foe prediction (model's or code's) into chances.

    The predicted action gets its confidence; the rest is spread as a
    generic attack on our actives, so a wrong read is never ignored.
    """
    ours = [_id(o) for o in ours if o]
    found = {}
    for foe, guess in prediction.items():
        confidence = min(1.0, max(0.0, float(guess.get("confidence") or CODE_CONFIDENCE)))
        rest = 1.0 - confidence
        expectation = FoeExpectation()
        action = guess.get("action")
        target = _id(guess.get("target"))
        if action == "protect":
            expectation.protect = confidence
        elif action == "support" and _id(guess.get("move")) in REDIRECTION:
            expectation.redirect = confidence
        elif action == "switch":
            expectation.switch = confidence
        elif action in ("attack", "fake_out") and target in ours:
            expectation.attacks[target] = confidence
            if action == "fake_out":
                expectation.fake_out[target] = confidence
        elif action in ("attack", "fake_out"):
            for o in ours:  # a spread attack, or a target we cannot name
                expectation.attacks[o] = expectation.attacks.get(o, 0.0) + confidence / max(1, len(ours))
                if action == "fake_out":
                    expectation.fake_out[o] = confidence / max(1, len(ours))
        for o in ours:
            expectation.attacks[o] = expectation.attacks.get(o, 0.0) + rest * ATTACK_SHARE_OF_REST / max(1, len(ours))
        found[foe] = expectation
    return found


READ_WEIGHT = 0.5  # how far the model's read moves us off the code's rule-based guess


def blend(read: dict[str, FoeExpectation], base: dict[str, FoeExpectation],
          weight: float = READ_WEIGHT) -> dict[str, FoeExpectation]:
    """Principles first, reads next: our line is scored against a mix of the
    model's read and the code's rule-based guess, so a confident-sounding read
    cannot erase a base rate. Match 286bbc2e, turn 8: the read said a
    first-turn Iron Valiant would attack (55%), which zeroed its 70% Protect
    odds; both our attacks went into its Protect."""
    found = {}
    for foe in set(read) | set(base):
        r, b = read.get(foe), base.get(foe)
        if r is None or b is None:
            found[foe] = r or b
            continue
        mix = lambda x, y: weight * x + (1 - weight) * y  # noqa: E731
        found[foe] = FoeExpectation(
            protect=mix(r.protect, b.protect),
            switch=mix(r.switch, b.switch),
            attacks={o: mix(r.attacks.get(o, 0.0), b.attacks.get(o, 0.0)) for o in set(r.attacks) | set(b.attacks)},
            fake_out={o: mix(r.fake_out.get(o, 0.0), b.fake_out.get(o, 0.0)) for o in set(r.fake_out) | set(b.fake_out)},
            redirect=mix(r.redirect, b.redirect),
        )
    return found


def focus(expected: dict[str, FoeExpectation], target: str | None, rate: float) -> None:
    """Their attackers double-target ``target`` (the one of ours they can KO
    together) at least ``rate`` of their attacks: that share of each foe's
    attacks on our other Pokemon moves onto it. ``rate`` is learned per
    opponent (style.focus_rate)."""
    if not target:
        return
    target = _id(target)
    for e in expected.values():
        others = {o: v for o, v in e.attacks.items() if o != target}
        moved = sum(v * rate for v in others.values())
        if moved <= 0:
            continue
        e.attacks = {o: (v * (1 - rate) if o != target else v) for o, v in e.attacks.items()}
        e.attacks[target] = e.attacks.get(target, 0.0) + moved


def escape(expected: dict[str, FoeExpectation], in_range, can_protect: dict[str, bool], rate: float,
           has_bench: bool) -> None:
    """A foe in our KO range escapes at least ``rate`` of the time: by Protect
    when it can, else by switching out (when they have a bench). The share
    comes off its attacks. Match b0e756ca: Basculegion Protected at 15%, then
    switched; Arcanine Protected at 16%, then switched to a Ghost - four of
    our finishing hits wasted while the read said they would attack."""
    for foe in in_range or ():
        e = expected.get(_id(foe))
        if e is None:
            continue
        mass = sum(e.attacks.values())
        if can_protect.get(_id(foe)):
            need, field = rate - e.protect, "protect"
        elif has_bench:
            need, field = rate - e.switch, "switch"
        else:
            continue
        if need <= 0 or mass <= 0:
            continue
        take = min(need, mass)
        keep = (mass - take) / mass
        e.attacks = {o: v * keep for o, v in e.attacks.items()}
        e.fake_out = {o: v * keep for o, v in e.fake_out.items()}
        setattr(e, field, getattr(e, field) + take)


def no_repeat_protect(expected: dict[str, FoeExpectation], protected: object) -> None:
    """A foe that just Protected will not Protect again (it works 1 time in 3):
    whatever the read said, that share becomes attacks on our actives."""
    for foe in protected or ():
        e = expected.get(_id(foe))
        if e is None or e.protect <= 0:
            continue
        share, e.protect = e.protect, 0.0
        targets = list(e.attacks)
        for o in targets:
            e.attacks[o] += share / len(targets)


def from_model(answer: dict, foes: list[str], ours: list[str]) -> dict[str, dict]:
    """The model's per-foe prediction, validated; raises ValueError if unusable."""
    rows = answer.get("foes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("foes must be a non-empty list")
    wanted = {_id(f) for f in foes}
    our_ids = {_id(o) for o in ours}
    found = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        foe = _id(row.get("species"))
        action = row.get("action")
        if foe not in wanted or action not in ACTIONS:
            continue
        target = _id(row.get("target"))
        found[foe] = {
            "action": action,
            "target": target if target in our_ids else None,
            "move": str(row.get("move") or ""),
            "confidence": row.get("confidence"),
            "why": str(row.get("why") or ""),
        }
    if not found:
        raise ValueError(f"no usable prediction for {sorted(wanted)}")
    return found
