"""Step 2 of move selection: what each opposing Pokemon will most likely do.

The code makes a first guess per foe, with its reasons:

- it carries Protect and did NOT Protect last turn -> it will probably Protect
  (a repeat only works 1 time in 3, so after a Protect it attacks);
- on its first turn with Fake Out -> Fake Out on whichever of ours it fears most;
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


STYLE_PROTECT_AT = 0.45  # below this style-blended chance, the code expects an attack instead


def code_guess(observation: dict, calc, tracker, table, protect_odds: dict[str, dict]) -> dict[str, dict]:
    """foe species id -> {"action", "target", "confidence", "why"}."""
    threats = calc.threats()  # foe name -> {our name: hit}
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
        carries = _carries_protect(entry, name, table, protect_odds.get(foe))
        moves = {_id(m) for m in table.common_moves(name)}
        styled = (protect_odds.get(foe) or {}).get("style_p")  # set once the opponent's style is read
        if carries and foe not in protected and styled is not None and styled < STYLE_PROTECT_AT:
            guess = {"action": "attack", "target": target,
                     "why": f"carries Protect, but this opponent holds it when it looks obvious ({styled:.0%})"}
        elif carries and foe not in protected:
            guess = {"action": "protect", "target": None, "why": "carries Protect and did not Protect last turn"}
        elif first_turn.get(foe) and "fakeout" in moves:
            guess = {"action": "fake_out", "target": target, "why": "first turn on the field with Fake Out"}
        else:
            reason = "protected last turn, so it attacks" if foe in protected else "no Protect expected"
            guess = {"action": "attack", "target": target, "why": f"{reason}; its STAB hits {target} hardest"}
        guess["confidence"] = round(styled, 2) if styled is not None and guess["action"] == "protect" else CODE_CONFIDENCE
        guesses[foe] = guess
    return guesses


def _carries_protect(entry: dict, name: str, table, odds: dict | None) -> bool:
    revealed = entry.get("revealed_moves") or entry.get("known_moves") or entry.get("moves")
    if isinstance(revealed, list) and revealed:
        ids = {_id(m.get("id") or m.get("name") if isinstance(m, dict) else m) for m in revealed}
        if ids & PROTECT_CHAIN:
            return True
        if len(ids) >= 4:
            return False
    common = {_id(m) for m in table.common_moves(name)}
    if common:
        return bool(common & PROTECT_CHAIN)
    # No usage data: most VGC Pokemon carry Protect; trust protect.py's read.
    return bool(odds and odds.get("likelihood", 0) >= 0.3)


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
