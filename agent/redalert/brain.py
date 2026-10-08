"""The high-level brain: choose the Mode. Deterministic core, optional model.

The brain decides *what posture* to play; the executor plays it fast. Its core
is a deterministic policy over the :class:`View` — correct, instant, and the
sole decider when no model is configured or the model is unavailable. An
optional LLM can be consulted for the genuinely judgment-heavy call (commit the
deathball or not) but it only ever picks among the SAME modes the policy would,
and the policy's safety rails (cheese, protect-lead) are never overridden by
the model. This is what makes the agent both "thinking" and robust: a model
problem can only cost a slightly-less-timely attack, never a stall or a bad
safety decision.

Throttling and the model call live in ``agent.py``; this module is pure policy
so it is trivially testable.
"""

from __future__ import annotations

from . import config as C
from . import scout
from .modes import Mode
from .observe import View


def choose_mode(view: View, *, cheese_since: float | None, model_pref: Mode | None = None) -> Mode:
    """The posture for this view.

    ``cheese_since`` is the game time cheese was first detected and still
    active (``None`` if not), so the caller can expire it. ``model_pref`` is an
    optional model suggestion, honored ONLY when it is a non-safety offensive
    choice the policy also considers reasonable.
    """
    if not view.ok:
        return Mode.OPEN

    # Safety rail 1: still opening (no construction yard / no barracks yet).
    if not view.has_building(C.CONSTRUCTION_YARD) or not view.has_building(C.BARRACKS):
        # Even while opening, an active cheese read forces defense.
        return Mode.DEFEND_RUSH if cheese_since is not None else Mode.OPEN

    # Safety rail 2: active cheese -> defend. Non-overridable by the model.
    if cheese_since is not None:
        return Mode.DEFEND_RUSH

    # Safety rail 3: near the time limit and ahead -> protect the lead.
    if _should_protect_lead(view):
        return Mode.PROTECT_LEAD

    # Offensive decision: do we commit the deathball?
    edge = _army_edge(view)
    threshold = _attack_threshold(view)
    policy_choice = Mode.ALL_IN if edge >= threshold else Mode.BUILD

    # A model may upgrade BUILD->ALL_IN or suggest HARASS, but only when we at
    # least have an army; it can never pull us out of a safety rail (handled
    # above) nor push an all-in with a token force.
    if model_pref in (Mode.ALL_IN, Mode.HARASS) and _has_real_army(view):
        if model_pref is Mode.ALL_IN and edge < 1.0:
            return Mode.BUILD  # don't let the model all-in while behind
        return model_pref

    return policy_choice


def _army_edge(view: View) -> float:
    """Our army value over the enemy value we can see. Large when enemy unseen
    (we assume we can push when nothing opposes us), small when outmatched."""
    ours = view.army_value
    theirs = view.enemy_army_value
    if theirs <= 0:
        return float("inf") if ours > 0 else 0.0
    return ours / theirs


def _attack_threshold(view: View) -> float:
    """Edge required to commit. Lower late (anti-passivity: being ahead at the
    tiebreak requires acting), higher early (don't throw away the deathball)."""
    if view.game_seconds >= C.ESCALATE_AFTER_SECONDS:
        return C.ESCALATED_ARMY_EDGE
    return C.ATTACK_ARMY_EDGE


def _has_real_army(view: View) -> bool:
    return view.army_value >= C.MIN_ATTACK_ARMY_VALUE or len(view.combat_units) >= 6


def _should_protect_lead(view: View) -> bool:
    if view.remaining_seconds is None:
        return False
    if view.remaining_seconds > C.PROTECT_LEAD_REMAINING_SECONDS:
        return False
    # Ahead on the tiebreak: positive trade differential, or more assets.
    trade = view.kills_cost - view.deaths_cost
    return trade > 0 or view.assets_value > 0 and view.enemy_army_value < view.army_value


def update_cheese_since(view: View, cheese_since: float | None) -> float | None:
    """Track the cheese window: set it when newly detected, keep it while the
    threat persists, and EXPIRE it after ``CHEESE_EXPIRE_SECONDS`` so a false
    alarm cannot cost us tempo for the rest of the game (strategy hole #3)."""
    verdict = scout.detect_cheese(view)
    if verdict is not None:
        return view.game_seconds if cheese_since is None else cheese_since
    if cheese_since is None:
        return None
    if view.game_seconds - cheese_since >= C.CHEESE_EXPIRE_SECONDS:
        return None  # threat never materialized / was repelled: back to normal
    return cheese_since
