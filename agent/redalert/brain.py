"""The brain: picks the army's Mode each view. Pure rules, instant, testable.

Safety first (opening, base under attack, protect a lead, retreat), then
attack or mass by army edge. The model advisor may suggest attack or hold;
it never overrides the safety rules.
"""

from __future__ import annotations

from . import config as C
from . import scout
from .modes import AGGRESSIVE, Mode
from .observe import View, dist2


def choose_mode(
    view: View,
    *,
    cheese_since: float | None,
    model_pref: Mode | None = None,
    prev_mode: Mode | None = None,
    scout_id: int | None = None,
    squad_ids: frozenset = frozenset(),
    losing_trades: bool = False,
    retreated_at: float | None = None,
    retreat_army: float | None = None,
    hunting: bool = False,
    alarm: bool = False,
    distraction: bool = False,
) -> Mode:
    """The posture for this view.

    ``cheese_since`` is the game time cheese was first detected and still
    active (``None`` if not), so the caller can expire it. ``model_pref`` is an
    optional model suggestion, honored ONLY when it is a non-safety offensive
    choice the policy also considers reasonable. ``prev_mode`` is last view's
    mode (for retreat), ``scout_id`` the roaming scout (not part of the army),
    ``losing_trades`` whether the recent exchange is going badly (agent.py
    measures it), ``retreated_at`` the game time of the last retreat and
    ``retreat_army`` our army value then (no new attack until it has grown).
    ``hunting``: their base is razed (scout.Hunter.hunting). ``alarm``: our
    scout was killed forward early (an early push may be coming).
    ``distraction``: our raid squad has drawn enemy fighters (raid.py).
    """
    if not view.ok:
        return Mode.OPEN

    # Safety rail 1: still opening (no construction yard / no barracks yet).
    if not view.has_building(C.CONSTRUCTION_YARD) or not view.has_building(C.BARRACKS):
        # Even while opening, an active cheese read forces defense.
        return _defend(view) if cheese_since is not None else Mode.OPEN

    # Safety rail 2: enemies at our base -> defend. Non-overridable by the model.
    if cheese_since is not None:
        return _defend(view)

    # Safety rail 2b: an early push is coming (the scout saw it massing):
    # hold at home, don't walk into it; pull back if already out.
    if scout.early_push_warning(view) or alarm:
        pulling_back = prev_mode in AGGRESSIVE or (prev_mode is Mode.RETREAT and not _regrouped(view, scout_id, squad_ids))
        return Mode.RETREAT if pulling_back else Mode.BUILD

    # Ruthless finish: their base is razed, or we dominate the trades, and
    # nothing we know of can stand up to our army -> destroy every building
    # for the win. No protect-lead, no model hold (matches 0561274e and
    # 654f1b2e went to the time limit this way, the latter 59k to 26k with
    # the enemy down to a 400 army and no harvesters).
    if (hunting or _dominant(view)) and _has_real_army(view) \
            and view.army_value >= C.MODEL_HOLD_MAX_EDGE * enemy_estimate(view):
        return Mode.ALL_IN

    # Safety rail 3: near the time limit and ahead -> protect the lead.
    if _should_protect_lead(view):
        return Mode.PROTECT_LEAD

    # Offensive decision: do we commit the deathball?
    edge = _army_edge(view)
    threshold = _attack_threshold(view)
    # Unseen enemy makes the edge infinite, so also require a real army: never
    # trickle units one at a time into the enemy base.
    cooling_down = retreated_at is not None and view.game_seconds - retreated_at < C.RETREAT_COOLDOWN_SECONDS
    behind_late = _catching_up(view)
    # No time to wait for the army to regrow when behind at the end.
    regrow_expired = retreated_at is not None and view.game_seconds - retreated_at >= C.RETREAT_REGROW_MAX_SECONDS
    regrown = (behind_late or regrow_expired or retreat_army is None
               or view.army_value >= C.REATTACK_ARMY_GROWTH * retreat_army)
    can_attack = edge >= threshold and _has_real_army(view) and not cooling_down and regrown
    # Into turrets only with the right army: a big infantry mass (kills them
    # fast) or tanks/artillery. (Match 654f1b2e: three attacks with a mixed
    # army into three flame towers, three losing trades and retreats.) An
    # attack already under way is left to the retreat rule.
    if can_attack and prev_mode not in AGGRESSIVE and not behind_late and not can_break_defenses(view):
        can_attack = False

    # Retreat: losing the fight we started -> pull back before we feed the
    # enemy kills_cost; hold the retreat until regrouped (or strong again).
    # The model can't override this.
    if prev_mode in AGGRESSIVE and (edge < C.RETREAT_ARMY_EDGE or losing_trades):
        return Mode.RETREAT
    if prev_mode is Mode.RETREAT and not can_attack:
        return Mode.BUILD if _regrouped(view, scout_id, squad_ids) else Mode.RETREAT

    # Early (before EARLY_ATTACK_SECONDS and a war factory) the opening army
    # stays home: it guards the harvesters while raids harass. What we see of
    # the enemy this early is a fraction of it, so no edge justifies the trade
    # (match b7fcbd4e: both seats attacked at 1:05 with 11 riflemen, "4x" what
    # they saw; the loser had 2 harvesters until 4:00).
    early = view.game_seconds < C.EARLY_ATTACK_SECONDS and not view.has_building(C.WAR_FACTORY)
    if early:
        can_attack = False
    # Their army is busy with our raid squad and our units outweigh theirs
    # (units only, no buildings): the main army goes down the centre now.
    if (distraction and not can_attack and not early and _has_real_army(view) and not cooling_down
            and our_unit_value(view) > enemy_unit_value(view) and (behind_late or can_break_defenses(view))):
        can_attack = True
    policy_choice = Mode.ALL_IN if can_attack else Mode.BUILD

    # The model may hold an attack back — but not when we clearly outnumber
    # them, or when behind at the end. It can't push one the numbers don't
    # support: its "attack" calls read the same fog-blind figures (08560d9b:
    # "a strong army value lead" at 3900 vs a real 5000-6000). It never
    # overrides the safety rails above.
    overwhelming = view.army_value >= C.MODEL_HOLD_MAX_EDGE * enemy_estimate(view) > 0
    if model_pref is Mode.BUILD and not ((overwhelming or behind_late) and can_attack):
        return Mode.BUILD

    return policy_choice


def _defend(view: View) -> Mode:
    """A rush in the opening (spam cheap units); later, defend the base
    without dropping the economy."""
    return Mode.DEFEND_RUSH if view.game_seconds < C.CHEESE_WINDOW_SECONDS else Mode.DEFEND_BASE


def enemy_estimate(view: View) -> float:
    """The enemy army through the fog: the most of what we see or remember,
    what we saw of it lately less our kills since (``enemy_army_peak``), and
    — both sides start equal on similar incomes — ``ENEMY_PRIOR_SHARE`` x our
    army adjusted by the trades (their losses are our kills). Match 08560d9b
    attacked with 3900 "against 1400"; they had 5000-6000, and showed 4000
    fifteen seconds before."""
    prior = C.ENEMY_PRIOR_SHARE * (view.army_value + view.deaths_cost - view.kills_cost)
    return max(view.enemy_army_value, view.enemy_army_peak, prior, 0.0)


def _army_edge(view: View) -> float:
    """Our army value over the estimated enemy army (:func:`enemy_estimate`).
    Large when nothing is known of them, small when outmatched."""
    ours = view.army_value
    theirs = enemy_estimate(view)
    if theirs <= 0:
        return float("inf") if ours > 0 else 0.0
    return ours / theirs


def _attack_threshold(view: View) -> float:
    """Edge required to commit. Lower late (anti-passivity: being ahead at the
    tiebreak requires acting), higher early (don't throw away the deathball)."""
    if view.game_seconds >= C.ESCALATE_AFTER_SECONDS:
        return C.ESCALATED_ARMY_EDGE
    return C.ATTACK_ARMY_EDGE


def _regrouped(view: View, scout_id: int | None, squad_ids: frozenset = frozenset()) -> bool:
    """Most of the army is back near the staging point (or there is nowhere
    to regroup / nothing left to regroup)."""
    stage = scout.choke_cell(view)
    army = [u for u in view.combat_units if u.id != scout_id and u.id not in squad_ids]   # scouts, raid, guards out
    if stage is None or not army:
        return True
    r2 = C.REGROUP_RADIUS_CELLS ** 2
    home = sum(1 for u in army if dist2(u.cell, stage) <= r2)
    return home >= C.REGROUPED_SHARE * len(army)


def _has_real_army(view: View) -> bool:
    """Enough value to attack with. By value only: six riflemen (600) used to
    count, and walked into the enemy one minute in."""
    return view.army_value >= C.MIN_ATTACK_ARMY_VALUE


def _should_protect_lead(view: View) -> bool:
    """Final 2 minutes with a lead of 1500+, or the final minute with any lead.

    The lead is the tiebreak's FIRST figure only, kills_cost - deaths_cost. (It
    used to also pass on "our army > the enemy army we can see", which fog
    makes small: match f9f06e34's losing seat sat out the last 3 minutes 3100
    behind on trades and lost by exactly that.)
    """
    if view.remaining_seconds is None or view.remaining_seconds > C.PROTECT_LEAD_REMAINING_SECONDS:
        return False
    if _overwhelming(view):
        return False   # they can't hurt us: keep killing (match 4bc1ee71 stopped on a beaten foe)
    lead = view.kills_cost - view.deaths_cost
    if view.remaining_seconds <= C.PROTECT_LEAD_LAST_SECONDS:
        return lead > 0
    return lead >= C.PROTECT_LEAD_MIN_MARGIN


def _dominant(view: View) -> bool:
    """Kills at least ``FINISH_KILL_RATIO`` x losses and ahead by
    ``FINISH_MIN_LEAD``: the game is ours to finish."""
    return (view.kills_cost >= C.FINISH_KILL_RATIO * view.deaths_cost
            and view.kills_cost - view.deaths_cost >= C.FINISH_MIN_LEAD)


def our_unit_value(view: View) -> float:
    """Our combat units' value (no buildings)."""
    return sum(view.unit_costs.get(u.type, C.DEFAULT_UNIT_COST) for u in view.combat_units)


def enemy_unit_value(view: View) -> float:
    """The enemy combat units we see or remember (no buildings, harvesters or MCV)."""
    skip = set(C.HARVESTER) | {C.MCV}
    seen = {a.id: a for a in list(view.known_enemy_units) + list(view.enemy_units)
            if a.id is not None and not a.wreck and a.type not in skip}
    return sum(view.unit_costs.get(a.type, C.DEFAULT_UNIT_COST) for a in seen.values())


def outgunned(view: View) -> bool:
    """The enemy units we know of are worth ``OUTGUNNED_MIN_VALUE`` and more
    than ours: units come before an MCV or another refinery (match e2f73d65:
    red bought a fourth MCV and three refineries with 7 fighters left, and
    blue's 25 walked through its base)."""
    enemy = enemy_unit_value(view)
    return enemy >= C.OUTGUNNED_MIN_VALUE and our_unit_value(view) < enemy


def can_break_defenses(view: View) -> bool:
    """No ground turrets guard the attack target, or our army can take them:
    ``INFANTRY_PER_DEFENSE`` infantry per turret (``INFANTRY_MASS_MIN`` at
    least), or ``ARMOR_PER_DEFENSE`` tanks per turret (siege units double)."""
    from .executor import _attack_target

    target = _attack_target(view)
    if target is None:
        return True
    r2 = C.DEFENSE_CLUSTER_CELLS ** 2
    known = {(a.id if a.id is not None else a.cell): a
             for a in list(view.enemy_buildings) + list(view.known_enemy_buildings)
             if a.type in C.GROUND_DEFENSE_TYPES and a.cell is not None and not a.wreck
             and dist2(a.cell, target) <= r2}
    n = len(known)
    if n == 0:
        return True
    infantry = sum(1 for u in view.combat_units if u.type not in C.VEHICLE_TYPES)
    armor = sum(2 if u.type in C.SIEGE_UNITS else 1 for u in view.combat_units
                if u.type in C.VEHICLE_TYPES and u.type not in ("jeep", "apc"))
    return (infantry >= max(C.INFANTRY_MASS_MIN, C.INFANTRY_PER_DEFENSE * n)
            or armor >= C.ARMOR_PER_DEFENSE * n)


def must_attack(view: View) -> bool:
    """Situations where nothing (an MCV escort included) should hold the army
    back: behind on trades at the end, or clearly overwhelming."""
    return _catching_up(view) or _overwhelming(view)


def _overwhelming(view: View) -> bool:
    """Our army is at least MODEL_HOLD_MAX_EDGE x a KNOWN enemy army (an
    unseen enemy is not a beaten one: fog hides armies)."""
    theirs = enemy_estimate(view)
    return (theirs > 0 and view.army_value >= C.MIN_ATTACK_ARMY_VALUE
            and view.army_value >= C.MODEL_HOLD_MAX_EDGE * theirs)


def _catching_up(view: View) -> bool:
    """Final window and level or behind on kills - deaths: the tiebreak is
    lost unless we take kills, so we can't afford to sit."""
    return (view.remaining_seconds is not None
            and view.remaining_seconds <= C.CATCH_UP_REMAINING_SECONDS
            and view.kills_cost - view.deaths_cost <= 0)


def update_cheese_since(view: View, cheese_since: float | None) -> float | None:
    """Track the threat window: the game time the threat was LAST seen,
    expiring ``CHEESE_EXPIRE_SECONDS`` after that, so a false alarm cannot cost
    us tempo for the rest of the game (strategy hole #3). Keyed on the last
    sighting, not the first: a long fight must not flip to attack the moment
    one view shows no enemy, then straight back (match e4caa28c flipped
    defend/attack within 2 s three times)."""
    verdict = scout.detect_cheese(view)
    if verdict is not None:
        return view.game_seconds
    if cheese_since is None:
        return None
    expire = C.CHEESE_EXPIRE_SECONDS if view.game_seconds < C.CHEESE_WINDOW_SECONDS else C.BASE_DEFEND_EXPIRE_SECONDS
    if view.game_seconds - cheese_since >= expire:
        return None  # threat never materialized / was repelled: back to normal
    return cheese_since
