"""The brain: picks the army's Mode each view. Pure rules, instant, testable.

Safety first (opening, base under attack, protect a lead, retreat), then
attack or mass by army edge. The model advisor may suggest attack or hold;
it never overrides the safety rules.
"""

from __future__ import annotations

from . import config as C
from . import scout, strength, terrain
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
    winning_trades: bool = False,
    traded_up: bool = False,
    retreated_at: float | None = None,
    retreat_army: float | None = None,
    hunting: bool = False,
    alarm: bool = False,
    distraction=None,
    attack_seconds: float | None = None,
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
    ``distraction``: (enemy ids, their value) our raid squads have drawn
    lately (raid.RaidCorps.drawing), or None.
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
    # (not straight back in after a retreat, unless hunting their last buildings:
    # b90e492f went again 51 s after one)
    just_retreated = retreated_at is not None and view.game_seconds - retreated_at < C.RETREAT_COOLDOWN_SECONDS
    if (hunting or ((_dominant(view) or beaten(view)) and not just_retreated)) and _has_real_army(view) \
            and army_odds(view) >= C.OVERWHELMING_ODDS:
        return Mode.ALL_IN

    # Safety rail 3: near the time limit and ahead -> protect the lead (but an
    # attack winning its trades keeps going: c7a982ac pulled back from a won fight).
    if _should_protect_lead(view) and not (prev_mode in AGGRESSIVE and winning_trades):
        return Mode.PROTECT_LEAD

    # Offensive decision: do we commit the deathball? Every fight is judged
    # the same way (strength.odds); their turrets at the target fight too
    # (attack_odds); the retreat rule below reads the armies.
    edge = army_odds(view)
    # The advisor's posture decides near each bar (advised: within
    # ADVISOR_BAND), not past it: it sees what the numbers miss (their army
    # out of position) but reads the same fog-blind figures (08560d9b: "a
    # strong army value lead" at 3900 vs a real 5000-6000). Never blind
    # (commander), never into turrets we can't break, never under parity.
    vote = vote_of(model_pref)
    # Through a choke they can hold (no way round): a column walks out of it
    # into their concave, which our fight model doesn't price — a higher bar,
    # unless the siege units come to shell it from out of its reach
    # (e2bc988e attacked through Sidestep's gap at 1.2x and lost 2050 for 200).
    threshold = attack_bar(view)
    choke = threshold / max(1e-9, _attack_threshold(view))
    strike = attack_odds(view)
    # Unseen enemy makes the edge infinite, so also require a real army: never
    # trickle units one at a time into the enemy base.
    cooling_down = retreated_at is not None and view.game_seconds - retreated_at < C.RETREAT_COOLDOWN_SECONDS
    behind_late = _catching_up(view)
    # No time to wait for the army to regrow when behind at the end.
    regrow_expired = retreated_at is not None and view.game_seconds - retreated_at >= C.RETREAT_REGROW_MAX_SECONDS
    regrown = (behind_late or regrow_expired or retreat_army is None
               or view.army_value >= C.REATTACK_ARMY_GROWTH * retreat_army)
    # A solid plan: build a real army first (PUSH_ARMY_VALUE), then push.
    built = view.army_value >= C.PUSH_ARMY_VALUE or behind_late
    # Decisive: having just traded clearly up (TRADE_LEAD_*), go now — their
    # army is down by what we killed — as long as the fight, their turrets at
    # the target included, is at least even; else the bar on what we know.
    # (The turrets are in the odds: match 654f1b2e attacked a mixed army into
    # three flame towers three times.)
    # (Its "hold" acts in one place, below, with the exceptions that override it.)
    # Trading up is no reason to go in: the counterattack after a won
    # defense went at about even and lost the army to their next push
    # (9cada749: in at 8:22 at 1.06 after trading 15250 for 10800, out 31 s
    # later; their 10:29 push took the war factory, refineries and depot).
    can_attack = (advised(strike, threshold, max(vote, 0), floor=C.ADVISOR_ATTACK_EDGE)
                  and _has_real_army(view) and built and not cooling_down and regrown)
    # A turtle waits for the siege: SIEGE_ATTACK_MIN siege units, or an army
    # that overwhelms army and turrets together (efefdca8: two attacks into
    # the ring, 8200 and 8650 lost) — not even when behind at the end.
    if (can_attack and prev_mode not in AGGRESSIVE and view.enemy_turtling
            and len(view.units_of(C.SIEGE_UNITS)) < C.SIEGE_ATTACK_MIN
            and strike < C.OVERWHELMING_ODDS):
        can_attack = False

    # Retreat: losing the fight we started -> pull back before we feed the
    # enemy kills_cost; hold the retreat until regrouped (or strong again).
    # The model can't override this.
    # Contact reveals their army and the estimate jumps: in the first seconds
    # of a fight only being clearly outmatched pulls back; then the trades
    # judge (match 5e840776: attacks retreated 6-30 s in, and were chased).
    floor = (C.RETREAT_EARLY_EDGE if attack_seconds is not None and attack_seconds < C.MIN_ENGAGEMENT_SECONDS
             else C.RETREAT_ARMY_EDGE)
    # An attack winning its trades keeps going: the estimate can dip on what
    # it sees, but we are trading up. Only being clearly outmatched stops it.
    if winning_trades:
        floor = min(floor, C.RETREAT_EARLY_EDGE)
    # A distraction attack is judged without the part of their army our raid
    # holds — the odds it started on (it retreated 15 s in otherwise).
    retreat_edge = edge
    if distraction:
        retreat_edge = max(edge, strength.army_odds(view, (), distraction[0], distraction[1]))
    # Near the line the advisor's posture decides (its "attack" fights on a
    # little below it, its "hold" pulls back a little above it); losing the
    # trades still ends it.
    if prev_mode in AGGRESSIVE and (not advised(retreat_edge, floor, vote) or losing_trades):
        return Mode.RETREAT
    # Committed: an attack under way goes on until the retreat rule calls it
    # off — not back to massing when the estimate dips under the attack bar
    # (match 82908a24 flipped attack/build nine times, every 25 s: the army
    # marched out and back). A model "hold" doesn't recall it either.
    if prev_mode in AGGRESSIVE:
        return Mode.ALL_IN
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
    # A real share of their army is busy with our raid squads: judged
    # without it, by the same bar — the main army goes at the rest now
    # (away from where the raid fights: _attack_target).
    busy_ids, busy_value = (distraction or (frozenset(), 0.0, None))[:2]
    if (busy_value >= max(C.DISTRACTION_MIN_VALUE, C.DISTRACTION_ARMY_SHARE * enemy_estimate(view))
            and not can_attack and not early and _has_real_army(view) and built and not cooling_down and regrown
            and attack_odds(view, busy_ids, busy_value) >= threshold):
        can_attack = True
    # A remote expansion of theirs we clearly beat: strike it, short of the
    # whole-army bar — they have to come out of their concave to save it
    # (d1615ac6: raids killed its harvesters for two minutes while the army,
    # 14850 against a believed 18900, sat at home).
    if (not can_attack and not early and built and not cooling_down and regrown and _has_real_army(view)
            and advised(edge, C.STRIKE_MIN_EDGE, max(vote, 0)) and expansion_strike(view, max(vote, 0)) is not None):
        can_attack = True
    # The turtle gate holds for every way in (a distraction or a strike set
    # it back on: the audit — the efefdca8 case the gate is there for).
    if (can_attack and prev_mode not in AGGRESSIVE and view.enemy_turtling
            and len(view.units_of(C.SIEGE_UNITS)) < C.SIEGE_ATTACK_MIN
            and strike < C.OVERWHELMING_ODDS):
        can_attack = False
    policy_choice = Mode.ALL_IN if can_attack else Mode.BUILD

    # The model may hold an attack back — but not when we clearly outnumber
    # them, or when behind at the end. It can't push one the numbers don't
    # support: its "attack" calls read the same fog-blind figures (08560d9b:
    # "a strong army value lead" at 3900 vs a real 5000-6000). It never
    # overrides the safety rails above.
    # Its "hold" stops an attack only near the bar (advised above): a clear
    # edge, behind at the end or overwhelming goes regardless.
    overwhelming = enemy_estimate(view) > 0 and edge >= C.OVERWHELMING_ODDS
    if (model_pref is Mode.BUILD and policy_choice is Mode.ALL_IN
            and not (overwhelming or behind_late
                     or (enemy_estimate(view) > 0 and strike >= threshold * (1 + C.ADVISOR_BAND)))):
        return Mode.BUILD

    return policy_choice


def advised(value: float, bar: float, vote: int, floor: float = 0.0) -> bool:
    """Whether an action whose number is ``value`` against the rules' ``bar``
    goes, with the advisor's ``vote`` (+1 its "attack", -1 its "hold", 0
    none) deciding only near the bar: within ``ADVISOR_BAND`` below it a yes
    fires it (never under ``floor``), within ``ADVISOR_BAND`` above it a no
    stops it; beyond the band the numbers decide. Bars are ratios (odds),
    so the band scales with any army and any map."""
    if vote > 0 and value >= max(floor, bar * (1 - C.ADVISOR_BAND)):
        return True
    if vote < 0 and value < bar * (1 + C.ADVISOR_BAND):
        return False
    return value >= bar


def vote_of(model_pref) -> int:
    """The advisor's posture as a vote: +1 attack, -1 hold, 0 none."""
    return 1 if model_pref is Mode.ALL_IN else -1 if model_pref is Mode.BUILD else 0


def attack_bar(view: View) -> float:
    """The odds an attack needs (:func:`_attack_threshold`), times
    ``CHOKE_ATTACK_FACTOR`` through a choke with no way round unless the
    siege comes."""
    choke = (C.CHOKE_ATTACK_FACTOR if terrain.choke_between(view)
             and len(view.units_of(C.SIEGE_UNITS)) < C.SIEGE_ATTACK_MIN and not _overwhelming(view) else 1.0)
    return _attack_threshold(view) * choke


def expansion_strike(view: View, vote: int = 0) -> tuple[int, int] | None:
    """The remote expansion of theirs (raid.remote_expansions) nearest our
    base that our whole army beats by ``STRIKE_LOCAL_ODDS`` — against the
    turrets and the units we know there, not their whole army — or None.
    The attack's target is the same one (executor._attack_target: a known
    expansion first, nearest our base)."""
    from .raid import remote_expansions

    if view.base_center is None or not view.combat_units:
        return None
    ours = [u.type for u in view.combat_units]
    r_t = (C.EXPAND_CLUSTER_CELLS + C.RAID_TURRET_CLEAR_CELLS) ** 2
    r_u = C.EXPAND_CLUSTER_CELLS ** 2
    for cell in sorted(remote_expansions(view), key=lambda c: dist2(c, view.base_center)):
        there = [a.type for a in list(view.enemy_buildings) + list(view.known_enemy_buildings)
                 if a.type in C.TURRETS and a.cell is not None and not a.wreck and dist2(a.cell, cell) <= r_t]
        there += [a.type for a in list(view.enemy_units) + list(view.known_enemy_units)
                  if a.type not in C.HARVESTER and a.type != C.MCV and a.cell is not None and not a.wreck
                  and dist2(a.cell, cell) <= r_u]
        if not there or advised(strength.odds(view, ours, there), C.STRIKE_LOCAL_ODDS, vote):
            return tuple(cell)
        return None                               # the nearest is too strong: not further afield
    return None


def _defend(view: View) -> Mode:
    """A rush in the opening (spam cheap units); later, defend the base
    without dropping the economy."""
    return Mode.DEFEND_RUSH if view.game_seconds < C.CHEESE_WINDOW_SECONDS else Mode.DEFEND_BASE


def enemy_estimate(view: View) -> float:
    """Their army's value as we believe it: the most we see or remember
    now, or saw lately, or what they could have built since we last saw it
    (``enemy_army_peak``: agent.py), less half our kills since. Units only.
    Decisions add trades (traded_up) and a scout's look before a blind
    attack (blind_estimate, which reads only what was actually seen)."""
    return max(view.enemy_army_value, view.enemy_army_peak, 0.0)


def behind_on_army(view: View) -> bool:
    """Past the opening, their army beats ours by ``BEHIND_ARMY_RATIO``
    (:func:`army_odds` at most its inverse): units come before economy and
    tech. Match 52a77fb1 banked 1200-3800 for 4 minutes, held for an MCV, a
    second war factory and refineries, while 2-3x behind."""
    if view.game_seconds < C.BEHIND_AFTER_SECONDS:
        return False
    return army_odds(view) <= 1.0 / C.BEHIND_ARMY_RATIO


def blind_estimate(view: View) -> bool:
    """We know too little of their army to commit: no fresh look at their
    base and less than ``BLIND_SEEN_SHARE`` x our army seen lately (match
    efefdca8 attacked 13150 with 700 seen; ee2ac21b attacked 5150 into a
    guess of 4440 and met 13500). A scout looks first (commander)."""
    if view.enemy_base_scouted:
        return False
    look = max(view.enemy_army_value, view.enemy_seen_recent, 0.0)
    return look < C.BLIND_SEEN_SHARE * view.army_value


def army_odds(view: View) -> float:
    """Our army against theirs (strength.army_odds: what we see or remember
    and the most we saw lately). Large when nothing is known of them."""
    return strength.army_odds(view)


def turrets_at_target(view: View) -> list[str]:
    """The enemy turrets we know within ``DEFENSE_CLUSTER_CELLS`` of our
    attack target, by type."""
    from .executor import route_goal

    target = route_goal(view)
    if target is None:
        return []
    r2 = C.DEFENSE_CLUSTER_CELLS ** 2
    known = {(a.id if a.id is not None else a.cell): a
             for a in list(view.enemy_buildings) + list(view.known_enemy_buildings)
             if a.type in C.TURRETS and a.cell is not None and not a.wreck
             and dist2(a.cell, target) <= r2}
    return [a.type for a in known.values()]


def attack_odds(view: View, busy_ids=frozenset(), busy_value: float = 0.0) -> float:
    """The odds of an attack: our army against theirs plus their turrets at
    the target, which fight too (match efefdca8: two attacks into a turtle's
    ring the army odds called even). ``busy_*``: their units held by our
    raid squads, left out."""
    return strength.army_odds(view, turrets_at_target(view), busy_ids, busy_value)


def _attack_threshold(view: View) -> float:
    """Edge required to commit. Lower late (anti-passivity: being ahead at the
    tiebreak requires acting), higher early (don't throw away the deathball)."""
    if _catching_up(view):
        return C.CATCH_UP_ARMY_EDGE      # behind at the end: a coin flip beats a sure loss
    if view.army_value >= C.BIG_ARMY_VALUE:
        return C.BIG_ARMY_EDGE           # a maxed army idling wastes its edge; production refills it
    if view.game_seconds >= C.ESCALATE_AFTER_SECONDS:
        return C.ESCALATED_ARMY_EDGE
    return C.ATTACK_ARMY_EDGE


def _regrouped(view: View, scout_id: int | None, squad_ids: frozenset = frozenset()) -> bool:
    """Most of the army is back at the main base's line (or there is
    nowhere to regroup / nothing left to regroup)."""
    stage = scout.home_line(view)        # where a retreat goes (executor._retreat)
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
    return strength.value(view, [u.type for u in view.combat_units])


def enemy_unit_value(view: View) -> float:
    """The enemy combat units we see or remember (no buildings, harvesters or MCV)."""
    return strength.value(view, strength.enemy_kinds(view))


def outgunned(view: View) -> bool:
    """The enemy fighters we see or remember (no turrets, buildings or
    harvesters) are worth ``OUTGUNNED_MIN_VALUE`` and beat ours
    (strength.odds under 1). Then units come before an MCV or another
    refinery (match e2f73d65: a fourth MCV and three refineries with 7
    fighters left)."""
    theirs = strength.enemy_kinds(view)
    return (strength.value(view, theirs) >= C.OUTGUNNED_MIN_VALUE
            and strength.odds(view, [u.type for u in view.combat_units], theirs) < 1.0)


def must_attack(view: View) -> bool:
    """Situations where nothing (an MCV escort included) should hold the army
    back: behind on trades at the end, or clearly overwhelming."""
    return _catching_up(view) or _overwhelming(view)


def beaten(view: View) -> bool:
    """Nothing left that can fight us: overwhelming (``_overwhelming``) and
    ``FINISH_MIN_LEAD`` up on trades. Then finish them, and a flank is safe.
    (A lead of 600 at 4:05 sent 18 units in "to finish": match b90e492f.)"""
    return _overwhelming(view) and view.kills_cost - view.deaths_cost >= C.FINISH_MIN_LEAD


def _overwhelming(view: View) -> bool:
    if (enemy_estimate(view) <= 0 and not view.enemy_base_scouted
            and view.kills_cost - view.deaths_cost < C.UNSEEN_LEAD_MIN):
        return False                   # nothing seen is not nothing there — unless we out-traded them by a lot
    return _overwhelming_known(view)


def _overwhelming_known(view: View) -> bool:
    """Our army's odds are at least ``OVERWHELMING_ODDS`` while ahead on
    trades (match 8db23b32 protected a lead at 19:24 against no army and
    four buildings instead of finishing)."""
    return (view.army_value >= C.MIN_ATTACK_ARMY_VALUE and view.kills_cost >= view.deaths_cost
            and army_odds(view) >= C.OVERWHELMING_ODDS)


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
