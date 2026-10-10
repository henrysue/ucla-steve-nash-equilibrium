"""Draft scoring: one score per card, with its reasons.

    total = plan + roles + coverage + versus + partners + strength

- plan: the team plan we are drafting toward (archetypes.py: core > member >
  flex, and a core piece first while we hold none);
- roles: what the six still needs - attackers (urgently when the picks are
  running out), one speed-control user, one Fake Out - and less of what it
  has too much of (a third pure support, a third Fake Out);
- coverage: team weaknesses the card covers or stacks;
- versus: the opponent's drafted Pokemon its STAB hits super-effectively,
  is walled by, or is threatened by; the table's counter lists both ways;
  tools that answer their inferred playstyle; and a weather/terrain setter
  that would overwrite our own;
- partners: the table names it and a roster member as partners;
- strength: draft points, or base offense where the table leaves them blank,
  deliberately a modest term (usage lists are topped by supports).

The only hard rules left are legality-like: the species clause, and a
second restricted/Mega ranked last (only one Mega Evolves per battle).
"""

from __future__ import annotations

from . import archetypes
from .typechart import TYPES, effectiveness

TEAM_SIZE = 6
TARGET_ATTACKERS = 3
MAX_PURE_SUPPORTS = 2

PARTNER_WEIGHT = 2.0
COVER_WEIGHT = 1.0
HIT_WEIGHT = 0.75
WALL_WEIGHT = 0.75
THREAT_WEIGHT = 0.5
COUNTER_WEIGHT = 0.75
MIN_COVERAGE_POWER = 60  # weaker attacks do not count as coverage in a matchup
POWER_WEIGHT = 0.1  # 20 draft points = 2.0, one partner link
OFFENSE_FLOOR, OFFENSE_SPAN = 90, 20  # base offense 130 = 2.0 when points are blank


def team_shape(roles: list[dict]) -> dict:
    return {
        "attackers": sum(r["attacker"] for r in roles),
        "pure_supports": sum(r["support"] and not r["attacker"] for r in roles),
        "speed_control": sum(r["speed_control"] for r in roles),
        "fake_out": sum(r["fake_out"] for r in roles),
    }


def strength(species: str, table) -> float:
    points = table.draft_points(species)
    if points:
        return points * POWER_WEIGHT
    return max(0.0, min(2.0, ((table.offense(species) or 0) - OFFENSE_FLOOR) / OFFENSE_SPAN))


def versus(species: str, opponent: list[str], table) -> tuple[float, dict]:
    """Matchup with the opponent's roster: our attacks into their types, theirs
    into ours (each side's real damaging moves when its set is known - a draft
    card - else its STAB types), plus the counter lists. A 4x threat counts
    double: match 8e3a639c led Tyranitar into Iron Hands' Close Combat and
    Archaludon's Body Press, which a STAB-type check cannot see."""
    own = table.species_types(species)
    hits, walled_by, threatened_by, threats = [], [], [], {}
    weight = 0.0
    for foe in opponent:
        theirs = table.species_types(foe)
        if not own or not theirs:
            continue
        best = max(effectiveness(t, theirs) for t in _attack_types(species, own, table))
        if best > 1.0:
            hits.append(foe)
        elif best < 1.0:
            walled_by.append(foe)
        worst, how = max(((effectiveness(t, own), name) for t, name in _attacks(foe, theirs, table)), default=(1.0, ""))
        if worst > 1.0:
            threatened_by.append(foe)
            threats[foe] = f"{how} {worst:g}x"
            weight += 2.0 if worst >= 4 else 1.0
    notes = table.matchup_notes([species], opponent)
    checks = sorted(name for name, answers in notes["our_pokemon_that_check_theirs"].items() if species in answers)
    checked_by = sorted(notes["their_pokemon_that_check_ours"].get(species, []))
    total = (
        HIT_WEIGHT * len(hits) - WALL_WEIGHT * len(walled_by) - THREAT_WEIGHT * weight
        + COUNTER_WEIGHT * (len(checks) - len(checked_by))
    )
    return total, {
        "hits_super_effectively": hits, "walled_by": walled_by, "threatened_by": threatened_by,
        "threat_moves": threats, "checks": checks, "checked_by": checked_by,
    }


def _attacks(species: str, types: list[str], table) -> list[tuple[str, str]]:
    """(type, move name) for each damaging move in its set, else its STAB types."""
    from . import movelist

    found = []
    for name in table.common_moves(species):
        move = movelist.lookup(name)
        # Real attacks only: not Fake Out-sized chip; power 0 = variable (Low Kick, Body Press...).
        if move and move["category"] in ("PHYSICAL", "SPECIAL") and (not move["power"] or move["power"] >= MIN_COVERAGE_POWER):
            found.append((move["type"], move["name"]))
    return found or [(t.upper(), f"{t.title()} STAB") for t in types]


def _attack_types(species: str, types: list[str], table) -> list[str]:
    return [t for t, _ in _attacks(species, types, table)]


def _role_need(roles: dict, shape: dict, picked: int) -> tuple[float, list[str]]:
    value, why = 0.0, []
    attackers_needed = TARGET_ATTACKERS - shape["attackers"]
    picks_left = TEAM_SIZE - picked
    if roles["attacker"] and attackers_needed > 0:
        urgent = attackers_needed >= picks_left
        value += 4.0 if urgent else 1.5
        why.append("attacker the team must have" if urgent else "attacker")
    elif not roles["attacker"] and attackers_needed >= picks_left:
        value -= 4.0
        why.append("the last picks must be attackers")
    if roles["speed_control"] and not shape["speed_control"]:
        value += 2.0
        why.append("first speed control")
    if roles["fake_out"]:
        if not shape["fake_out"]:
            value += 1.0
            why.append("first Fake Out")
        elif shape["fake_out"] >= 2:
            value -= 1.0
            why.append("third Fake Out")
    if roles["support"] and not roles["attacker"]:
        if picked == 0:
            value -= 3.0
            why.append("a first pick should be a win condition, not a support")
        elif shape["pure_supports"] >= MAX_PURE_SUPPORTS:
            value -= 4.0
            why.append(f"already {shape['pure_supports']} pure supports")
        elif shape["pure_supports"] >= 1:
            value -= 1.0
            why.append("second pure support")
    return value, why


def _coverage(species: str, mine: list[str], table) -> tuple[float, dict]:
    shared = {}
    for attack in TYPES:
        count = sum(1 for name in mine if (t := table.species_types(name)) and effectiveness(attack, t) > 1.0)
        if count >= 2:
            shared[attack] = count
    own = table.species_types(species)
    covers = sorted(t for t in shared if own and effectiveness(t, own) < 1.0)
    stacks = sorted(t for t in shared if own and effectiveness(t, own) > 1.0)
    return COVER_WEIGHT * (len(covers) - len(stacks)), {
        "covers": [t.title() for t in covers], "stacks": [t.title() for t in stacks],
    }


def score(species: str, roles: dict, mine: list[str], mine_roles: list[dict], theirs: list[str],
          plan: dict | None, runner_up: dict | None, styles: dict, table) -> dict:
    """One card's fit: a total and the reasons for it."""
    plan_bonus, plan_slot = archetypes.card_bonus(species, plan, runner_up, table, mine)
    role_value, role_why = _role_need(roles, team_shape(mine_roles), len(mine))
    cover_value, cover_detail = _coverage(species, mine, table)
    type_value, type_detail = versus(species, theirs, table)
    counter_value, counter_why = archetypes.counter_bonus(species, styles, table)
    clash_value, clash_why = archetypes.field_clash(species, mine, plan, table)
    links = table.partner_notes([*mine, species])
    partners = sorted(
        set(links.get(species, [])) | {n for n, theirs_ in links.items() if species in theirs_ and n != species}
    )
    power = strength(species, table)
    total = (
        plan_bonus + role_value + cover_value + type_value + counter_value + clash_value
        + PARTNER_WEIGHT * len(partners) + power
    )
    return {
        "total": round(total, 2),
        "plan": {"bonus": plan_bonus, "slot": plan_slot},
        "roles": {"value": round(role_value, 2), "why": role_why},
        "coverage": {"value": cover_value, **cover_detail},
        "versus": {
            "value": round(type_value + counter_value + clash_value, 2),
            **type_detail,
            "answers_their_style": counter_why,
            **({"field_clash": clash_why} if clash_why else {}),
        },
        "partners": partners,
        "strength": round(power, 2),
    }
