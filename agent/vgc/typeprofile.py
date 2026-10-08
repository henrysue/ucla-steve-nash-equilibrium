"""Deterministic opponent type-profile: strategy hints from team typing.

We cannot see the opponent's moves, items, or EVs, but we DO know the types
of every Pokemon they have revealed (from the live observation, backed by the
format table). Those types alone carry a lot of strategic signal, and it is
pure, fast arithmetic to extract — so this module computes it once in code
rather than making the model re-derive it every turn:

- **Shared weaknesses**: attacking types that hit SEVERAL of the opponent's
  revealed Pokemon super-effectively. These are the team's structural soft
  spots — the types our own attackers most want to click.
- **Offensive STAB coverage**: the set of types the opponent can hit us with
  for same-type-attack bonus. A proxy for their damage, since the types they
  carry constrain what their moves can be.
- **Common resistances/immunities**: types that most of their team shrugs off,
  so we do not lean a game plan on a move half their side resists.

Directionality matters and is respected: a "weakness of X" means an ATTACK of
some type lands on X super-effectively — ``effectiveness(attacker, X_types)``,
attacker first (see typechart). Everything is a PRIOR from revealed typing: it
says nothing about Tera (which changes type mid-game), abilities that grant
immunities (Levitate, Flash Fire), or coverage moves off a Pokemon's own
types. The output says so, so the model treats it as a hint, not a fact.

Nothing here raises; an empty or untyped team yields ``None`` and the caller
shows nothing.
"""

from __future__ import annotations

from .typechart import TYPES, effectiveness

# Thresholds for what is worth telling the model. A weakness shared by at
# least this many revealed Pokemon is a *team* soft spot rather than one
# member's problem; coverage/resistance shown by this fraction is a *team*
# trait. Kept conservative so the profile stays short and decisive.
_MIN_SHARED_WEAKNESS = 2       # >= 2 of their revealed mons weak to the type
_RESIST_FRACTION = 0.5         # type resisted/immune by >= half the team


def _as_type_list(value: object) -> list[str]:
    out = []
    for t in value if isinstance(value, list) else []:
        if isinstance(t, str) and t.strip().upper() in TYPES:
            out.append(t.strip().upper())
    return out


def opponent_profile(team_types: dict) -> dict | None:
    """A compact strategy profile for the opponent's revealed team.

    ``team_types`` maps species name -> list of type names (already resolved
    by the caller, preferring the live observation and falling back to the
    format table). Returns a prompt-ready dict, or ``None`` when no revealed
    Pokemon has a known type.
    """
    members = {
        species: types
        for species, raw in (team_types or {}).items()
        if isinstance(species, str) and species and (types := _as_type_list(raw))
    }
    if not members:
        return None

    n = len(members)

    # Shared defensive weaknesses: for each attacking type, how many revealed
    # mons it hits super-effectively (>1x), and the worst single multiplier.
    weaknesses = []
    for atk in TYPES:
        hits = [sp for sp, types in members.items() if effectiveness(atk, types) > 1.0]
        if len(hits) >= _MIN_SHARED_WEAKNESS:
            worst = max(effectiveness(atk, members[sp]) for sp in hits)
            weaknesses.append({
                "type": _title(atk),
                "hits": len(hits),
                "of": n,
                "max_multiplier": worst,
                "species": list(hits),
            })
    # Most-shared first, then hardest-hitting.
    weaknesses.sort(key=lambda w: (w["hits"], w["max_multiplier"]), reverse=True)

    # Offensive STAB coverage: the distinct types they can hit us with for
    # STAB, i.e. the union of their members' types.
    coverage = sorted({_title(t) for types in members.values() for t in types})

    # Types most of the team resists or is immune to (<1x on every-member avg
    # is too strict; use "resisted or immune by >= half the team").
    resisted = []
    for atk in TYPES:
        resist_count = sum(1 for types in members.values() if effectiveness(atk, types) < 1.0)
        if resist_count >= max(2, round(_RESIST_FRACTION * n)):
            resisted.append({"type": _title(atk), "resisted_by": resist_count, "of": n})
    resisted.sort(key=lambda r: r["resisted_by"], reverse=True)

    return {
        "revealed_count": n,
        "shared_weaknesses": weaknesses,
        "their_stab_coverage": coverage,
        "commonly_resisted": resisted,
        "basis": (
            "PRIOR from the opponent's REVEALED typing only. 'shared_weaknesses' are attacking "
            "types that hit several of their mons super-effectively (click these); "
            "'their_stab_coverage' is what they can hit you with for STAB; 'commonly_resisted' is "
            "what to avoid relying on. It ignores Tera (which changes type), ability immunities "
            "(Levitate, Flash Fire, etc.), and coverage moves off-type. Treat as a hint, verify "
            "against the live observation."
        ),
    }


def _title(type_name: str) -> str:
    return type_name[:1].upper() + type_name[1:].lower()
