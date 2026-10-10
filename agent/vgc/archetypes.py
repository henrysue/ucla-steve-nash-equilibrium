"""Team plans to draft toward, the opponent's inferred playstyle, and counters.

The draft builds toward one of the plans in ``data/team_plans.json``
(user-written teams first among equals) instead of down a usage list. Each
pick:

1. the opponent's picks are read for a PLAYSTYLE (sun, rain, sand, snow,
   tailwind, trick_room, hyper_offense, balance) from their setters,
   abusers and team shape, with the evidence kept;
2. every plan is scored for what we already hold, what is still on the
   board, what the opponent has taken, and how its playstyle matches up
   with theirs ("beats" / "loses_to"); two members held commits us;
3. each card earns a plan bonus (core > member > flex) and a counter bonus
   for tools that answer their style (Taunt vs Trick Room, a rival weather
   setter vs weather, Intimidate and Fake Out vs hyper offense...).

Everything is shown to the model with its reasons; the totals also order
the board and drive the no-model pick.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DEFAULT_PLANS = Path(__file__).resolve().parents[2] / "data" / "team_plans.json"
STYLES = (
    "sun", "rain", "sand", "snow", "electric_terrain", "tailwind", "trick_room", "hyper_offense", "balance",
    "cycling", "stat_punish",
)
FIELD_STYLES = ("sun", "rain", "sand", "snow", "electric_terrain")

PLAN_BONUS = {"core": 6.0, "member": 3.0, "flex": 1.5}
CORE_FIRST = 2.0  # extra for a core piece while we hold none: lose it and the plan is gone
RUNNER_UP_BONUS = {"core": 1.5, "member": 1.0, "flex": 0.0}
COMMITTED_AT = 2  # members held before the plan is ours

WEATHER_ABILITIES = {
    "drought": "sun", "orichalcumpulse": "sun", "drizzle": "rain", "sandstream": "sand", "snowwarning": "snow",
    "hadronengine": "electric_terrain", "electricsurge": "electric_terrain",
}
ABUSER_ABILITIES = {
    "chlorophyll": "sun", "protosynthesis": "sun", "solarpower": "sun", "swiftswim": "rain", "sandrush": "sand",
    "sandforce": "sand", "slushrush": "snow", "quarkdrive": "electric_terrain", "surgesurfer": "electric_terrain",
}
ABILITY_TAGS = {
    "intimidate": "intimidate", "defiant": "stat_punish", "competitive": "stat_punish", "regenerator": "regenerator",
    "lightningrod": "lightning_rod", "grassysurge": "terrain_setter", "psychicsurge": "terrain_setter",
    "mistysurge": "terrain_setter", "electricsurge": "terrain_setter", "hadronengine": "terrain_setter",
    "clearbody": "anti_intimidate", "mirrorarmor": "anti_intimidate", "fullmetalbody": "anti_intimidate",
    "whitesmoke": "anti_intimidate", "innerfocus": "anti_intimidate", "owntempo": "anti_intimidate",
    "oblivious": "anti_intimidate", "scrappy": "anti_intimidate", "guarddog": "anti_intimidate",
}
MOVE_TAGS = {
    "tailwind": "tailwind", "trickroom": "trick_room", "taunt": "taunt", "imprison": "imprison",
    "fakeout": "fake_out", "ragepowder": "redirection", "followme": "redirection", "wideguard": "wide_guard",
    "icywind": "speed_control", "electroweb": "speed_control", "thunderwave": "speed_control",
    "swordsdance": "setup", "dragondance": "setup", "nastyplot": "setup", "quiverdance": "setup",
    "calmmind": "setup", "bulkup": "setup", "haze": "anti_setup", "clearsmog": "anti_setup", "encore": "anti_setup",
    "partingshot": "pivot", "uturn": "pivot", "flipturn": "pivot", "voltswitch": "pivot", "teleport": "pivot",
}
LEARNABLE_WEIGHT = 0.5  # a move it CAN learn counts half as much as one it runs
# Their style -> our tags that answer it, and how much.
COUNTERS = {
    "sun": {"rain": 2.0, "sand": 2.0, "snow": 1.5},
    "rain": {"sun": 2.0, "sand": 1.5, "snow": 1.5, "redirection": 0.5},
    "sand": {"rain": 1.5, "sun": 1.5, "snow": 1.5},
    "snow": {"sun": 2.0, "rain": 1.0, "sand": 1.5},
    "electric_terrain": {"terrain_setter": 2.0, "lightning_rod": 1.0, "sand": 0.5},
    "tailwind": {"trick_room": 2.0, "stat_punish": 1.0, "tailwind": 1.0, "speed_control": 1.0, "priority": 1.0},
    "trick_room": {"taunt": 2.0, "imprison": 2.0, "fake_out": 1.0, "trick_room": 1.0, "priority_block": 0.5},
    "hyper_offense": {"intimidate": 1.5, "fake_out": 1.0, "redirection": 1.0, "wide_guard": 1.0},
    "balance": {"setup": 1.5, "taunt": 1.0, "anti_setup": 0.5},
    "cycling": {"stat_punish": 2.0, "anti_intimidate": 1.0, "taunt": 1.0, "setup": 0.5},
    "stat_punish": {"trick_room": 1.0, "setup": 1.0},
}


def _id(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


@lru_cache(maxsize=4)
def load_plans(path: str = str(DEFAULT_PLANS)) -> tuple[dict, ...]:
    """The team plans, or none if the file is missing or malformed."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    return tuple(p for p in data.get("plans", []) if isinstance(p, dict) and p.get("name") and p.get("members"))


def family(name: str, table) -> str:
    """The species family a name belongs to: the table's base species, or
    the first word for forms the table does not list ("Charizard Y")."""
    if table.row(name) is not None:
        return table.base_species(name)
    return _id(re.split(r"[\s\-]", str(name).strip())[0]) if name else ""


def matches(member: str, species: str, table) -> bool:
    """Whether a draft card ``species`` fills the plan slot ``member``."""
    if _id(member) == _id(species):
        return True
    if table.row(member) is None or _id(member) == table.base_species(member):
        if family(member, table) != family(species, table):
            return False
        # A form qualifier must be in the card's name: "Charizard Y" is
        # Charizard-Mega-Y (Drought), never Mega X or the base form.
        wanted = set(re.split(r"[^a-z0-9]+", str(member).lower())) - {""}
        have = set(re.split(r"[^a-z0-9]+", str(species).lower())) - {""}
        known_base = table.row(member) is not None and _id(member) == table.base_species(member)
        return wanted <= have or known_base
    return False


def _slot(plan: dict, species: str, table) -> str | None:
    for kind in ("core", "member", "flex"):
        key = "members" if kind == "member" else kind
        if any(matches(m, species, table) for m in plan.get(key) or []):
            return kind
    return None


def declared_roles(species: str, table, plans=None) -> set[str]:
    """Roles a plan declares for this species (forms the table cannot tag)."""
    found = set()
    for plan in plans if plans is not None else load_plans():
        for member, roles in (plan.get("roles") or {}).items():
            if matches(member, species, table):
                found |= set(roles)
    return found


def tags(species: str, table, *, learnable: bool = False) -> set[str]:
    """Playstyle-relevant tools a species brings: from its abilities, common
    moves and role text, plus (``learnable=True``) moves it can learn but
    may not run. Weather comes from abilities only: nearly everything can
    learn Rain Dance, and nobody plays it as rain."""
    found = set()
    abilities = {_id(a) for a in table.abilities(species)}
    for ability, style in WEATHER_ABILITIES.items():
        if ability in abilities:
            found |= {style, f"sets_{style}"}
    if re.search(r"(mega)?y$", _id(species)) and family(species, table) == "charizard":
        found |= {"sun", "sets_sun"}  # Mega Charizard Y: Drought
    for ability, style in ABUSER_ABILITIES.items():
        if ability in abilities:
            found.add(f"{style}_abuser")
    for ability, tag in ABILITY_TAGS.items():
        if ability in abilities:
            found.add(tag)
    if abilities & {"psychicsurge", "armortail", "queenlymajesty", "dazzling"}:
        found.add("priority_block")
    moves = {_id(m) for m in table.common_moves(species)}
    role = re.sub(r"under trick room", "", table.field(species, "role").lower())
    moves |= {_id(phrase) for phrase in re.split(r"[;,/+]", role)}
    if learnable:
        moves |= {_id(m) for m in table.field(species, "support_moves").split(",")}
    for move in moves:
        for key, tag in MOVE_TAGS.items():
            if key in move:
                found.add(tag)
    if table.field(species, "priority_moves"):
        found.add("priority")
    stats = table.base_stats(species)
    if stats and stats["spe"] <= 50:
        found.add("slow")
    return found


def infer_playstyle(roster: list[str], table, plans=None) -> dict[str, dict]:
    """style -> {"confidence": 0..1, "evidence": [...]}, strongest first,
    only styles with real evidence. Read from the opponent's picks so far."""
    if not roster:
        return {}
    per = {name: tags(name, table) for name in roster}
    roles = {name: table.roles(name) for name in roster}
    found: dict[str, dict] = {}

    def add(style: str, amount: float, why: str) -> None:
        entry = found.setdefault(style, {"confidence": 0.0, "evidence": []})
        entry["confidence"] = min(1.0, entry["confidence"] + amount)
        entry["evidence"].append(why)

    for name, t in per.items():
        for style in FIELD_STYLES:
            label = style.replace("_", " ")
            if f"sets_{style}" in t:
                add(style, 0.6, f"{name} sets {label}")
            if f"{style}_abuser" in t:
                add(style, 0.2, f"{name} benefits from {label}")
        if "tailwind" in t:
            add("tailwind", 0.45, f"{name} sets Tailwind")
        if "trick_room" in t:
            add("trick_room", 0.45, f"{name} sets Trick Room")
    slow = [n for n, t in per.items() if "slow" in t and roles[n]["attacker"]]
    if slow and "trick_room" in found:
        add("trick_room", 0.15 * len(slow), f"slow attackers: {', '.join(slow)}")
    attackers = sum(r["attacker"] for r in roles.values())
    pure = sum(r["support"] and not r["attacker"] for r in roles.values())
    if len(roster) >= 3 and attackers >= max(3, len(roster) - 1) and pure <= 1 and "trick_room" not in found:
        add("hyper_offense", 0.5, f"{attackers} attackers, {pure} pure supports")
    bulk = [n for n, t in per.items() if t & {"intimidate", "redirection"}]
    if len(bulk) >= 2:
        add("balance", 0.25 * len(bulk), f"Intimidate/redirection: {', '.join(bulk)}")
    cyclers = [n for n, t in per.items() if "intimidate" in t]
    pivots = [n for n, t in per.items() if t & {"pivot", "regenerator"} and n not in cyclers]
    if len(cyclers) >= 2 or (cyclers and pivots):
        everyone = cyclers + pivots
        add("cycling", 0.2 * len(everyone), f"Intimidate/pivot/Regenerator: {', '.join(everyone)}")
    punish = [n for n, t in per.items() if "stat_punish" in t]
    if len(punish) >= 2:
        add("stat_punish", 0.3 * len(punish), f"Defiant/Competitive: {', '.join(punish)}")
    for plan in plans if plans is not None else load_plans():
        held = [n for n in roster if _slot(plan, n, table) in ("core", "member")]
        has_core = any(_slot(plan, n, table) == "core" for n in held)
        # A core piece is required: generic members (Incineroar, Amoonguss)
        # appear in half the plans and say nothing about the style alone.
        if has_core and len(held) >= 2 and plan.get("playstyle") in STYLES:
            add(plan["playstyle"], 0.3, f"{', '.join(held)} match the '{plan['name']}' plan")
    return dict(sorted(found.items(), key=lambda kv: -kv[1]["confidence"]))


OPENING_SPREAD = 2.0  # on pick one, any plan this close to the best may be chosen


def choose_plan(mine: list[str], theirs: list[str], board: list[str], styles: dict, table, plans=None, rng=None):
    """The plan to draft toward, the runner-up, and a status for the prompt.

    With rng, the very first pick (nothing held, nothing seen) chooses
    at random among the plans within OPENING_SPREAD of the best, so the
    opening is not the same every match.
    """
    plans = plans if plans is not None else load_plans()
    scored = []
    for plan in plans:
        held = [n for n in mine if _slot(plan, n, table)]
        taken = [n for n in theirs if _slot(plan, n, table)]
        core_held = [n for n in held if _slot(plan, n, table) == "core"]
        core_lost = [n for n in taken if _slot(plan, n, table) == "core"]
        if core_lost and not core_held and len(core_lost) >= len(plan.get("core") or []):
            continue  # every core piece is gone: this plan cannot be built
        available = [n for n in board if _slot(plan, n, table)]
        if board and not core_held and not any(_slot(plan, n, table) == "core" for n in available):
            continue  # no core piece is held or left on the board: it cannot be started
        score = 3 * len(core_held) + 2 * len(held) + min(len(available), 4) - 1.5 * len(taken)
        score += sum(2 * styles[s]["confidence"] for s in plan.get("beats") or [] if s in styles)
        score -= sum(2 * styles[s]["confidence"] for s in plan.get("loses_to") or [] if s in styles)
        if len(held) >= COMMITTED_AT:
            score += 3
        if plan.get("source") == "user":
            score += 0.5  # the user's own teams win ties
        scored.append((score, plan, held, taken, available))
    if not scored:
        return None, None, {}
    scored.sort(key=lambda item: -item[0])
    if rng is not None and not mine and not theirs:
        close = [item for item in scored if item[0] >= scored[0][0] - OPENING_SPREAD]
        chosen = rng.choice(close)
        scored.remove(chosen)
        scored.insert(0, chosen)
    score, plan, held, taken, available = scored[0]
    need = [m for m in plan["members"] if not any(matches(m, n, table) for n in mine)]
    status = {
        "name": plan["name"],
        "playstyle": plan.get("playstyle"),
        "win_condition": plan.get("win_condition"),
        "game_plan": plan.get("game_plan"),
        "held": held,
        "still_needed": need,
        "on_board_now": available,
        "taken_by_opponent": taken,
        "committed": len(held) >= COMMITTED_AT,
        "score": round(score, 2),
    }
    runner_up = scored[1][1] if len(scored) > 1 and len(held) < COMMITTED_AT else None
    return plan, runner_up, status


def card_bonus(
    species: str, plan: dict | None, runner_up: dict | None, table, mine: list[str] = ()
) -> tuple[float, str | None]:
    """Bonus for fitting the target plan (or, before commitment, the runner-up)."""
    if plan:
        slot = _slot(plan, species, table)
        if slot:
            bonus = PLAN_BONUS[slot]
            if slot == "core" and not any(_slot(plan, n, table) == "core" for n in mine):
                bonus += CORE_FIRST
            return bonus, f"{slot} of '{plan['name']}'"
    if runner_up:
        slot = _slot(runner_up, species, table)
        if slot and RUNNER_UP_BONUS[slot]:
            return RUNNER_UP_BONUS[slot], f"{slot} of runner-up '{runner_up['name']}'"
    return 0.0, None


def counter_bonus(species: str, styles: dict, table) -> tuple[float, list[str]]:
    """Bonus for tools that answer the opponent's inferred playstyle."""
    runs, can = tags(species, table), tags(species, table, learnable=True)
    total, reasons = 0.0, []
    for style, info in styles.items():
        options = {
            tag: weight * (1.0 if tag in runs else LEARNABLE_WEIGHT)
            for tag, weight in COUNTERS.get(style, {}).items()
            if tag in can
        }
        if options:
            tag, weight = max(options.items(), key=lambda kv: kv[1])
            total += weight * info["confidence"]
            reasons.append(f"{tag.replace('_', ' ')} vs their {style.replace('_', ' ')}")
    return round(total, 2), reasons


CLASH_PENALTY = 4.0


def field_clash(species: str, mine: list[str], plan: dict | None, table) -> tuple[float, str | None]:
    """Penalty for a card that sets a weather/terrain at odds with ours.

    A rain setter on a sun team overwrites our own sun (the simulated
    Champions drafts took Mega Charizard Y and then Pelipper).
    """
    sets = {style for style in FIELD_STYLES if f"sets_{style}" in tags(species, table)}
    if not sets:
        return 0.0, None
    ours = {style for name in mine for style in FIELD_STYLES if f"sets_{style}" in tags(name, table)}
    if plan and plan.get("playstyle") in FIELD_STYLES:
        ours.add(plan["playstyle"])
    if ours and not sets & ours:
        mine_label = "/".join(sorted(s.replace("_", " ") for s in ours))
        return -CLASH_PENALTY, f"sets {'/'.join(sorted(sets))} over our {mine_label}"
    return 0.0, None
