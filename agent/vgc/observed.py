"""What actually happened last turn, and whether our read of the opponent was right.

The server sends no battle log, so the turn is reconstructed by comparing
two observations: HP, faints, who left the field, newly revealed moves (and
PP spent, when the server shows it), plus the move tracker's Protect read.
Each foe's predicted action is then checked against that: "hit", "miss", or
"unknown" when the evidence does not say. Logged per decision and totalled in
the match summary, so read accuracy can be measured instead of guessed.

Also reads the field (``side_conditions``, ``opponent_side_conditions``,
``fields``, ``weather``), whose exact shape the platform docs do not show: any
dict key, list item or string naming an effect counts.
"""

from __future__ import annotations

import re

from . import movelist

HP_EPSILON = 0.005
# Moves that cost the user HP: its HP loss that turn is not someone else's attack.
SELF_DAMAGING = frozenset({
    "woodhammer", "bravebird", "flareblitz", "wavecrash", "doubleedge", "volttackle", "headsmash", "lightofruin",
    "wildcharge", "takedown", "submission", "headcharge", "chloroblast", "steelbeam", "mindblown",
})
PROTECT_IDS = frozenset({
    "protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "burningbulwark", "silktrap", "obstruct",
})


def _id(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def conditions(value: object) -> set[str]:
    """Effect ids named anywhere in a field/side-conditions value."""
    found: set[str] = set()
    if isinstance(value, dict):
        # A key's presence means it is up: the value is a start turn or a
        # layer count, and a lead's terrain starts on turn 0 (match 07a3cb1f
        # showed no terrain at all while Psychic Surge and Grassy Surge led).
        for key, inner in value.items():
            if inner is not None and inner is not False:
                found.add(_id(str(key).split(".")[-1]))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            found |= conditions(item) if isinstance(item, (dict, list)) else {_id(str(item).split(".")[-1])}
    elif isinstance(value, str) and value:
        found.add(_id(value.split(".")[-1]))
    return {f for f in found if f}


def field_state(observation: dict) -> dict | None:
    """Ours/theirs side effects and field effects, or ``None`` when the server
    sent none of the fields (old observations, tests)."""
    keys = ("side_conditions", "opponent_side_conditions", "fields", "weather")
    if not any(k in observation for k in keys):
        return None
    return {
        "ours": sorted(conditions(observation.get("side_conditions"))),
        "theirs": sorted(conditions(observation.get("opponent_side_conditions"))),
        "field": sorted(conditions(observation.get("fields")) | conditions(observation.get("weather"))),
    }


def _hp(entry: dict) -> float | None:
    for key in ("hp_fraction", "current_hp_fraction"):
        value = entry.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value) / (100.0 if value > 1 else 1.0)
    current, maximum = entry.get("current_hp"), entry.get("max_hp")
    if isinstance(current, (int, float)) and isinstance(maximum, (int, float)) and maximum:
        return current / maximum
    return None


def _moves(entry: dict) -> dict[str, int | None]:
    """Known move id -> PP left (``None`` when not shown)."""
    raw = entry.get("moves") or entry.get("known_moves") or entry.get("revealed_moves") or []
    found: dict[str, int | None] = {}
    items = raw.items() if isinstance(raw, dict) else ((None, m) for m in raw) if isinstance(raw, list) else ()
    for key, move in items:
        if isinstance(move, dict):
            name = move.get("id") or move.get("name") or key
            pp = move.get("current_pp", move.get("pp"))
            found[_id(name)] = pp if isinstance(pp, int) else None
        elif move or key:
            found[_id(move if isinstance(move, str) else key)] = None
    return {k: v for k, v in found.items() if k}


def _side(entries: object) -> dict[str, dict]:
    out = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        species = _id(entry.get("species") or entry.get("name"))
        if species:
            out[species] = {
                "hp": _hp(entry), "status": entry.get("status") or None, "boosts": entry.get("boosts") or {},
                "fainted": bool(entry.get("fainted")) or _hp(entry) == 0, "moves": _moves(entry),
            }
    return out


def snapshot(observation: dict) -> dict:
    """The parts of one observation that show what happened."""
    bench = {}
    team = observation.get("opponent_team")
    for entry in team.values() if isinstance(team, dict) else team if isinstance(team, list) else []:
        if isinstance(entry, dict) and (entry.get("species") or entry.get("name")):
            bench[_id(entry.get("species") or entry.get("name"))] = bool(entry.get("fainted")) or _hp(entry) == 0
    return {
        "ours": _side(observation.get("active_pokemon")),
        "theirs": _side(observation.get("opponent_active_pokemon")),
        "their_fainted": bench,
        "field": field_state(observation),
        "raw_field": {k: observation[k] for k in ("side_conditions", "opponent_side_conditions", "fields", "weather")
                      if k in observation},
    }


def last_turn(before: dict, after: dict, prediction: dict | None, protected: dict,
              our_switches: set[str] | None = None, our_moves: dict[str, str] | None = None) -> dict:
    """What happened between two snapshots, and each prediction's verdict.
    ``our_switches``: species ids we switched out ourselves (left the field,
    but not hit); the hit aimed at their slot lands on whoever came in."""
    our_switches = {_id(s) for s in our_switches or ()}
    # Our own recoil is not evidence of being attacked (match 07a3cb1f: Wood
    # Hammer recoil was read as Indeedee's attack landing).
    self_hurt = {_id(m) for m, move in (our_moves or {}).items() if _id(move) in SELF_DAMAGING}
    theirs = {}
    for foe, was in before["theirs"].items():
        now = after["theirs"].get(foe)
        row: dict = {}
        if now is None:
            fainted = after["their_fainted"].get(foe)
            row["left_field"] = "fainted" if fainted or (was["hp"] or 0) <= 0 else "switched or fainted"
        else:
            if was["hp"] is not None and now["hp"] is not None:
                row["hp_change"] = round(now["hp"] - was["hp"], 3)
            new = sorted(set(now["moves"]) - set(was["moves"]))
            spent = sorted(m for m, pp in now["moves"].items()
                           if pp is not None and was["moves"].get(m) is not None and pp < was["moves"][m])
            if new:
                row["revealed"] = new
            if spent:
                row["used"] = spent
            if now["fainted"]:
                row["left_field"] = "fainted"
        if foe in protected:
            row["protected"] = protected[foe]
        theirs[foe] = row
    for foe in after["theirs"]:
        if foe not in before["theirs"]:
            theirs.setdefault(foe, {})["entered"] = True
    ours = {}
    for mon, was in before["ours"].items():
        now = after["ours"].get(mon)
        if now is None:
            ours[mon] = {"left_field": "switched" if mon in our_switches else "fainted"}
        elif was["hp"] is not None and now["hp"] is not None:
            ours[mon] = {"hp_change": round(now["hp"] - was["hp"], 3)}
    for mon, now in after["ours"].items():
        if mon not in before["ours"]:
            ours[mon] = {"entered": True, "hp": now["hp"]}
    report = {"theirs": theirs, "ours": ours}
    if after.get("field") is not None:
        report["field"] = after["field"]
    if after.get("raw_field"):
        report["raw_field"] = after["raw_field"]  # the server's own shape, until it is confirmed
    if prediction:
        hurt = {m: row for m, row in ours.items() if m not in self_hurt or row.get("left_field")}
        report["reads"] = {foe: verdict(guess, theirs.get(_id(foe), {}), hurt, our_switches) for foe, guess in prediction.items()
                           if isinstance(guess, dict)}
    return report


def verdict(guess: dict, foe: dict, ours: dict, our_switches: set[str] = frozenset()) -> dict:
    """Was this foe's predicted action what it did? hit / miss / unknown."""
    action = guess.get("action")
    moves = set(foe.get("revealed", [])) | set(foe.get("used", []))
    # A move it showed this turn beats our HP-based guess (match 915ba9e1:
    # Landorus's U-turn was logged as a correct Protect read).
    protected = bool(moves & PROTECT_IDS) or ("protected" in foe and not moves)
    left = foe.get("left_field")
    switched = left == "switched or fainted" and not moves
    attacked = any(_damaging(m) for m in moves)
    status_only = bool(moves - PROTECT_IDS) and not attacked  # e.g. Follow Me: it did not attack
    target = _id(guess.get("target"))
    if action == "protect":
        result = "hit" if protected else "miss" if (switched or attacked or foe.get("hp_change", 0) < -HP_EPSILON) else "unknown"
    elif action == "switch":
        result = "hit" if switched else "miss" if left is None else "unknown"
    elif action == "fake_out" and moves and "fakeout" not in moves:
        result = "miss"  # it showed another move this turn (match 07a3cb1f: Iron Hands used Close Combat)
    elif action == "fake_out" and "fakeout" not in moves:
        # No Fake Out shown: damage on the target may be its partner's (match
        # 666b209d: Chi-Yu's Overheat on Iron Hands credited Rillaboom's Fake Out,
        # which had flinched Dragonite).
        result = "miss" if (protected or switched) else "unknown"
    elif action in ("attack", "fake_out"):
        # Hurt: lost HP, fainted, or came in already hurt (it took a hit aimed
        # at the slot we switched; that counts for the target we switched out).
        took = {m for m, row in ours.items()
                if row.get("hp_change", 0) < -HP_EPSILON or row.get("left_field") == "fainted"
                or (row.get("entered") and (row.get("hp") or 1.0) < 1 - HP_EPSILON)}
        if any(ours[m].get("entered") for m in took if m in ours):
            took |= our_switches
        if protected or switched or status_only:
            result = "miss"
        elif attacked or took:
            result = "hit" if not target or target in took or not took else "hit, other target"
        else:
            result = "unknown"
    else:  # speed_control / setup / support
        status = any((movelist.lookup(m) or {}).get("category") == "STATUS" for m in moves - PROTECT_IDS)
        result = "hit" if status else "miss" if (protected or switched or attacked) else "unknown"
    return {"predicted": action + (f" -> {guess['target']}" if guess.get("target") else ""), "result": result}


def _damaging(move_id: str) -> bool:
    move = movelist.lookup(move_id) or {}
    return move.get("category") in ("PHYSICAL", "SPECIAL")


def tally(reads: list[dict]) -> dict:
    """Totals per predicted action: hit / miss / unknown, and the hit rate where known."""
    out: dict[str, dict] = {}
    for read in reads:
        action = str(read.get("predicted", "?")).split(" ")[0]
        result = "hit" if str(read.get("result", "")).startswith("hit") else read.get("result", "unknown")
        row = out.setdefault(action, {"hit": 0, "miss": 0, "unknown": 0})
        row[result] = row.get(result, 0) + 1
    for row in out.values():
        known = row["hit"] + row["miss"]
        row["hit_rate"] = round(row["hit"] / known, 2) if known else None
    return out
