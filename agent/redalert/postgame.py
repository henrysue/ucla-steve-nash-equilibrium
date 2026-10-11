"""Between games: Claude Opus reads the match log against an opponent and
writes down what it learned — their playstyle and opening, the decisions that
worked or failed for each side (they left their harvesters unguarded; they
killed our raid squads cheaply), a short summary, and one high-level way to
beat them. It is saved in the opponent's profile (opponents.py); the next
game against them, the advisor gets it as an addendum to the plan
(``Advisor.addendum``).

Runs at the end of a tournament match (``RedAlertAgent.on_match_end``, at
most ``REDALERT_ANALYST_TIMEOUT`` seconds), or by hand between games::

    python -m agent.redalert.postgame matchlogs/<match>.jsonl --opponent NAME [--save]

``REDALERT_ANALYST=off`` switches it off; ``REDALERT_ANALYST_MODEL`` picks the
model (default Claude Opus).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

ANALYST_MODEL = "claude-opus-5-5"
TIMEOUT_SECONDS = 90.0

SYSTEM = """\
You are a top Command & Conquer: Red Alert (OpenRA) coach. You review one 1v1 match from our \
agent's side, from a digest of its match log, to help it beat this opponent next time.

Our agent's plan, in brief: the pros' double-refinery opening into a war factory; scouts round \
their base early; the army masses in a concave at the main choke and holds it; it attacks when \
its strength clearly beats theirs (their turrets count), or right after winning a fight; raid \
squads hit their expansions; it expands to the natural and beyond; about half the army's value in \
tanks; turrets on the road in when an aggressive enemy is read; a model advisor reads the enemy \
and can set posture (attack/hold), expansion timing, balance (more army / more economy), raid \
targets and turret side.

From the digest (samples every 15 s, our mode changes, what our scouts saw first and when, the \
advisor's reads, our raids and attacks with their trades, buildings lost, the result):
- playstyle: rush, turtle, macro, standard or a mix — in a phrase, with the evidence.
- opening: their build and timings as far as we saw them (refineries, barracks, war factory, \
turrets, first push).
- worked_for_them / failed_for_them: decisions of theirs that paid off or cost them (e.g. they \
killed our raiders cheaply; their push into our turrets failed; they never defended harassment).
- worked_for_us / failed_for_us: the same for our agent.
- how_to_beat: ONE high-level suggestion in 1-2 sentences that fits our plan's levers (when to \
attack or hold, raid or not, turrets, expand early or late, unit mix, scouting). Simple, not a \
build order.
- summary: 2-4 sentences.
Base everything on the digest; say "unclear" rather than invent. Keep each list item short (one \
line, at most 4 items). If earlier analyses of this opponent are given, refine them: keep what \
this game confirms, correct what it contradicts.\
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "playstyle": {"type": "string"},
        "opening": {"type": "string"},
        "worked_for_them": {"type": "array", "items": {"type": "string"}},
        "failed_for_them": {"type": "array", "items": {"type": "string"}},
        "worked_for_us": {"type": "array", "items": {"type": "string"}},
        "failed_for_us": {"type": "array", "items": {"type": "string"}},
        "how_to_beat": {"type": "string"},
    },
    "required": ["summary", "playstyle", "opening", "worked_for_them", "failed_for_them",
                 "worked_for_us", "failed_for_us", "how_to_beat"],
    "additionalProperties": False,
}
_LISTS = ("worked_for_them", "failed_for_them", "worked_for_us", "failed_for_us")


def _rows(path) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except OSError:
        return []
    return out


def digest(path) -> dict:
    """The match log, cut down to what a coach reads (a few KB): who and
    where, a 15 s timeline, our mode changes, first sightings, the advisor's
    reads as they changed, raids and attacks with their trades, buildings
    lost, and the result."""
    rows = _rows(path)
    start = next((r for r in rows if r.get("kind") == "start"), {})
    timeline, armies = [], []
    for r in rows:
        if r.get("kind") != "sample":
            continue
        timeline.append([r.get("t"), r.get("mode"), (r.get("cash") or 0) + (r.get("ore") or 0), r.get("harvesters"),
                         r.get("army"), r.get("enemy_seen"), r.get("kills_cost"), r.get("deaths_cost"),
                         r.get("raid_squad"), len(r.get("expansions_done") or [])])
        pos = r.get("positions") or {}
        if pos and (r.get("t") or 0) % 60 < 15:              # army make-ups once a minute
            armies.append({"t": r.get("t"),
                           "ours": dict(Counter(u[0] for u in pos.get("units") or [])),
                           "theirs_in_sight": dict(Counter(e[0] for e in pos.get("enemies") or []))})
    advice, last = [], None
    for r in rows:
        if r.get("kind") != "advice":
            continue
        key = (r.get("enemy_style"), r.get("posture"), r.get("plan"))
        if key != last:
            last = key
            advice.append({"t": round(r.get("made_at") or 0), "enemy_style": r.get("enemy_style"),
                           "posture": r.get("posture"), "confidence": r.get("confidence"),
                           "plan": r.get("plan"), "evidence": r.get("evidence")})
    first_seen: dict = {}
    for r in rows:
        if r.get("kind") == "scouted":
            for it in r.get("items") or []:
                first_seen.setdefault(it.get("type"), [it.get("t"), it.get("kind"), it.get("cell")])
    end = next((r for r in reversed(rows) if r.get("kind") == "end"), {})
    pick = lambda kind: [{k: v for k, v in r.items() if k != "kind"} for r in rows if r.get("kind") == kind]  # noqa: E731
    return {
        "match": {"map": start.get("map"), "our_faction": start.get("faction"),
                  "their_faction": start.get("enemy_faction"), "minutes": start.get("max_minutes")},
        "timeline_columns": ["t", "mode", "bank", "harvesters", "army", "enemy_seen", "kills", "deaths",
                             "raiders", "expansions"],
        "timeline": timeline,
        "army_makeup": armies[:20],
        "mode_changes": pick("mode")[:40],
        "first_seen": first_seen,
        "advisor_reads": advice[:25],
        "raids": pick("raid_result")[:20],
        "attacks": pick("attack_result")[:12],
        "buildings_lost": pick("event")[:30],
        "result": {"result": end.get("result"), "military": end.get("military")},
    }


_FILLER = re.compile(r"^(that is|keep it|end|thank you|no further notes|done)\b", re.I)


def _tidy(value, sentences: int, limit: int) -> str:
    """Whole sentences only — at most ``sentences`` of them within ``limit``
    characters — without filler (the model once padded its suggestion with
    "Keep it. — End. Thank you. . . .")."""
    text = " ".join(str(value or "").split()).strip(" {}")
    parts = [p.strip(" -—") for p in re.split(r"(?<=[.!?])(?<!e\.g\.)(?<!i\.e\.)\s+", text)]
    parts = [p for p in parts if p.strip(". ()") and not (len(p) < 40 and _FILLER.match(p.strip("( ")))]
    out = ""
    for p in parts[:sentences]:
        if len(out) + len(p) + 1 > limit:
            break
        out = f"{out} {p}".strip()
    if not out and parts:                       # one long sentence: cut it at a word
        out = parts[0][:limit].rsplit(" ", 1)[0] + "…"
    return out


def _clean(answer) -> dict | None:
    if not isinstance(answer, dict):
        return None
    out = {k: _tidy(answer.get(k), n, c) for k, n, c in (("summary", 4, 700), ("playstyle", 2, 300),
                                                         ("opening", 3, 450), ("how_to_beat", 2, 400))}
    for k in _LISTS:
        out[k] = [_tidy(v, 1, 200) for v in (answer.get(k) or [])[:4] if isinstance(v, str) and v.strip()]
        out[k] = [v for v in out[k] if v]
    return out if out["summary"] or out["how_to_beat"] else None


def analyze(path, provider, previous=(), timeout: float = TIMEOUT_SECONDS) -> dict | None:
    """Claude's analysis of the match log at ``path`` (with the earlier
    analyses of this opponent in ``previous`` to refine), or None."""
    data = digest(path)
    if not data["timeline"]:
        return None
    earlier = [r["analysis"] for r in previous if isinstance(r, dict) and isinstance(r.get("analysis"), dict)][-2:]
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps({"digest": data, "earlier_analyses": earlier},
                                                separators=(",", ":"), default=str)},
    ]
    return _clean(provider.complete_structured(messages, "red_alert_opponent_analysis", SCHEMA,
                                               timeout=timeout, effort="medium"))


def addendum(records) -> str:
    """The latest analysis in an opponent's records, as an addendum to the
    advisor's plan ("" without one)."""
    done = [r for r in records or () if isinstance(r, dict) and isinstance(r.get("analysis"), dict)]
    if not done:
        return ""
    a = done[-1]["analysis"]
    items = lambda k: "; ".join(a.get(k) or []) or "unclear"  # noqa: E731
    return (
        f"AGAINST THIS OPPONENT (our analysis after {len(done)} past game(s) with them — an addendum to the "
        "plan above, not a replacement; this game's scouting wins where they differ):\n"
        f"- Summary: {a.get('summary') or 'unclear'}\n"
        f"- Playstyle: {a.get('playstyle') or 'unclear'}. Opening: {a.get('opening') or 'unclear'}\n"
        f"- Worked for them: {items('worked_for_them')}. Failed for them: {items('failed_for_them')}\n"
        f"- Worked for us: {items('worked_for_us')}. Failed for us: {items('failed_for_us')}\n"
        f"- How to beat them: {a.get('how_to_beat') or 'unclear'}"
    )


def provider_from_env(log=print):
    """The analyst's model (Claude Opus unless ``REDALERT_ANALYST_MODEL``), or
    None when switched off (``REDALERT_ANALYST=off``) or no credentials."""
    if os.environ.get("REDALERT_ANALYST", "").strip().lower() in ("off", "0", "false", "no"):
        return None
    try:
        from agent.vgc.providers import AnthropicProvider, anthropic_credentials
        if not anthropic_credentials():
            return None
        return AnthropicProvider(model=os.environ.get("REDALERT_ANALYST_MODEL") or ANALYST_MODEL)
    except Exception as exc:  # noqa: BLE001 - the analysis is optional
        log(f"[redalert] post-game analyst unavailable: {type(exc).__name__}: {exc}")
        return None


def launch(path, name: str, log=print) -> bool:
    """Run the analysis of the match log at ``path`` in a process of its own
    that outlives this one (the supervisor stops a finished match's worker
    within seconds — 9ff91807's analysis died with it), merging into the
    opponent's latest record. Its output goes to ``postgame.log`` beside the
    profiles. False when it couldn't start (or ``REDALERT_ANALYST_DETACH=0``):
    the caller analyses in place."""
    import subprocess
    import sys

    from . import opponents

    if os.environ.get("REDALERT_ANALYST_DETACH", "1").strip().lower() in ("0", "off", "false", "no"):
        return False
    directory = opponents.profile_dir()
    if directory is None:
        return False
    try:
        directory.mkdir(parents=True, exist_ok=True)
        out = open(directory / "postgame.log", "a", encoding="utf-8")    # noqa: SIM115 - handed to the child
        flags = {}
        if os.name == "nt":
            flags["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            flags["start_new_session"] = True
        subprocess.Popen([sys.executable, "-m", "agent.redalert.postgame", str(path), "--opponent", name, "--merge"],
                         cwd=str(Path(__file__).resolve().parents[2]), stdout=out, stderr=out,
                         stdin=subprocess.DEVNULL, **flags)
        out.close()
        log(f"[redalert] post-game analysis of {name} started in the background (log: {directory / 'postgame.log'})")
        return True
    except OSError as exc:
        log(f"[redalert] post-game analysis couldn't start in the background ({exc}); analysing now")
        return False


def timeout_from_env() -> float:
    try:
        return max(10.0, float(os.environ.get("REDALERT_ANALYST_TIMEOUT") or TIMEOUT_SECONDS))
    except ValueError:
        return TIMEOUT_SECONDS


def main(argv=None) -> int:
    from dotenv import load_dotenv

    from . import opponents

    load_dotenv()                                   # credentials and the log directory, as the agent has them
    p = argparse.ArgumentParser(description="Analyse a Red Alert match log against an opponent (Claude Opus).")
    p.add_argument("log", type=Path, help="the match log (.jsonl)")
    p.add_argument("--opponent", required=True, help="the opponent's display name")
    p.add_argument("--save", action="store_true", help="add the analysis to the opponent's profile")
    p.add_argument("--merge", action="store_true", help="merge it into the opponent's latest record (the agent's way)")
    args = p.parse_args(argv)
    provider = provider_from_env()
    if provider is None:
        print("no analyst model (credentials missing or REDALERT_ANALYST=off)")
        return 1
    history = opponents.load([args.opponent])
    analysis = analyze(args.log, provider, history, timeout_from_env())
    if analysis is None:
        print("no analysis (empty log or no answer)")
        return 1
    print(json.dumps(analysis, indent=1))
    if args.merge:
        ok = opponents.merge_last([args.opponent], {"analysis": analysis})
        print(f"merged into {args.opponent}'s latest record" if ok else "not saved (no profile directory)")
    elif args.save:
        ok = opponents.save([args.opponent], {"source": str(args.log.name), "analysis": analysis})
        print("saved to the opponent's profile" if ok else "not saved (no profile directory)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
