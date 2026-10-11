"""What we learned about each opponent, across games.

At the end of a match the agent writes one record per opponent — their
opening, when they first pushed, how the advisor read them, the result —
and at the start of the next game against the same opponent the advisor is
handed the last few (advisor memory ``opponent_history``): in a tournament
series it starts from what they did last time instead of from nothing.

Records live in ``REDALERT_PROFILE_DIR``, else ``<REDALERT_LOG_DIR>/opponents``
(off when neither is set), one JSON list per opponent, keyed by a filesystem-
safe form of their display name. Reading or writing never raises: a broken
or missing file just means no history.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

KEEP = 8          # records kept per opponent
SHOWN = 3         # records handed to the advisor


def profile_dir() -> Path | None:
    raw = os.environ.get("REDALERT_PROFILE_DIR")
    if not raw:
        logs = os.environ.get("REDALERT_LOG_DIR") or os.environ.get("VGC_LOG_DIR")
        raw = str(Path(logs) / "opponents") if logs else None
    return Path(raw).expanduser() if raw else None


def _key(name) -> str | None:
    if not isinstance(name, str):
        return None
    key = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:60]
    return key or None


def _path(names, directory: Path | None) -> Path | None:
    key = next((k for k in (_key(n) for n in names or ()) if k), None)
    return directory / f"{key}.json" if directory is not None and key else None


def load(names, directory: Path | None = None) -> list[dict]:
    """The last ``SHOWN`` records for the first named opponent, oldest first."""
    path = _path(names, directory if directory is not None else profile_dir())
    if path is None:
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [r for r in records if isinstance(r, dict)][-SHOWN:] if isinstance(records, list) else []


def merge_last(names, patch: dict, directory: Path | None = None) -> bool:
    """Merge ``patch`` into the first named opponent's latest record (the
    post-game analysis joins the game it is about); a new record when there
    is none. False when there is nowhere to write or it failed."""
    directory = directory if directory is not None else profile_dir()
    path = _path(names, directory)
    if path is None:
        return False
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        records = [r for r in records if isinstance(r, dict)] if isinstance(records, list) else []
    except (OSError, ValueError):
        records = []
    if not records:
        return save(names, patch, directory)
    records[-1] = {**records[-1], **patch}
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records[-KEEP:], indent=1, default=str), encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError:
        return False


def save(names, record: dict, directory: Path | None = None) -> bool:
    """Append ``record`` to the first named opponent's file (the last
    ``KEEP`` kept); False when there is nowhere to write or it failed."""
    directory = directory if directory is not None else profile_dir()
    path = _path(names, directory)
    if path is None:
        return False
    try:
        directory.mkdir(parents=True, exist_ok=True)
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            records = [r for r in records if isinstance(r, dict)] if isinstance(records, list) else []
        except (OSError, ValueError):
            records = []
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps((records + [record])[-KEEP:], indent=1, default=str), encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError:
        return False
