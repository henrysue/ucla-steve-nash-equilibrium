"""The draft-meta reference table: what to draft, and what beats what.

Loads ``data/vgc_2025_reg_i_draft_doubles.csv`` — the tournament's own
format, ``gen9vgc2025regi`` (override with ``VGC_META_CSV``) — and serves it
back in prompt-sized pieces. The table carries per-Pokémon usage, estimated
draft points, doubles role, learnable support/spread/priority moves,
defensive profile, named counters and checks, synergistic partners, whether
the Pokémon is a restricted legendary, any item it requires, and the Worlds
Top 8 teams it appeared on — which is exactly the knowledge Section 7 wants
for draft and Team Preview, and that Section 2 wants for inferring an
opponent's likely sets.

Columns are resolved through alias lists rather than fixed names, because
the table is replaced when the format rotates and the headers move with it
(Reg MC's ``Mega?``/``Mega Stone`` became Reg I's ``Restricted
Legendary?``/``Required Item``, and the usage column is named after whatever
source measured it). A column that no longer exists simply drops out of the
briefs instead of raising.

Three rules govern how the table is used:

- **The live observation wins.** The table is prior knowledge about a
  format; the server is authoritative about this match. A card's real moves
  and item come from the observation, never from here.
- **Absence is normal.** A species missing from the table, or a blank cell
  for a Pokémon below the usage cutoff, yields ``None`` and the agent
  reasons without it. Species ids are matched the way Showdown forms them
  (lowercase, punctuation stripped), with a base-species fallback so an
  unlisted form still finds its base entry.
- **Rows are data, never instructions.** They are quoted into prompts as
  reference material.
"""

from __future__ import annotations

import csv
import os
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_CSV = DATA_DIR / "vgc_2025_reg_i_draft_doubles.csv"

# Short keys, because the real headers are long and these go into prompts
# under a clock. Each maps to candidate column names, most current first; a
# trailing "*" makes an alias a prefix match.
SPECIES = "species"
BASE_SPECIES = "base_species"
DRAFT_POINTS = "draft_points"
COUNTERS = "countered_by"
PARTNERS = "partners"
RESTRICTED = "restricted"

_SHARED_FIELDS: dict[str, tuple[str, ...]] = {
    SPECIES: ("Pokemon",),
    BASE_SPECIES: ("Base Species",),
    "type1": ("Primary Type",),
    "type2": ("Secondary Type",),
    "speed": ("Spe",),
    RESTRICTED: ("Restricted Legendary?",),
    "required_item": ("Required Item", "Mega Stone"),
    "tier": ("Usage Tier",),
    "usage_pct": ("Usage %*",),
    DRAFT_POINTS: ("Est. Draft Points*",),
    "role": ("Doubles Role",),
    "support_moves": ("Key Doubles Support Moves (learnable)",),
    "priority_moves": ("Priority Attacks (learnable)",),
    COUNTERS: ("Counters & Checks",),
    PARTNERS: ("Synergistic Partners",),
}

_DRAFT_FIELDS = {**_SHARED_FIELDS, "bst": ("BST",)}

_BATTLE_FIELDS = {
    **_SHARED_FIELDS,
    "hp": ("HP",),
    "atk": ("Atk",),
    "def": ("Def",),
    "spa": ("SpA",),
    "spd": ("SpD",),
    "weak_4x": ("4x Weak",),
    "weak_2x": ("2x Weak",),
    "resists": ("Resists (0.5x / 0.25x)",),
    "immune": ("Immune",),
    "abilities": ("Abilities",),
    "ability_notes": ("Ability-Dependent Matchup Notes",),
    "common_item": ("Most Common Item",),
    "common_moves": ("Most Common Moves",),
    "spread_moves": ("Spread Attacks (learnable, top BP)",),
    "cores": ("2025 Worlds Top 8 Teams Featuring It", "Proven Tournament Cores / Notes"),
}

# Resolved for logic, not shown in a brief: base species drives name
# matching, and "restricted: No" on 439 of 472 rows is pure prompt noise.
_HIDDEN = frozenset({BASE_SPECIES})

_TRUTHY = frozenset({"yes", "y", "true", "1"})


def species_id(name: str) -> str:
    """Showdown's id form: lowercase, punctuation and spaces stripped."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _resolve(fields: dict[str, tuple[str, ...]], available: list[str]) -> dict[str, str]:
    """Map each short key to the first of its aliases present in the file."""
    resolved: dict[str, str] = {}
    for key, aliases in fields.items():
        for alias in aliases:
            if alias.endswith("*"):
                match = next((column for column in available if column.startswith(alias[:-1])), None)
                if match:
                    resolved[key] = match
                    break
            elif alias in available:
                resolved[key] = alias
                break
    return resolved


class MetaTable:
    """Indexed access to the reference table."""

    def __init__(self, rows: list[dict]) -> None:
        columns = list(rows[0].keys()) if rows else []
        self._draft = _resolve(_DRAFT_FIELDS, columns)
        self._battle = _resolve(_BATTLE_FIELDS, columns)
        self._name = self._draft.get(SPECIES, "Pokemon")
        self._base = self._draft.get(BASE_SPECIES)

        self._rows: dict[str, dict] = {}
        self._by_base: dict[str, dict] = {}
        for row in rows:
            name = (row.get(self._name) or "").strip()
            if not name:
                continue
            self._rows.setdefault(species_id(name), row)
            base = (row.get(self._base) or "").strip() if self._base else ""
            # Base-species fallback, but never shadowing a real entry: a
            # form's row must not become the answer for the plain species.
            if base and species_id(base) not in self._rows:
                self._by_base.setdefault(species_id(base), row)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def columns(self) -> dict[str, str]:
        """Which real column each short key resolved to (diagnostics)."""
        return dict(self._battle)

    def row(self, species: str | None) -> dict | None:
        if not species:
            return None
        key = species_id(species)
        return self._rows.get(key) or self._by_base.get(key)

    def _value(self, row: dict, key: str) -> str:
        column = self._battle.get(key) or self._draft.get(key)
        return (row.get(column) or "").strip() if column else ""

    def draft_points(self, species: str | None) -> int | None:
        """Estimated draft cost, or ``None`` when the table leaves it blank
        (which it does for everything below the usage cutoff)."""
        row = self.row(species)
        if row is None:
            return None
        try:
            return int(self._value(row, DRAFT_POINTS))
        except ValueError:
            return None

    def is_restricted(self, species: str | None) -> bool:
        """Whether this is a restricted legendary under the loaded format."""
        row = self.row(species)
        return row is not None and self._value(row, RESTRICTED).lower() in _TRUTHY

    def restricted_in(self, names: list[str]) -> list[str]:
        """Which of ``names`` are restricted legendaries, in order."""
        return [name for name in dict.fromkeys(names) if self.is_restricted(name)]

    def brief(self, species: str | None, *, battle: bool = False) -> dict | None:
        """A compact view of one species, or ``None`` if it is not listed."""
        row = self.row(species)
        if row is None:
            return None
        fields = self._battle if battle else self._draft
        brief = {}
        for key, column in fields.items():
            if key in _HIDDEN:
                continue
            value = (row.get(column) or "").strip()
            if not value or value.lower() == "none":
                continue
            # A "No" here is true of almost every row; only the Yes informs.
            if key == RESTRICTED:
                if value.lower() in _TRUTHY:
                    brief[key] = True
                continue
            brief[key] = value
        if species_id(species) != species_id(row.get(self._name) or ""):
            brief["note"] = (
                f"no exact entry for {species!r}; showing base species {row.get(self._name)!r}"
            )
        return brief

    def briefs(self, names: list[str], *, battle: bool = False) -> dict[str, dict]:
        found = {}
        for name in dict.fromkeys(names):
            brief = self.brief(name, battle=battle)
            if brief is not None:
                found[name] = brief
        return found

    # -- cross-referencing ----------------------------------------------------

    def names_mentioned(self, text: str, candidates: list[str]) -> list[str]:
        """Which ``candidates`` are named in a free-text counters/partners list.

        Matched on base-species words, so prose that says "Landorus" finds a
        ``Landorus-Therian`` candidate even though the table spells it the
        other way round. Deliberately generous: this is a hint shown
        alongside the raw text, not a verdict.
        """
        if not text:
            return []
        lowered = text.lower()
        squashed = re.sub(r"[^a-z0-9]", "", lowered)
        hits = []
        for candidate in candidates:
            row = self.row(candidate)
            base = (self._value(row, BASE_SPECIES) if row else "") or candidate
            if not base:
                continue
            if re.search(rf"\b{re.escape(base.lower())}\b", lowered):
                hits.append(candidate)
            elif species_id(base) and species_id(base) in squashed:
                hits.append(candidate)
        return hits

    def matchup_notes(self, ours: list[str], theirs: list[str]) -> dict:
        """Who answers whom, read off the table's counter lists.

        For each of their Pokémon: which of ours the table names as a
        counter or check. And the mirror — which of theirs is named as a
        counter to each of ours, so the agent sees its own liabilities too.
        """
        our_answers: dict[str, list[str]] = {}
        threats_to_us: dict[str, list[str]] = {}
        for name in theirs:
            row = self.row(name)
            if row is None:
                continue
            answers = self.names_mentioned(self._value(row, COUNTERS), ours)
            if answers:
                our_answers[name] = answers
        for name in ours:
            row = self.row(name)
            if row is None:
                continue
            threats = self.names_mentioned(self._value(row, COUNTERS), theirs)
            if threats:
                threats_to_us[name] = threats
        return {
            "our_pokemon_that_check_theirs": our_answers,
            "their_pokemon_that_check_ours": threats_to_us,
        }

    def partner_notes(self, roster: list[str]) -> dict[str, list[str]]:
        """Which members of a roster the table lists as partners for each other."""
        links: dict[str, list[str]] = {}
        for name in roster:
            row = self.row(name)
            if row is None:
                continue
            partners = [
                other
                for other in self.names_mentioned(self._value(row, PARTNERS), roster)
                if species_id(other) != species_id(name)
            ]
            if partners:
                links[name] = partners
        return links


@lru_cache(maxsize=1)
def table() -> MetaTable:
    """Load the reference table once per process.

    ``VGC_META_CSV`` wins; otherwise the format's own file, falling back to
    a lone CSV in ``data/`` if that file has been renamed. A missing or
    unreadable table yields an empty one rather than an error: the agent
    plays without the reference, it just plays less informed.
    """
    for path in _candidate_paths():
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error, UnicodeDecodeError):
            continue
        if rows:
            return MetaTable(rows)
    return MetaTable([])


def _candidate_paths() -> list[Path]:
    override = os.environ.get("VGC_META_CSV")
    if override:
        return [Path(override)]
    if DEFAULT_CSV.exists():
        return [DEFAULT_CSV]
    try:
        found = sorted(DATA_DIR.glob("*.csv"))
    except OSError:
        return []
    # Only an unambiguous single file; two tables and a guess is worse than none.
    return found if len(found) == 1 else []
