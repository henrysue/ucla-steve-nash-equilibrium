"""Exact sets from the draft board.

Each draft card is a fixed, pre-built set (species, item, ability, nature,
EVs, four moves), and the opponent drafts from the same 18-card board. So
every Pokemon they bring is a set we have seen: no need to guess its moves
from usage data or assume a generic 90-power STAB attack (match e4546428:
Chi-Yu's real Overheat with Beads of Ruin KO'd Urshifu on turn 1).

``CardSets`` remembers every card on every board and persists them to
``data/card_sets.json`` (``VGC_CARD_SETS`` overrides; empty turns the file
off), so a card the opponent took before our first look at the board is
still known from an earlier match. ``overlay(table)`` puts the sets in front
of a reference table: ``common_moves``, ``common_item`` and ``abilities``
answer from the card when there is one, and ``card_set`` returns the whole
set for exact stats.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .meta import DATA_DIR, species_id

DEFAULT_PATH = DATA_DIR / "card_sets.json"
FIELDS = ("card_id", "species", "item", "ability", "nature", "level", "evs", "ivs", "moves")

# Nature -> (raised stat, lowered stat); the five neutral natures are absent.
NATURES = {
    "lonely": ("atk", "def"), "brave": ("atk", "spe"), "adamant": ("atk", "spa"), "naughty": ("atk", "spd"),
    "bold": ("def", "atk"), "relaxed": ("def", "spe"), "impish": ("def", "spa"), "lax": ("def", "spd"),
    "timid": ("spe", "atk"), "hasty": ("spe", "def"), "jolly": ("spe", "spa"), "naive": ("spe", "spd"),
    "modest": ("spa", "atk"), "mild": ("spa", "def"), "quiet": ("spa", "spe"), "rash": ("spa", "spd"),
    "calm": ("spd", "atk"), "gentle": ("spd", "def"), "sassy": ("spd", "spe"), "careful": ("spd", "spa"),
}


def nature_factor(nature: object, stat: str) -> float:
    up, down = NATURES.get(str(nature or "").strip().lower(), (None, None))
    return 1.1 if stat == up else 0.9 if stat == down else 1.0


def _path() -> Path | None:
    value = os.environ.get("VGC_CARD_SETS")
    if value is None:
        return DEFAULT_PATH
    return Path(value) if value.strip() else None


def _known(value: object) -> bool:
    """Not a placeholder: Team Preview lists the opponent's Pokemon with
    ``"item": "unknown_item"`` and no moves (match 8a882a3f, where that
    overwrote their real draft cards)."""
    if value in (None, "", [], {}):
        return False
    return not (isinstance(value, str) and value.strip().lower().startswith("unknown"))


def _clean(card: dict) -> dict | None:
    """The set's known fields, or ``None`` when it shows no moves at all."""
    species = card.get("species") or card.get("name")
    moves = card.get("moves")
    if not isinstance(species, str) or not isinstance(moves, list):
        return None
    names = [m.get("name") or m.get("id") if isinstance(m, dict) else m for m in moves]
    names = [m for m in names if isinstance(m, str) and _known(m)]
    if not names:
        return None
    found = {k: card[k] for k in FIELDS if k in card and k != "moves" and _known(card[k])}
    found["species"] = species
    found["moves"] = names
    return found


def _merge(old: dict | None, new: dict) -> dict:
    """A draft card (it has a card_id) is the whole truth and replaces what
    we had; anything else (a Team Preview roster entry) only fills gaps."""
    if not old or "card_id" in new:
        return new
    merged = dict(old)
    for key, value in new.items():
        if key not in merged or (key == "moves" and len(value) > len(merged["moves"])):
            merged[key] = value
    return merged


class CardSets:
    """species id -> the card's set; this match's boards first, then saved ones."""

    def __init__(self, path: Path | None = None, *, persist: bool = True) -> None:
        self._path = _path() if path is None and persist else path
        self._saved: dict[str, dict] = self._load()
        self._seen: dict[str, dict] = {}  # this match only: certain

    def _load(self) -> dict[str, dict]:
        if not self._path:
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {k: found for k, v in data.items() if isinstance(v, dict) and (found := _clean(v))}

    def __len__(self) -> int:
        return len({**self._saved, **self._seen})

    def remember(self, cards: object) -> int:
        """Record every card in a board (``available_cards``) or roster list.
        Returns how many sets were new or changed in the saved catalog."""
        changed = 0
        for card in cards if isinstance(cards, list) else []:
            found = _clean(card) if isinstance(card, dict) else None
            if not found:
                continue
            key = species_id(found["species"])
            self._seen[key] = _merge(self._seen.get(key) or self._saved.get(key), found)
            if self._saved.get(key) != self._seen[key]:
                self._saved[key] = self._seen[key]
                changed += 1
        if changed:
            self._save()
        return changed

    def _save(self) -> None:
        if not self._path:
            return
        try:
            self._path.write_text(json.dumps(dict(sorted(self._saved.items())), indent=1) + "\n", encoding="utf-8")
        except OSError:
            pass  # the catalog is a convenience; never cost a move over it

    def get(self, species: str | None) -> dict | None:
        key = species_id(species or "")
        return self._seen.get(key) or self._saved.get(key) if key else None

    def seen_this_match(self, species: str | None) -> bool:
        return species_id(species or "") in self._seen

    def overlay(self, table) -> "CardTable":
        return CardTable(table.base if isinstance(table, CardTable) else table, self)


class CardTable:
    """A reference table with the card sets in front of its guesses."""

    def __init__(self, base, cards: CardSets) -> None:
        self.base = base
        self.cards = cards

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __len__(self) -> int:
        return len(self.base)

    def card_set(self, species: str | None) -> dict | None:
        return self.cards.get(species)

    def common_moves(self, species: str | None) -> list[str]:
        card = self.cards.get(species)
        return list(card["moves"]) if card else self.base.common_moves(species)

    def common_item(self, species: str | None) -> str:
        card = self.cards.get(species)
        return card.get("item") or "" if card else self.base.common_item(species)

    def abilities(self, species: str | None) -> list[str]:
        card = self.cards.get(species)
        return [card["ability"]] if card and card.get("ability") else self.base.abilities(species)

    def roles(self, species: str | None, moves: list[str] | None = None) -> dict:
        return self.base.roles(species, moves if moves is not None else self.common_moves(species))

    def brief(self, species: str | None, *, battle: bool = False) -> dict | None:
        found = self.base.brief(species, battle=battle)
        card = self.cards.get(species)
        if not card or found is None:
            return found
        found = dict(found)
        found["exact_set"] = {k: card[k] for k in ("item", "ability", "nature", "evs", "moves") if k in card}
        for key in ("common_moves", "common_item", "abilities"):
            found.pop(key, None)  # the card replaces the usage guesses
        return found

    def briefs(self, names: list[str], *, battle: bool = False) -> dict[str, dict]:
        found = {}
        for name in dict.fromkeys(names):
            brief = self.brief(name, battle=battle)
            if brief is not None:
                found[name] = brief
        return found
