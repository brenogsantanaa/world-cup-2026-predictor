"""Stable canonical IDs for teams and players.

Goal: any record from any source maps to one ``team_id`` / ``player_id``, so we
can join facts about the same entity regardless of spelling.

Design choices:

- **Deterministic IDs from the match key** (a short hash). This makes IDs stable
  and reproducible across runs without persisting a counter, so two pipeline runs
  on the same data produce identical IDs.
- **Aliases merge keys.** When we learn that "Cristiano Ronaldo" and "Cristiano
  Ronaldo dos Santos Aveiro" are the same person, we point the alias key at the
  canonical key; both then resolve to one id.
- Team ids reuse the existing canonical team names (``soccer.teams``), so the
  registry never invents a team the rest of the codebase doesn't know.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from sports_predictor.canonical.names import display_name, name_key, team_key
from sports_predictor.soccer.teams import normalize_team_name


def _hash_id(prefix: str, key: str) -> str:
    return f"{prefix}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


def team_id(name: str) -> str:
    """Stable id for a team, derived from its canonical name."""
    return _hash_id("tm", team_key(name))


@dataclass
class CanonicalRegistry:
    """Resolves source names to stable canonical ids, with alias merging."""

    # alias match key -> canonical match key
    _player_aliases: dict[str, str] = field(default_factory=dict)
    # canonical match key -> chosen display name (first non-abbreviated seen)
    _player_display: dict[str, str] = field(default_factory=dict)

    # ----- teams ----------------------------------------------------------- #
    def team(self, name: str) -> dict:
        """Return ``{'team_id', 'name'}`` for a (possibly non-canonical) team name."""
        canonical = normalize_team_name(name)
        return {"team_id": team_id(canonical), "name": canonical}

    # ----- players --------------------------------------------------------- #
    def add_player_alias(self, alias: str, canonical: str) -> None:
        """Record that ``alias`` refers to the same player as ``canonical``."""
        c_key = self._resolve_key(name_key(canonical))
        self._player_aliases[name_key(alias)] = c_key
        self._player_display.setdefault(c_key, display_name(canonical))

    def _resolve_key(self, key: str) -> str:
        """Follow alias chains to the canonical key (one hop is the norm)."""
        seen = set()
        while key in self._player_aliases and key not in seen:
            seen.add(key)
            key = self._player_aliases[key]
        return key

    def player(self, name: str) -> dict:
        """Return ``{'player_id', 'name', 'key'}`` for a player name.

        Registers the player on first sight; the first display name seen for a key
        is kept (deterministic). Abbreviated spellings like "F. Last" key
        differently from "First Last" and are *not* auto-merged -- use
        :meth:`add_player_alias` when you know two spellings are the same person.
        """
        key = self._resolve_key(name_key(name))
        self._player_display.setdefault(key, display_name(name))
        return {"player_id": _hash_id("plr", key), "name": self._player_display[key], "key": key}

    def player_id(self, name: str) -> str:
        return self.player(name)["player_id"]
