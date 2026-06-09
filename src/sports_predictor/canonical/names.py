"""Name normalization for cross-source matching.

The hard part of merging football sources is that the same person appears as
"Kylian Mbappé", "Kylian Mbappe", "K. Mbappé", "Mbappe, Kylian", … . We never try
to *display* a corrected name; instead we compute a stable **match key** that
collapses these variants so the registry can decide they are the same entity.

``name_key`` is intentionally aggressive (accent-folded, punctuation-stripped,
lowercased, whitespace-collapsed) because under-matching (treating one player as
two) is the costlier error here. ``display_name`` just tidies whitespace for
human-facing output. Team names reuse the existing canonical map in
``soccer.teams`` so the two layers never diverge.
"""

from __future__ import annotations

import re
import unicodedata

from sports_predictor.soccer.teams import normalize_team_name

_WHITESPACE_RE = re.compile(r"\s+")
_NON_KEY_RE = re.compile(r"[^a-z0-9 ]+")


def strip_accents(text: str) -> str:
    """Remove diacritics: 'Mbappé' -> 'Mbappe', 'Özil' -> 'Ozil'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def display_name(name: str) -> str:
    """Trim and collapse internal whitespace; preserve original casing/accents."""
    if name is None:
        raise ValueError("name cannot be None")
    return _WHITESPACE_RE.sub(" ", str(name).strip())


def name_key(name: str) -> str:
    """Return an accent/punctuation-insensitive match key for a player name.

    Handles "Last, First" by swapping to "First Last" before keying, so
    "Mbappé, Kylian" and "Kylian Mbappe" collapse to the same key.
    """
    if name is None:
        raise ValueError("name cannot be None")
    cleaned = display_name(name)
    if "," in cleaned:
        last, _, first = cleaned.partition(",")
        cleaned = f"{first.strip()} {last.strip()}".strip()
    folded = strip_accents(cleaned).casefold()
    # Apostrophes join ("N'Golo" -> "ngolo"); other punctuation becomes a space
    # ("C." -> "c ", "Jean-Pierre" -> "jean pierre").
    folded = folded.replace("'", "").replace("\u2019", "").replace("`", "")
    folded = _NON_KEY_RE.sub(" ", folded)
    return _WHITESPACE_RE.sub(" ", folded).strip()


def team_key(name: str) -> str:
    """Canonical team match key (folded form of the canonical team name)."""
    return name_key(normalize_team_name(name))
