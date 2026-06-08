"""Canonical national-team names.

Different sources spell the same country differently ("USA" vs "United States",
"Korea Republic" vs "South Korea", "IR Iran" vs "Iran"). Merging two sources on
mismatched names silently drops rows, which is one of the most common and most
damaging data bugs (see ``DATA_SOURCES.md`` §4).

To avoid that, we pick a single *canonical* spelling for every team and translate
everything else to it. We adopt the spellings used by the international results
dataset (martj42 / "International football results from 1872 to present") as the
canonical set, because that dataset is our backbone source. Aliases below map the
spellings used by other sources (FIFA ranking, eloratings.net) onto it.

Only add an alias you are confident about. A wrong mapping merges two different
countries together, which is worse than leaving a name unmatched.
"""

from __future__ import annotations

import re

# Aliases -> canonical spelling.
# Keys are matched case-insensitively after whitespace is normalized, so list the
# variant exactly once in any convenient casing.
TEAM_ALIASES: dict[str, str] = {
    # United States
    "usa": "United States",
    "united states of america": "United States",
    # Koreas (FIFA uses "Korea Republic" / "Korea DPR")
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "korea dpr": "North Korea",
    "dpr korea": "North Korea",
    # Iran (FIFA/AFC use "IR Iran")
    "ir iran": "Iran",
    "iran (islamic republic of)": "Iran",
    # China (FIFA uses "China PR"; results dataset also uses "China PR")
    "china": "China PR",
    "china (pr)": "China PR",
    # Ivory Coast (FIFA uses "Côte d'Ivoire")
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    # Cape Verde (FIFA uses "Cabo Verde")
    "cabo verde": "Cape Verde",
    # Turkey (FIFA now uses "Türkiye")
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    # Ireland
    "ireland": "Republic of Ireland",
    # Bosnia
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    # Czechia (results dataset uses "Czech Republic")
    "czechia": "Czech Republic",
    # Congo
    "congo dr": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "congo-brazzaville": "Congo",
    # North Macedonia (was "Macedonia" pre-2019)
    "macedonia": "North Macedonia",
    "fyr macedonia": "North Macedonia",
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_team_name(name: str) -> str:
    """Return the canonical spelling for a team name.

    Trims and collapses internal whitespace, then applies the alias map
    (case-insensitively). Unknown names are returned cleaned but unchanged, so a
    new spelling is preserved rather than discarded; use
    :func:`find_unmapped_teams` to surface names that need attention.
    """
    if name is None:
        raise ValueError("team name cannot be None")
    cleaned = _WHITESPACE_RE.sub(" ", str(name).strip())
    return TEAM_ALIASES.get(cleaned.casefold(), cleaned)


def find_unmapped_teams(names: object, known: set[str]) -> set[str]:
    """Return the canonical names in ``names`` that are not in ``known``.

    Use this as a sanity check when joining a new source against an existing
    canonical set: any returned name is a spelling we have not reconciled yet and
    would silently fail to merge.
    """
    return {normalize_team_name(n) for n in names} - known
