"""Understat parser: per-player season xG/xA for the top European leagues.

Understat embeds its data as a JSON string inside a ``<script>`` tag, e.g.::

    var playersData = JSON.parse('[{\\x22id\\x22:\\x22001\\x22, ... }]');

so we extract that string, undo the ``\\xNN`` escaping, and ``json.loads`` it.
Fetching is done by :class:`scrapers.base.CachedFetcher`; this module only parses
bytes/str, so it is fully testable against a saved sample page.

Understat covers club football (top-5 leagues). For the World Cup it is an
*indirect* signal -- a national-team player's club xG form -- which is why it
feeds the canonical player table and is only later linked to nations via squads.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from sports_predictor.canonical.registry import CanonicalRegistry

SOURCE = "understat"

_PLAYERS_RE = re.compile(r"playersData\s*=\s*JSON\.parse\('(.*?)'\)\s*;", re.S)

# Minutes below this make per-90 rates unreliable -> low_data flag.
MIN_MINUTES = 450


def _decode_understat_blob(escaped: str) -> str:
    """Undo Understat's ``\\xNN`` escaping into clean UTF-8 text."""
    # \xNN are UTF-8 *bytes*; decode escapes, re-bytes via latin-1, then UTF-8.
    return escaped.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")


def parse_players(html: str | bytes) -> pd.DataFrame:
    """Parse an Understat league page into a per-player DataFrame.

    Columns: source, understat_id, player_name, team_title, games, minutes,
    goals, assists, xg, xa. Numeric fields are coerced to numbers.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")

    match = _PLAYERS_RE.search(html)
    if not match:
        raise ValueError("could not find playersData in the page (Understat layout changed?)")

    records = json.loads(_decode_understat_blob(match.group(1)))

    df = pd.DataFrame.from_records(records)
    rename = {
        "id": "understat_id",
        "time": "minutes",
        "xG": "xg",
        "xA": "xa",
    }
    df = df.rename(columns=rename)

    df["source"] = SOURCE
    for col in ("games", "minutes", "goals", "assists", "xg", "xa"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    columns = [
        "source", "understat_id", "player_name", "team_title",
        "games", "minutes", "goals", "assists", "xg", "xa",
    ]
    return df[[c for c in columns if c in df.columns]]


def to_canonical(players: pd.DataFrame, registry: CanonicalRegistry | None = None) -> pd.DataFrame:
    """Attach canonical ``player_id`` and one feature (``xg_per_90``).

    ``xg_per_90`` is NaN with ``low_data = 1`` when minutes are below
    :data:`MIN_MINUTES` (a per-90 rate off a tiny sample is noise, not signal) --
    never a fabricated zero.
    """
    registry = registry or CanonicalRegistry()
    out = players.copy()
    out["player_id"] = out["player_name"].map(lambda n: registry.player(n)["player_id"])

    enough = out["minutes"] >= MIN_MINUTES
    out["xg_per_90"] = (out["xg"] / (out["minutes"] / 90.0)).where(enough)
    out["low_data"] = (~enough).astype(float)
    return out
