"""FBref parser: detailed per-player stats (minutes, goals, assists, xG, xA).

FBref (Sports Reference) is our most detailed match-stats source. Two quirks the
parser must handle:

1. **Tables are wrapped in HTML comments** (``<!-- ... -->``) to defer rendering,
   so we re-parse comment contents to find them.
2. **Cells are keyed by ``data-stat`` attributes** (stable across layout tweaks),
   so we read by ``data-stat`` rather than column position.

Fetching is handled by :class:`scrapers.base.CachedFetcher`; this module only
parses bytes/str and is tested against a saved sample page.

Terms of use: Sports Reference restricts automated access and rate-limits hard --
fetch politely, cache verbatim, and pull only what you need.
"""

from __future__ import annotations

import re

import pandas as pd
from bs4 import BeautifulSoup, Comment

from sports_predictor.canonical.registry import CanonicalRegistry

SOURCE = "fbref"
MIN_MINUTES = 450  # below this, per-90 rates are noise -> low_data

_PLAYER_ID_RE = re.compile(r"/players/([0-9a-f]+)/")

# FBref data-stat -> our column name
_STAT_MAP = {
    "player": "player_name",
    "nationality": "nationality",
    "position": "position",
    "minutes": "minutes",
    "goals": "goals",
    "assists": "assists",
    "xg": "xg",
    "xg_assist": "xa",
}
_NUMERIC = ["minutes", "goals", "assists", "xg", "xa"]


def _find_table(soup: BeautifulSoup, table_id: str):
    """Find a table by id, including ones hidden inside HTML comments."""
    table = soup.find("table", id=table_id)
    if table is not None:
        return table
    for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
        if table_id in comment:
            inner = BeautifulSoup(comment, "html.parser").find("table", id=table_id)
            if inner is not None:
                return inner
    return None


def _cell(tr, stat: str) -> str | None:
    el = tr.find(attrs={"data-stat": stat})
    if el is None:
        return None
    text = el.get_text(strip=True)
    return text or None


def parse_player_stats(html: str | bytes, table_id: str = "stats_standard") -> pd.DataFrame:
    """Parse an FBref stats table into a per-player DataFrame.

    Columns: source, fbref_id, player_name, nationality, position, minutes,
    goals, assists, xg, xa. Repeated mid-table header rows are skipped.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = _find_table(soup, table_id)
    if table is None:
        raise ValueError(f"no FBref table id={table_id!r} found (layout changed?)")

    rows = []
    for tr in table.select("tbody > tr"):
        classes = tr.get("class") or []
        if "thead" in classes or "spacer" in classes:
            continue  # repeated header / spacer row
        player_el = tr.find(attrs={"data-stat": "player"})
        if player_el is None or not player_el.get_text(strip=True):
            continue

        link = player_el.find("a")
        href = link.get("href", "") if link else ""
        id_match = _PLAYER_ID_RE.search(href)

        record = {"source": SOURCE, "fbref_id": id_match.group(1) if id_match else None}
        for stat, col in _STAT_MAP.items():
            record[col] = _cell(tr, stat)
        rows.append(record)

    df = pd.DataFrame(rows, columns=["source", "fbref_id", *_STAT_MAP.values()])
    for col in _NUMERIC:
        df[col] = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce")
    return df


def to_canonical(stats: pd.DataFrame, registry: CanonicalRegistry | None = None) -> pd.DataFrame:
    """Attach canonical ``player_id`` and an ``xg_per_90`` feature.

    ``xg_per_90`` is NaN with ``low_data = 1`` for players below
    :data:`MIN_MINUTES` -- never a fabricated value off a tiny sample.
    """
    registry = registry or CanonicalRegistry()
    out = stats.copy()
    out["player_id"] = out["player_name"].map(lambda n: registry.player(n)["player_id"])
    enough = out["minutes"] >= MIN_MINUTES
    out["xg_per_90"] = (out["xg"] / (out["minutes"] / 90.0)).where(enough)
    out["low_data"] = (~enough).astype(float)
    return out
