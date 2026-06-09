"""Transfermarkt parser: national-team squad lists with market value / age / position.

Transfermarkt is the authority for squad composition and market value (see the
conflict-resolution priorities in ``canonical.conflicts``). Squad pages are nested
HTML tables, so we parse with BeautifulSoup (stdlib ``html.parser`` backend) rather
than fragile regexes. Fetching is handled by :class:`scrapers.base.CachedFetcher`;
this module only parses bytes/str and is tested against a saved sample page.

Terms of use: Transfermarkt restricts automated access -- fetch politely (slow
rate limit), cache verbatim, and pull only the squad pages you need.
"""

from __future__ import annotations

import re

import pandas as pd
from bs4 import BeautifulSoup

from sports_predictor.canonical.registry import CanonicalRegistry

SOURCE = "transfermarkt"

_SPIELER_ID_RE = re.compile(r"/spieler/(\d+)")
_AGE_RE = re.compile(r"\((\d+)\)")


def parse_market_value(text: str | None) -> float | None:
    """'€180.00m' -> 180000000.0, '€500k' -> 500000.0, '-'/'' -> None."""
    if text is None:
        return None
    t = text.replace("\u20ac", "").replace("\xa0", " ").strip()
    if t in ("", "-", "\u2014", "?"):
        return None
    t = t.replace(",", "")
    mult = 1.0
    low = t.lower()
    if low.endswith("bn"):
        mult, t = 1e9, t[:-2]
    elif low.endswith("m"):
        mult, t = 1e6, t[:-1]
    elif low.endswith("k"):
        mult, t = 1e3, t[:-1]
    try:
        return float(t) * mult
    except ValueError:
        return None


def _age(text: str | None) -> float | None:
    if not text:
        return None
    m = _AGE_RE.search(text)
    if m:
        return float(m.group(1))
    try:
        return float(text.strip())
    except ValueError:
        return None


def parse_squad(html: str | bytes) -> pd.DataFrame:
    """Parse a Transfermarkt squad page into a per-player DataFrame.

    Columns: source, transfermarkt_id, player_name, position, age, nationality,
    market_value_eur. Missing market values are NaN (never fabricated).
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.items")
    if table is None:
        raise ValueError("no squad table (table.items) found (Transfermarkt layout changed?)")

    rows = []
    for tr in table.select("tbody > tr"):
        posrela = tr.select_one("td.posrela")
        if posrela is None:
            continue  # spacer / non-player row
        name_a = posrela.select_one("a")
        if name_a is None:
            continue

        href = name_a.get("href", "")
        id_match = _SPIELER_ID_RE.search(href)
        inline_rows = posrela.select("table.inline-table tr")
        position = inline_rows[-1].get_text(strip=True) if len(inline_rows) > 1 else None

        flag = tr.select_one("img.flaggenrahmen")
        nationality = flag.get("title") if flag is not None else None

        value_cell = tr.select_one("td.rechts")
        market_value = parse_market_value(value_cell.get_text(strip=True)) if value_cell else None

        # Age cell: the zentriert cell containing a "(NN)" age.
        age = None
        for td in tr.select("td.zentriert"):
            if _AGE_RE.search(td.get_text()):
                age = _age(td.get_text())
                break

        rows.append(
            {
                "source": SOURCE,
                "transfermarkt_id": id_match.group(1) if id_match else None,
                "player_name": name_a.get_text(strip=True),
                "position": position,
                "age": age,
                "nationality": nationality,
                "market_value_eur": market_value,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "source", "transfermarkt_id", "player_name", "position",
            "age", "nationality", "market_value_eur",
        ],
    )


def to_canonical(squad: pd.DataFrame, registry: CanonicalRegistry | None = None) -> pd.DataFrame:
    """Attach canonical ``player_id`` / ``team_id`` and a ``low_data`` flag.

    The carried feature is ``market_value_eur``; ``low_data = 1`` marks players
    with no market value, so downstream aggregation can degrade gracefully instead
    of treating "unknown" as "worth zero".
    """
    registry = registry or CanonicalRegistry()
    out = squad.copy()
    out["player_id"] = out["player_name"].map(lambda n: registry.player(n)["player_id"])
    out["team_id"] = out["nationality"].map(
        lambda t: registry.team(t)["team_id"] if isinstance(t, str) and t else None
    )
    out["team"] = out["nationality"].map(
        lambda t: registry.team(t)["name"] if isinstance(t, str) and t else None
    )
    out["low_data"] = out["market_value_eur"].isna().astype(float)
    return out
