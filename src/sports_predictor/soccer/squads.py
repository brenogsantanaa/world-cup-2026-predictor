"""National-team squad assembly -- the linchpin between club stats and nations.

Club-stat sources (Understat, FBref) know players, not nationalities. To turn a
player's club form into a *national-team* feature we need to know which nation
each player represents. That mapping comes from national-team **squad lists**
(Transfermarkt is the natural source -- we already parse its squad pages).

This module assembles parsed squad pages into one canonical table and exposes:

- :func:`assemble_squads` — one row per (nation, player) with canonical ids.
- :func:`player_nation_map` — ``player_id -> team_id`` (each senior player has
  exactly one national team).
- :func:`attach_nation` — join a club-stats table to nations via ``player_id``;
  players not in any squad get NaN + ``in_squad = 0`` (never fabricated).
- :func:`squad_value` — per-nation aggregate market value with honest coverage /
  ``low_data`` reporting.

Fetching the actual pages is the scrapers' job (cache-first); this module works
from already-parsed frames so it is fully testable without the network.
"""

from __future__ import annotations

import pandas as pd

from sports_predictor.canonical.registry import CanonicalRegistry
from sports_predictor.scrapers import transfermarkt

SQUAD_COLUMNS = [
    "team_id", "team", "player_id", "player_name",
    "position", "age", "market_value_eur", "source",
]


def assemble_squads(
    parsed_squads, registry: CanonicalRegistry | None = None
) -> pd.DataFrame:
    """Combine parsed Transfermarkt squad frames into one canonical squad table.

    ``parsed_squads`` is an iterable of DataFrames in ``transfermarkt.parse_squad``
    schema. Duplicate (nation, player) rows are collapsed, keeping the highest
    known market value.
    """
    registry = registry or CanonicalRegistry()
    frames = [transfermarkt.to_canonical(df, registry) for df in parsed_squads]
    if not frames:
        return pd.DataFrame(columns=SQUAD_COLUMNS)

    squads = pd.concat(frames, ignore_index=True)
    squads = squads.sort_values("market_value_eur", ascending=False, na_position="last")
    squads = squads.drop_duplicates(["team_id", "player_id"], keep="first")
    return squads[SQUAD_COLUMNS].reset_index(drop=True)


def player_nation_map(squads: pd.DataFrame) -> dict[str, str]:
    """Return ``{player_id: team_id}`` for every player in the squads table."""
    return dict(zip(squads["player_id"], squads["team_id"]))


def attach_nation(player_stats: pd.DataFrame, squads: pd.DataFrame) -> pd.DataFrame:
    """Attach national ``team_id`` / ``team`` to a club-stats table by player_id.

    Players absent from every squad get NaN nation and ``in_squad = 0`` -- the
    honest "we don't know which nation" state, not a fabricated assignment.
    """
    nation = player_nation_map(squads)
    team_name = dict(zip(squads["player_id"], squads["team"]))
    out = player_stats.copy()
    out["team_id"] = out["player_id"].map(nation)
    out["team"] = out["player_id"].map(team_name)
    out["in_squad"] = out["team_id"].notna().astype(float)
    return out


def squad_value(squads: pd.DataFrame) -> pd.DataFrame:
    """Per-nation aggregate market value with explicit coverage / low_data.

    ``total_value_eur`` sums *known* values only; ``coverage`` is the share of the
    squad with a value, and ``low_data = 1`` when under half the squad is valued,
    so a sparsely-covered nation is flagged rather than silently understated.
    """
    grouped = squads.groupby(["team_id", "team"], dropna=False)
    out = grouped.agg(
        total_value_eur=("market_value_eur", "sum"),
        n_players=("player_id", "size"),
        n_valued=("market_value_eur", lambda s: int(s.notna().sum())),
    ).reset_index()
    out["coverage"] = out["n_valued"] / out["n_players"]
    out["low_data"] = (out["coverage"] < 0.5).astype(float)
    return out.sort_values("total_value_eur", ascending=False).reset_index(drop=True)
