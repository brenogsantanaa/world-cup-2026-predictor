"""Chronological train/test splitting.

Sports results are a time series: we want to train on the past and test on the
future, exactly as we would have to in real life. A random split lets the model
peek at later matches while scoring earlier ones, which inflates results and is a
form of leakage (``DATA_SOURCES.md`` §6). These helpers split strictly by time.
"""

from __future__ import annotations

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    date_column: str = "date",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into earlier (train) and later (test) parts by date.

    The most recent ``test_fraction`` of rows become the test set; the rest are
    train. Rows are sorted by ``date_column`` first, so the input order does not
    matter. Returns two new DataFrames (the input is not modified).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if date_column not in df.columns:
        raise KeyError(f"date_column {date_column!r} not found in DataFrame")

    ordered = df.sort_values(date_column, kind="stable").reset_index(drop=True)
    split_at = int(len(ordered) * (1.0 - test_fraction))
    train = ordered.iloc[:split_at].reset_index(drop=True)
    test = ordered.iloc[split_at:].reset_index(drop=True)
    return train, test


def split_by_cutoff_date(
    df: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so that train is strictly before ``cutoff`` and test is on/after it.

    Useful for honest backtests, e.g. "train on everything before the 2022 World
    Cup, test on it." Returns two new DataFrames.
    """
    if date_column not in df.columns:
        raise KeyError(f"date_column {date_column!r} not found in DataFrame")

    cutoff_ts = pd.Timestamp(cutoff)
    dates = pd.to_datetime(df[date_column])
    train = df[dates < cutoff_ts].reset_index(drop=True)
    test = df[dates >= cutoff_ts].reset_index(drop=True)
    return train, test
