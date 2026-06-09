"""Ingest international goalscorers.

Source: "International football results from 1872 to present" (martj42) -- the same
repo and license as our match results, so it shares provenance and trust.
    Repo:    https://github.com/martj42/international_results
    File:    goalscorers.csv (raw, served from GitHub)
    License: CC0-1.0 (public domain dedication) -- free to use and cache.
    Trust:   Tier 2 in DATA_SOURCES.md. One row per goal: date, the two teams,
             the team it counted for, the scorer, the minute, and own-goal /
             penalty flags. Coverage is uneven (goals are recorded reliably for
             well-covered fixtures, sparsely for obscure ones), so any feature
             built from it must degrade gracefully -- see player_features.py.

Pipeline (DATA_SOURCES.md §4): fetch -> save RAW verbatim (with as-of date) ->
clean -> save processed. The raw file is never edited; all normalization happens
in :func:`clean`, so the processed table is always reproducible from the cache.

Run end to end::

    python -m sports_predictor.soccer.goalscorers
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from sports_predictor.core.paths import PROCESSED_DIR, RAW_DIR, ensure_dir
from sports_predictor.soccer.teams import normalize_team_name

SOURCE_NAME = "international_results"
SOURCE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
)

RAW_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "team",
    "scorer",
    "minute",
    "own_goal",
    "penalty",
]

PROCESSED_FILENAME = "goalscorers.parquet"


def _today_iso() -> str:
    return date.today().isoformat()


def fetch_raw(
    raw_dir: Path = RAW_DIR,
    url: str = SOURCE_URL,
    asof: str | None = None,
    timeout: int = 60,
) -> Path:
    """Download goalscorers.csv and save it verbatim, returning the saved path."""
    asof = asof or _today_iso()
    out_dir = ensure_dir(Path(raw_dir) / SOURCE_NAME)
    out_path = out_dir / f"goalscorers_{asof}.csv"

    request = urllib.request.Request(url, headers={"User-Agent": "sports-predictor/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"fetch failed: HTTP {response.status} from {url}")
        payload = response.read()

    out_path.write_bytes(payload)

    manifest = out_dir / f"goalscorers_{asof}.source.txt"
    sha256 = hashlib.sha256(payload).hexdigest()
    manifest.write_text(
        f"source: {SOURCE_NAME}\n"
        f"file: goalscorers.csv\n"
        f"url: {url}\n"
        f"downloaded_at_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"asof: {asof}\n"
        f"bytes: {len(payload)}\n"
        f"sha256: {sha256}\n"
        f"license: CC0-1.0\n"
    )
    return out_path


def latest_raw(raw_dir: Path = RAW_DIR) -> Path:
    """Return the most recent cached goalscorers snapshot, or raise if none."""
    out_dir = Path(raw_dir) / SOURCE_NAME
    snapshots = sorted(out_dir.glob("goalscorers_*.csv"))
    if not snapshots:
        raise FileNotFoundError(
            f"no goalscorers snapshot in {out_dir}. Run fetch_raw() (or `python -m "
            f"sports_predictor.soccer.goalscorers`) first."
        )
    return snapshots[-1]


def load_raw(path: Path) -> pd.DataFrame:
    """Read a cached raw snapshot into a DataFrame, validating its columns."""
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"raw file {path} is missing expected columns {missing}. The upstream "
            f"schema may have changed; review before trusting this data."
        )
    return df


def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce a CSV truthy column (True/False, true/false, 1/0) to bool."""
    if series.dtype == bool:
        return series
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .isin({"true", "1", "yes"})
    )


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw goalscorers into a canonical per-goal schema.

    Output columns:
        date        datetime64 (UTC midnight)
        home_team   canonical name
        away_team   canonical name
        team        canonical name -- the team the goal counted *for*
        scorer      player name (trimmed; whitespace collapsed)
        minute      raw minute string/number (kept as-is; may be missing)
        own_goal    bool
        penalty     bool

    Note: for an own goal the ``scorer`` is a player on the *opposing* team, so
    own goals must be excluded when crediting a player's scoring record (handled
    in player_features.py, not here -- this stage stays faithful to the source).
    """
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", utc=True)

    for col in ("home_team", "away_team", "team"):
        df[col] = df[col].map(normalize_team_name)

    df["scorer"] = df["scorer"].astype("string").str.strip().str.replace(
        r"\s+", " ", regex=True
    )
    df["minute"] = df["minute"].astype("string").str.strip()

    df["own_goal"] = _to_bool(df["own_goal"])
    df["penalty"] = _to_bool(df["penalty"])

    df = df.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)

    columns = ["date", "home_team", "away_team", "team", "scorer", "own_goal", "penalty", "minute"]
    cleaned = df[columns]
    _sanity_check(cleaned)
    return cleaned


def _sanity_check(df: pd.DataFrame) -> None:
    problems: list[str] = []

    today = pd.Timestamp.now(tz="UTC").normalize()
    if (df["date"] > today).any():
        problems.append("goal dates in the future")
    if (df["date"] < pd.Timestamp("1872-01-01", tz="UTC")).any():
        problems.append("goal dates before the first international (1872)")

    # The scoring team should be one of the two teams on the pitch.
    off_pitch = (df["team"] != df["home_team"]) & (df["team"] != df["away_team"])
    if off_pitch.any():
        problems.append(f"{int(off_pitch.sum())} goals credited to a team not in the match")

    if problems:
        raise ValueError("sanity check failed: " + "; ".join(problems))


def build(
    refresh: bool = True,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Run the full pipeline and write the cleaned table to ``processed_dir``."""
    raw_path = fetch_raw(raw_dir=raw_dir) if refresh else latest_raw(raw_dir=raw_dir)
    raw = load_raw(raw_path)
    cleaned = clean(raw)

    out_dir = ensure_dir(Path(processed_dir))
    out_path = out_dir / PROCESSED_FILENAME
    cleaned.to_parquet(out_path, index=False)

    print(
        f"raw snapshot: {raw_path}\n"
        f"goal rows:    {len(cleaned):,} "
        f"({cleaned['date'].min().date()} -> {cleaned['date'].max().date()})\n"
        f"own goals:    {int(cleaned['own_goal'].sum()):,}\n"
        f"penalties:    {int(cleaned['penalty'].sum()):,}\n"
        f"distinct scorers: {cleaned['scorer'].nunique():,}\n"
        f"written:      {out_path}"
    )
    return cleaned


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="reuse the latest cached raw snapshot instead of downloading.",
    )
    args = parser.parse_args()
    build(refresh=not args.no_refresh)


if __name__ == "__main__":
    _main()
