"""Ingest historical FIFA/Coca-Cola Men's World Ranking (as-of publication date).

The FIFA ranking is a Tier-1 source in DATA_SOURCES.md and -- unlike Elo/form --
is computed by FIFA itself (fixed periods, confederation weights), so it is a
partly *independent* strength signal worth testing.

Source: the well-known community time series of the official ranking, one row per
team per publication date.
    URL:     a GitHub-hosted mirror of the Kaggle "FIFA ranking" dataset
    License: the rankings themselves are FIFA's published facts; this CSV is a
             community redistribution. Provenance (URL + SHA-256 + as-of) is
             recorded; verify terms before any redistribution of our own.
    Trust:   Tier 1 for the numbers, Tier 2 for the delivery mechanism. Spot
             check a couple of well-known dates against FIFA's site.

IMPORTANT COVERAGE CAVEAT: this snapshot covers **1993-08 to 2018-06** only.
Matches after mid-2018 (including the 2022 World Cup) therefore have *no* FIFA
ranking available and must degrade gracefully (NaN + low-data flag), never a
fabricated value. See fifa_features.py.

Pipeline (DATA_SOURCES.md §4): fetch -> raw cache (with as-of) -> clean -> parquet.

Run::

    python -m sports_predictor.soccer.fifa_ranking
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

SOURCE_NAME = "fifa_ranking"
SOURCE_URL = (
    "https://raw.githubusercontent.com/Vaiosue/ETL_Project/HEAD/Resources/fifa_ranking.csv"
)

# We only depend on these columns; the source has more (year-average breakdowns).
RAW_COLUMNS = ["rank", "country_full", "total_points", "rank_date"]

PROCESSED_FILENAME = "fifa_ranking.parquet"


def _today_iso() -> str:
    return date.today().isoformat()


def fetch_raw(
    raw_dir: Path = RAW_DIR,
    url: str = SOURCE_URL,
    asof: str | None = None,
    timeout: int = 60,
) -> Path:
    """Download the ranking CSV and save it verbatim, returning the saved path."""
    asof = asof or _today_iso()
    out_dir = ensure_dir(Path(raw_dir) / SOURCE_NAME)
    out_path = out_dir / f"fifa_ranking_{asof}.csv"

    request = urllib.request.Request(url, headers={"User-Agent": "sports-predictor/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"fetch failed: HTTP {response.status} from {url}")
        payload = response.read()

    out_path.write_bytes(payload)

    manifest = out_dir / f"fifa_ranking_{asof}.source.txt"
    sha256 = hashlib.sha256(payload).hexdigest()
    manifest.write_text(
        f"source: {SOURCE_NAME}\n"
        f"url: {url}\n"
        f"downloaded_at_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"asof: {asof}\n"
        f"bytes: {len(payload)}\n"
        f"sha256: {sha256}\n"
        f"license: FIFA official ranking (facts); community CSV redistribution -- verify terms\n"
    )
    return out_path


def latest_raw(raw_dir: Path = RAW_DIR) -> Path:
    out_dir = Path(raw_dir) / SOURCE_NAME
    snapshots = sorted(out_dir.glob("fifa_ranking_*.csv"))
    if not snapshots:
        raise FileNotFoundError(
            f"no FIFA ranking snapshot in {out_dir}. Run fetch_raw() (or `python -m "
            f"sports_predictor.soccer.fifa_ranking`) first."
        )
    return snapshots[-1]


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"raw file {path} is missing expected columns {missing}. The upstream "
            f"schema may have changed; review before trusting this data."
        )
    return df


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize into a canonical (team, date, rank, points) table.

    Output columns:
        date     publication date (datetime64, UTC) -- the as-of date
        team     canonical national-team name
        rank     integer FIFA rank on that date
        points   FIFA ranking points on that date
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["rank_date"], utc=True)
    df["team"] = df["country_full"].map(normalize_team_name)
    df["rank"] = df["rank"].astype(int)
    df["points"] = df["total_points"].astype(float)

    df = df[["date", "team", "rank", "points"]].sort_values(
        ["date", "rank"], kind="stable"
    )

    # A team can appear at most once per publication; extra rows are upstream
    # data-entry duplicates (the source lists "Sudan" twice on some dates). Keep
    # the first (best-ranked) and report the count.
    dup_mask = df.duplicated(subset=["date", "team"], keep="first")
    dropped_duplicates = int(dup_mask.sum())
    df = df[~dup_mask].reset_index(drop=True)
    df.attrs["dropped_duplicates"] = dropped_duplicates

    _sanity_check(df)
    return df


def _sanity_check(df: pd.DataFrame) -> None:
    problems: list[str] = []
    if (df["rank"] < 1).any():
        problems.append("non-positive ranks present")
    if (df["points"] < 0).any():
        problems.append("negative ranking points present")
    # A given team should appear at most once per publication date.
    dup = df.duplicated(subset=["date", "team"]).sum()
    if dup:
        problems.append(f"{dup} duplicate (date, team) rows")
    if problems:
        raise ValueError("sanity check failed: " + "; ".join(problems))


def build(
    refresh: bool = True,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    raw_path = fetch_raw(raw_dir=raw_dir) if refresh else latest_raw(raw_dir=raw_dir)
    raw = load_raw(raw_path)
    cleaned = clean(raw)

    out_dir = ensure_dir(Path(processed_dir))
    out_path = out_dir / PROCESSED_FILENAME
    cleaned.to_parquet(out_path, index=False)

    print(
        f"raw snapshot: {raw_path}\n"
        f"rows:         {len(cleaned):,} "
        f"({cleaned['date'].min().date()} -> {cleaned['date'].max().date()})\n"
        f"publications: {cleaned['date'].nunique():,}\n"
        f"teams:        {cleaned['team'].nunique():,}\n"
        f"dropped (duplicates): {cleaned.attrs.get('dropped_duplicates', 0):,}\n"
        f"written:      {out_path}"
    )
    return cleaned


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-refresh", action="store_true", help="reuse cached snapshot.")
    args = parser.parse_args()
    build(refresh=not args.no_refresh)


if __name__ == "__main__":
    _main()
