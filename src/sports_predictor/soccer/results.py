"""Ingest international match results.

Source: "International football results from 1872 to present" (martj42).
    Repo:    https://github.com/martj42/international_results
    File:    results.csv (raw, served from GitHub)
    License: CC0-1.0 (public domain dedication) -- free to use and cache.
    Trust:   Tier 2 in DATA_SOURCES.md. Widely used, community maintained.
             Contains every men's full international, with score, venue, and a
             neutral-ground flag. Spot-check against a second source before
             trusting any single surprising row.

Pipeline (DATA_SOURCES.md §4):
    fetch -> save RAW verbatim (with as-of date) -> clean -> save processed.

The raw file is never edited. All normalization happens in :func:`clean`, so a
result is always reproducible from the cached raw snapshot.

Run end to end::

    python -m sports_predictor.soccer.results
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sports_predictor.core.paths import PROCESSED_DIR, RAW_DIR, ensure_dir
from sports_predictor.soccer.teams import normalize_team_name

SOURCE_NAME = "international_results"
SOURCE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# Columns we expect in the raw file. If the upstream schema changes, fail loudly
# rather than silently producing a wrong table.
RAW_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]

PROCESSED_FILENAME = "matches.parquet"


def _today_iso() -> str:
    return date.today().isoformat()


def fetch_raw(
    raw_dir: Path = RAW_DIR,
    url: str = SOURCE_URL,
    asof: str | None = None,
    timeout: int = 60,
) -> Path:
    """Download the results CSV and save it verbatim, returning the saved path.

    The file is named ``results_<asof>.csv`` where ``asof`` is the download date
    (ISO 8601). We save the bytes exactly as received, before any parsing, so the
    raw snapshot is an auditable record of what the source served.
    """
    asof = asof or _today_iso()
    out_dir = ensure_dir(Path(raw_dir) / SOURCE_NAME)
    out_path = out_dir / f"results_{asof}.csv"

    request = urllib.request.Request(url, headers={"User-Agent": "sports-predictor/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"fetch failed: HTTP {response.status} from {url}")
        payload = response.read()

    out_path.write_bytes(payload)

    # A tiny manifest records provenance next to the data.
    manifest = out_dir / f"results_{asof}.source.txt"
    sha256 = hashlib.sha256(payload).hexdigest()
    manifest.write_text(
        f"source: {SOURCE_NAME}\n"
        f"url: {url}\n"
        f"downloaded_at_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"asof: {asof}\n"
        f"bytes: {len(payload)}\n"
        f"sha256: {sha256}\n"
        f"license: CC0-1.0\n"
    )
    return out_path


def latest_raw(raw_dir: Path = RAW_DIR) -> Path:
    """Return the most recent cached raw snapshot, or raise if none exist."""
    out_dir = Path(raw_dir) / SOURCE_NAME
    snapshots = sorted(out_dir.glob("results_*.csv"))
    if not snapshots:
        raise FileNotFoundError(
            f"no raw snapshot in {out_dir}. Run fetch_raw() (or `python -m "
            f"sports_predictor.soccer.results`) first."
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


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw results into the canonical per-match schema.

    Output columns:
        match_id      stable id (hash of date + teams + score)
        date          datetime64 (UTC midnight; the source has date-only values)
        home_team     canonical name
        away_team     canonical name
        home_score    int
        away_score    int
        neutral       bool (True when played on neutral ground)
        tournament    raw tournament label (e.g. "FIFA World Cup")
        is_friendly   bool (tournament == "Friendly")
        is_world_cup  bool (World Cup finals or qualifiers)
        city, country str (venue)
        result        "H" / "D" / "A"  -- the OUTCOME, a label only

    NOTE ON LEAKAGE: ``home_score``, ``away_score`` and ``result`` describe what
    happened in the match. They are valid prediction targets but must never be
    fed to a model as input features (DATA_SOURCES.md §6).
    """
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", utc=True)

    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)

    # Scores: drop rows without a recorded result (e.g. future fixtures), then int.
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"])
    dropped_no_score = before - len(df)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df["neutral"] = df["neutral"].astype(bool)

    # Two national teams cannot play each other twice on the same day with an
    # identical score, so exact matches on this key are upstream data-entry
    # duplicates (e.g. the same game logged with two spellings of the venue).
    key_cols = ["date", "home_team", "away_team", "home_score", "away_score"]
    dup_mask = df.duplicated(subset=key_cols, keep="first")
    dropped_duplicates = int(dup_mask.sum())
    df = df[~dup_mask]

    df["tournament"] = df["tournament"].astype("string").str.strip()
    df["city"] = df["city"].astype("string").str.strip()
    df["country"] = df["country"].astype("string").str.strip()

    lowered = df["tournament"].str.casefold()
    df["is_friendly"] = (lowered == "friendly").astype(bool)
    df["is_world_cup"] = lowered.str.contains("fifa world cup", na=False).astype(bool)

    df["result"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["H", "A"],
        default="D",
    )
    df["result"] = df["result"].astype("string")

    df["match_id"] = [
        _match_id(d, h, a, hs, as_)
        for d, h, a, hs, as_ in zip(
            df["date"], df["home_team"], df["away_team"], df["home_score"], df["away_score"]
        )
    ]

    # Chronological order is required before computing any rolling feature later.
    df = df.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)

    columns = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "neutral",
        "tournament",
        "is_friendly",
        "is_world_cup",
        "city",
        "country",
        "result",
    ]
    cleaned = df[columns]
    cleaned.attrs["dropped_no_score"] = dropped_no_score
    cleaned.attrs["dropped_duplicates"] = dropped_duplicates
    _sanity_check(cleaned)
    return cleaned


def _match_id(date_val, home: str, away: str, home_score: int, away_score: int) -> str:
    key = f"{pd.Timestamp(date_val).date()}|{home}|{away}|{home_score}|{away_score}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _sanity_check(df: pd.DataFrame) -> None:
    """Fail loudly on impossible data; the contract values trust over volume."""
    problems: list[str] = []

    if (df["home_score"] < 0).any() or (df["away_score"] < 0).any():
        problems.append("negative scores present")

    if (df["home_team"] == df["away_team"]).any():
        problems.append("a team is listed against itself")

    today = pd.Timestamp.now(tz="UTC").normalize()
    if (df["date"] > today).any():
        problems.append("match dates in the future")
    if (df["date"] < pd.Timestamp("1872-01-01", tz="UTC")).any():
        problems.append("match dates before the first international (1872)")

    dup = df["match_id"].duplicated().sum()
    if dup:
        problems.append(f"{dup} duplicate match_id values")

    if problems:
        raise ValueError("sanity check failed: " + "; ".join(problems))


def build(
    refresh: bool = True,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Run the full pipeline and write the cleaned table to ``processed_dir``.

    With ``refresh=True`` a fresh raw snapshot is downloaded; otherwise the most
    recent cached snapshot is reused (offline / reproducible runs).
    """
    raw_path = fetch_raw(raw_dir=raw_dir) if refresh else latest_raw(raw_dir=raw_dir)
    raw = load_raw(raw_path)
    cleaned = clean(raw)

    out_dir = ensure_dir(Path(processed_dir))
    out_path = out_dir / PROCESSED_FILENAME
    cleaned.to_parquet(out_path, index=False)

    print(
        f"raw snapshot: {raw_path}\n"
        f"cleaned rows: {len(cleaned):,} "
        f"({cleaned['date'].min().date()} -> {cleaned['date'].max().date()})\n"
        f"dropped (no score): {cleaned.attrs.get('dropped_no_score', 0):,}\n"
        f"dropped (duplicates): {cleaned.attrs.get('dropped_duplicates', 0):,}\n"
        f"world cup matches: {int(cleaned['is_world_cup'].sum()):,}\n"
        f"written: {out_path}"
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
