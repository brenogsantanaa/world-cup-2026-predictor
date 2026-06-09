"""Tests for FIFA-ranking ingestion and leakage-safe as-of features."""

import numpy as np
import pandas as pd

from sports_predictor.soccer.fifa_features import build_fifa_features
from sports_predictor.soccer.fifa_ranking import clean


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


RANKINGS = pd.DataFrame(
    {
        "date": [_ts("2020-01-01"), _ts("2020-06-01"), _ts("2020-01-01")],
        "team": ["A", "A", "B"],
        "rank": [5, 3, 10],
        "points": [100.0, 120.0, 80.0],
    }
)


def _matches(rows):
    return pd.DataFrame(
        [
            {"match_id": mid, "date": _ts(d), "home_team": h, "away_team": a}
            for mid, d, h, a in rows
        ]
    )


def test_asof_uses_latest_publication_before_match():
    matches = _matches([("m1", "2020-03-01", "A", "B")])
    feats = build_fifa_features(matches, RANKINGS)
    # March match uses the Jan publication (rank 5), not the June one.
    assert feats.loc["m1", "home_fifa_rank"] == 5
    assert feats.loc["m1", "away_fifa_rank"] == 10
    assert feats.loc["m1", "fifa_rank_diff"] == -5  # home better (lower rank)
    assert feats.loc["m1", "fifa_points_diff"] == 20.0
    assert feats.loc["m1", "home_fifa_low_data"] == 0.0


def test_same_day_publication_is_excluded():
    matches = _matches([("m2", "2020-06-01", "A", "B")])
    feats = build_fifa_features(matches, RANKINGS)
    # A publication dated the same day as the match must NOT be used (leakage).
    assert feats.loc["m2", "home_fifa_rank"] == 5  # still the Jan value


def test_missing_ranking_is_nan_and_flagged():
    matches = _matches([("m3", "2019-01-01", "A", "B")])
    feats = build_fifa_features(matches, RANKINGS)
    # No publication exists before this match -> NaN + flag, never a fake number.
    assert np.isnan(feats.loc["m3", "home_fifa_rank"])
    assert np.isnan(feats.loc["m3", "fifa_rank_diff"])
    assert feats.loc["m3", "home_fifa_low_data"] == 1.0
    assert feats.loc["m3", "away_fifa_low_data"] == 1.0


def test_row_count_preserved():
    matches = _matches([("m1", "2020-03-01", "A", "B"), ("m3", "2019-01-01", "A", "B")])
    feats = build_fifa_features(matches, RANKINGS)
    assert len(feats) == 2
    assert set(feats.index) == {"m1", "m3"}


def test_clean_canonicalizes_team_names():
    raw = pd.DataFrame(
        {
            "rank": [1],
            "country_full": ["Korea Republic"],
            "total_points": [1500.0],
            "rank_date": ["2018-06-07"],
        }
    )
    cleaned = clean(raw)
    assert cleaned.loc[0, "team"] == "South Korea"
