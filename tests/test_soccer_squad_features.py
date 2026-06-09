"""Tests for team-match aggregation of player profiles + backtest harness."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sports_predictor.canonical.registry import team_id as team_id_of
from sports_predictor.soccer.squad_features import (
    aggregate_nation_features,
    build_team_match_features,
    run_slice_backtest,
)

FR = team_id_of("France")
BR = team_id_of("Brazil")


def _player_table():
    # France: one star (high xG, valued), one mid, one uncovered reserve.
    # Brazil: two valued, one with xG.
    return pd.DataFrame(
        {
            "team_id": [FR, FR, FR, BR, BR],
            "player_id": ["p1", "p2", "p3", "p4", "p5"],
            "xg_per_90": [0.9, 0.1, np.nan, 0.7, np.nan],
            "market_value_eur": [180e6, 80e6, np.nan, 200e6, 18e6],
        }
    )


def test_aggregate_nation_features_values_and_coverage():
    agg = aggregate_nation_features(_player_table()).set_index("team_id")
    fr = agg.loc[FR]
    assert fr["squad_size"] == 3
    assert fr["squad_value_eur"] == pytest.approx(260e6)  # known values only
    assert fr["top_xg90"] == pytest.approx(0.9)
    assert fr["value_coverage"] == pytest.approx(2 / 3)
    # one of two xG-covered France players is "in form" (>=0.4): 0.9 yes, 0.1 no
    assert fr["share_in_form"] == pytest.approx(0.5)


def test_aggregate_low_data_flag_when_sparse():
    sparse = pd.DataFrame(
        {
            "team_id": [BR, BR, BR, BR],
            "player_id": ["a", "b", "c", "d"],
            "xg_per_90": [0.5, np.nan, np.nan, np.nan],
            "market_value_eur": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    agg = aggregate_nation_features(sparse).set_index("team_id")
    assert agg.loc[BR, "low_data"] == 1.0  # only 1/4 covered


def test_build_team_match_features_diffs_and_missing():
    agg = aggregate_nation_features(_player_table())
    matches = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "home_team": ["France", "France"],
            "away_team": ["Brazil", "Narnia"],  # Narnia has no squad data
        }
    )
    feats = build_team_match_features(matches, agg)
    # m1: France vs Brazil, both covered; value diff = 260m - 218m
    assert feats.loc["m1", "squad_value_eur_diff"] == pytest.approx(260e6 - 218e6)
    assert feats.loc["m1", "home_squad_low_data"] == 0.0
    assert feats.loc["m1", "away_squad_low_data"] == 0.0
    # m2: away team unknown -> NaN feature + low_data flag, never fabricated
    assert np.isnan(feats.loc["m2", "away_squad_value_eur"])
    assert feats.loc["m2", "away_squad_low_data"] == 1.0


def test_run_slice_backtest_reports_prerequisite_when_no_data(tmp_path):
    result = run_slice_backtest(asof_features_path=tmp_path / "nope.parquet")
    assert result["ran"] is False
