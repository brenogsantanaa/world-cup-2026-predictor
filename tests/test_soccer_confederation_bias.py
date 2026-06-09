"""Tests for the confederation Elo-bias analysis."""

import numpy as np
import pandas as pd

from sports_predictor.soccer.confederation_bias import (
    confederation_residuals,
    estimate_elo_offsets,
)


def test_residuals_zero_when_perfectly_calibrated():
    df = pd.DataFrame(
        {
            "home_conf": ["UEFA", "CONMEBOL", "AFC"],
            "away_conf": ["CONMEBOL", "AFC", "UEFA"],
            "elo_expected_home": [0.6, 0.4, 0.5],
            "actual_home": [0.6, 0.4, 0.5],
        }
    )
    res = confederation_residuals(df)
    assert np.allclose(res["mean_residual"], 0.0)


def test_overrated_confederation_has_negative_residual():
    # AFC is always expected to win but never does -> strongly negative residual.
    df = pd.DataFrame(
        {
            "home_conf": ["AFC"] * 50,
            "away_conf": ["UEFA"] * 50,
            "elo_expected_home": [0.7] * 50,
            "actual_home": [0.0] * 50,
        }
    )
    res = confederation_residuals(df)
    assert res.loc["AFC", "mean_residual"] < 0
    assert res.loc["UEFA", "mean_residual"] > 0


def test_offset_recovers_planted_bias():
    # AFC teams are rated 100 above UEFA but are actually equal (draws), so AFC is
    # overrated by ~100 Elo; the fit should return a strongly negative offset.
    n = 300
    df = pd.DataFrame(
        {
            "home_conf": ["AFC"] * n,
            "away_conf": ["UEFA"] * n,
            "home_elo_pre": [1600.0] * n,
            "away_elo_pre": [1500.0] * n,
            "neutral": [True] * n,
            "actual_home": [0.5] * n,
        }
    )
    offsets = estimate_elo_offsets(df, reference="UEFA")
    assert offsets["UEFA"] == 0.0
    assert offsets["AFC"] < -50  # detects the inflation (true value ~ -100)
