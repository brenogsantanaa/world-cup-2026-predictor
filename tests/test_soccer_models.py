"""Tests for the unified MatchClassifier (logistic + xgboost)."""

import numpy as np
import pandas as pd
import pytest

from sports_predictor.soccer.models import MatchClassifier

FEATURES = ["elo_diff", "home_win_rate_5", "away_win_rate_5"]


def _training_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 200, n)
    X = pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "home_win_rate_5": rng.random(n),
            "away_win_rate_5": rng.random(n),
        }
    )
    # Stronger home team (positive elo_diff) leans toward H, weaker toward A.
    y = pd.Series(np.where(elo_diff > 80, "H", np.where(elo_diff < -80, "A", "D")))
    return X, y


@pytest.mark.parametrize("name", ["logistic", "xgboost"])
def test_predict_proba_shape_and_normalization(name):
    X, y = _training_data()
    clf = MatchClassifier(name).fit(X[FEATURES], y)
    proba = clf.predict_proba(X[FEATURES])
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


@pytest.mark.parametrize("name", ["logistic", "xgboost"])
def test_columns_are_in_HDA_order(name):
    X, y = _training_data()
    clf = MatchClassifier(name).fit(X[FEATURES], y)
    # A strongly home-favored row should put most mass on column 0 (H).
    strong_home = pd.DataFrame(
        {"elo_diff": [600.0], "home_win_rate_5": [0.9], "away_win_rate_5": [0.1]}
    )
    proba = clf.predict_proba(strong_home)[0]
    assert proba.argmax() == 0  # H is column 0


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        MatchClassifier("randomforest")
