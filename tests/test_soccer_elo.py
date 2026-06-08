"""Tests for the Elo rating engine."""

import pandas as pd

from sports_predictor.soccer.elo import (
    DEFAULT_RATING,
    compute_elo,
    expected_score,
    goal_diff_multiplier,
    k_factor,
)


def test_expected_score_is_symmetric_and_fair():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1600, 1400) + expected_score(1400, 1600) == 1.0
    # Home advantage tilts the expected score upward.
    assert expected_score(1500, 1500, home_advantage=100) > 0.5


def test_goal_diff_multiplier_grows_with_margin():
    assert goal_diff_multiplier(0) == 1.0
    assert goal_diff_multiplier(1) == 1.0
    assert goal_diff_multiplier(2) == 1.5
    assert goal_diff_multiplier(3) == (11 + 3) / 8
    assert goal_diff_multiplier(5) > goal_diff_multiplier(3)


def test_k_factor_tiers():
    assert k_factor("Friendly") == 20
    assert k_factor("FIFA World Cup") == 60
    assert k_factor("FIFA World Cup qualification") == 40
    assert k_factor("UEFA Euro") == 50
    assert k_factor("UEFA Euro qualification") == 40
    assert k_factor("UEFA Nations League") == 30
    assert k_factor(None) == 30


def _matches(rows):
    df = pd.DataFrame(
        rows,
        columns=["date", "home_team", "away_team", "home_score", "away_score", "neutral", "tournament"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["match_id"] = [f"m{i}" for i in range(len(df))]
    return df


def test_first_match_uses_base_rating():
    df, _ = compute_elo(_matches([("2020-01-01", "A", "B", 1, 0, False, "Friendly")]))
    assert df.loc[0, "home_elo_pre"] == DEFAULT_RATING
    assert df.loc[0, "away_elo_pre"] == DEFAULT_RATING


def test_update_is_zero_sum_and_winner_gains():
    df, ratings = compute_elo(_matches([("2020-01-01", "A", "B", 3, 0, True, "Friendly")]))
    # Neutral venue, A wins big -> A above base, B below base, sum preserved.
    assert ratings["A"] > DEFAULT_RATING
    assert ratings["B"] < DEFAULT_RATING
    assert round(ratings["A"] + ratings["B"], 6) == round(2 * DEFAULT_RATING, 6)


def test_pre_match_rating_reflects_only_earlier_matches():
    # A beats B, then plays C. A's rating entering match 2 must be > base (it won
    # match 1) and must NOT depend on match 2's result. This is the leakage test.
    df, _ = compute_elo(
        _matches(
            [
                ("2020-01-01", "A", "B", 2, 0, True, "Friendly"),
                ("2020-02-01", "A", "C", 0, 5, True, "Friendly"),
            ]
        )
    )
    assert df.loc[1, "home_elo_pre"] > DEFAULT_RATING  # earned in match 1 only


def test_repeated_wins_increase_rating_monotonically():
    rows = [(f"2020-0{i}-01", "A", "B", 1, 0, True, "Friendly") for i in range(1, 6)]
    df, _ = compute_elo(_matches(rows))
    pre = list(df["home_elo_pre"])
    assert pre == sorted(pre)  # A's pre-match rating never decreases while winning


def test_input_is_not_mutated():
    original = _matches([("2020-01-01", "A", "B", 1, 0, False, "Friendly")])
    snapshot = original.copy()
    compute_elo(original)
    assert original.equals(snapshot)
