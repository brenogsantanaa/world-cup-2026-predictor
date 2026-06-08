"""Tests for the per-match feature builders, focused on leakage safety."""

import pandas as pd

from sports_predictor.soccer.features import (
    add_head_to_head,
    add_team_rolling_features,
    build_modeling_table,
)


def _matches(rows):
    df = pd.DataFrame(
        rows,
        columns=["date", "home_team", "away_team", "home_score", "away_score"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = False
    df["tournament"] = "Friendly"
    df["match_id"] = [f"m{i}" for i in range(len(df))]
    return df


def test_rolling_form_uses_only_prior_matches():
    # Team A: win (2-0), loss (0-1), then the match we inspect (a draw).
    df = _matches(
        [
            ("2020-01-01", "A", "B", 2, 0),
            ("2020-01-08", "A", "C", 0, 1),
            ("2020-01-15", "A", "D", 1, 1),
        ]
    )
    out = add_team_rolling_features(df, windows=(5,)).set_index("match_id")

    # Match 3's home (A) features must reflect only matches 1 and 2.
    assert out.loc["m2", "home_win_rate_5"] == 0.5  # 1 win in 2 prior games
    assert out.loc["m2", "home_points_avg_5"] == 1.5  # (3 + 0) / 2
    assert out.loc["m2", "home_goals_for_avg_5"] == 1.0  # (2 + 0) / 2
    assert out.loc["m2", "home_goals_against_avg_5"] == 0.5  # (0 + 1) / 2


def test_first_ever_match_has_no_history():
    df = _matches([("2020-01-01", "A", "B", 2, 0)])
    out = add_team_rolling_features(df, windows=(5,)).set_index("match_id")
    assert pd.isna(out.loc["m0", "home_win_rate_5"])
    assert pd.isna(out.loc["m0", "away_win_rate_5"])


def test_changing_current_result_does_not_change_current_features():
    base = _matches(
        [
            ("2020-01-01", "A", "B", 2, 0),
            ("2020-01-08", "A", "C", 0, 1),
            ("2020-01-15", "A", "D", 1, 1),
        ]
    )
    flipped = base.copy()
    flipped.loc[2, ["home_score", "away_score"]] = [9, 0]  # blow out match 3

    a = add_team_rolling_features(base, windows=(5,)).set_index("match_id")
    b = add_team_rolling_features(flipped, windows=(5,)).set_index("match_id")
    # The features for match 3 must be identical: they depend only on the past.
    assert a.loc["m2", "home_win_rate_5"] == b.loc["m2", "home_win_rate_5"]
    assert a.loc["m2", "home_goals_for_avg_5"] == b.loc["m2", "home_goals_for_avg_5"]


def test_rest_days_counts_gap_since_previous_match():
    df = _matches(
        [
            ("2020-01-01", "A", "B", 1, 0),
            ("2020-01-08", "A", "C", 1, 0),  # 7 days later
        ]
    )
    out = add_team_rolling_features(df, windows=(5,)).set_index("match_id")
    assert pd.isna(out.loc["m0", "home_rest_days"])  # no previous match
    assert out.loc["m1", "home_rest_days"] == 7


def test_head_to_head_excludes_current_match():
    df = _matches(
        [
            ("2020-01-01", "A", "B", 1, 0),  # A wins
            ("2020-06-01", "B", "A", 2, 2),  # draw
            ("2021-01-01", "A", "B", 0, 1),  # B wins
        ]
    )
    out = add_head_to_head(df).set_index("match_id")

    assert out.loc["m0", "h2h_matches"] == 0
    assert pd.isna(out.loc["m0", "h2h_home_win_rate"])  # never met before
    assert out.loc["m1", "h2h_matches"] == 1
    assert out.loc["m1", "h2h_home_win_rate"] == 0.0  # home=B, B had not won
    assert out.loc["m2", "h2h_matches"] == 2
    assert out.loc["m2", "h2h_home_win_rate"] == 0.5  # home=A won 1 of 2 priors


def test_build_modeling_table_preserves_rows_and_targets():
    df = _matches(
        [
            ("2020-01-01", "A", "B", 1, 0),
            ("2020-01-08", "C", "A", 2, 2),
        ]
    )
    df["result"] = ["H", "D"]
    table = build_modeling_table(df)
    assert len(table) == len(df)
    # Targets survive; Elo + form + h2h columns are present.
    for col in ["result", "elo_diff", "home_win_rate_5", "h2h_matches"]:
        assert col in table.columns
