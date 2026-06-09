"""Tests for the leakage-safe player-profile feature layer."""

import numpy as np
import pandas as pd

from sports_predictor.soccer.player_features import build_player_features


def _matches(rows):
    return pd.DataFrame(
        [
            {
                "match_id": mid,
                "date": pd.Timestamp(date, tz="UTC"),
                "home_team": home,
                "away_team": away,
            }
            for mid, date, home, away in rows
        ]
    )


def _goals(rows):
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(date, tz="UTC"),
                "home_team": home,
                "away_team": away,
                "team": team,
                "scorer": scorer,
                "own_goal": own,
                "penalty": pen,
            }
            for date, home, away, team, scorer, own, pen in rows
        ]
    )


# A: home in every match and scores; B: away and never scores.
MATCHES = _matches([
    ("m1", "2020-01-01", "A", "B"),
    ("m2", "2020-02-01", "A", "B"),
    ("m3", "2020-03-01", "A", "B"),
])
GOALS = _goals([
    ("2020-01-01", "A", "B", "A", "p1", False, False),   # m1: p1
    ("2020-02-01", "A", "B", "A", "p1", False, False),   # m2: p1
    ("2020-02-01", "A", "B", "A", "p2", False, True),    # m2: p2 (penalty)
])


def test_aggregation_preserves_rows_and_ids():
    feats = build_player_features(MATCHES, GOALS)
    assert len(feats) == len(MATCHES)
    assert set(feats.index) == set(MATCHES["match_id"])


def test_first_appearance_has_empty_history_not_zeros():
    feats = build_player_features(MATCHES, GOALS)
    # A's first ever match: no prior goals -> NaN profile + low-data flag (not 0).
    assert np.isnan(feats.loc["m1", "home_p_recent_goals"])
    assert np.isnan(feats.loc["m1", "home_p_goal_concentration"])
    assert feats.loc["m1", "home_p_low_data"] == 1.0


def test_window_uses_only_prior_matches():
    feats = build_player_features(MATCHES, GOALS)
    # At m2, A's window is just m1 (one goal by p1).
    assert feats.loc["m2", "home_p_recent_goals"] == 1.0
    assert feats.loc["m2", "home_p_top_scorer_goals"] == 1.0
    assert feats.loc["m2", "home_p_goal_concentration"] == 1.0  # single scorer
    # At m3, the window is m1+m2: p1 twice, p2 once -> 3 goals, top scorer 2.
    assert feats.loc["m3", "home_p_recent_goals"] == 3.0
    assert feats.loc["m3", "home_p_top_scorer_goals"] == 2.0
    assert feats.loc["m3", "home_p_penalty_share"] == 1 / 3
    # squad_experience = career goals (as of m3) of window scorers: p1=2, p2=1.
    assert feats.loc["m3", "home_p_squad_experience"] == 3.0


def test_future_goals_do_not_change_past_features():
    extra = _goals([("2020-03-01", "A", "B", "A", "p1", False, False)])  # a goal in m3
    feats_a = build_player_features(MATCHES, GOALS)
    feats_b = build_player_features(MATCHES, pd.concat([GOALS, extra], ignore_index=True))
    # m3's own goals must not affect m1/m2 features.
    for mid in ("m1", "m2"):
        a = feats_a.loc[mid].fillna(-1)
        b = feats_b.loc[mid].fillna(-1)
        pd.testing.assert_series_equal(a, b)


def test_team_that_never_scores_gets_nan_not_zero():
    feats = build_player_features(MATCHES, GOALS)
    # B never scores -> away profile is NaN with low-data flag, never a faked 0.
    for mid in ("m1", "m2", "m3"):
        assert np.isnan(feats.loc[mid, "away_p_recent_goals"])
        assert feats.loc[mid, "away_p_low_data"] == 1.0


def test_own_goals_are_not_credited_to_the_scorer():
    goals = _goals([
        ("2020-01-01", "A", "B", "A", "own_scorer", True, False),  # own goal credited to A
        ("2020-02-01", "A", "B", "A", "p1", False, False),
    ])
    feats = build_player_features(MATCHES, goals)
    # By m3 the window covers m1 (own goal, excluded) + m2 (p1's real goal), so
    # only the one legitimate goal counts -- the own goal is never credited.
    assert feats.loc["m3", "home_p_recent_goals"] == 1.0
    assert feats.loc["m3", "home_p_top_scorer_goals"] == 1.0
