"""Build the per-match modeling table.

Turns the clean ``matches`` table into one row per match with **pre-kickoff**
features only. Every feature here obeys the single most important rule in
``DATA_SOURCES.md`` §6: a feature for match *M* may use information from matches
that finished *before* M, and never from M itself.

The mechanism that guarantees this is ``shift(1)`` inside each team's own
chronological history: before any rolling/expanding window is taken, we drop the
current match, so a window can only ever see earlier matches.

Public entry point: :func:`build_modeling_table`.
"""

from __future__ import annotations

import pandas as pd

from sports_predictor.core.paths import PROCESSED_DIR, ensure_dir
from sports_predictor.soccer.elo import compute_elo

# How many recent matches the rolling-form windows look back over.
FORM_WINDOWS = (5, 10)

MATCHES_FILENAME = "matches.parquet"
MODEL_TABLE_FILENAME = "model_table.parquet"


def build_modeling_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Return ``matches`` enriched with Elo, recent-form, rest-days, and H2H.

    The result is sorted chronologically and keyed by ``match_id``. Outcome
    columns (scores, ``result``) are preserved as targets but are never used as
    inputs by the feature builders.
    """
    elo_df, _ = compute_elo(matches)
    with_form = add_team_rolling_features(elo_df, windows=FORM_WINDOWS)
    with_h2h = add_head_to_head(with_form)
    return with_h2h.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)


def _team_perspective(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode each match into two rows, one from each team's point of view.

    This lets us compute a team's form over its own match history regardless of
    whether it played home or away in any given game.
    """
    home = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "team": matches["home_team"],
            "is_home": True,
            "goals_for": matches["home_score"],
            "goals_against": matches["away_score"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "team": matches["away_team"],
            "is_home": False,
            "goals_for": matches["away_score"],
            "goals_against": matches["home_score"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["win"] = (long["goals_for"] > long["goals_against"]).astype(int)
    draw = long["goals_for"] == long["goals_against"]
    # League-style points: 3 win / 1 draw / 0 loss.
    long["points"] = long["win"] * 3 + draw.astype(int)
    return long


def add_team_rolling_features(
    matches: pd.DataFrame,
    windows: tuple[int, ...] = FORM_WINDOWS,
) -> pd.DataFrame:
    """Attach recent-form and rest-days features for the home and away teams.

    For each window N (e.g. 5, 10) and each team we compute, over that team's N
    most recent *previous* matches:
        win_rate_N, points_avg_N, goals_for_avg_N, goals_against_avg_N
    plus rest_days (days since the team's previous match of any kind).

    The first match a team ever plays has no history, so its features are NaN
    (an honest "unknown" rather than a fabricated zero).
    """
    long = _team_perspective(matches)
    long = long.sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)
    grouped = long.groupby("team", sort=False)

    def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
        # shift(1) excludes the current match -> no leakage.
        return series.shift(1).rolling(window, min_periods=1).mean()

    feature_cols: list[str] = []
    for window in windows:
        for source, name in [
            ("win", "win_rate"),
            ("points", "points_avg"),
            ("goals_for", "goals_for_avg"),
            ("goals_against", "goals_against_avg"),
        ]:
            col = f"{name}_{window}"
            long[col] = grouped[source].transform(lambda s, w=window: _rolling_mean(s, w))
            feature_cols.append(col)

    long["rest_days"] = (long["date"] - grouped["date"].shift(1)).dt.days
    feature_cols.append("rest_days")

    home_feats = (
        long[long["is_home"]].set_index("match_id")[feature_cols].add_prefix("home_")
    )
    away_feats = (
        long[~long["is_home"]].set_index("match_id")[feature_cols].add_prefix("away_")
    )

    out = matches.set_index("match_id").join(home_feats).join(away_feats)
    return out.reset_index()


def add_head_to_head(matches: pd.DataFrame) -> pd.DataFrame:
    """Attach head-to-head history between the two teams, before this match.

    Adds:
        h2h_matches         number of prior meetings (any venue)
        h2h_home_win_rate   fraction of prior meetings won by *this* match's
                            home team (NaN when the teams have never met)

    Draws count toward h2h_matches but not toward either side's win rate.
    """
    df = matches.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)

    # A venue-independent key for the pairing, so A-vs-B and B-vs-A share history.
    pair = [tuple(sorted((h, a))) for h, a in zip(df["home_team"], df["away_team"])]
    df["_pair"] = pair
    df["_a_is_home"] = [h == p[0] for h, p in zip(df["home_team"], pair)]

    a_win = (df["_a_is_home"] & (df["home_score"] > df["away_score"])) | (
        ~df["_a_is_home"] & (df["away_score"] > df["home_score"])
    )
    b_win = (df["_a_is_home"] & (df["home_score"] < df["away_score"])) | (
        ~df["_a_is_home"] & (df["away_score"] < df["home_score"])
    )
    df["_a_win"] = a_win.astype(int)
    df["_b_win"] = b_win.astype(int)

    grouped = df.groupby("_pair", sort=False)

    def _prior_sum(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding().sum()

    prior_a_wins = grouped["_a_win"].transform(_prior_sum)
    prior_b_wins = grouped["_b_win"].transform(_prior_sum)
    prior_meetings = grouped.cumcount()  # number of earlier rows in this pairing

    # Prior wins by the team that is home in the current match.
    prior_home_wins = prior_a_wins.where(df["_a_is_home"], prior_b_wins)

    df["h2h_matches"] = prior_meetings
    df["h2h_home_win_rate"] = (prior_home_wins / prior_meetings).where(prior_meetings > 0)

    return df.drop(columns=["_pair", "_a_is_home", "_a_win", "_b_win"])


def _main() -> None:
    matches_path = PROCESSED_DIR / MATCHES_FILENAME
    if not matches_path.exists():
        raise FileNotFoundError(
            f"{matches_path} not found. Run `python -m sports_predictor.soccer.results` first."
        )
    matches = pd.read_parquet(matches_path)
    table = build_modeling_table(matches)

    out_path = ensure_dir(PROCESSED_DIR) / MODEL_TABLE_FILENAME
    table.to_parquet(out_path, index=False)

    feature_cols = [
        c
        for c in table.columns
        if c.startswith(("home_", "away_", "elo_", "h2h_")) and c not in {"home_team", "away_team"}
    ]
    print(
        f"matches in:  {len(matches):,}\n"
        f"rows out:    {len(table):,}\n"
        f"feature cols: {len(feature_cols)}\n"
        f"written:     {out_path}"
    )


if __name__ == "__main__":
    _main()
