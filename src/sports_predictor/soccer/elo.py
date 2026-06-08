"""Elo ratings for national teams, computed from match history.

We compute our own Elo instead of scraping eloratings.net, for three reasons:
reproducibility (rebuilt from the cached raw results every run), leakage safety
(we walk matches in date order and attach the rating *as it stood before* each
match), and provenance (no scraping, no team-name alignment with an outside
source). The method follows the documented "World Football Elo Ratings" scheme;
eloratings.net is then a useful cross-check, exactly as DATA_SOURCES.md intends.

Algorithm, applied to matches in chronological order:

    expected_home = 1 / (1 + 10 ** (-(R_home + H - R_away) / 400))

where H is a home-field bonus that is dropped on neutral ground (most World Cup
matches). After the match the winner gains and the loser loses:

    delta = K * G * (actual_home - expected_home)

K scales with match importance (a World Cup final moves ratings more than a
friendly) and G scales with the margin of victory. The update is zero-sum: the
away team's rating change is exactly the negative of the home team's.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_RATING = 1500.0
HOME_ADVANTAGE = 100.0


def expected_score(rating_home: float, rating_away: float, home_advantage: float = 0.0) -> float:
    """Probability-like expected score for the home team (between 0 and 1).

    ``expected_score(a, b) + expected_score(b, a)`` always equals 1.
    """
    return 1.0 / (1.0 + 10.0 ** (-((rating_home + home_advantage) - rating_away) / 400.0))


def goal_diff_multiplier(goal_difference: int) -> float:
    """Margin-of-victory multiplier G (World Football Elo).

    A one-goal win counts as 1.0; bigger wins count for progressively more, with
    diminishing returns so a blowout does not dominate.
    """
    gd = abs(int(goal_difference))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def k_factor(tournament: str | None) -> int:
    """Match-importance weight K, inferred from the tournament name.

    Tiers follow the World Football Elo convention. Unknown competitive matches
    default to 30 (a generic tournament); this is intentionally conservative and
    easy to tune later.
    """
    t = (tournament or "").casefold()
    if t == "friendly":
        return 20
    is_qualifier = "qualif" in t
    if "world cup" in t and not is_qualifier:
        return 60
    if "confederations cup" in t:
        return 50
    continental_finals = (
        "uefa euro",
        "copa américa",
        "copa america",
        "african cup of nations",
        "africa cup of nations",
        "afc asian cup",
        "gold cup",
        "concacaf championship",
        "ofc nations cup",
    )
    if any(name in t for name in continental_finals) and not is_qualifier:
        return 50
    if is_qualifier:
        return 40
    return 30


def compute_elo(
    matches: pd.DataFrame,
    base_rating: float = DEFAULT_RATING,
    home_advantage: float = HOME_ADVANTAGE,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Attach pre-match Elo features to every match.

    Returns ``(df, final_ratings)`` where ``df`` is ``matches`` (chronologically
    ordered) with added columns:

        home_elo_pre        home team's rating *before* the match
        away_elo_pre        away team's rating *before* the match
        elo_diff            home_elo_pre - away_elo_pre
        elo_expected_home   expected score for the home team (0..1), neutral-aware

    and ``final_ratings`` maps each team to its rating after the last match (the
    current strength estimate, useful for the tournament simulator later).

    Leakage safety: the four columns above are recorded *before* the result is
    applied, so they only ever reflect earlier matches.
    """
    required = {"date", "match_id", "home_team", "away_team", "home_score", "away_score", "neutral"}
    missing = required - set(matches.columns)
    if missing:
        raise KeyError(f"compute_elo missing required columns: {sorted(missing)}")

    df = matches.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)

    ratings: dict[str, float] = {}
    home_pre: list[float] = []
    away_pre: list[float] = []
    expected: list[float] = []

    for row in df.itertuples(index=False):
        r_home = ratings.get(row.home_team, base_rating)
        r_away = ratings.get(row.away_team, base_rating)
        adv = 0.0 if row.neutral else home_advantage

        e_home = expected_score(r_home, r_away, home_advantage=adv)
        home_pre.append(r_home)
        away_pre.append(r_away)
        expected.append(e_home)

        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score < row.away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        k = k_factor(row.tournament if "tournament" in df.columns else None)
        g = goal_diff_multiplier(row.home_score - row.away_score)
        delta = k * g * (actual_home - e_home)

        ratings[row.home_team] = r_home + delta
        ratings[row.away_team] = r_away - delta

    df["home_elo_pre"] = home_pre
    df["away_elo_pre"] = away_pre
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
    df["elo_expected_home"] = expected
    return df, ratings
