"""Tests for cleaning international results into the canonical schema."""

import pandas as pd
import pytest

from sports_predictor.soccer.results import clean


def _raw():
    """A tiny fake of the upstream results.csv, deliberately out of date order."""
    return pd.DataFrame(
        {
            "date": ["2022-12-18", "1872-11-30", "2014-07-08"],
            "home_team": ["Argentina", "Scotland", "Brazil"],
            "away_team": ["France", "England", "Germany"],
            "home_score": [3, 0, 1],
            "away_score": [3, 0, 7],
            "tournament": ["FIFA World Cup", "Friendly", "FIFA World Cup"],
            "city": ["Lusail", "Glasgow", "Belo Horizonte"],
            "country": ["Qatar", "Scotland", "Brazil"],
            "neutral": [True, False, False],
        }
    )


def test_clean_sorts_chronologically():
    out = clean(_raw())
    assert list(out["date"]) == sorted(out["date"])


def test_result_label_is_correct():
    out = clean(_raw()).set_index("home_team")
    assert out.loc["Scotland", "result"] == "D"  # 0-0
    assert out.loc["Argentina", "result"] == "D"  # 3-3
    assert out.loc["Brazil", "result"] == "A"  # 1-7, away win


def test_flags_and_types():
    out = clean(_raw()).set_index("home_team")
    assert out.loc["Brazil", "is_world_cup"]
    assert not out.loc["Scotland", "is_world_cup"]
    assert out.loc["Scotland", "is_friendly"]
    assert out["neutral"].dtype == bool
    assert out["home_score"].dtype == "int64"


def test_team_names_are_canonicalized():
    raw = _raw()
    raw.loc[0, "home_team"] = "USA"
    out = clean(raw)
    assert "United States" in set(out["home_team"])
    assert "USA" not in set(out["home_team"])


def test_match_id_is_unique_and_stable():
    out = clean(_raw())
    assert out["match_id"].is_unique
    # Stable across runs: same input -> same ids.
    assert list(out["match_id"]) == list(clean(_raw())["match_id"])


def test_rows_without_scores_are_dropped():
    raw = _raw()
    raw.loc[1, "home_score"] = None  # an unplayed/future fixture
    out = clean(raw)
    assert len(out) == 2


def test_exact_duplicate_matches_are_dropped():
    raw = _raw()
    # Same match logged twice with a different venue spelling (an upstream quirk).
    dup_row = raw.iloc[[0]].copy()
    dup_row["city"] = "Different Stadium"
    raw = pd.concat([raw, dup_row], ignore_index=True)

    out = clean(raw)
    assert len(out) == 3  # the duplicate is removed
    assert out.attrs["dropped_duplicates"] == 1


def test_sanity_check_rejects_team_against_itself():
    raw = _raw()
    raw.loc[0, "away_team"] = "Argentina"
    with pytest.raises(ValueError):
        clean(raw)
