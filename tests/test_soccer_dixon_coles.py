"""Tests for the Dixon-Coles goal model.

These use tiny synthetic datasets so they run fast and the expected behaviour is
checkable by hand. The themes mirror the model's promises: correct probability
shapes, the low-score correction is *local*, neutral venues drop home advantage,
recovered strengths track the truth, and the fit is leakage-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sports_predictor.soccer.dixon_coles import (
    LABELS,
    MAX_GOALS,
    DixonColesModel,
    _tau,
)


def _make_matches(n_per_pair: int = 12, seed: int = 0) -> pd.DataFrame:
    """Synthetic neutral-venue matches from known attack strengths.

    Six teams with a clear strength ladder; goals drawn from independent Poisson
    using log-rate = attack[home] - defense[away]. Defense fixed equal so attack
    alone drives scoring, which makes strength recovery easy to assert.
    """
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D", "E", "F"]
    attack = {"A": 0.6, "B": 0.4, "C": 0.1, "D": -0.1, "E": -0.4, "F": -0.6}
    defense = dict.fromkeys(teams, 0.0)
    base = 0.2  # so mean attack ~0 keeps rates ~ exp(0.2) ~ 1.2 goals

    rows = []
    start = pd.Timestamp("2015-01-01", tz="UTC")
    day = 0
    for _ in range(n_per_pair):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = np.exp(base + attack[h] - defense[a])
                mu = np.exp(base + attack[a] - defense[h])
                rows.append(
                    {
                        "date": start + pd.Timedelta(days=day),
                        "home_team": h,
                        "away_team": a,
                        "home_score": int(rng.poisson(lam)),
                        "away_score": int(rng.poisson(mu)),
                        "neutral": True,
                    }
                )
                day += 1
    return pd.DataFrame(rows)


def test_scoreline_matrix_sums_to_one():
    model = DixonColesModel(max_age_years=50).fit(_make_matches(), cutoff="2017-01-01")
    mat = model.scoreline_matrix("A", "B", neutral=True)
    assert mat.shape == (MAX_GOALS + 1, MAX_GOALS + 1)
    assert mat.min() >= 0.0
    assert mat.sum() == pytest.approx(1.0, abs=1e-9)


def test_outcome_proba_sums_to_one():
    model = DixonColesModel(max_age_years=50).fit(_make_matches(), cutoff="2017-01-01")
    p = model.outcome_proba("A", "F", neutral=True)
    assert sum(p) == pytest.approx(1.0, abs=1e-9)
    # The much stronger team should be favoured.
    p_home, _, p_away = p
    assert p_home > p_away


def test_predict_proba_rows_sum_to_one():
    model = DixonColesModel(max_age_years=50).fit(_make_matches(), cutoff="2017-01-01")
    fixtures = pd.DataFrame(
        {"home_team": ["A", "C", "F"], "away_team": ["F", "D", "A"], "neutral": [True, True, True]}
    )
    proba = model.predict_proba(fixtures)
    assert proba.shape == (3, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_tau_only_touches_four_low_score_cells():
    # Build a small grid of scores; only (0,0),(0,1),(1,0),(1,1) may differ from 1.
    hg = np.array([0, 0, 1, 1, 2, 3, 0, 2])
    ag = np.array([0, 1, 0, 1, 0, 2, 2, 1])
    lam = np.full(hg.shape, 1.3)
    mu = np.full(hg.shape, 1.1)
    out = _tau(hg, ag, lam, mu, rho=-0.1)
    special = (hg <= 1) & (ag <= 1)
    assert np.allclose(out[~special], 1.0)  # untouched cells
    assert not np.allclose(out[special], 1.0)  # corrected cells changed


def test_dc_correction_localized_in_scoreline_matrix():
    model = DixonColesModel(max_age_years=50).fit(_make_matches(), cutoff="2017-01-01")
    lam, mu = model.expected_goals("A", "B", neutral=True)
    mat = model.scoreline_matrix("A", "B", neutral=True)

    no_rho = DixonColesModel(**{**model.__dict__})
    no_rho.rho = 0.0
    mat0 = no_rho.scoreline_matrix("A", "B", neutral=True)

    # Differences (before renormalisation effects) concentrate on the 4 low cells.
    diff = np.abs(mat - mat0)
    low = diff[:2, :2].sum()
    high = diff.sum() - low
    assert low > high  # the correction is concentrated in the low-score corner


def test_neutral_drops_home_advantage():
    model = DixonColesModel(max_age_years=50).fit(_make_matches(), cutoff="2017-01-01")
    model.gamma = 0.5  # force a clear home edge
    lam_neutral, _ = model.expected_goals("C", "D", neutral=True)
    lam_home, _ = model.expected_goals("C", "D", neutral=False)
    # At home, expected goals are scaled by exp(gamma) relative to neutral.
    assert lam_home == pytest.approx(lam_neutral * np.exp(0.5), rel=1e-9)


def test_recovers_strength_ordering():
    model = DixonColesModel(xi=0.0, max_age_years=50).fit(_make_matches(n_per_pair=20), cutoff="2017-01-01")
    # True order A > B > C > D > E > F. Recovered attack should be monotone-ish.
    attacks = [model.attack[t] for t in ["A", "B", "C", "D", "E", "F"]]
    # Spearman-style: strictly decreasing on average -> A strongest, F weakest.
    assert model.attack["A"] > model.attack["F"]
    assert attacks[0] > attacks[2] > attacks[5]


def test_fit_is_leakage_safe():
    """A match on/after the cutoff must not influence the fitted strengths."""
    base = _make_matches(seed=1)
    model_a = DixonColesModel(max_age_years=50).fit(base, cutoff="2017-01-01")

    future = base.copy()
    future.loc[len(future)] = {
        "date": pd.Timestamp("2100-01-01", tz="UTC"),
        "home_team": "A",
        "away_team": "F",
        "home_score": 9,
        "away_score": 0,
        "neutral": True,
    }
    model_b = DixonColesModel(max_age_years=50).fit(future, cutoff="2017-01-01")
    # Cutoff excludes the 2100 match, so attacks must be identical.
    assert model_a.attack["A"] == pytest.approx(model_b.attack["A"], abs=1e-9)


def test_unknown_team_falls_back_and_is_flagged():
    model = DixonColesModel(max_age_years=50).fit(_make_matches(), cutoff="2017-01-01")
    p = model.outcome_proba("A", "Atlantis", neutral=True)  # Atlantis never seen
    assert sum(p) == pytest.approx(1.0, abs=1e-9)
    assert "Atlantis" in model.low_data_teams
