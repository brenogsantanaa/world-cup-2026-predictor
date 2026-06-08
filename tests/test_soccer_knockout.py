"""Tests for the knockout-draw conversion."""

import numpy as np
import pytest

from sports_predictor.soccer.knockout import (
    advance_probabilities,
    even_share,
    proportional_share,
)


def test_advancement_probabilities_sum_to_one():
    # The draw mass must be fully reassigned, nothing created or lost.
    for p_h, p_d, p_a in [(0.5, 0.3, 0.2), (0.1, 0.6, 0.3), (0.33, 0.34, 0.33)]:
        home, away = advance_probabilities(p_h, p_d, p_a)
        assert home + away == pytest.approx(1.0)


def test_favorite_gets_at_least_half_the_draw_mass():
    # Home is the favorite (higher regulation win prob).
    p_h, p_d, p_a = 0.5, 0.3, 0.2
    home_only_regulation = p_h
    home_adv, _ = advance_probabilities(p_h, p_d, p_a, strategy="proportional")
    gained = home_adv - home_only_regulation
    assert gained >= 0.5 * p_d  # favorite never receives less than half the draw mass


def test_proportional_share_tracks_relative_strength():
    # 0.6 vs 0.2 regulation wins -> home should get 0.6/0.8 = 0.75 of the draw.
    assert proportional_share(0.6, 0.2) == pytest.approx(0.75)
    assert even_share(0.6, 0.2) == 0.5


def test_even_strategy_splits_draw_mass_exactly_evenly():
    p_h, p_d, p_a = 0.5, 0.4, 0.1
    home_adv, away_adv = advance_probabilities(p_h, p_d, p_a, strategy="even")
    assert home_adv == pytest.approx(p_h + 0.5 * p_d)
    assert away_adv == pytest.approx(p_a + 0.5 * p_d)


def test_all_draw_prediction_falls_back_to_even():
    # No regulation wins for either side -> proportional must not divide by zero.
    home_adv, away_adv = advance_probabilities(0.0, 1.0, 0.0, strategy="proportional")
    assert home_adv == pytest.approx(0.5)
    assert away_adv == pytest.approx(0.5)


def test_works_vectorized_over_arrays():
    p_h = np.array([0.5, 0.1])
    p_d = np.array([0.3, 0.6])
    p_a = np.array([0.2, 0.3])
    home, away = advance_probabilities(p_h, p_d, p_a)
    assert np.allclose(home + away, 1.0)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        advance_probabilities(0.5, 0.3, 0.2, strategy="momentum")


def test_custom_callable_strategy_is_accepted():
    # Give the whole draw mass to the home team.
    home_adv, away_adv = advance_probabilities(0.4, 0.3, 0.3, strategy=lambda h, a: 1.0)
    assert home_adv == pytest.approx(0.7)
    assert away_adv == pytest.approx(0.3)
