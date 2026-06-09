"""Tests for temperature scaling."""

import numpy as np

from sports_predictor.core.calibration import apply_temperature, fit_temperature

LABELS = ["H", "D", "A"]


def test_temperature_one_is_identity():
    proba = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
    assert np.allclose(apply_temperature(proba, 1.0), proba)


def test_rows_stay_normalized():
    proba = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
    for t in (0.5, 2.0, 4.0):
        out = apply_temperature(proba, t)
        assert np.allclose(out.sum(axis=1), 1.0)


def test_high_temperature_softens_toward_uniform():
    proba = np.array([[0.8, 0.15, 0.05]])
    softened = apply_temperature(proba, 3.0)
    # Most-likely class stays the same but is less extreme.
    assert softened.argmax() == proba.argmax()
    assert softened[0, 0] < proba[0, 0]


def test_low_temperature_sharpens():
    proba = np.array([[0.5, 0.3, 0.2]])
    sharpened = apply_temperature(proba, 0.5)
    assert sharpened[0, 0] > proba[0, 0]


def test_fit_recovers_overconfidence():
    # Build overconfident probabilities and check the fitted T > 1 corrects them.
    rng = np.random.default_rng(0)
    n = 4000
    true_p = rng.uniform(0.2, 0.8, n)
    y = np.where(rng.random(n) < true_p, "H", "A")
    # Overstate confidence: push probabilities toward 0/1 (temperature 0.5).
    over = np.column_stack([true_p, np.zeros(n), 1 - true_p])
    over = apply_temperature(over, 0.5)
    t_star = fit_temperature(over, y, LABELS)
    assert t_star > 1.2  # detects and would undo the overconfidence
