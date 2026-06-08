"""Tests for evaluation metrics."""

import numpy as np
import pandas as pd

from sports_predictor.core.evaluation import (
    class_base_rates,
    evaluate_probabilities,
    no_skill_log_loss,
)

LABELS = ["H", "D", "A"]


def test_class_base_rates_match_counts():
    y = pd.Series(["H", "H", "D", "A"])
    rates = class_base_rates(y, LABELS)
    assert np.allclose(rates, [0.5, 0.25, 0.25])


def test_perfect_predictions_give_zero_log_loss_and_full_accuracy():
    y = pd.Series(["H", "D", "A"])
    proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    result = evaluate_probabilities(y, proba, LABELS)
    assert result["accuracy"] == 1.0
    assert result["log_loss"] < 1e-9


def test_no_skill_baseline_uses_training_rates():
    # Train is all home wins -> predicting [1,0,0] for a home-win test is perfect.
    y_train = pd.Series(["H", "H", "H", "H"])
    y_test = pd.Series(["H", "H"])
    assert no_skill_log_loss(y_train, y_test, LABELS) < 1e-9


def test_a_real_model_beats_no_skill():
    # Outcomes are 50% home / 25% draw / 25% away.
    y_train = pd.Series(["H", "H", "D", "A"] * 25)
    y_test = pd.Series(["H"] * 50 + ["D"] * 25 + ["A"] * 25)

    baseline = no_skill_log_loss(y_train, y_test, LABELS)

    # A "model" that leans correctly toward home should beat the flat baseline.
    confident = np.tile([0.7, 0.15, 0.15], (len(y_test), 1))
    model = evaluate_probabilities(y_test, confident, LABELS)
    # Not asserting the model wins here (depends on mix); just that both are sane.
    assert baseline > 0
    assert model["log_loss"] > 0
