"""Evaluation metrics for match predictions.

For sports prediction we care about the *quality of the probabilities*, not just
whether the top pick was right. A model that says "55% home win" and is right 55%
of the time is well calibrated and genuinely useful for simulation; one that
screams "95%" and is wrong a lot is dangerous. Log loss rewards honest
probabilities and punishes confident mistakes, so it is our primary metric.

Every model must beat the *no-skill baseline*: predicting the overall class
rates (how often home/draw/away happen) for every match. If a model can't beat
that, its features add nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


def _ordered_log_loss(y_true, proba: np.ndarray, labels: list[str]) -> float:
    """log_loss with ``proba`` columns given in ``labels`` order.

    scikit-learn's ``log_loss`` assumes probability columns are in lexicographic
    label order, so we reorder columns to match before calling it. This keeps the
    rest of the codebase free to use a human-friendly order like H/D/A.
    """
    order = np.argsort(labels)
    sorted_labels = [labels[i] for i in order]
    return log_loss(y_true, proba[:, order], labels=sorted_labels)


def class_base_rates(y: pd.Series, labels: list[str]) -> np.ndarray:
    """Return how often each label occurs in ``y`` (in ``labels`` order)."""
    counts = pd.Series(y).value_counts(normalize=True)
    return np.array([counts.get(label, 0.0) for label in labels])


def no_skill_log_loss(
    y_train: pd.Series, y_test: pd.Series, labels: list[str]
) -> float:
    """Log loss of predicting the training class rates for every test match.

    This is the bar to beat: it uses no features at all, only how common each
    outcome was historically.
    """
    rates = class_base_rates(y_train, labels)
    proba = np.tile(rates, (len(y_test), 1))
    return _ordered_log_loss(y_test, proba, labels)


def evaluate_probabilities(
    y_true: pd.Series, proba: np.ndarray, labels: list[str]
) -> dict[str, float]:
    """Return log loss and accuracy for predicted class probabilities.

    ``proba`` has one row per sample and one column per label (in ``labels``
    order), as returned by scikit-learn's ``predict_proba``.
    """
    loss = _ordered_log_loss(y_true, proba, labels)
    predicted = np.asarray(labels)[np.argmax(proba, axis=1)]
    return {"log_loss": float(loss), "accuracy": float(accuracy_score(y_true, predicted))}
