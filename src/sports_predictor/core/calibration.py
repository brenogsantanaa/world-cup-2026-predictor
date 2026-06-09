"""Temperature scaling for probability calibration.

Temperature scaling softens (or sharpens) predicted class probabilities without
changing which class is most likely. We raise each probability to the power
``1/T`` and renormalize::

    p_i' = p_i ** (1/T) / sum_j p_j ** (1/T)

- ``T = 1`` leaves probabilities unchanged.
- ``T > 1`` softens them toward uniform (less confident) -- the fix for an
  overconfident model.
- ``T < 1`` sharpens them (more confident).

It has a single parameter, so it can be fit on a held-out slice without
overfitting, and because it is monotonic it never hurts accuracy. We use it to
check whether a slice (e.g. World Cup matches) is systematically over- or
under-confident, and to correct it if so.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from sports_predictor.core.evaluation import _ordered_log_loss

_EPS = 1e-12


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Return ``proba`` rescaled by ``temperature`` (rows renormalized to 1)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = np.clip(proba, _EPS, 1.0) ** (1.0 / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(
    proba: np.ndarray,
    y_true,
    labels: list[str],
    bounds: tuple[float, float] = (0.25, 5.0),
) -> float:
    """Find the temperature that minimizes log loss on ``(proba, y_true)``.

    ``T* > 1`` means the inputs were overconfident on this data; ``T* < 1`` means
    underconfident; ``T* ~ 1`` means already calibrated.
    """

    def neg_objective(t: float) -> float:
        return _ordered_log_loss(y_true, apply_temperature(proba, t), labels)

    result = minimize_scalar(neg_objective, bounds=bounds, method="bounded")
    return float(result.x)
