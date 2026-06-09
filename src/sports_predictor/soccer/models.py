"""Match-outcome models and a logistic-vs-XGBoost bake-off.

The baseline (``baseline.py``) is a logistic regression. Here we add gradient
boosting (XGBoost), which can model feature *interactions* a linear model cannot,
and compare the two honestly.

Two principles guide the comparison:

1. Judge on the slices that matter (neutral venue, World Cup finals), not just the
   global log loss, which is flattered by lopsided qualifiers.
2. Watch calibration. Boosted trees often improve log loss but come out the box
   *less* calibrated than logistic regression, so we print reliability for both
   and refuse to celebrate a sharper-but-overconfident model.

:class:`MatchClassifier` wraps either model behind one interface whose
``predict_proba`` always returns columns in ``LABELS`` order (H, D, A), so the
rest of the codebase (including the simulator) does not care which is underneath.

Run the comparison::

    python -m sports_predictor.soccer.models
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sports_predictor.core.evaluation import evaluate_probabilities, no_skill_log_loss
from sports_predictor.soccer.baseline import (
    FULL_FEATURES,
    LABELS,
    TARGET,
    _print_calibration,
    load_model_table,
    prepare,
)
from sports_predictor.core.splitting import chronological_split

LABEL_TO_INT = {"H": 0, "D": 1, "A": 2}


class MatchClassifier:
    """A 3-way match model (``logistic`` or ``xgboost``) with a unified API.

    ``predict_proba`` always returns an array whose columns are in ``LABELS``
    order (H, D, A), regardless of the underlying library's class ordering.
    """

    def __init__(self, name: str = "logistic", **params):
        self.name = name
        if name == "logistic":
            self._estimator = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000)
            )
            self._encodes_labels = False
        elif name == "xgboost":
            from xgboost import XGBClassifier

            defaults = dict(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=len(LABELS),
                n_jobs=4,
                random_state=0,
            )
            defaults.update(params)
            self._estimator = XGBClassifier(**defaults)
            self._encodes_labels = True
        else:
            raise ValueError(f"unknown model {name!r}; use 'logistic' or 'xgboost'")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MatchClassifier":
        if self._encodes_labels:
            self._estimator.fit(X, y.map(LABEL_TO_INT))
        else:
            self._estimator.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self._estimator.predict_proba(X)
        if self._encodes_labels:
            # Encoded classes 0,1,2 already map to H,D,A in order.
            return proba
        order = [list(self._estimator.classes_).index(label) for label in LABELS]
        return proba[:, order]


def _slice_masks(test: pd.DataFrame) -> dict[str, np.ndarray]:
    is_neutral = (test["neutral"] == 1).to_numpy()
    is_wc = test["is_world_cup"].astype(bool).to_numpy()
    return {
        "all": np.ones(len(test), dtype=bool),
        "neutral": is_neutral,
        "WC finals~": is_wc & is_neutral,
    }


def run_comparison(test_fraction: float = 0.2) -> dict:
    table = load_model_table()
    full = prepare(table, FULL_FEATURES, carry=("is_world_cup",))
    train, test = chronological_split(full, date_column="date", test_fraction=test_fraction)
    y_train, y_test = train[TARGET], test[TARGET].reset_index(drop=True)
    masks = _slice_masks(test)

    print(
        f"train {len(train):,} (-> {train['date'].max().date()})  |  "
        f"test {len(test):,} (from {test['date'].min().date()})\n"
    )

    # Header: log loss per slice + global accuracy.
    slice_names = list(masks)
    header = f"{'model':<14}" + "".join(f"{s + ' LL':>14}" for s in slice_names) + f"{'acc':>8}"
    print(header)
    print("-" * len(header))

    # No-skill baseline row (log loss only).
    base_cells = "".join(
        f"{no_skill_log_loss(y_train, y_test[m], LABELS):>14.4f}" for m in masks.values()
    )
    print(f"{'no-skill':<14}{base_cells}{'-':>8}")

    results: dict[str, dict] = {}
    proba_by_model: dict[str, np.ndarray] = {}
    for name in ("logistic", "xgboost"):
        clf = MatchClassifier(name).fit(train[FULL_FEATURES], y_train)
        proba = clf.predict_proba(test[FULL_FEATURES])
        proba_by_model[name] = proba

        cells = ""
        slice_metrics = {}
        for s, mask in masks.items():
            m = evaluate_probabilities(y_test[mask], proba[mask], LABELS)
            slice_metrics[s] = m
            cells += f"{m['log_loss']:>14.4f}"
        acc = evaluate_probabilities(y_test, proba, LABELS)["accuracy"]
        results[name] = slice_metrics
        print(f"{name:<14}{cells}{acc:>8.1%}")

    # Calibration side by side, so a sharper-but-overconfident model is visible.
    for name in ("logistic", "xgboost"):
        _print_calibration(
            y_test, proba_by_model[name], bins=10, title=f"calibration: {name}"
        )

    return results


def _main() -> None:
    run_comparison()


if __name__ == "__main__":
    _main()
