"""Baseline match-outcome model.

A first, deliberately simple model: multinomial logistic regression predicting
the 3-way outcome Home win / Draw / Away win from the pre-kickoff features in the
modeling table. The point is not raw power but to prove, with an honest
chronological backtest, that the engineered features carry real signal.

We train on the earlier matches and test on the later ones (never a random
split: that would let the model see the future). We then compare three things:

    1. no-skill baseline   - predict historical class rates, no features
    2. Elo-only model      - logistic regression on Elo alone
    3. full model          - logistic regression on all features

If (3) clearly beats (2) beats (1) on log loss, the feature work paid off.

Run:

    python -m sports_predictor.soccer.baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sports_predictor.core.evaluation import evaluate_probabilities, no_skill_log_loss
from sports_predictor.core.paths import MODELS_DIR, PROCESSED_DIR, ensure_dir
from sports_predictor.core.splitting import chronological_split

TARGET = "result"
LABELS = ["H", "D", "A"]

# Pre-kickoff features only. (Outcome columns such as scores/result are excluded.)
ELO_FEATURES = ["elo_diff", "elo_expected_home"]
FORM_FEATURES = [
    "home_win_rate_5",
    "home_points_avg_5",
    "home_goals_for_avg_5",
    "home_goals_against_avg_5",
    "home_win_rate_10",
    "home_goals_for_avg_10",
    "home_goals_against_avg_10",
    "home_rest_days",
    "away_win_rate_5",
    "away_points_avg_5",
    "away_goals_for_avg_5",
    "away_goals_against_avg_5",
    "away_win_rate_10",
    "away_goals_for_avg_10",
    "away_goals_against_avg_10",
    "away_rest_days",
]
H2H_FEATURES = ["h2h_matches", "h2h_home_win_rate"]
CONTEXT_FEATURES = ["neutral"]

FULL_FEATURES = ELO_FEATURES + FORM_FEATURES + H2H_FEATURES + CONTEXT_FEATURES

MODEL_FILENAME = "baseline_logreg.joblib"


def load_model_table(processed_dir=PROCESSED_DIR) -> pd.DataFrame:
    path = processed_dir / "model_table.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m sports_predictor.soccer.features` first."
        )
    return pd.read_parquet(path)


def prepare(
    table: pd.DataFrame, features: list[str], carry: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Select features + target and make them model-ready (no NaNs).

    - ``neutral`` becomes 0/1.
    - ``h2h_home_win_rate`` is NaN when the teams have never met; we fill it with
      0.5 (a neutral prior) and keep ``h2h_matches`` so the model can tell a
      first meeting from an established rivalry.
    - rows still missing form/rest features (a team's first few ever matches) are
      dropped, since those features are genuinely undefined.

    ``carry`` lists extra columns to keep for slicing/reporting (e.g.
    ``is_world_cup``) without using them as model inputs.
    """
    df = table.copy()
    if "neutral" in features:
        df["neutral"] = df["neutral"].astype(int)
    if "h2h_home_win_rate" in features:
        df["h2h_home_win_rate"] = df["h2h_home_win_rate"].fillna(0.5)

    keep = list(dict.fromkeys(features + [TARGET, "date", *carry]))
    df = df[keep].dropna(subset=features + [TARGET]).reset_index(drop=True)
    return df


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    model.fit(train[features], train[TARGET])
    # Align probability columns to LABELS order regardless of class ordering.
    proba = model.predict_proba(test[features])
    proba = proba[:, [list(model.classes_).index(label) for label in LABELS]]
    return model, proba


def run(test_fraction: float = 0.2, save: bool = True) -> dict:
    table = load_model_table()

    full = prepare(table, FULL_FEATURES, carry=("is_world_cup",))
    train, test = chronological_split(full, date_column="date", test_fraction=test_fraction)

    y_train, y_test = train[TARGET], test[TARGET]

    baseline_ll = no_skill_log_loss(y_train, y_test, LABELS)

    _, elo_proba = _fit_predict(train, test, ELO_FEATURES)
    elo_metrics = evaluate_probabilities(y_test, elo_proba, LABELS)

    model, full_proba = _fit_predict(train, test, FULL_FEATURES)
    full_metrics = evaluate_probabilities(y_test, full_proba, LABELS)

    _print_report(train, test, baseline_ll, elo_metrics, full_metrics)
    _print_calibration(y_test, full_proba)

    slice_results = _evaluate_slices(train, test, full_proba)

    if save:
        import joblib

        out = ensure_dir(MODELS_DIR) / MODEL_FILENAME
        joblib.dump({"model": model, "features": FULL_FEATURES, "labels": LABELS}, out)
        print(f"\nsaved model -> {out}")

    return {
        "no_skill_log_loss": baseline_ll,
        "elo_only": elo_metrics,
        "full": full_metrics,
        "slices": slice_results,
    }


def _evaluate_slices(
    train: pd.DataFrame, test: pd.DataFrame, full_proba: np.ndarray
) -> dict:
    """Report the full model on the slices that matter for the World Cup.

    The global number is dominated by easy, lopsided qualifiers. What we actually
    care about is performance on **neutral-venue** matches and **World Cup**
    matches, so we evaluate the same predictions on those held-out subsets.
    Each slice is compared to its own no-skill baseline (training class rates vs
    that slice's outcomes), so "is the model adding value *here*" is honest.
    """
    y_train = train[TARGET]
    y_test = test[TARGET].reset_index(drop=True)

    is_neutral = (test["neutral"] == 1).to_numpy()
    is_wc = test["is_world_cup"].astype(bool).to_numpy()
    slices = {
        "all test": np.ones(len(test), dtype=bool),
        "neutral venue": is_neutral,
        "world cup (+qual)": is_wc,
        # Neutral + World Cup approximates the finals themselves (qualifiers are
        # mostly home/away), i.e. the conditions we ultimately predict.
        "WC finals~": is_wc & is_neutral,
    }

    print(f"\n{'=' * 60}\nperformance by slice (full model on held-out test)")
    print(f"{'slice':<16}{'n':>7}{'no-skill':>11}{'model LL':>11}{'acc':>8}")
    results: dict[str, dict] = {}
    for name, mask in slices.items():
        n = int(mask.sum())
        if n == 0:
            continue
        y_slice = y_test[mask]
        proba_slice = full_proba[mask]
        base = no_skill_log_loss(y_train, y_slice, LABELS)
        metrics = evaluate_probabilities(y_slice, proba_slice, LABELS)
        results[name] = {"n": n, "no_skill": base, **metrics}
        print(
            f"{name:<16}{n:>7,}{base:>11.4f}"
            f"{metrics['log_loss']:>11.4f}{metrics['accuracy']:>8.1%}"
        )

    # Calibration specifically on the World Cup slice, our true target.
    wc_mask = slices["world cup (+qual)"]
    if wc_mask.sum() >= 50:
        _print_calibration(
            y_test[wc_mask], full_proba[wc_mask], bins=5, title="calibration (World Cup slice)"
        )
    return results


def _print_report(train, test, baseline_ll, elo_metrics, full_metrics) -> None:
    rates = train[TARGET].value_counts(normalize=True)
    print(
        f"rows: {len(train) + len(test):,}  "
        f"(train {len(train):,} -> {train['date'].max().date()}, "
        f"test {len(test):,} from {test['date'].min().date()})\n"
        f"train outcome mix: H {rates.get('H', 0):.0%} / "
        f"D {rates.get('D', 0):.0%} / A {rates.get('A', 0):.0%}\n"
        f"{'-' * 56}\n"
        f"{'model':<22}{'log loss':>12}{'accuracy':>12}\n"
        f"{'no-skill baseline':<22}{baseline_ll:>12.4f}{'-':>12}\n"
        f"{'Elo only':<22}{elo_metrics['log_loss']:>12.4f}{elo_metrics['accuracy']:>11.1%}\n"
        f"{'full features':<22}{full_metrics['log_loss']:>12.4f}{full_metrics['accuracy']:>11.1%}"
    )


def _print_calibration(
    y_test: pd.Series,
    proba: np.ndarray,
    bins: int = 10,
    title: str = "calibration (home-win prob)",
) -> None:
    """Reliability of the home-win probability: predicted vs actual, by bucket."""
    p_home = proba[:, LABELS.index("H")]
    actual_home = (np.asarray(y_test) == "H").astype(int)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_home, edges) - 1, 0, bins - 1)

    print(f"\n{title}:\n  predicted -> actual   (n)")
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        print(
            f"  {p_home[mask].mean():>6.0%}    -> {actual_home[mask].mean():>5.0%}"
            f"   ({mask.sum():>5})"
        )


def _main() -> None:
    run()


if __name__ == "__main__":
    _main()
