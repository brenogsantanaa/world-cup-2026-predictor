"""Aggregate player profiles into team-match features + a slice backtest harness.

This is the *additive* player layer: the Elo/form/rest/H2H backbone stays primary,
and squad-derived signal is layered on top only if it earns its place on the
neutral/World-Cup slices.

Two paths, and the difference matters for leakage:

- :func:`build_team_match_features` with ``static_snapshot=True`` attaches a single
  *current* squad snapshot to matches. That is correct **only for forward (2026)
  prediction** -- using today's squad/club form for a 2015 match would be
  anachronistic. It is *not* a valid backtest input.
- A real backtest needs **as-of** team-match features (each squad's club form as it
  stood before kickoff), saved to ``data/processed/squad_match_features.parquet``.
  Building that requires historical, time-stamped club stats per player -- i.e.
  real cached pages across seasons. Until that exists, :func:`run_slice_backtest`
  reports the prerequisite instead of inventing a verdict.

Graceful degradation throughout: nations with no covered players get NaN + a
``low_data`` flag; aggregates sum only known values; coverage is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sports_predictor.canonical.registry import team_id as team_id_of
from sports_predictor.core.paths import PROCESSED_DIR

IN_FORM_XG90 = 0.4   # an attacker with >=0.4 xG/90 is "in form"
TOP_N = 3            # aggregate over a squad's most productive N players

NATION_FEATURES = ["squad_value_eur", "top_xg90", "mean_top_xg90", "share_in_form"]


def aggregate_nation_features(player_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a per-player table into per-nation squad features.

    ``player_table`` needs columns ``team_id`` and any of ``market_value_eur`` /
    ``xg_per_90``. Returns one row per ``team_id`` with squad aggregates plus
    coverage and a ``low_data`` flag (set when under half the squad is covered).
    """
    def _agg(group: pd.DataFrame) -> pd.Series:
        xg = group.get("xg_per_90", pd.Series(dtype=float)).dropna()
        val = group.get("market_value_eur", pd.Series(dtype=float)).dropna()
        n = len(group)
        top = xg.nlargest(TOP_N)
        return pd.Series(
            {
                "squad_size": n,
                "squad_value_eur": val.sum() if len(val) else np.nan,
                "value_coverage": len(val) / n if n else np.nan,
                "top_xg90": xg.max() if len(xg) else np.nan,
                "mean_top_xg90": top.mean() if len(top) else np.nan,
                "share_in_form": (xg >= IN_FORM_XG90).mean() if len(xg) else np.nan,
                "xg_coverage": len(xg) / n if n else np.nan,
            }
        )

    out = player_table.groupby("team_id", dropna=False).apply(_agg, include_groups=False)
    out = out.reset_index()
    cov = out[["value_coverage", "xg_coverage"]].fillna(0.0).max(axis=1)
    out["low_data"] = (cov < 0.5).astype(float)
    return out


def build_team_match_features(
    matches: pd.DataFrame,
    nation_features: pd.DataFrame,
    static_snapshot: bool = True,
) -> pd.DataFrame:
    """Attach each side's squad aggregates to matches (home/away + differences).

    ``nation_features`` must be keyed by ``team_id`` (e.g. from
    :func:`aggregate_nation_features`). When ``static_snapshot`` is True this is a
    forward-prediction artifact only (see module docstring); historical backtests
    require as-of features.
    """
    feats = nation_features.set_index("team_id")
    cols = [c for c in NATION_FEATURES if c in feats.columns]

    out = pd.DataFrame(index=matches["match_id"])
    out.index.name = "match_id"
    home_id = matches["home_team"].map(team_id_of).to_numpy()
    away_id = matches["away_team"].map(team_id_of).to_numpy()

    for col in cols:
        home_vals = feats[col].reindex(home_id).to_numpy()
        away_vals = feats[col].reindex(away_id).to_numpy()
        out[f"home_{col}"] = home_vals
        out[f"away_{col}"] = away_vals
        out[f"{col}_diff"] = home_vals - away_vals

    low_map = feats["low_data"] if "low_data" in feats.columns else None

    def side_low_data(ids):
        if low_map is None:
            return np.ones(len(ids))
        flag = low_map.reindex(ids).to_numpy()  # NaN when the nation is absent
        return np.where(np.isnan(flag), 1.0, flag)

    out["home_squad_low_data"] = side_low_data(home_id)
    out["away_squad_low_data"] = side_low_data(away_id)
    out.attrs["static_snapshot"] = static_snapshot
    return out


def run_slice_backtest(
    asof_features_path=PROCESSED_DIR / "squad_match_features.parquet",
    test_fraction: float = 0.2,
) -> dict:
    """Backtest backbone vs backbone+squad on neutral/WC slices -- if data exists.

    Requires a *leakage-safe, as-of* team-match feature table at
    ``asof_features_path`` (each squad's club form before kickoff). If it is not
    present, this prints the prerequisite and returns ``{"ran": False}`` rather
    than producing a misleading verdict from a current-snapshot.
    """
    path = asof_features_path
    if not path.exists():
        print(
            "Squad-feature backtest harness is wired and ready, but NOT run:\n"
            f"  missing as-of feature table: {path}\n"
            "  Prerequisite: historical, time-stamped club stats per player (real\n"
            "  cached Understat/FBref/Transfermarkt pages across seasons), aggregated\n"
            "  to each squad's pre-kickoff club form. A current snapshot would be\n"
            "  anachronistic for past matches, so no verdict is fabricated here.\n"
            "  Once that parquet exists, this compares neutral/WC log loss and keeps\n"
            "  the features only if they improve those slices."
        )
        return {"ran": False, "reason": "missing as-of squad features"}

    # --- real path (exercised once as-of data is cached) -------------------- #
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from sports_predictor.core.evaluation import evaluate_probabilities
    from sports_predictor.core.splitting import chronological_split
    from sports_predictor.soccer.baseline import FULL_FEATURES, LABELS, TARGET, load_model_table, prepare

    feats = pd.read_parquet(path)
    table = load_model_table().merge(feats, on="match_id", how="left")
    extra = [c for c in feats.columns if c != "match_id"]

    full = prepare(table, FULL_FEATURES, carry=("is_world_cup", *extra))
    train, test = chronological_split(full, date_column="date", test_fraction=test_fraction)
    for col in extra:
        fill = 1.0 if col.endswith("low_data") else train[col].median()
        train[col] = train[col].fillna(fill)
        test[col] = test[col].fillna(fill)

    y_train, y_test = train[TARGET], test[TARGET].reset_index(drop=True)
    masks = {
        "neutral": (test["neutral"] == 1).to_numpy(),
        "WC finals~": (test["is_world_cup"].astype(bool) & (test["neutral"] == 1)).to_numpy(),
    }

    def fit_eval(features):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(train[features], y_train)
        proba = model.predict_proba(test[features])
        return proba[:, [list(model.classes_).index(l) for l in LABELS]]

    base, aug = fit_eval(FULL_FEATURES), fit_eval(list(FULL_FEATURES) + extra)
    results = {"ran": True}
    helps = True
    for s, m in masks.items():
        b = evaluate_probabilities(y_test[m], base[m], LABELS)["log_loss"]
        a = evaluate_probabilities(y_test[m], aug[m], LABELS)["log_loss"]
        results[s] = {"backbone": b, "backbone+squad": a, "delta": a - b}
        helps = helps and (a < b)
        print(f"  {s:<12} backbone {b:.4f} -> +squad {a:.4f}  ({a - b:+.4f})")
    results["helps"] = helps
    print(f"  verdict: {'keep' if helps else 'leave OFF by default'}")
    return results
