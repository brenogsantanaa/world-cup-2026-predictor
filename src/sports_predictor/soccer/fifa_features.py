"""FIFA-ranking features, joined as-of each match date (leakage-safe).

For every match we attach each team's **most recent FIFA ranking published
strictly before kickoff** (`pandas.merge_asof`, backward, exact matches
disallowed). This is the contract's "use the value as it stood on the match date"
rule (DATA_SOURCES.md §6): a match never sees a ranking published on or after it.

Features (both sides + differences):
    home_fifa_rank / away_fifa_rank        integer rank (lower = better)
    home_fifa_points / away_fifa_points    ranking points
    fifa_rank_diff                         home_rank - away_rank (negative = home better)
    fifa_points_diff                       home_points - away_points
    home_fifa_low_data / away_fifa_low_data  1.0 when no ranking was available

Graceful degradation: the source only covers 1993-2018, so pre-1993 and
post-mid-2018 matches (incl. the 2022 World Cup) have no ranking. Those emit NaN
and ``low_data = 1`` -- never a fabricated value -- and callers impute the NaNs
from training statistics while keeping the flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sports_predictor.core.paths import PROCESSED_DIR, ensure_dir

MATCHES_FILENAME = "matches.parquet"
RANKING_FILENAME = "fifa_ranking.parquet"
FIFA_FEATURES_FILENAME = "fifa_features.parquet"

FIFA_FEATURES = [
    "home_fifa_rank",
    "away_fifa_rank",
    "home_fifa_points",
    "away_fifa_points",
    "fifa_rank_diff",
    "fifa_points_diff",
    "home_fifa_low_data",
    "away_fifa_low_data",
]


def _asof_join(long: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Attach each (team, date) row its latest ranking published strictly before."""
    rk = rankings[["date", "team", "rank", "points"]].sort_values("date", kind="stable")
    left = long.sort_values("date", kind="stable")
    return pd.merge_asof(
        left,
        rk,
        on="date",
        by="team",
        direction="backward",
        allow_exact_matches=False,
    )


def build_fifa_features(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Return one row per match (indexed by ``match_id``) of FIFA-ranking features."""
    home = matches[["match_id", "date", "home_team"]].rename(columns={"home_team": "team"})
    home["side"] = "home"
    away = matches[["match_id", "date", "away_team"]].rename(columns={"away_team": "team"})
    away["side"] = "away"
    long = pd.concat([home, away], ignore_index=True)

    merged = _asof_join(long, rankings)

    wide = merged.pivot(index="match_id", columns="side", values=["rank", "points"])
    wide.columns = [f"{side}_fifa_{stat}" for stat, side in wide.columns]
    wide = wide.reindex(matches["match_id"])

    out = pd.DataFrame(index=matches["match_id"])
    out["home_fifa_rank"] = wide["home_fifa_rank"]
    out["away_fifa_rank"] = wide["away_fifa_rank"]
    out["home_fifa_points"] = wide["home_fifa_points"]
    out["away_fifa_points"] = wide["away_fifa_points"]
    out["fifa_rank_diff"] = out["home_fifa_rank"] - out["away_fifa_rank"]
    out["fifa_points_diff"] = out["home_fifa_points"] - out["away_fifa_points"]
    out["home_fifa_low_data"] = out["home_fifa_rank"].isna().astype(float)
    out["away_fifa_low_data"] = out["away_fifa_rank"].isna().astype(float)
    out.index.name = "match_id"
    return out


def build_and_save(processed_dir=PROCESSED_DIR) -> pd.DataFrame:
    matches = pd.read_parquet(processed_dir / MATCHES_FILENAME)
    rankings = pd.read_parquet(processed_dir / RANKING_FILENAME)
    feats = build_fifa_features(matches, rankings)
    out_path = ensure_dir(processed_dir) / FIFA_FEATURES_FILENAME
    feats.reset_index().to_parquet(out_path, index=False)
    return feats


def _report_coverage(matches: pd.DataFrame, feats: pd.DataFrame) -> None:
    from sports_predictor.soccer.teams import confederation_of

    df = matches.merge(feats.reset_index(), on="match_id", how="left")
    overall = df["home_fifa_low_data"].fillna(1.0).mean()
    wc = df[df["is_world_cup"].astype(bool)]
    wc_rate = wc["home_fifa_low_data"].fillna(1.0).mean() if len(wc) else float("nan")
    print("\nFIFA coverage (share of matches with NO home ranking available):")
    print(f"  all matches:       {overall:.1%}")
    print(f"  World Cup matches: {wc_rate:.1%}")

    df["conf"] = df["home_team"].map(confederation_of)
    by_conf = df.groupby("conf")["home_fifa_low_data"].agg(
        low_data_rate=lambda s: s.fillna(1.0).mean(), n="size"
    )
    print("\n  no-ranking rate by home confederation:")
    print(f"    {'confederation':<14}{'low-data':>10}{'n':>9}")
    for conf, r in by_conf.sort_values("low_data_rate").iterrows():
        print(f"    {conf:<14}{r['low_data_rate']:>10.1%}{int(r['n']):>9,}")


def run_experiment(test_fraction: float = 0.2) -> dict:
    """Does FIFA ranking help over the backbone? Judge on neutral/WC slices."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from sports_predictor.core.evaluation import evaluate_probabilities, no_skill_log_loss
    from sports_predictor.core.splitting import chronological_split
    from sports_predictor.soccer.baseline import (
        FULL_FEATURES,
        LABELS,
        TARGET,
        load_model_table,
        prepare,
    )

    matches = pd.read_parquet(PROCESSED_DIR / MATCHES_FILENAME)
    feats = pd.read_parquet(PROCESSED_DIR / FIFA_FEATURES_FILENAME)
    table = load_model_table().merge(feats, on="match_id", how="left")

    _report_coverage(matches, feats.set_index("match_id"))

    full = prepare(table, FULL_FEATURES, carry=("is_world_cup", *FIFA_FEATURES))
    train, test = chronological_split(full, date_column="date", test_fraction=test_fraction)

    augmented = list(FULL_FEATURES)
    for col in FIFA_FEATURES:
        if col.endswith("_low_data"):
            train[col] = train[col].fillna(1.0)
            test[col] = test[col].fillna(1.0)
        else:
            med = train[col].median()
            train[col] = train[col].fillna(med)
            test[col] = test[col].fillna(med)
        augmented.append(col)

    y_train = train[TARGET]
    y_test = test[TARGET].reset_index(drop=True)
    is_neutral = (test["neutral"] == 1).to_numpy()
    is_wcf = (test["is_world_cup"].astype(bool) & (test["neutral"] == 1)).to_numpy()
    masks = {"all": np.ones(len(test), bool), "neutral": is_neutral, "WC finals~": is_wcf}

    def fit_eval(features):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(train[features], y_train)
        proba = model.predict_proba(test[features])
        return proba[:, [list(model.classes_).index(l) for l in LABELS]]

    proba_base = fit_eval(FULL_FEATURES)
    proba_aug = fit_eval(augmented)

    print("\nlog loss by slice  (backbone = Elo/form/rest/H2H):")
    header = f"  {'model':<20}" + "".join(f"{s + ' LL':>14}" for s in masks)
    print(header)
    print("  " + "-" * (len(header) - 2))
    base_cells = "".join(
        f"{no_skill_log_loss(y_train, y_test[m], LABELS):>14.4f}" for m in masks.values()
    )
    print(f"  {'no-skill':<20}{base_cells}")
    results = {}
    for name, proba in [("backbone", proba_base), ("backbone + FIFA", proba_aug)]:
        cells = ""
        results[name] = {}
        for s, m in masks.items():
            ll = evaluate_probabilities(y_test[m], proba[m], LABELS)["log_loss"]
            results[name][s] = ll
            cells += f"{ll:>14.4f}"
        print(f"  {name:<20}{cells}")

    print("\n  delta (backbone + FIFA) - backbone, negative = better:")
    deltas = {}
    for s in ("neutral", "WC finals~"):
        deltas[s] = results["backbone + FIFA"][s] - results["backbone"][s]
        print(f"    {s:<12}{deltas[s]:>+10.4f}")

    # Nuanced verdict: a meaningful change is >0.001 log loss either way.
    helps_wc = deltas["WC finals~"] < -0.001
    hurts_neutral = deltas["neutral"] > 0.001
    results["helps_wc"] = helps_wc
    print(
        "\n  verdict: FIFA ranking "
        + (
            "IMPROVES the World Cup slice"
            if helps_wc
            else "does not improve the World Cup slice"
        )
        + (" but is flat on neutral" if not hurts_neutral else " and worsens neutral")
        + ".\n  NOTE: the source ends mid-2018, so live 2026 matches would have NO\n"
        "  ranking. Left OFF by default until a current FIFA-ranking source is wired\n"
        "  in; the WC-slice gain says it's worth enabling once coverage is fixed."
    )
    return results


def _main() -> None:
    feats = build_and_save()
    low = feats["home_fifa_low_data"].mean()
    print(
        f"feature rows:  {len(feats):,}\n"
        f"home no-rank:  {low:.1%}\n"
        f"written:       {PROCESSED_DIR / FIFA_FEATURES_FILENAME}"
    )
    run_experiment()


if __name__ == "__main__":
    _main()
