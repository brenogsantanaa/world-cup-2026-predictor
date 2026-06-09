"""Player-profile features, aggregated to one row per match.

This is an **additive** signal layer on top of the team-level backbone (Elo,
form, rest, H2H). It is built only from data we already have: the goalscorers
table (``goalscorers.parquet``), which is one row per goal.

What we can and cannot derive
-----------------------------
The source records *goals*, not *appearances* (we have no lineups). So a true
"goals per appearance" is not computable without fabricating an appearances
denominator, and we deliberately do not invent one. What the goal stream *does*
let us measure honestly, per team and as-of each match date, is the **scoring
profile** of a team's recent goals -- signal the team-level features (which only
see scorelines) cannot capture:

    recent_goals        goals the team scored in its last K matches
    top_scorer_goals    most by any single player in that window (star presence)
    goal_concentration  HHI of goals across scorers (1 = one player scores them all)
    penalty_share       fraction of those goals that were penalties
    squad_experience    summed career international goals of the window's scorers
                        (as-of the match date) -- a "proven goalscorers present" proxy

Leakage safety
--------------
A feature for match *M* is computed from goals scored strictly *before* M. We do
one global chronological pass: features for M are read from each team's rolling
window and the players' career totals **before** M's own goals are folded in.

Graceful degradation
--------------------
Coverage is uneven. When a team has no goals in its window we emit NaN (never a
fabricated zero) and set ``low_data = 1``; when it has some but fewer than
``MIN_WINDOW_GOALS`` we still emit values but flag ``low_data = 1``. The model
must remain usable for such teams via the team-level backbone alone, so callers
impute the NaNs (from training statistics) and keep the ``low_data`` flag.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

from sports_predictor.core.paths import PROCESSED_DIR, ensure_dir

# Window over each team's most recent matches, and the minimum goals in it before
# we trust the profile.
WINDOW_MATCHES = 20
MIN_WINDOW_GOALS = 3

GOALSCORERS_FILENAME = "goalscorers.parquet"
MATCHES_FILENAME = "matches.parquet"
PLAYER_FEATURES_FILENAME = "player_features.parquet"

# Feature names per side (prefixed home_p_ / away_p_ in the output).
_PROFILE_FIELDS = [
    "recent_goals",
    "top_scorer_goals",
    "goal_concentration",
    "penalty_share",
    "squad_experience",
]
PLAYER_FEATURE_FIELDS = _PROFILE_FIELDS + ["low_data"]


def _link_goals_to_matches(goals: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Attach ``match_id`` to each goal and keep only player-credited goals.

    Own goals are dropped (the scorer plays for the *other* team, so the goal is
    not a striker credential) along with goals whose fixture we cannot match.
    """
    keys = ["date", "home_team", "away_team"]
    linked = goals.merge(matches[["match_id", *keys]], on=keys, how="inner")
    linked = linked[(~linked["own_goal"].astype(bool)) & linked["scorer"].notna()]
    return linked


def _goals_by_match_team(linked: pd.DataFrame) -> dict[tuple[str, str], list[tuple[str, bool]]]:
    by: dict[tuple[str, str], list[tuple[str, bool]]] = defaultdict(list)
    for r in linked.itertuples(index=False):
        by[(r.match_id, r.team)].append((r.scorer, bool(r.penalty)))
    return by


def _profile(window_counts: Counter, total: int, pen: int, career: dict) -> dict:
    """Compute the scoring-profile features from a team's current window."""
    if total <= 0:
        feats = {f: np.nan for f in _PROFILE_FIELDS}
        feats["low_data"] = 1.0
        return feats
    return {
        "recent_goals": float(total),
        "top_scorer_goals": float(max(window_counts.values())),
        "goal_concentration": float(sum((c / total) ** 2 for c in window_counts.values())),
        "penalty_share": pen / total,
        "squad_experience": float(sum(career.get(s, 0) for s in window_counts)),
        "low_data": 1.0 if total < MIN_WINDOW_GOALS else 0.0,
    }


def build_player_features(
    matches: pd.DataFrame, goals: pd.DataFrame, window: int = WINDOW_MATCHES
) -> pd.DataFrame:
    """Return one row per match (indexed by ``match_id``) of player-profile features.

    Columns: ``home_p_*`` and ``away_p_*`` for each field in
    :data:`PLAYER_FEATURE_FIELDS`. Row count equals ``len(matches)``.
    """
    matches = matches.sort_values(["date", "match_id"], kind="stable")
    goals_by = _goals_by_match_team(_link_goals_to_matches(goals, matches))

    career: dict[str, int] = defaultdict(int)
    win: dict[str, deque] = defaultdict(deque)
    win_counts: dict[str, Counter] = defaultdict(Counter)
    win_total: dict[str, int] = defaultdict(int)
    win_pen: dict[str, int] = defaultdict(int)

    rows: list[dict] = []
    for m in matches.itertuples(index=False):
        row = {"match_id": m.match_id}
        for side, team in (("home", m.home_team), ("away", m.away_team)):
            feats = _profile(win_counts[team], win_total[team], win_pen[team], career)
            for field, value in feats.items():
                row[f"{side}_p_{field}"] = value
        rows.append(row)

        # Fold this match's goals into history *after* reading features (no leakage).
        for team in (m.home_team, m.away_team):
            events = goals_by.get((m.match_id, team), [])
            entry: Counter = Counter()
            entry_pen = 0
            for scorer, penalty in events:
                entry[scorer] += 1
                entry_pen += 1 if penalty else 0
                career[scorer] += 1

            dq = win[team]
            dq.append((entry, entry_pen, sum(entry.values())))
            win_counts[team].update(entry)
            win_total[team] += sum(entry.values())
            win_pen[team] += entry_pen
            if len(dq) > window:
                old_entry, old_pen, old_total = dq.popleft()
                for scorer, c in old_entry.items():
                    win_counts[team][scorer] -= c
                    if win_counts[team][scorer] <= 0:
                        del win_counts[team][scorer]
                win_total[team] -= old_total
                win_pen[team] -= old_pen

    return pd.DataFrame(rows).set_index("match_id")


# Model-facing feature names (both sides).
PLAYER_FEATURES = [
    f"{side}_p_{field}" for side in ("home", "away") for field in PLAYER_FEATURE_FIELDS
]


def build_and_save(processed_dir=PROCESSED_DIR) -> pd.DataFrame:
    matches = pd.read_parquet(processed_dir / MATCHES_FILENAME)
    goals = pd.read_parquet(processed_dir / GOALSCORERS_FILENAME)
    feats = build_player_features(matches, goals)
    out_path = ensure_dir(processed_dir) / PLAYER_FEATURES_FILENAME
    feats.reset_index().to_parquet(out_path, index=False)
    return feats


def _report_coverage(matches: pd.DataFrame, feats: pd.DataFrame) -> None:
    """Show how much of the data the player profiles actually cover, honestly."""
    from sports_predictor.soccer.teams import confederation_of

    df = matches.merge(feats.reset_index(), on="match_id", how="left")
    overall = df["home_p_low_data"].fillna(1.0).mean()
    wc = df[df["is_world_cup"].astype(bool)]
    wc_rate = wc["home_p_low_data"].fillna(1.0).mean() if len(wc) else float("nan")

    print(f"\ncoverage (share of matches with an unreliable/empty home profile):")
    print(f"  all matches:      {overall:.1%}")
    print(f"  World Cup matches: {wc_rate:.1%}")

    df["conf"] = df["home_team"].map(confederation_of)
    by_conf = df.groupby("conf")["home_p_low_data"].agg(
        low_data_rate=lambda s: s.fillna(1.0).mean(), n="size"
    )
    print("\n  low-data rate by home confederation:")
    print(f"    {'confederation':<14}{'low-data':>10}{'n':>9}")
    for conf, r in by_conf.sort_values("low_data_rate").iterrows():
        print(f"    {conf:<14}{r['low_data_rate']:>10.1%}{int(r['n']):>9,}")


def run_experiment(test_fraction: float = 0.2) -> dict:
    """Does the player layer help? Compare backbone vs backbone+player on slices.

    We only keep these features if they improve the **neutral** and **World Cup**
    slices (the conditions we ultimately predict); the global number is dominated
    by lopsided qualifiers and is not the test that matters.
    """
    import numpy as np
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
    feats = pd.read_parquet(PROCESSED_DIR / PLAYER_FEATURES_FILENAME)
    table = load_model_table().merge(feats, on="match_id", how="left")

    _report_coverage(matches, feats.set_index("match_id"))

    # Keep player columns through prepare without letting NaNs drop rows.
    full = prepare(table, FULL_FEATURES, carry=("is_world_cup", *PLAYER_FEATURES))
    train, test = chronological_split(full, date_column="date", test_fraction=test_fraction)

    # Impute player NaNs from TRAIN statistics only (no leakage); flags stay 0/1.
    augmented = list(FULL_FEATURES)
    for col in PLAYER_FEATURES:
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
        proba = proba[:, [list(model.classes_).index(l) for l in LABELS]]
        return proba

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
    for name, proba in [("backbone", proba_base), ("backbone + player", proba_aug)]:
        cells = ""
        results[name] = {}
        for s, m in masks.items():
            ll = evaluate_probabilities(y_test[m], proba[m], LABELS)["log_loss"]
            results[name][s] = ll
            cells += f"{ll:>14.4f}"
        print(f"  {name:<20}{cells}")

    # Verdict on the slices that matter.
    print("\n  delta (backbone + player) - backbone, negative = better:")
    for s in ("neutral", "WC finals~"):
        d = results["backbone + player"][s] - results["backbone"][s]
        print(f"    {s:<12}{d:>+10.4f}")
    helps = all(
        results["backbone + player"][s] < results["backbone"][s]
        for s in ("neutral", "WC finals~")
    )
    print(
        f"\n  verdict: player features {'IMPROVE' if helps else 'do NOT improve'} "
        f"the neutral/WC slices -> {'keep' if helps else 'leave OFF by default'}."
    )
    results["helps"] = helps
    return results


def _main() -> None:
    matches_path = PROCESSED_DIR / MATCHES_FILENAME
    goals_path = PROCESSED_DIR / GOALSCORERS_FILENAME
    for p in (matches_path, goals_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run the results and goalscorers pipelines first."
            )

    feats = build_and_save()
    low = feats["home_p_low_data"].mean()
    print(
        f"feature rows:   {len(feats):,}\n"
        f"home low-data:  {low:.1%} of matches have an unreliable home profile\n"
        f"written:        {PROCESSED_DIR / PLAYER_FEATURES_FILENAME}"
    )
    run_experiment()


if __name__ == "__main__":
    _main()
