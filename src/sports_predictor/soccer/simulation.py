"""Monte Carlo tournament simulation.

Turns the single-match model into tournament-level answers (group advancement,
champion odds) by playing the bracket thousands of times and counting outcomes.

The flow:

1. Freeze each team's strength **as of a cutoff date** (Elo + recent form), so a
   backtest of a past tournament uses only information available before it began.
2. Precompute a 3-way probability for every possible pairing of the tournament's
   teams on a neutral ground. Because "home" is meaningless at a World Cup, each
   pairing is predicted both ways and averaged, removing any residual home bias.
3. Simulate the group stage (round robin, sampled results -> points) to get the
   two qualifiers per group, then the single-elimination knockout, converting the
   draw into an advancement via :mod:`sports_predictor.soccer.knockout`.
4. Repeat N times and aggregate per-team probabilities for each round.

Run the 2022 backtest::

    python -m sports_predictor.soccer.simulation
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sports_predictor.core.paths import PROCESSED_DIR
from sports_predictor.soccer.baseline import FULL_FEATURES, LABELS, TARGET, prepare
from sports_predictor.soccer.elo import DEFAULT_RATING, compute_elo, expected_score
from sports_predictor.soccer.features import _team_perspective
from sports_predictor.soccer.knockout import advance_probabilities
from sports_predictor.soccer.tournaments import (
    WC_2010,
    WC_2014,
    WC_2018,
    WC_2022,
    Tournament,
)

# Typical rest between World Cup matches (days). Applied symmetrically, so it
# never biases one side; it just keeps the feature in a realistic range.
REST_DEFAULT = 4.0

ROUND_NAMES = ["R16", "QF", "SF", "Final"]

# Per-team strength-uncertainty sigma (logit units) used by default. Chosen by
# `tune_strength_sigma` over the 2010-2022 World Cups: champion log loss is nearly
# flat and marginally prefers a small value, so we apply a mild humility prior
# rather than the aggressive softening it would take to match bookmaker spreads
# (which the outcome backtest does not support). See PROJECT_REPORT.md.
DEFAULT_STRENGTH_SIGMA = 0.2


def _utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    """Coerce a date/timestamp to a UTC Timestamp (idempotent)."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# --------------------------------------------------------------------------- #
# 1. Team state as of a cutoff date
# --------------------------------------------------------------------------- #
def compute_team_states(matches: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Per-team strength snapshot using only matches strictly before ``cutoff``.

    Columns mirror the per-team features used by the model: current Elo, plus
    last-5 and last-10 form (win rate, points, goals for/against).
    """
    cutoff_ts = _utc(cutoff)
    pre = matches[matches["date"] < cutoff_ts]
    if pre.empty:
        raise ValueError(f"no matches before cutoff {cutoff_ts.date()}")

    _, ratings = compute_elo(pre)
    long = _team_perspective(pre).sort_values(["team", "date"], kind="stable")

    rows: dict[str, dict] = {}
    for team, g in long.groupby("team", sort=False):
        last5, last10 = g.tail(5), g.tail(10)
        rows[team] = {
            "elo": ratings.get(team, DEFAULT_RATING),
            "win_rate_5": last5["win"].mean(),
            "points_avg_5": last5["points"].mean(),
            "goals_for_avg_5": last5["goals_for"].mean(),
            "goals_against_avg_5": last5["goals_against"].mean(),
            "win_rate_10": last10["win"].mean(),
            "goals_for_avg_10": last10["goals_for"].mean(),
            "goals_against_avg_10": last10["goals_against"].mean(),
        }
    states = pd.DataFrame.from_dict(rows, orient="index")
    return states.fillna(states.mean(numeric_only=True))


def compute_pair_h2h(
    matches: pd.DataFrame, cutoff: str | pd.Timestamp, teams: list[str]
) -> tuple[dict, dict]:
    """Head-to-head meetings and per-side win rate before ``cutoff``.

    Returns ``(meetings, win_rate)`` where ``meetings[(x, y)]`` is the number of
    prior meetings and ``win_rate[(x, y)]`` is x's win fraction in them (0.5 if
    they have never met), both keyed by ordered pair.
    """
    cutoff_ts = _utc(cutoff)
    tset = set(teams)
    pre = matches[(matches["date"] < cutoff_ts)]
    pre = pre[pre["home_team"].isin(tset) & pre["away_team"].isin(tset)]

    meet: Counter = Counter()
    wins: Counter = Counter()
    for r in pre.itertuples(index=False):
        h, a = r.home_team, r.away_team
        meet[(h, a)] += 1
        meet[(a, h)] += 1
        if r.home_score > r.away_score:
            wins[(h, a)] += 1
        elif r.home_score < r.away_score:
            wins[(a, h)] += 1

    win_rate = {}
    for (x, y), n in meet.items():
        win_rate[(x, y)] = wins[(x, y)] / n if n else 0.5
    return dict(meet), win_rate


# --------------------------------------------------------------------------- #
# 2. Matchup probabilities
# --------------------------------------------------------------------------- #
def _matchup_row(home, away, states, meet, win_rate, neutral) -> dict:
    sx, sy = states.loc[home], states.loc[away]
    adv = 0.0 if neutral else 100.0
    return {
        "elo_diff": sx.elo - sy.elo,
        "elo_expected_home": expected_score(sx.elo, sy.elo, adv),
        "home_win_rate_5": sx.win_rate_5,
        "home_points_avg_5": sx.points_avg_5,
        "home_goals_for_avg_5": sx.goals_for_avg_5,
        "home_goals_against_avg_5": sx.goals_against_avg_5,
        "home_win_rate_10": sx.win_rate_10,
        "home_goals_for_avg_10": sx.goals_for_avg_10,
        "home_goals_against_avg_10": sx.goals_against_avg_10,
        "home_rest_days": REST_DEFAULT,
        "away_win_rate_5": sy.win_rate_5,
        "away_points_avg_5": sy.points_avg_5,
        "away_goals_for_avg_5": sy.goals_for_avg_5,
        "away_goals_against_avg_5": sy.goals_against_avg_5,
        "away_win_rate_10": sy.win_rate_10,
        "away_goals_for_avg_10": sy.goals_for_avg_10,
        "away_goals_against_avg_10": sy.goals_against_avg_10,
        "away_rest_days": REST_DEFAULT,
        "h2h_matches": meet.get((home, away), 0),
        "h2h_home_win_rate": win_rate.get((home, away), 0.5),
        "neutral": 1 if neutral else 0,
    }


def build_pair_probabilities(teams, model, states, meet, win_rate) -> dict:
    """Precompute order-averaged neutral-venue probabilities for every pairing.

    Returns ``pair[(a, b)] = (p_a_win, p_draw, p_b_win)`` for ``a < b``. We
    predict each pairing both ways (a-home and b-home) and average to cancel any
    residual home-side artifact the model learned.
    """
    pairs = list(itertools.combinations(sorted(teams), 2))
    rows = [_matchup_row(a, b, states, meet, win_rate, True) for a, b in pairs]
    rows += [_matchup_row(b, a, states, meet, win_rate, True) for a, b in pairs]

    proba = model.predict_proba(pd.DataFrame(rows)[FULL_FEATURES])
    order = [list(model.classes_).index(label) for label in LABELS]
    proba = proba[:, order]  # columns now [H, D, A]

    m = len(pairs)
    ab, ba = proba[:m], proba[m:]
    pair: dict = {}
    for i, (a, b) in enumerate(pairs):
        # ab[i] = [a wins, draw, b wins]; ba[i] = [b wins, draw, a wins]
        p_a = (ab[i, 0] + ba[i, 2]) / 2
        p_d = (ab[i, 1] + ba[i, 1]) / 2
        p_b = (ab[i, 2] + ba[i, 0]) / 2
        total = p_a + p_d + p_b
        pair[(a, b)] = (p_a / total, p_d / total, p_b / total)
    return pair


def make_lookup(pair_probs: dict):
    """Return ``lookup(x, y) -> (p_x_win, p_draw, p_y_win)`` for any order."""

    def lookup(x, y):
        if (x, y) in pair_probs:
            return pair_probs[(x, y)]
        p_y, p_d, p_x = pair_probs[(y, x)]
        return (p_x, p_d, p_y)

    return lookup


def perturbed_lookup(base_lookup, offsets: dict):
    """Wrap a lookup so each team's strength is shifted by ``offsets[team]``.

    The shift is applied in logit space to the *decisive* part of the prediction
    (who wins given it is not a draw); the draw probability is left unchanged.
    Used to inject correlated rating uncertainty: a per-simulation draw of
    ``offsets`` makes a team stronger or weaker across *all* of its matches at
    once, which is how real "this team is over/underrated" uncertainty behaves.
    """

    def lookup(x, y):
        p_x, p_d, p_y = base_lookup(x, y)
        delta = offsets.get(x, 0.0) - offsets.get(y, 0.0)
        if delta == 0.0:
            return p_x, p_d, p_y
        decisive = p_x + p_y
        if decisive <= 0:
            return p_x, p_d, p_y
        # Shift the win/lose split by delta in logit space.
        logit = np.log(p_x / p_y) + delta
        cx = 1.0 / (1.0 + np.exp(-logit))
        return decisive * cx, p_d, decisive * (1.0 - cx)

    return lookup


# --------------------------------------------------------------------------- #
# 3. Simulating one tournament
# --------------------------------------------------------------------------- #
def _sample_outcome(probs, u: float) -> int:
    """0 = first team wins, 1 = draw, 2 = second team wins."""
    if u < probs[0]:
        return 0
    if u < probs[0] + probs[1]:
        return 1
    return 2


def simulate_group(teams, lookup, strength, rng) -> tuple[str, str]:
    """Round-robin a group; return (winner, runner-up).

    Tie-breaking simplification: real World Cup tiebreakers use goal difference
    and goals scored, which need a scoreline model we do not have yet. Here ties
    on points are broken by team strength (Elo), then randomly. A goal-based
    tiebreak is a documented future upgrade.
    """
    points = dict.fromkeys(teams, 0)
    for x, y in itertools.combinations(teams, 2):
        outcome = _sample_outcome(lookup(x, y), rng.random())
        if outcome == 0:
            points[x] += 3
        elif outcome == 1:
            points[x] += 1
            points[y] += 1
        else:
            points[y] += 3

    ranked = sorted(teams, key=lambda t: (points[t], strength[t], rng.random()), reverse=True)
    return ranked[0], ranked[1]


def play_tie(x, y, lookup, strategy, rng) -> str:
    """Play a single knockout tie; return the team that advances."""
    p_x_win, p_draw, p_y_win = lookup(x, y)
    p_x_adv, _ = advance_probabilities(p_x_win, p_draw, p_y_win, strategy=strategy)
    return x if rng.random() < p_x_adv else y


def simulate_tournament(tournament, lookup, strength, strategy, rng) -> dict:
    """Play one full tournament; return furthest round reached per team."""
    qualifiers: dict = {}
    reached: dict[str, str] = {}
    for group, teams in tournament.groups.items():
        winner, runner_up = simulate_group(teams, lookup, strength, rng)
        qualifiers[(group, 1)] = winner
        qualifiers[(group, 2)] = runner_up
        reached[winner] = "R16"
        reached[runner_up] = "R16"

    ties = [(qualifiers[s1], qualifiers[s2]) for s1, s2 in tournament.r16]
    for round_name in ROUND_NAMES:
        winners = [play_tie(x, y, lookup, strategy, rng) for x, y in ties]
        next_round = "Champion" if round_name == "Final" else ROUND_NAMES[ROUND_NAMES.index(round_name) + 1]
        for w in winners:
            reached[w] = next_round
        if round_name != "Final":
            ties = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]

    return reached


# --------------------------------------------------------------------------- #
# 4. Monte Carlo
# --------------------------------------------------------------------------- #
# Each round implies all earlier ones (a finalist also reached the SF, etc.).
_ROUND_ORDER = ["R16", "QF", "SF", "Final", "Champion"]


def monte_carlo(
    tournament,
    lookup,
    strength,
    strategy="proportional",
    n=20000,
    seed=0,
    strength_sigma=0.0,
) -> pd.DataFrame:
    """Run ``n`` simulations; return per-team probabilities for each round.

    ``strength_sigma`` (logit units) is the standard deviation of the per-team,
    per-simulation strength perturbation. 0 disables it (ratings treated as
    exact); larger values inject more rating uncertainty and soften the favorites'
    odds toward the field.
    """
    rng = np.random.default_rng(seed)
    counts = {team: Counter() for team in tournament.teams}
    teams = tournament.teams

    for _ in range(n):
        if strength_sigma > 0:
            offsets = {t: rng.normal(0.0, strength_sigma) for t in teams}
            sim_lookup = perturbed_lookup(lookup, offsets)
        else:
            sim_lookup = lookup
        reached = simulate_tournament(tournament, sim_lookup, strength, strategy, rng)
        for team, furthest in reached.items():
            depth = _ROUND_ORDER.index(furthest)
            for r in _ROUND_ORDER[: depth + 1]:
                counts[team][r] += 1

    table = pd.DataFrame(
        {
            team: {r: counts[team][r] / n for r in _ROUND_ORDER}
            for team in tournament.teams
        }
    ).T
    table.columns = ["reach_R16", "reach_QF", "reach_SF", "reach_Final", "win"]
    return table.sort_values("win", ascending=False)


# --------------------------------------------------------------------------- #
# 5. The 2022 backtest
# --------------------------------------------------------------------------- #
def _train_model(model_table: pd.DataFrame, cutoff: pd.Timestamp):
    train = prepare(model_table[model_table["date"] < cutoff], FULL_FEATURES)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(train[FULL_FEATURES], train[TARGET])
    return model, len(train)


def _build_simulation_inputs(matches, model_table, tournament, cutoff):
    """Train the model and assemble (lookup, strength) for a tournament/cutoff.

    These inputs do not depend on ``strength_sigma``, so they can be built once
    and reused across many simulations (e.g. when calibrating sigma).
    """
    cutoff_ts = _utc(cutoff)
    model, n_train = _train_model(model_table, cutoff_ts)
    states = compute_team_states(matches, cutoff_ts)

    missing = [t for t in tournament.teams if t not in states.index]
    if missing:
        raise ValueError(f"no pre-cutoff data for: {missing}")

    meet, win_rate = compute_pair_h2h(matches, cutoff_ts, tournament.teams)
    pair_probs = build_pair_probabilities(tournament.teams, model, states, meet, win_rate)
    return make_lookup(pair_probs), states["elo"].to_dict(), n_train


def run_backtest(
    tournament: Tournament = WC_2022,
    cutoff: str = "2022-11-20",
    n: int = 20000,
    strength_sigma: float = DEFAULT_STRENGTH_SIGMA,
) -> dict:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    model_table = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")

    lookup, strength, n_train = _build_simulation_inputs(matches, model_table, tournament, cutoff)

    print(
        f"{tournament.name} backtest  (cutoff {cutoff}, trained on {n_train:,} "
        f"matches, {n:,} sims, strength_sigma={strength_sigma})\n"
    )

    results = {}
    for strategy in ("proportional", "even"):
        results[strategy] = monte_carlo(
            tournament, lookup, strength, strategy=strategy, n=n, strength_sigma=strength_sigma
        )

    _print_results(results["proportional"])
    _print_strategy_comparison(results["proportional"], results["even"])
    return results


# --------------------------------------------------------------------------- #
# 6. Calibrating the strength perturbation across multiple World Cups
# --------------------------------------------------------------------------- #
# Past 32-team World Cups with their pre-tournament cutoff and actual winner.
WORLD_CUPS = [
    (WC_2010, "2010-06-11", "Spain"),
    (WC_2014, "2014-06-12", "Germany"),
    (WC_2018, "2018-06-14", "France"),
    (WC_2022, "2022-11-20", "Argentina"),
]


def tune_strength_sigma(
    sigmas=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    n: int = 4000,
    world_cups=WORLD_CUPS,
    seed: int = 0,
) -> dict:
    """Pick the perturbation sigma that best predicts past champions.

    For each candidate sigma we simulate every World Cup in ``world_cups`` and
    score it by the champion log loss ``-log P(actual winner)``, averaged over
    tournaments. Lower is better: a sigma that is too small is overconfident on
    the wrong favorite (heavy loss when an underdog wins), while too large washes
    everything toward uniform. The minimum is the best-calibrated middle ground.

    Caveat: only a handful of tournaments exist, so this is a low-power estimate
    -- treat the result as a sensible default, not a precise optimum.
    """
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    model_table = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")

    # Build per-tournament inputs once; they are independent of sigma.
    inputs = []
    for tour, cutoff, champ in world_cups:
        lookup, strength, _ = _build_simulation_inputs(matches, model_table, tour, cutoff)
        inputs.append((tour, champ, lookup, strength))

    champ_names = [c for _, _, c in world_cups]
    header = f"{'sigma':>7}" + "".join(f"{c[:9]:>11}" for c in champ_names) + f"{'mean LL':>11}"
    print("champion log loss (-log P[actual winner]) by perturbation sigma:\n")
    print(header)
    print("-" * len(header))

    scores: dict[float, float] = {}
    for sigma in sigmas:
        losses = []
        cells = ""
        for tour, champ, lookup, strength in inputs:
            table = monte_carlo(
                tour, lookup, strength, strategy="proportional", n=n, seed=seed,
                strength_sigma=sigma,
            )
            p = max(float(table.loc[champ, "win"]) if champ in table.index else 0.0, 1e-6)
            loss = -np.log(p)
            losses.append(loss)
            cells += f"{loss:>11.3f}"
        mean_ll = float(np.mean(losses))
        scores[sigma] = mean_ll
        marker = ""
        print(f"{sigma:>7.2f}{cells}{mean_ll:>11.3f}{marker}")

    best = min(scores, key=scores.get)
    print(f"\nbest sigma = {best:.2f}  (mean champion log loss {scores[best]:.3f})")
    return scores


def _print_results(table: pd.DataFrame, top: int = 12) -> None:
    print(f"champion odds (proportional knockout rule), top {top}:")
    print(f"  {'team':<16}{'champion':>10}{'final':>9}{'semi':>9}{'R16':>8}")
    for team, row in table.head(top).iterrows():
        print(
            f"  {team:<16}{row['win']:>9.1%}{row['reach_Final']:>9.1%}"
            f"{row['reach_SF']:>9.1%}{row['reach_R16']:>8.1%}"
        )


def _print_strategy_comparison(proportional: pd.DataFrame, even: pd.DataFrame, top: int = 8) -> None:
    print("\ndoes the knockout rule matter? champion odds: proportional vs even")
    print(f"  {'team':<16}{'proportional':>13}{'even':>8}{'delta':>8}")
    for team in proportional.head(top).index:
        p = proportional.loc[team, "win"]
        e = even.loc[team, "win"]
        print(f"  {team:<16}{p:>12.1%}{e:>8.1%}{(p - e) * 100:>+7.1f}pp")


def _main() -> None:
    run_backtest()


if __name__ == "__main__":
    _main()
