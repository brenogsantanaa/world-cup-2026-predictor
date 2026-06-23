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
from sports_predictor.soccer.dixon_coles import DEFAULT_XI, DixonColesModel
from sports_predictor.soccer.elo import DEFAULT_RATING, compute_elo, expected_score
from sports_predictor.soccer.features import _team_perspective
from sports_predictor.soccer.knockout import advance_probabilities
from sports_predictor.soccer.tournaments import (
    WC_2010,
    WC_2014,
    WC_2018,
    WC_2022,
    WC_2026,
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


def simulate_group_ranked(teams, lookup, strength, rng) -> tuple[list[str], dict[str, int]]:
    """Round-robin a group; return (teams ranked best-first, points by team).

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
    return ranked, points


def simulate_group(teams, lookup, strength, rng) -> tuple[str, str]:
    """Round-robin a group; return (winner, runner-up)."""
    ranked, _ = simulate_group_ranked(teams, lookup, strength, rng)
    return ranked[0], ranked[1]


def play_tie(x, y, lookup, strategy, rng) -> str:
    """Play a single knockout tie; return the team that advances."""
    p_x_win, p_draw, p_y_win = lookup(x, y)
    p_x_adv, _ = advance_probabilities(p_x_win, p_draw, p_y_win, strategy=strategy)
    return x if rng.random() < p_x_adv else y


def simulate_tournament(tournament, lookup, strength, strategy, rng) -> dict:
    """Play one full tournament; return furthest round reached per team."""
    if tournament.is_48:
        return _simulate_48(tournament, lookup, strength, strategy, rng)
    return _simulate_32(tournament, lookup, strength, strategy, rng)


def _simulate_32(tournament, lookup, strength, strategy, rng) -> dict:
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


# Labels assigned to the WINNER of each 48-team knockout round (R32 onward); the
# 32 qualifiers start at "R32". Length = len(bracket) + 1.
_LABELS_48 = ["R16", "QF", "SF", "Final", "Champion"]


def _assign_thirds(best, tournament, rng) -> dict[int, str]:
    """Assign the best third-placed teams to the R32 third-slots.

    ``best`` is the qualifying thirds, already ordered best-first, as
    ``(points, strength, group, team)``. We seed them into the third-slots in
    bracket order, skipping a slot whose host-group winner is from the same group
    (FIFA never lets a group winner face their own group's third). This is a
    transparent stand-in for FIFA's full 495-case combination table and only
    affects R32 pairings. Returns ``{r32_index: team}``.
    """
    host_group = {}
    for idx in tournament.third_place_slots:
        slot_a, slot_b = tournament.r32[idx]
        host = slot_a if slot_a[0] == "W" else slot_b
        host_group[idx] = host[1]

    available = [(group, team) for _, _, group, team in best]
    assign: dict[int, str] = {}
    for idx in tournament.third_place_slots:
        pick = next(
            (i for i, (g, _) in enumerate(available) if g != host_group[idx]),
            0 if available else None,
        )
        if pick is None:
            continue
        _, team = available.pop(pick)
        assign[idx] = team
    return assign


def _resolve_slot(slot, qualifiers, third_assign, idx):
    kind = slot[0]
    if kind == "W":
        return qualifiers[(slot[1], 1)]
    if kind == "R":
        return qualifiers[(slot[1], 2)]
    return third_assign[idx]  # "3"


def _simulate_48(tournament, lookup, strength, strategy, rng) -> dict:
    reached: dict[str, str] = {}
    qualifiers: dict = {}
    thirds = []
    for group, teams in tournament.groups.items():
        ranked, points = simulate_group_ranked(teams, lookup, strength, rng)
        qualifiers[(group, 1)] = ranked[0]
        qualifiers[(group, 2)] = ranked[1]
        reached[ranked[0]] = "R32"
        reached[ranked[1]] = "R32"
        thirds.append((points[ranked[2]], strength[ranked[2]], group, ranked[2]))

    best = sorted(thirds, key=lambda x: (x[0], x[1], rng.random()), reverse=True)[: tournament.n_thirds]
    for _, _, _, team in best:
        reached[team] = "R32"
    third_assign = _assign_thirds(best, tournament, rng)

    first_ties = [
        (
            _resolve_slot(slot_a, qualifiers, third_assign, idx),
            _resolve_slot(slot_b, qualifiers, third_assign, idx),
        )
        for idx, (slot_a, slot_b) in enumerate(tournament.r32)
    ]

    winners = [play_tie(x, y, lookup, strategy, rng) for x, y in first_ties]
    for w in winners:
        reached[w] = _LABELS_48[0]
    prev = winners
    for ri, round_conn in enumerate(tournament.bracket):
        prev = [play_tie(prev[i], prev[j], lookup, strategy, rng) for i, j in round_conn]
        for w in prev:
            reached[w] = _LABELS_48[ri + 1]

    return reached


# --------------------------------------------------------------------------- #
# 4. Monte Carlo
# --------------------------------------------------------------------------- #
# Each round implies all earlier ones (a finalist also reached the SF, etc.).
_ROUND_ORDER = ["R16", "QF", "SF", "Final", "Champion"]
_ROUND_ORDER_48 = ["R32", "R16", "QF", "SF", "Final", "Champion"]


def _round_order(tournament) -> list[str]:
    return _ROUND_ORDER_48 if tournament.is_48 else _ROUND_ORDER


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
    round_order = _round_order(tournament)

    for _ in range(n):
        if strength_sigma > 0:
            offsets = {t: rng.normal(0.0, strength_sigma) for t in teams}
            sim_lookup = perturbed_lookup(lookup, offsets)
        else:
            sim_lookup = lookup
        reached = simulate_tournament(tournament, sim_lookup, strength, strategy, rng)
        for team, furthest in reached.items():
            depth = round_order.index(furthest)
            for r in round_order[: depth + 1]:
                counts[team][r] += 1

    table = pd.DataFrame(
        {team: {r: counts[team][r] / n for r in round_order} for team in teams}
    ).T
    table.columns = [f"reach_{r}" for r in round_order[:-1]] + ["win"]
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


def build_pair_probabilities_dc(teams, dc_model: DixonColesModel) -> dict:
    """Neutral-venue (a-win, draw, b-win) for every pairing from a goal model.

    The Dixon-Coles model is already symmetric on neutral ground (no home term),
    so unlike the classifier path we do not need to predict both orders and
    average -- one call per pairing is exact.
    """
    pair: dict = {}
    for a, b in itertools.combinations(sorted(teams), 2):
        pair[(a, b)] = dc_model.outcome_proba(a, b, neutral=True)
    return pair


def _build_dc_inputs(matches, tournament, cutoff, xi: float = DEFAULT_XI):
    """Fit Dixon-Coles and assemble (lookup, strength) for a tournament/cutoff.

    Strength (used only for group tie-breaks and the perturbation) is the team's
    overall rating attack+defense; teams unseen in the fit window fall back to the
    league-average strength the model already uses.
    """
    cutoff_ts = _utc(cutoff)
    dc = DixonColesModel(xi=xi).fit(matches, cutoff_ts)

    def overall(team: str) -> float:
        atk, dfn = dc._strength(team)
        return atk + dfn

    strength = {t: overall(t) for t in tournament.teams}
    lookup = make_lookup(build_pair_probabilities_dc(tournament.teams, dc))
    return lookup, strength, dc


# Default ensemble weight on the Dixon-Coles goal model; the rest goes to the Elo
# backbone. The two models have opposite blind spots (Elo is win-based and blind
# to scoring style; Dixon-Coles is goal-based and skewed by confederation scoring
# levels), so blending cancels both and beats either alone.
#
# Tuned by `tune_ensemble_weight` on the 256 pooled WC-finals matches of the
# 2010-2022 World Cups (leakage-safe, the same higher-power metric `tune_xi`
# uses): blending cuts WC-finals match log loss from 0.9961 (backbone-only) to a
# broad, flat minimum of ~0.988 across w_dc in [0.30, 0.50] (optimum 0.40). We
# take 0.35 rather than the bare optimum: it is within sampling noise on the WC
# slice (0.9880 vs 0.9878) while avoiding the slight degradation that w_dc >= 0.40
# causes on the broader neutral/all-match slices. See PROJECT_REPORT.md.
DEFAULT_ENSEMBLE_W_DC = 0.35


def _build_ensemble_inputs(matches, model_table, tournament, cutoff, w_dc=DEFAULT_ENSEMBLE_W_DC):
    """Blend the Elo backbone and the Dixon-Coles goal model per pairing.

    Each model produces (a-win, draw, b-win) on neutral ground for every pairing;
    we take a weighted average and renormalise. ``w_dc`` is the weight on the goal
    model. Strength (tie-breaks/perturbation) stays the Elo rating.
    """
    cutoff_ts = _utc(cutoff)
    model, n_train = _train_model(model_table, cutoff_ts)
    states = compute_team_states(matches, cutoff_ts)
    meet, win_rate = compute_pair_h2h(matches, cutoff_ts, tournament.teams)
    backbone = build_pair_probabilities(tournament.teams, model, states, meet, win_rate)

    dc = DixonColesModel().fit(matches, cutoff_ts)
    dc_pairs = build_pair_probabilities_dc(tournament.teams, dc)
    pair = blend_pair_probabilities(backbone, dc_pairs, w_dc)
    return make_lookup(pair), states["elo"].to_dict(), n_train, dc


def blend_pair_probabilities(backbone: dict, dc_pairs: dict, w_dc: float) -> dict:
    """Weighted average of two pairwise (win, draw, lose) tables, renormalised.

    ``w_dc`` is the weight on the Dixon-Coles entry, ``1 - w_dc`` on the backbone.
    Each blended triple is renormalised so it remains a valid probability.
    """
    pair: dict = {}
    for key, b in backbone.items():
        d = dc_pairs[key]
        blended = [(1.0 - w_dc) * bi + w_dc * di for bi, di in zip(b, d)]
        s = sum(blended)
        pair[key] = tuple(x / s for x in blended)
    return pair


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
# 5b. Forward prediction (a real, not backtested, tournament)
# --------------------------------------------------------------------------- #
WC2026_ODDS_PARQUET = "wc2026_odds.parquet"
WC2026_ODDS_CSV = "wc2026_odds.csv"
WC2026_LABELS = {
    "backbone": "backbone (Elo + form) + international features, no club data",
    "dixon_coles": "Dixon-Coles goal model (attack/defense, time-decay), no club data",
    "ensemble": "ensemble: Elo backbone + Dixon-Coles goal model (50/50), no club data",
}


def run_forward(
    tournament: Tournament = WC_2026,
    cutoff: str = "2026-06-11",
    n: int = 20000,
    strength_sigma: float = DEFAULT_STRENGTH_SIGMA,
    seed: int = 0,
    save: bool = True,
    model_kind: str = "backbone",
) -> pd.DataFrame:
    """Predict a real, upcoming tournament (no known result to backtest against).

    ``model_kind`` selects the engine:

    * ``"backbone"``     - the Elo + form/H2H logistic model (original path).
    * ``"dixon_coles"``  - the bivariate Poisson goal model.

    Either way the run uses no club data (that enrichment is separate and
    additive). Saves the odds table to processed parquet + CSV, suffixed by model.
    """
    if model_kind not in WC2026_LABELS:
        raise ValueError(f"unknown model_kind {model_kind!r}; use {list(WC2026_LABELS)}")

    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")

    if model_kind == "dixon_coles":
        lookup, strength, dc = _build_dc_inputs(matches, tournament, cutoff)
        n_train = dc.n_matches
        extra = f"  gamma={dc.gamma:.3f} rho={dc.rho:.3f}"
    elif model_kind == "ensemble":
        model_table = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
        lookup, strength, n_train, dc = _build_ensemble_inputs(
            matches, model_table, tournament, cutoff
        )
        extra = f"  w_dc={DEFAULT_ENSEMBLE_W_DC}"
    else:
        model_table = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
        lookup, strength, n_train = _build_simulation_inputs(
            matches, model_table, tournament, cutoff
        )
        extra = ""

    table = monte_carlo(
        tournament, lookup, strength, strategy="proportional", n=n, seed=seed,
        strength_sigma=strength_sigma,
    )

    label = WC2026_LABELS[model_kind]
    print(
        f"{tournament.name} forward prediction  (as-of {tournament.as_of}, cutoff "
        f"{cutoff})\n"
        f"  model: {model_kind}{extra}\n"
        f"  trained on {n_train:,} matches, {n:,} sims, strength_sigma="
        f"{strength_sigma}\n"
        f"  source: {label}\n"
    )
    _print_results(table)

    if save:
        suffix = "" if model_kind == "backbone" else f"_{model_kind}"
        parquet = WC2026_ODDS_PARQUET.replace(".parquet", f"{suffix}.parquet")
        csv = WC2026_ODDS_CSV.replace(".csv", f"{suffix}.csv")
        export = table.copy()
        export.insert(0, "team", export.index)
        export = export.rename(columns={"win": "champion"})
        export["source"] = label
        export["as_of"] = tournament.as_of
        export = export.reset_index(drop=True).round(6)
        export.to_parquet(PROCESSED_DIR / parquet, index=False)
        export.to_csv(PROCESSED_DIR / csv, index=False)
        print(f"\nsaved: {PROCESSED_DIR / parquet}\n       {PROCESSED_DIR / csv}")
    return table


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


# --------------------------------------------------------------------------- #
# 6b. Calibrating the Dixon-Coles time-decay (xi) across past World Cups
# --------------------------------------------------------------------------- #
# Candidate per-day decay rates. 0 = no decay (all history weighted equally);
# bigger = forgets the past faster. Half-life in years = ln(2) / xi / 365.25, so
# this grid spans roughly "no decay" down to a ~0.7-year half-life.
DC_XI_GRID = (0.0, 0.0002, 0.0005, 0.0008, 0.0012, 0.0018, 0.0026)


def _wc_finals_slice(matches: pd.DataFrame, cutoff, days: int = 45) -> pd.DataFrame:
    """The actual World Cup finals matches in the ``days`` after ``cutoff``.

    Leakage-safe by construction: these all kick off on/after the cutoff, so a
    model fit strictly before the cutoff has never seen them.
    """
    cut = _utc(cutoff)
    dates = pd.to_datetime(matches["date"])
    if dates.dt.tz is None:
        dates = dates.dt.tz_localize("UTC")
    mask = (
        (matches["tournament"] == "FIFA World Cup")
        & (dates >= cut)
        & (dates <= cut + pd.Timedelta(days=days))
    )
    return matches[mask]


def tune_xi(
    xis=DC_XI_GRID,
    n: int = 4000,
    world_cups=WORLD_CUPS,
    seed: int = 0,
    strength_sigma: float = DEFAULT_STRENGTH_SIGMA,
    days: int = 45,
    verbose: bool = True,
) -> dict:
    """Pick the Dixon-Coles time-decay ``xi`` by backtesting past World Cups.

    For each candidate ``xi`` we refit the goal model before each World Cup and
    score it two ways, both measured only on World Cup matches and both
    leakage-safe (the model never sees the tournament it is judged on):

    * **match log loss** - pooled over every actual finals match of the 2010-2022
      World Cups (256 matches). This is the higher-power signal and the one we
      select on; log loss rewards honest probabilities, not just the top pick.
    * **champion log loss** - ``-log P[actual winner]`` from the Monte Carlo
      simulation, averaged over the four tournaments. Only four data points, so
      it is noisy; we report it as a sanity check, not the deciding metric.

    Each ``xi`` is fit once per tournament and reused for both signals. Returns
    ``{xi: {"match_ll", "champion_ll", "gamma", "rho"}}``.

    Caveat: four tournaments is a small sample. Treat the winner as a
    well-supported default, not a precise optimum.
    """
    from sports_predictor.core.evaluation import evaluate_probabilities
    from sports_predictor.soccer.dixon_coles import LABELS as DC_LABELS

    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    matches = matches.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)

    if verbose:
        print(
            "Dixon-Coles xi tuning  (lower log loss = better)\n"
            f"  scored on {len(world_cups)} World Cups, {n:,} sims each, "
            f"strength_sigma={strength_sigma}\n"
            f"{'-' * 64}\n"
            f"{'xi':>8}{'half-life':>11}{'match LL':>11}{'champ LL':>11}"
            f"{'gamma':>8}{'rho':>8}"
        )

    results: dict[float, dict] = {}
    for xi in xis:
        ys, ps, champ_losses, gammas, rhos = [], [], [], [], []
        for tour, cutoff, champ in world_cups:
            dc = DixonColesModel(xi=xi).fit(matches, cutoff)
            gammas.append(dc.gamma)
            rhos.append(dc.rho)

            finals = _wc_finals_slice(matches, cutoff, days)
            if len(finals):
                ps.append(dc.predict_proba(finals))
                ys.append(finals["result"].to_numpy())

            lookup = make_lookup(build_pair_probabilities_dc(tour.teams, dc))
            strength = {t: sum(dc._strength(t)) for t in tour.teams}
            table = monte_carlo(
                tour, lookup, strength, strategy="proportional", n=n, seed=seed,
                strength_sigma=strength_sigma,
            )
            p = max(float(table.loc[champ, "win"]) if champ in table.index else 0.0, 1e-6)
            champ_losses.append(-np.log(p))

        y_true = pd.Series(np.concatenate(ys))
        proba = np.concatenate(ps)
        match_ll = evaluate_probabilities(y_true, proba, DC_LABELS)["log_loss"]
        champ_ll = float(np.mean(champ_losses))
        gamma, rho = float(np.mean(gammas)), float(np.mean(rhos))
        results[xi] = {"match_ll": match_ll, "champion_ll": champ_ll, "gamma": gamma, "rho": rho}

        if verbose:
            hl = "inf" if xi == 0 else f"{np.log(2) / xi / 365.25:.1f}y"
            print(f"{xi:>8.4f}{hl:>11}{match_ll:>11.4f}{champ_ll:>11.4f}{gamma:>8.3f}{rho:>8.3f}")

    best = min(results, key=lambda k: results[k]["match_ll"])
    if verbose:
        print(
            f"\nbest xi = {best:.4f}  (match log loss {results[best]['match_ll']:.4f}, "
            f"half-life {'inf' if best == 0 else f'{np.log(2) / best / 365.25:.1f}y'})\n"
            f"selected on match log loss; champion log loss shown only as a sanity check."
        )
    return results


# --------------------------------------------------------------------------- #
# 6c. Tuning the ensemble weight + an honest match-level bake-off
# --------------------------------------------------------------------------- #
# Candidate weights on the Dixon-Coles goal model (rest goes to the Elo backbone).
ENSEMBLE_W_GRID = (0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0)


def _wc_match_predictions(matches, model_table, world_cups=WORLD_CUPS, days: int = 45):
    """Pooled, leakage-safe backbone & Dixon-Coles match probabilities on past WCs.

    For each World Cup we fit both models strictly before its cutoff and predict
    the actual finals matches (joined to ``model_table`` by ``match_id`` for the
    backbone's features). Returns ``(y_true, p_backbone, p_dc)`` pooled over every
    finals match of every tournament in ``world_cups`` -- the same higher-power
    signal ``tune_xi`` selects on, reused so the ensemble weight is tuned on the
    slice that matters rather than on the held-out test set.
    """
    from sports_predictor.soccer.baseline import FULL_FEATURES, TARGET, prepare

    clean = matches.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    ys, p_bb, p_dc = [], [], []
    for tour, cutoff, _champ in world_cups:
        cutoff_ts = _utc(cutoff)
        model, _ = _train_model(model_table, cutoff_ts)
        order = [list(model.classes_).index(label) for label in LABELS]

        finals = _wc_finals_slice(clean, cutoff, days)
        feat = model_table.merge(finals[["match_id"]], on="match_id", how="inner")
        feat = prepare(feat, FULL_FEATURES, carry=("home_team", "away_team", "neutral"))
        if feat.empty:
            continue

        dc = DixonColesModel().fit(clean, cutoff_ts)
        ys.append(feat[TARGET].to_numpy())
        p_bb.append(model.predict_proba(feat[FULL_FEATURES])[:, order])
        p_dc.append(dc.predict_proba(feat[["home_team", "away_team", "neutral"]]))

    return (
        pd.Series(np.concatenate(ys)),
        np.concatenate(p_bb),
        np.concatenate(p_dc),
    )


def _blend(p_bb, p_dc, w_dc):
    blended = (1.0 - w_dc) * p_bb + w_dc * p_dc
    return blended / blended.sum(axis=1, keepdims=True)


def tune_ensemble_weight(
    weights=ENSEMBLE_W_GRID, world_cups=WORLD_CUPS, days: int = 45, verbose: bool = True
) -> dict:
    """Pick the Elo-vs-Dixon-Coles blend weight by pooled WC-finals match log loss.

    Leakage-safe: every model is fit before the World Cup it is scored on. ``w_dc``
    is the weight on the goal model; 0 is the pure Elo backbone, 1 is pure
    Dixon-Coles. Lower log loss is better. Returns ``{w_dc: match_ll}``.
    """
    from sports_predictor.core.evaluation import evaluate_probabilities

    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    model_table = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
    y, p_bb, p_dc = _wc_match_predictions(matches, model_table, world_cups, days)

    if verbose:
        print(
            "Ensemble weight tuning  (pooled WC-finals match log loss, "
            f"{len(y)} matches; lower = better)\n"
            f"{'w_dc':>7}{'match LL':>11}"
        )
    scores: dict[float, float] = {}
    for w in weights:
        ll = evaluate_probabilities(y, _blend(p_bb, p_dc, w), LABELS)["log_loss"]
        scores[w] = ll
        if verbose:
            print(f"{w:>7.2f}{ll:>11.4f}")

    best = min(scores, key=scores.get)
    if verbose:
        print(
            f"\nbest w_dc = {best:.2f}  (match log loss {scores[best]:.4f}; "
            f"backbone-only {scores[weights[0]]:.4f})"
        )
    return scores


def run_match_bakeoff(world_cups=WORLD_CUPS, days: int = 45, w_dc: float | None = None) -> dict:
    """Honest match-level bake-off on past WC finals: backbone, XGBoost, DC, ensemble.

    All models are fit strictly before each tournament and scored on its actual
    finals matches, pooled. Reports log loss + accuracy so the goal model and the
    ensemble are compared on equal, leakage-safe footing.
    """
    from sports_predictor.core.evaluation import evaluate_probabilities
    from sports_predictor.soccer.baseline import FULL_FEATURES, TARGET, prepare
    from sports_predictor.soccer.models import MatchClassifier

    if w_dc is None:
        w_dc = DEFAULT_ENSEMBLE_W_DC
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    model_table = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
    clean = matches.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)

    y, p_bb, p_dc = _wc_match_predictions(matches, model_table, world_cups, days)

    # XGBoost, fit per-WC on the same features for a fair pooled comparison.
    p_xgb = []
    for tour, cutoff, _champ in world_cups:
        cutoff_ts = _utc(cutoff)
        train = prepare(model_table[model_table["date"] < cutoff_ts], FULL_FEATURES)
        clf = MatchClassifier("xgboost").fit(train[FULL_FEATURES], train[TARGET])
        finals = _wc_finals_slice(clean, cutoff, days)
        feat = model_table.merge(finals[["match_id"]], on="match_id", how="inner")
        feat = prepare(feat, FULL_FEATURES, carry=("home_team", "away_team", "neutral"))
        if not feat.empty:
            p_xgb.append(clf.predict_proba(feat[FULL_FEATURES]))
    p_xgb = np.concatenate(p_xgb)

    preds = {
        "backbone (Elo+form)": p_bb,
        "xgboost": p_xgb,
        "dixon_coles": p_dc,
        f"ensemble (w_dc={w_dc:g})": _blend(p_bb, p_dc, w_dc),
    }
    print(
        f"Match-level bake-off on {len(y)} pooled WC-finals matches "
        f"(leakage-safe, fit before each WC)\n"
        f"{'model':<24}{'log loss':>10}{'accuracy':>10}"
    )
    results = {}
    for name, proba in preds.items():
        m = evaluate_probabilities(y, proba, LABELS)
        results[name] = m
        print(f"{name:<24}{m['log_loss']:>10.4f}{m['accuracy']:>10.1%}")
    return results


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
    import sys

    args = sys.argv[1:]
    if args and args[0] == "forward":
        # `forward` -> backbone; `forward dc` -> goal model; `forward ens` -> ensemble.
        kinds = {"dc": "dixon_coles", "dixon_coles": "dixon_coles", "ens": "ensemble", "ensemble": "ensemble"}
        model_kind = kinds.get(args[1], "backbone") if len(args) > 1 else "backbone"
        run_forward(model_kind=model_kind)
    elif args and args[0] in ("tune-xi", "tune_xi"):
        tune_xi()
    elif args and args[0] in ("tune-sigma", "tune_sigma"):
        tune_strength_sigma()
    elif args and args[0] in ("tune-weight", "tune_weight", "tune-w"):
        tune_ensemble_weight()
    elif args and args[0] in ("bakeoff", "bake-off"):
        run_match_bakeoff()
    else:
        run_backtest()


if __name__ == "__main__":
    _main()
