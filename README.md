# World Cup Prediction Engine

An end-to-end machine learning project that predicts international football match
outcomes and uses those probabilities to simulate the **FIFA World Cup**.

The guiding principle lives in [`DATA_SOURCES.md`](DATA_SOURCES.md): **trust before
volume.** Every data point must be traceable to a known source, with a known
license and a known "as-of" date, and no feature may use information that did not
exist before kickoff.

> The repository keeps a small NBA scaffold from when the pipeline was first
> prototyped. New work targets soccer / the World Cup; the NBA module is optional.

## What we are predicting

1. **Match level:** the probability that a team wins / draws / loses a single
   international match.
2. **Tournament level:** feed those match probabilities into a Monte Carlo
   simulation to estimate group advancement and World Cup champion odds.

## Project structure

- `data/raw/`: original downloaded data, cached verbatim with an as-of date.
- `data/processed/`: cleaned tables built from raw data.
- `notebooks/`: guided exploration.
- `src/sports_predictor/core/`: sport-agnostic ML utilities (splitting, training,
  evaluation, simulation).
- `src/sports_predictor/soccer/`: soccer data ingestion, team-name mapping, and
  feature engineering.
- `src/sports_predictor/nba/`: legacy NBA scaffolding (optional).
- `models/`: saved trained models.
- `tests/`: automated checks for the reusable code.
- `app/`: future app.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Data pipeline

The ingestion contract (see `DATA_SOURCES.md` §4) is always:

```text
external source
  -> fetch (throttled, error-handled)
  -> save RAW response to data/raw/<source>/<name>_<asof_date>.csv   (never edited)
  -> clean / normalize into a standard per-match schema
  -> save to data/processed/
  -> build the per-match modeling table
```

The first integrated source is the openly licensed
[international football results](https://github.com/martj42/international_results)
dataset (Tier 2 in `DATA_SOURCES.md`): every men's international match from 1872
to today, no API key required.

Fetch and clean it:

```bash
python -m sports_predictor.soccer.results
```

This writes a raw snapshot to `data/raw/international_results/` and a cleaned,
canonical table to `data/processed/matches.parquet`.

Then build the per-match modeling table (Elo + form + rest + head-to-head):

```bash
python -m sports_predictor.soccer.features
```

This reads `data/processed/matches.parquet` and writes
`data/processed/model_table.parquet`, one row per match with **pre-kickoff**
features only.

## Features (all leakage-safe, pre-kickoff)

- **Elo** (`soccer/elo.py`): our own World-Football-Elo computation, neutral-aware
  home advantage, K-factor by tournament importance. The rating attached to a
  match is the one *before* it was played. Current ratings cross-check cleanly
  against eloratings.net (Spain, Argentina, France, England, Brazil at the top).
- **Recent form** (`soccer/features.py`): last-5 / last-10 win rate, points avg,
  goals for/against avg, per team.
- **Rest days** since each team's previous match.
- **Head-to-head** record between the two teams before this match.

## Status

- [x] Project scaffold and data contract (`DATA_SOURCES.md`)
- [x] Canonical team-name mapping
- [x] International results ingestion (fetch -> raw cache -> cleaned table)
- [x] Chronological train/test split (leakage-safe)
- [x] Elo as-of-match-date features (own computation, validated vs eloratings.net)
- [x] Recent-form / rest-days / head-to-head features
- [x] Baseline 3-way match model (logistic regression) + calibration check
- [x] Knockout-draw conversion (proportional / even split)
- [x] Monte Carlo tournament simulator + 2022 World Cup backtest
- [x] XGBoost evaluated — ties logistic, so logistic is retained (the model is
      not the bottleneck; Elo already captures the signal). See `soccer/models.py`.
- [ ] Tournament-level calibration (shrinkage) for top-team overconfidence
- [ ] Goalscorer-derived player signal (`goalscorers.csv`)
- [ ] FIFA-ranking as-of-match-date features

### Baseline backtest (train ≤2016, test 2016→2026, ~9.8k matches)

| model | log loss | accuracy |
|---|---|---|
| no-skill (class rates) | 1.0525 | — |
| Elo only | 0.8834 | 59.8% |
| full features | **0.8712** | 59.8% |

The full model beats the no-skill baseline by ~17% on log loss and is **well
calibrated** — across home-win probability buckets, predicted ≈ actual (e.g. the
"55%" bucket wins 52%, the "85%" bucket wins 83%). Good calibration is exactly
what the tournament simulator will need, and it's strong evidence there's no
leakage. Run it with `python -m sports_predictor.soccer.baseline`.

## Tournament simulation

`python -m sports_predictor.soccer.simulation` runs a Monte Carlo simulation
(`soccer/simulation.py`). It freezes each team's strength as of a cutoff date,
precomputes a neutral-venue 3-way probability for every pairing (predicted both
ways and averaged), then plays the group stage and knockout bracket thousands of
times. Knockout ties use the draw-conversion rule from `soccer/knockout.py`.

**2022 World Cup backtest** (train on the 45,400 matches before kickoff, 20k sims,
proportional rule) — top champion odds:

| team | champion | reach final | reach semi |
|---|---|---|---|
| Brazil | 32% | 43% | 63% |
| Argentina | 28% | 39% | 62% |
| France | 6% | 17% | 33% |
| Spain | 6% | 13% | 25% |
| Portugal | 5% | 15% | 26% |

The top favorites match the pre-tournament market consensus, and the eventual
winner (Argentina) and runner-up (France) both land in the top three. The
knockout rule is material: the **proportional** split lifts the strongest teams
(Brazil +7.5pp, Argentina +6.1pp champion odds vs the **even** 50/50 split),
since favoring the favorite compounds across rounds. Known limitation: the very
top teams are somewhat overconfident (Elo-driven), which tournament-level
calibration/shrinkage should address.
