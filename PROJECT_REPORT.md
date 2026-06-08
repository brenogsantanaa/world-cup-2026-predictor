# World Cup Prediction Engine — Full Report

_Last updated: 2026-06-08_

A from-scratch, end-to-end machine learning pipeline that predicts **international
football match outcomes** (Home win / Draw / Away win) and is being built toward
**FIFA World Cup tournament simulation**.

When this work started, the repo was an empty scaffold framed around *NBA*. Based
on `DATA_SOURCES.md` and the project direction, the whole project was **pivoted to
soccer / the World Cup**, and the first real, working, tested vertical slice was
built — from raw data all the way to a calibrated, backtested model.

Guiding principle throughout (from `DATA_SOURCES.md`): **trust before volume** —
every number traceable to a source, and **no data leakage** (a feature for a match
may only use information that existed before kickoff).

---

## 1. Architecture

```
src/sports_predictor/
├── core/                  # sport-agnostic, reusable
│   ├── paths.py           # central data/model paths
│   ├── splitting.py       # leakage-safe chronological splits
│   └── evaluation.py      # log loss / accuracy / no-skill baseline
├── soccer/                # soccer-specific
│   ├── teams.py           # canonical national-team name mapping
│   ├── results.py         # ingestion: fetch -> raw cache -> clean table
│   ├── elo.py             # our own Elo rating engine
│   ├── features.py        # per-match modeling table (Elo+form+rest+H2H)
│   └── baseline.py        # 3-way logistic regression + backtest
└── nba/                   # legacy scaffold, now optional
tests/                     # 6 test files, 34 tests
```

~1,030 lines of production code + ~400 lines of tests. Every module has a single
responsibility, and stable logic lives in `src/` (not notebooks).

---

## 2. Project hygiene fixed

- Restored the deleted `pyproject.toml`; made it soccer-first (NBA moved to an
  optional `[nba]` extra), added `pyarrow`.
- Rewrote `README.md` around the World Cup goal, the ingestion contract, the
  feature list, and the live results table.
- Diagnosed that the `.venv` had a stale path (created at an old location) and
  worked around it via `python -m pip`.

---

## 3. Data pipeline & provenance

**Source:** "International football results from 1872 to present" (martj42),
**CC0 license**, Tier 2 in `DATA_SOURCES.md`. No API key. Every men's full
international with score, venue, and neutral-ground flag.

**Contract enforced:**

- Raw bytes saved **verbatim** before any transform, named with an as-of date:
  `data/raw/international_results/results_2026-06-08.csv` (3.7 MB).
- A **provenance manifest** is written alongside it (URL, UTC timestamp, byte
  count, SHA-256, license).
- Cleaning is fully separate and reproducible from the raw cache.

**Cleaned output:** `data/processed/matches.parquet` (1.5 MB)

- **49,372 matches**, 1872-11-30 -> 2026-06-07
- **9,735 World Cup matches**
- Dropped **72** rows with no score (unplayed fixtures) and **1** genuine upstream
  duplicate (a Gibraltar–Cayman match logged twice with two spellings of the same
  venue — caught automatically by the sanity check).

Canonical schema:
`match_id, date (UTC), home_team, away_team, home_score, away_score, neutral,
tournament, is_friendly, is_world_cup, city, country, result`.
Scores / `result` are clearly labeled **targets only**, never inputs.

---

## 4. Features (all pre-kickoff, leakage-safe)

`data/processed/model_table.parquet` (4.4 MB) — 49,372 rows × 26 feature columns:

- **Elo** (`elo.py`): our own World-Football-Elo computation — neutral-aware home
  advantage, K-factor by tournament importance (friendly 20 -> WC final 60),
  margin-of-victory multiplier. The rating attached to a match is the one *before*
  kickoff. Produces `home_elo_pre, away_elo_pre, elo_diff, elo_expected_home`.
- **Recent form**: last-5 and last-10 win rate, points avg, goals for/against avg,
  for both teams.
- **Rest days** since each team's previous match.
- **Head-to-head**: prior meetings count + home team's win rate in prior meetings.

The leakage guarantee is structural: every rolling/expanding window is preceded by
`shift(1)` within each team's own history, so a window can only ever see earlier
matches.

---

## 5. Validation & results

### Elo cross-check (sanity vs reality)

Top current ratings came out **Spain -> Argentina -> France -> England -> Brazil
-> Colombia -> Portugal** — matching eloratings.net and the `post-assets` chart.
Predictiveness: home favorites by >50 Elo win **69.9%** (n≈21k); underdogs win
**26.5%** (n≈19k).

### Feature health (matches since 2024)

NaN rates ≈ 0% for Elo/form/rest; `h2h_home_win_rate` is NaN ~10.6% — correctly,
those are first-ever meetings (honest "unknown", not a fabricated value).

### Baseline backtest

Honest chronological split — train ≤ 2016-03-25 (39,256 matches), test 2016 -> 2026
(9,814 matches). Train outcome mix: H 49% / D 23% / A 28%.

| model | log loss | accuracy |
|---|---|---|
| no-skill (predict class rates) | 1.0525 | — |
| Elo only | 0.8834 | 59.8% |
| **full features** | **0.8712** | 59.8% |

Log loss is the primary metric (probability quality matters more than win/lose for
simulation). The full model beats no-skill by **~17%**.

### Calibration (the strongest quality signal)

Predicted home-win probability ≈ actual, across every bucket:

```
predicted -> actual   (n)
   5%    ->   5%   ( 555)
  15%    ->  11%   ( 770)
  25%    ->  22%   (1006)
  35%    ->  33%   (1204)
  45%    ->  41%   (1345)
  55%    ->  52%   (1361)
  65%    ->  63%   (1191)
  75%    ->  74%   (1032)
  85%    ->  83%   ( 881)
  94%    ->  93%   ( 469)
```

Near-perfect calibration is exactly what the Monte Carlo simulator will need, and
it is strong evidence of **no leakage** (a leaky model is overconfident and
miscalibrated). The fitted model is saved to `models/baseline_logreg.joblib`.

---

## 6. Tests — all 34 passing (~1.5s)

**`test_core_splitting.py`** (4) — chronological split puts latest rows in test;
doesn't mutate input; covers all rows; rejects bad fractions; cutoff split is
strict.

**`test_core_evaluation.py`** (4) — class base rates; perfect predictions -> 0 log
loss / 100% accuracy; no-skill baseline uses training rates; sanity of model vs
baseline.

**`test_soccer_teams.py`** (4) — known aliases map to canonical (USA -> United
States, etc.); case/whitespace-insensitive; unknown names preserved (not dropped);
`find_unmapped_teams` reports only unknowns.

**`test_soccer_results.py`** (8) — chronological sort; correct H/D/A result label;
flag/dtype correctness; team-name canonicalization; unique & stable `match_id`;
rows without scores dropped; exact duplicate matches dropped; sanity check rejects
a team listed against itself.

**`test_soccer_elo.py`** (8) — expected score symmetric & fair; goal-diff
multiplier grows with margin; K-factor tiers; first match uses base rating; update
is zero-sum & winner gains; **pre-match rating reflects only earlier matches
(leakage test)**; repeated wins increase rating monotonically; input not mutated.

**`test_soccer_features.py`** (6) — rolling form uses only prior matches;
first-ever match has no history; **changing the current result doesn't change the
current match's features (leakage test)**; rest-days counts the gap; head-to-head
excludes the current match; modeling table preserves rows & targets.

---

## 7. How to run the whole pipeline

```bash
python -m sports_predictor.soccer.results    # fetch + clean -> matches.parquet
python -m sports_predictor.soccer.features   # build -> model_table.parquet
python -m sports_predictor.soccer.baseline   # backtest + calibration + save model
pytest                                        # 34 tests
```

---

## 8. Honest caveats

- **Elo dominates**: the full feature set only edges out Elo-only (0.8834 ->
  0.8712). Expected — Elo is derived from the same match history, so it already
  absorbs most of the signal. XGBoost (interactions) is the natural next lever.
- The test set mixes many lopsided minnow-vs-giant qualifiers, which are easy to
  call and flatter the headline numbers — not wrong, just context.
- No player-level data yet (deliberately). Full lineups / player ratings aren't
  available from a clean, licensed source; `goalscorers.csv` (CC0) is the honest
  next addition.

---

## 9. Status & next steps

**Done:** data contract -> ingestion -> canonical names -> Elo -> form/rest/H2H ->
leakage-safe split -> calibrated baseline, all tested.

**Next:**

1. **Phase B** — ingest `goalscorers.csv` for pre-match player/attack signal
   (top-scorer form, goals-per-game trends, penalty tendencies).
2. Stronger model (XGBoost) + probability-calibration tuning.
3. Monte Carlo tournament simulation (group advancement, bracket, champion odds).

> Note: the data/model artifacts and `pyproject.toml` exist locally but nothing
> has been committed to git yet (the repo currently has zero commits).
