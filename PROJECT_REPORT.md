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

---

## 10. Update — model bake-off & tournament-level calibration

### 10.1 Logistic vs XGBoost (it's a tie)

Same chronological backtest, log loss by slice (lower is better):

| model | all | neutral | WC finals | acc |
|---|---|---|---|---|
| no-skill | 1.0525 | 1.0962 | 1.1068 | — |
| logistic | **0.8712** | **0.9140** | 0.8530 | 59.8% |
| xgboost | 0.8739 | 0.9156 | **0.8527** | 59.9% |

Differences are noise-level and both models are well calibrated. Elo already
encodes the signal in a near-linear way, so there are few interactions for trees
to exploit. **Conclusion: keep logistic** (simpler, equally accurate/calibrated,
interpretable). The model is not the bottleneck. `MatchClassifier`
(`soccer/models.py`) keeps both behind one interface; run the bake-off with
`python -m sports_predictor.soccer.models`.

### 10.2 Why champion odds looked overconfident — and the real fix

The 2022 backtest put Brazil + Argentina ≈ 60% combined (market ≈ 35%). The
instinct is to "calibrate" the match model, but the diagnosis says otherwise:

| slice | optimal temperature | log loss change |
|---|---|---|
| all | 1.01 | none |
| neutral | 1.00 | none |
| WC finals | 0.93 | 0.8530 → 0.8521 |

Optimal temperature ≈ 1 everywhere, so **per-match probabilities are already
calibrated** — naive temperature scaling would do nothing. The tournament
overconfidence is *structural*: (a) the proportional knockout rule favours the
favourite by design, and (b) the sim treats each Elo rating as exact, so a
favourite's 7-game path looks more certain than it is. Markets price in the
chance a team is simply *overrated* (correlated across all its matches).

**Fix — strength-uncertainty perturbation** (`soccer/simulation.py`): each
simulation, every team gets one strength offset `~ N(0, sigma)` (logit units)
applied to *all* its matches, injecting that correlated uncertainty. Temperature
tooling lives in `core/calibration.py`.

### 10.3 Calibrating sigma over 2010–2022 (and an honest result)

`tune_strength_sigma` scores each sigma by champion log loss `-log P(actual
winner)` across the four 32-team World Cups:

| sigma | mean champion LL |
|---|---|
| 0.00 | 1.962 |
| **0.20** | **1.950** |
| 0.40 | 1.956 |
| 0.60 | 2.024 |
| 0.80 | 2.102 |
| 1.00 | 2.160 |

The curve is nearly flat and marginally prefers a **small** sigma. Effect on the
2022 top-2 combined: σ=0 → 60.0%, σ=0.2 → 59.3%, σ=0.5 → 53.3%, σ=0.8 → 45.7%.
So matching the market would need σ≈0.8, but **the outcome data rejects that** —
favourites usually do win (3 of these 4 champions were top-rated), so heavy
softening hurts. We therefore set a **mild humility prior `DEFAULT_STRENGTH_SIGMA
= 0.2`** rather than chasing bookmaker spreads.

Caveat: only four tournaments exist, so this is a low-power estimate — a sensible
default, not a precise optimum. More history (and the 2026 bracket) will sharpen
it.

### 10.4 Calibrating the Dixon-Coles time-decay xi

The goal model weights each historical match by `exp(-xi * age_in_days)`, so `xi`
sets how fast the past is forgotten. The original `0.0005` (~3.8-year half-life)
was a reasonable guess; `simulation.tune_xi` replaces it with a backtested value.
For each candidate `xi` it refits the model strictly before each 2010–2022 World
Cup and scores it two leakage-safe ways on World Cup matches only:

| xi | half-life | match LL (256 finals) | champion LL (4 WCs) |
|---|---|---|---|
| 0.0000 | inf | 1.0242 | 2.723 |
| 0.0002 | 9.5y | 1.0189 | 2.473 |
| 0.0005 | 3.8y | 1.0041 | **2.346** |
| **0.0008** | **2.4y** | **1.0024** | 2.473 |
| 0.0012 | 1.6y | 1.0032 | 2.470 |
| 0.0018 | 1.1y | 1.0269 | 2.444 |
| 0.0026 | 0.7y | 1.0270 | 2.441 |

Match log loss — pooled over every actual finals match, so the higher-power
signal — bottoms out in a **broad, shallow basin from ~0.0005 to ~0.0012** with
its minimum at **0.0008**, well below both the no-decay (1.0242) and fast-decay
(1.0270) extremes. Champion log loss, with only four data points, is far noisier
and marginally prefers 0.0005; the gap is within sampling noise. We select on the
match metric and set **`DEFAULT_XI = 0.0008`**.

Honest read: this is a small, well-supported nudge, not a dramatic gain — the main
finding is that the decay band is right and the extremes are wrong. Caveat as
above: four tournaments is low power.

### 10.5 Test count

Suite is now **60 passing** (added `test_soccer_models.py`,
`test_core_calibration.py`, and strength-perturbation tests in
`test_soccer_simulation.py`). Work is committed in logical units.

---

## 11. Update — player-profile feature layer (Phase B)

Built an **additive** player signal from `goalscorers.csv` (same martj42 repo /
CC0 license as results), following the existing conventions: raw cache +
provenance manifest, cleaning separate from raw, leakage-safe features, tests.

### 11.1 Ingestion (`soccer/goalscorers.py`)

Fetch → verbatim raw snapshot (`goalscorers_<asof>.csv` + SHA-256 manifest) →
`clean` → `goalscorers.parquet`. 47,601 goals (1916–2026), 15,334 distinct
scorers, 922 own goals, 3,249 penalties. Team names canonicalized via the shared
`teams.py` map; own goals are kept in the raw table but excluded from player
credit downstream.

### 11.2 What is (and isn't) derivable

The source records **goals, not appearances** (no lineups), so a true
"goals per appearance" can't be computed without inventing a denominator — we
deliberately don't. What the goal stream *does* give, per team and as-of each
match (`soccer/player_features.py`):

| feature | meaning |
|---|---|
| `recent_goals` | goals in the team's last 20 matches |
| `top_scorer_goals` | most by any single player in that window (star presence) |
| `goal_concentration` | HHI of goals across scorers (1 = one-man team) |
| `penalty_share` | fraction of those goals from penalties |
| `squad_experience` | summed career intl goals (as-of date) of the window's scorers |

**Leakage safety:** one global chronological pass; a match's features are read
*before* its own goals are folded into the windows / career totals.

**Graceful degradation:** no goals in window → NaN + `low_data = 1` (never a
fabricated zero); some but `< 3` → values emitted but still flagged. Callers
impute NaNs from *training* statistics and keep the flag, so low-data teams still
work off the team backbone alone.

### 11.3 Coverage (honest)

Share of matches with an empty/unreliable home profile:

| scope | low-data rate |
|---|---|
| all matches | 31.3% |
| **World Cup matches** | **13.0%** |

By home confederation: CONMEBOL 6.5%, UEFA 15.5%, CAF 20.5%, CONCACAF 21.0%,
AFC 27.1%, OFC 35.7%, Unknown 50.7% — coverage is good exactly where we care
(major nations, World Cups) and thin for obscure fixtures.

### 11.4 Does it help? No — and that's the finding

Backbone (Elo/form/rest/H2H) vs backbone + player, log loss by slice:

| model | all | neutral | WC finals |
|---|---|---|---|
| no-skill | 1.0525 | 1.0962 | 1.1068 |
| backbone | 0.8712 | 0.9140 | **0.8530** |
| backbone + player | 0.8710 | 0.9137 | 0.8566 |

Delta on the slices that matter: neutral −0.0004 (negligible), WC finals **+0.0036
(worse)**. The player profiles overlap with team form/Elo (a team that scores a
lot already shows strong form), so on the small WC slice they add noise.

**Verdict (per the spec): leave the player layer OFF by default.** The pipeline,
parquet, coverage report and tests remain so it can be revisited (e.g. with real
lineups/appearances, or opponent-adjusted finishing) — but we don't ship a
feature that doesn't earn its place.

### 11.5 Tests & count

Added `test_soccer_player_features.py` (leakage / future goals don't leak; first
appearance is empty not zero; aggregation preserves rows & ids; never-scored team
→ NaN not zero; own goals not credited). Full suite: **66 passing**.

---

## 12. Update — cross-confederation Elo bias check

**Hypothesis:** our Elo is a near-closed system per confederation (teams mostly
play within their own pool), so a confederation could hoard rating and look strong
on paper. We suspected CONMEBOL was overrated, inflating Brazil/Argentina.

**Method (`soccer/confederation_bias.py`):** on inter-confederation matches only,
compare each side's actual score (W/D/L = 1/0.5/0) to its neutral-aware,
pre-match Elo-expected score. Mean residual `actual − expected` per confederation
is the bias; we also fit additive per-confederation Elo offsets (UEFA = reference)
that would zero the residuals.

**Result (2002+), residual `actual − expected`:**

| confederation | mean residual | verdict |
|---|---|---|
| OFC | −0.129 | overrated |
| AFC | −0.052 | overrated |
| CONCACAF | −0.049 | overrated |
| UEFA | +0.017 | well-calibrated |
| CONMEBOL | +0.048 | (slightly) underrated |
| CAF | +0.054 | underrated |

Implied Elo offset vs UEFA: CONMEBOL +16, CAF +12, CONCACAF −55, AFC −61,
OFC −155. CONMEBOL vs UEFA *at World Cups* (73 matches): actual 0.507 vs expected
0.537 (−0.030, within noise).

**The hypothesis is rejected.** CONMEBOL is *not* overrated (fair-to-slightly
underrated; roughly even vs UEFA at World Cups) and UEFA is well-calibrated — so
Brazil/Argentina's high ratings are essentially **earned**, not a confederation
artifact. The genuinely overrated pools are the *weaker* ones (OFC/AFC/CONCACAF),
which farm rating among themselves. Correcting that would, if anything, make
favourites slightly *stronger*, not weaker — so it does not explain the
simulator's favourite-heaviness (that was structural, fixed in §10).

**Should we apply a correction?** Out-of-sample test (fit offsets pre-2018, test
2018+ inter-confed matches): Brier all 0.1451 → 0.1440 (−0.0010, negligible);
Brier **World Cup 0.1863 → 0.1901 (+0.0038, worse)**. The bias is real in-sample
but does **not generalise** to better World Cup predictions, so we **do not** wire
a confederation adjustment into the model. Kept as a diagnostic tool.

Adds `test_soccer_confederation_bias.py`. Full suite: **69 passing**.

> Net theme across §10–12: XGBoost, the player layer, and a confederation
> correction all failed to beat the Elo+form backbone on the slices that matter.
> The backbone is strong; the remaining frontier is genuinely new information or
> the live 2026 product, not more modelling on the same data.

---

## 13. Update — FIFA ranking as-of features (first slice win)

The one signal **not** derived from our own match results. Ingested the
historical official ranking time series (`soccer/fifa_ranking.py`): 57,754 rows,
286 publication dates, 216 teams, **1993-08 → 2018-06**, with raw cache +
provenance manifest (upstream "Sudan" double-entries deduped, 39 rows).

Features (`soccer/fifa_features.py`) attach, via leakage-safe `merge_asof`
(backward, same-day excluded), each team's latest ranking **published strictly
before kickoff**: `home/away_fifa_rank`, `…_points`, `fifa_rank_diff`,
`fifa_points_diff`, plus `low_data` flags. Missing → NaN + flag (callers impute
from training stats), never fabricated.

**Coverage** (no home ranking available): all 43.5%, World Cup 29.5% (the 2022 WC
is entirely uncovered — source ends 2018). By confederation 31–51%.

**Result — the first feature to help the slice that matters:**

| model | all | neutral | WC finals |
|---|---|---|---|
| backbone | 0.8712 | 0.9140 | 0.8530 |
| backbone + FIFA | 0.8702 | 0.9142 | **0.8479** |

Δ neutral +0.0001 (noise), **Δ WC finals −0.0050** — the biggest single-feature
gain on the World Cup slice so far, despite 29.5% of WC matches having no ranking.

**Decision:** left **OFF by default** — not because it lacks signal, but because
the source ends mid-2018, so for the live 2026 product there would be *no*
ranking at all. The WC-slice gain says it's worth enabling the moment a current
FIFA-ranking source is wired in. Adds `test_soccer_fifa.py`. Suite: **74 passing**.

---

## 14. Update — broad multi-source data build (canonical layer + scrapers)

Built incrementally (one green commit per slice) into a foundation for unifying
CC0 data with scraped club/squad data.

**Canonical reconciliation layer** (`canonical/`) — the core. Accent/
transliteration-safe match keys (`names.py`: Mbappé→mbappe, N'Golo→ngolo,
"Last, First" swaps); deterministic, reproducible `player_id`/`team_id` with alias
merging and *safe* non-merge of abbreviated names (`registry.py`); documented
per-field source priority (FBref→match stats, Understat→xG, Transfermarkt→value)
with disagreement **logging** rather than silent overwrites (`conflicts.py`).

**Scrapers** (`scrapers/`) — one isolated parser per site, fetching fully separated
from parsing. `base.py` is cache-first (re-runs never re-hit a site), rate-limited,
retried with backoff, sends a real User-Agent, and writes a provenance manifest
(URL, UTC ts, bytes, SHA-256). Three sites, each end-to-end (cache→parse→canonical
→one feature): **Understat** (embedded JSON incl. `\xNN`/accents), **Transfermarkt**
(nested squad tables via BeautifulSoup, market-value m/k/bn parsing), **FBref**
(tables hidden in HTML comments, cells by `data-stat`). A test proves the *same*
player resolves to one `player_id` across FBref and Transfermarkt.

**Squad assembly** (`soccer/squads.py`) — the linchpin: `player_id → nation` from
national-team squad pages, so club stats attach to countries. Players outside every
squad get NaN + `in_squad=0`; squad value sums only *known* values with coverage +
`low_data` flags.

**Team-match aggregation + backtest harness** (`soccer/squad_features.py`) —
leakage-safe per-nation aggregates (squad value, top xG/90, mean-top-N, share in
form) attached to matches as home/away/diff features with `low_data`. The harness
deliberately **refuses to fabricate a backtest verdict from a current snapshot**
(anachronistic for past matches); it runs only against a real *as-of* feature
table and otherwise prints the data prerequisite.

**Live-fetch reality:** an actual Understat pull from the build environment returned
a stripped page without inline data, and Transfermarkt/FBref are Cloudflare-
protected. The cache-first, fixture-tested design means parsing/tests/architecture
are unaffected and ready to run the moment real pages are cached. All parsers are
verified against committed sample pages. Suite: **125 passing**.
