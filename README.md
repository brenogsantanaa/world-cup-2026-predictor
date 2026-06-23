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
- `src/sports_predictor/canonical/`: cross-source reconciliation — accent-safe
  name keys, stable `player_id`/`team_id`, per-field conflict resolution.
- `src/sports_predictor/scrapers/`: one isolated, cache-first parser per site
  (Understat, Transfermarkt, FBref); fetching is separate from parsing.
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

### Optional: API-Football (live 2026 club data)

For the **forward-looking 2026 prediction** only, club form can be enriched via
[API-Football](https://www.api-sports.io/). Set your key in the environment
(never hardcoded):

```bash
export API_FOOTBALL_KEY="your-key-here"
```

The client (`soccer/apifootball.py`) is **cache-first** and **quota-aware**: the
free tier allows ~100 requests/day, so it throttles, backs off on 429s, stops
cleanly when the daily limit is hit, and is **resumable** — re-running skips
anything already cached (cache hits don't consume quota), so a full squad pull can
span several days. All raw JSON is saved verbatim to `data/raw/apifootball/` with a
provenance manifest. Without the key the client refuses live calls but still parses
any cached responses. This data is **never** fed into the historical backtest
(that would be leakage) — only the 2026 forward simulation.

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
- [x] Tournament-level calibration. Per-match probabilities are already calibrated
      (optimal temperature ≈ 1 on every slice), so the tournament overconfidence is
      *structural*. Added per-simulation **team strength-uncertainty perturbation**
      (`strength_sigma`) and calibrated it over the 2010–2022 World Cups by champion
      log loss. See `core/calibration.py`, `soccer/simulation.py`, PROJECT_REPORT.md.
- [x] Goalscorer-derived player signal (`goalscorers.csv`) — ingested + a
      leakage-safe player-profile layer (`soccer/player_features.py`). Honest
      backtest verdict: it does **not** improve the neutral/WC slices (overlaps
      with team form/Elo), so it's **off by default**. See PROJECT_REPORT.md §11.
- [x] Cross-confederation Elo bias check (`soccer/confederation_bias.py`) —
      CONMEBOL is **not** overrated (UEFA well-calibrated); the weaker pools
      (OFC/AFC/CONCACAF) are, but correcting it doesn't generalise to World Cup
      predictions, so it's left out. See PROJECT_REPORT.md §12.
- [x] FIFA-ranking as-of-match-date features (`soccer/fifa_ranking.py`,
      `soccer/fifa_features.py`) — leakage-safe `merge_asof`. **Improves the
      World Cup slice** (−0.0050 log loss), flat on neutral. Off by default only
      because the source ends mid-2018 (no live 2026 coverage). See §13.
- [x] Broad-data foundation: **canonical reconciliation layer**
      (`canonical/`) + cache-first scrapers (`scrapers/`) for **Understat**,
      **Transfermarkt**, and **FBref**, each built incrementally end-to-end
      (cache → parse → canonical id → one feature) and tested against committed
      sample pages. The same player resolves to one `player_id` across sources.
- [x] National-team squad assembly (`soccer/squads.py`) — the linchpin mapping
      `player_id → nation`, so club stats become national-team features. Players
      not in any squad degrade to NaN + `in_squad=0`.
- [x] Team-match aggregation of squad profiles + slice-backtest harness
      (`soccer/squad_features.py`) — leakage-safe aggregation (squad value, top
      xG/90, share in form) with coverage/`low_data` flags. The backtest is wired
      and tested but only *runs* on a real as-of feature table (current snapshots
      would be anachronistic for past matches), so it reports the data
      prerequisite rather than fabricating a verdict.

#### Multi-source scraping architecture

Each site has an isolated parser so one site's HTML change can't break the
others. Fetching (`scrapers/base.py`) is cache-first, rate-limited, retried with
backoff, sends a real User-Agent, and writes a provenance manifest (URL, UTC ts,
bytes, SHA-256); parsing reads only from cache, so runs are reproducible. Cross-
source facts are unified by `canonical/` (accent/transliteration-safe name keys,
deterministic ids, documented per-field source priority with disagreement
logging). Missing data degrades to NaN + a `low_data` flag — never a fabricated
zero. ToS note: Transfermarkt and FBref restrict automated access; fetch politely
and cache aggressively.

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

## 2026 World Cup prediction (first real, forward-looking run)

The real thing: the **48-team 2026 World Cup** (12 groups, top two + eight best
third-placed teams → Round of 32 → R16 → QF → SF → Final), with the actual final
draw (as-of 2026-04-23, play-offs resolved) and FIFA's published bracket wiring
encoded in `soccer/tournaments.py`. Run it:

```bash
python -m sports_predictor.soccer.simulation forward
```

This trains on every international before the 11 Jun 2026 cutoff and simulates the
tournament 20,000 times. Three engines are selectable (each writes its own
labelled `data/processed/wc2026_odds*.parquet` + CSV):

```bash
python -m sports_predictor.soccer.simulation forward        # backbone (Elo+form)
python -m sports_predictor.soccer.simulation forward dc      # Dixon-Coles goal model
python -m sports_predictor.soccer.simulation forward ens     # ensemble (best; default w_dc=0.35)
```

**Headline prediction — the ensemble** (Elo backbone + Dixon-Coles, `w_dc=0.35`),
the most accurate engine on the leakage-safe WC-finals bake-off (see below):

| team | champion | final | semi | R16 |
|---|---|---|---|---|
| Argentina | 27.6% | 38.3% | 51.0% | 75.0% |
| Spain | 17.1% | 29.6% | 42.9% | 73.7% |
| Brazil | 9.5% | 17.2% | 33.2% | 69.5% |
| Colombia | 6.7% | 13.6% | 24.0% | 73.2% |
| England | 6.6% | 13.3% | 27.4% | 72.5% |
| France | 6.0% | 12.4% | 25.0% | 67.2% |
| Ecuador | 4.8% | 10.7% | 22.1% | 65.5% |
| Portugal | 3.3% | 8.1% | 16.0% | 62.2% |
| Japan | 2.1% | 5.4% | 13.4% | 44.4% |
| Netherlands | 2.1% | 5.6% | 13.8% | 46.8% |

Adding the goal model lifts the South-American attacking sides (Argentina to #1,
Brazil up) and softens the backbone's Spain-heaviness — consistent with both
market consensus and the bake-off, where the ensemble beats every single model.

### Which engine? An honest, leakage-safe bake-off

Fit each model strictly before each of the 2010–2022 World Cups and score it on
those tournaments' **actual 256 finals matches** (`simulation bakeoff`):

| model | log loss | accuracy |
|---|---|---|
| backbone (Elo+form) | 0.9961 | 53.1% |
| xgboost | 1.0063 | 52.3% |
| dixon_coles | 1.0025 | 53.5% |
| **ensemble (w_dc=0.35)** | **0.9880** | **55.5%** |

Neither the Dixon-Coles goal model nor XGBoost beats the Elo backbone *alone*, but
**blending the backbone with the goal model beats all three** on both log loss and
accuracy. The weight is tuned by `simulation tune-weight` (broad flat optimum over
`w_dc ∈ [0.30, 0.50]`); 0.35 is chosen as a robust point that avoids the slight
non-WC-slice cost of heavier weights. This is the first *modeling* change (not new
data) to beat the backbone on the World Cup slice.

**Caveat:** the exact FIFA third-place combination table (which group's third fills
which R32 slot, 1 of 495 cases) is approximated by seeding the best thirds with
same-group avoidance; this only shifts R32 matchups and is a documented,
easy-to-revise choice. Club-data enrichment (API-Football) is additive and will be
reported as a separate with/without comparison.
