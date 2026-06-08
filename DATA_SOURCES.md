# Data Instructions for the Sports Prediction Engine (World Cup focus)

> This file tells Cursor (or any AI assistant) **what data we use, where it comes
> from, and the rules for trusting it.** The guiding principle: use as much
> useful data as we can, but **never add a data point we cannot trace to a
> trustworthy source.** Coverage is good; provenance is non-negotiable.

---

## 0. Golden Rules (read first)

1. **Trust before volume.** More features only help if each one is correct and
   verifiable. A wrong feature is worse than a missing one.
2. **Every dataset must have a known source, a known license, and a known
   "as-of" date.** If we cannot say where a number came from, we do not use it.
3. **Cache raw responses to disk before transforming them.** Raw data goes in
   `data/raw/`, cleaned data in `data/processed/`. Notebooks and the model must
   read from local cache, not re-hit the network every run (reproducibility).
4. **No data leakage.** A feature for a given match may only use information that
   existed *before kickoff*. This is the single most common way to build a model
   that looks great and is secretly useless.
5. **No claims we can't back.** If a data source isn't available or is flaky,
   say so in code comments and skip it. Do not invent or estimate values to fill
   gaps without clearly labeling them as estimates.
6. **Respect each source's terms of use and rate limits.** Cache aggressively,
   throttle requests, and prefer official APIs and open datasets over scraping.

---

## 1. What we are predicting

- **Primary target:** the probability that a given team wins (or draws/loses) a
  single international match.
- **Downstream goal:** feed those match probabilities into a Monte Carlo
  tournament simulation to estimate group-stage advancement and World Cup
  champion probabilities.
- We start by sharpening the pipeline on **NBA data** (already scaffolded), then
  point the same architecture at **international soccer**.

---

## 2. Data sources (ranked by trustworthiness)

Cursor: before integrating any source, **verify it is currently reachable and
check its license/terms.** Note the access method and any API key requirement in
code comments. Prefer the higher-tier sources.

### Tier 1 — Authoritative / official
| Source | What we get | Access | Trust notes |
|---|---|---|---|
| **FIFA/Coca-Cola Men's World Ranking** (official) | Official team ranking + points | Published tables on FIFA's site | Authoritative for ranking. Updates on fixed dates; record the publication date. |
| **football-data.org** | Fixtures, results, competitions | REST API (free tier, API key required) | Reputable, documented API. Mind the free-tier rate limits. |

### Tier 2 — Well-established community / open data
| Source | What we get | Access | Trust notes |
|---|---|---|---|
| **eloratings.net** (World Football Elo Ratings) | Elo rating per national team, match history | Public website (no official API) | Widely cited, methodology is documented. No API — if we scrape, do it gently, cache locally, and record the as-of date. |
| **openfootball / football.db** | Historical results, fixtures, squads | Open data repos (GitHub) | Open-licensed, good for backfilling history. Verify how current it is. |
| **Kaggle: International football results (1872–present)** | Historical match results, scores, venues, tournament/friendly flag, neutral-venue flag | CSV download | Community-maintained; spot-check against a second source before trusting blindly. |

### Tier 3 — Useful but handle with care
| Source | What we get | Access | Trust notes |
|---|---|---|---|
| **Transfermarkt** | Squad market value (proxy for squad quality) | Website (scraping) | Valuations are estimates, not facts. Check terms of use before scraping; label these features clearly as proxies. |
| **Sportsbook odds** | Market-implied probabilities | Various | For **validation/comparison only**, never as a training label. Not financial advice. |

> If a source isn't in this list, add it here with the same columns before using
> it. The table is the source of truth for what's allowed.

---

## 3. Features we want (the "as much data as we can" list)

Group these into a single per-match modeling table. **Every feature must be
computable using only pre-kickoff information.**

**Team strength & rating**
- Elo rating of each team (home/away) at match date
- FIFA ranking and ranking points at match date
- Rating difference between the two teams

**Recent form (rolling, time-aware)**
- Win/draw/loss rate over last N matches (e.g., 5 and 10)
- Goals scored / conceded average over last N matches
- Points-per-game trend
- Current win/unbeaten streak length

**Match context**
- Home / away / neutral venue (World Cup is mostly neutral — flag this)
- Confederation of each team
- Competitive vs friendly match flag
- Tournament stage (group, knockout) where applicable
- Rest days since each team's previous match (derived from the schedule)

**Head-to-head**
- Historical H2H record between the two teams
- Recent H2H results (last few meetings)

**Squad quality (proxy, Tier-3, label as estimate)**
- Aggregate squad market value
- Average squad age / caps (if reliably available)

> Start with the features we can build from Tier 1–2 sources. Add Tier-3 proxies
> only after the core model works, and mark them as estimates.

---

## 4. Ingestion flow (do this for every source)

```
External source
   -> fetch (throttled, with error handling)
   -> save RAW response to data/raw/<source>/<descriptive_name>_<asof_date>.csv|json
   -> clean/normalize into a standard schema
   -> save to data/processed/
   -> build the per-match modeling table
```

Rules:
- **Always save the raw pull first**, with the as-of date in the filename, before
  any cleaning. This makes every result reproducible and auditable.
- Normalize team names to a **single canonical mapping** (e.g., "USA" vs "United
  States", "Korea Republic" vs "South Korea"). Keep the mapping in one place
  (`src/sports_predictor/soccer/`). Name mismatches are a top source of silent
  data-merging bugs.
- Store dates in **UTC, ISO 8601**. Sort chronologically before computing any
  rolling feature.

---

## 5. Trustworthiness checklist (run before trusting a dataset)

For each source, Cursor should verify and note in comments:

- [ ] **Provenance:** where it came from + URL + access date.
- [ ] **License / terms:** are we allowed to use and cache it?
- [ ] **As-of date:** what point in time does the data represent?
- [ ] **Coverage:** which teams/years are present? Where are the gaps?
- [ ] **Sanity checks:** row counts reasonable, no impossible scores, dates in
      range, no duplicate matches, team names map cleanly.
- [ ] **Cross-check:** at least one key field (e.g., a recent result) matches a
      second independent source.
- [ ] **Missing-value policy:** how are nulls handled, and is that explicit?

If any box can't be checked, either fix it or drop the source. Don't paper over
gaps with guesses.

---

## 6. Anti-leakage rules (critical)

- A feature for match `M` may only use matches that **finished before `M`'s
  kickoff.** Use chronological/expanding windows, never the full dataset.
- **Never** include the final score, result, or anything derived from `M` itself
  as an input feature.
- Use a **chronological train/test split** (train on earlier matches, test on
  later ones), not a random split. Sports data is a time series.
- When backfilling ratings (Elo, FIFA points), use the value **as it stood on
  the match date**, not today's value.

---

## 7. What NOT to do

- Do not hardcode data values into the codebase; always load from cached files.
- Do not mix data sources without reconciling team names and dates first.
- Do not use betting odds as a training label (validation only).
- Do not present Tier-3 proxies (market value, etc.) as hard facts.
- Do not skip the raw-cache step "to save time" — it breaks reproducibility.
- Do not add a feature you cannot trace to a source in Section 2.

---

## 8. Suggested first steps for Cursor

1. Stand up the soccer data module skeleton in `src/sports_predictor/soccer/`.
2. Integrate **one Tier-1/Tier-2 results source** end to end (fetch → raw cache
   → cleaned table), with the trustworthiness checklist filled in.
3. Add **Elo** and **FIFA ranking** as-of-match-date features.
4. Build the canonical team-name mapping.
5. Write small tests on tiny fake DataFrames for: rolling-form (no current-match
   leakage), rest-days calculation, and the chronological split.
6. Only then expand to more sources and Tier-3 proxies.

Keep changes small and explain each one. Build trust in the data before
building cleverness in the model.
