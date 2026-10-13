# World Cup Prediction Engine: Results

Main results of the project. Every number comes from a leakage-safe backtest: a
model is only ever fit on matches played before the period it is scored on. Log
loss is the main metric (lower is better), because the tournament simulator needs
good probabilities, not just the right pick.

---

## 1. Data

| dataset | source | coverage |
|---|---|---|
| International results | [martj42/international_results](https://github.com/martj42/international_results), CC0 | 49,372 matches, 1872-11-30 to 2026-06-07 |
| Goalscorers | same repository, CC0 | 47,601 goals (1916 to 2026), 15,334 scorers |
| FIFA ranking history | official ranking time series | 57,754 rows, 216 teams, 1993-08 to 2018-06 |

Every raw download is saved unchanged, with an as-of date and a provenance
manifest (URL, UTC timestamp, size, SHA-256), before any cleaning. Cleaning
dropped 72 unplayed fixtures and 1 duplicate match that was logged under two
spellings of the same venue. In the FIFA ranking, 39 duplicated "Sudan" rows were
removed.

---

## 2. Baseline match model

Chronological split: train up to 2016-03-25 (39,256 matches), test 2016 to 2026
(9,814 matches). Features: pre-match Elo, last-5 and last-10 form, rest days and
head-to-head record. Model: 3-way logistic regression.

| model | log loss | accuracy |
|---|---|---|
| no-skill (class rates) | 1.0525 | n/a |
| Elo only | 0.8834 | 59.8% |
| **full features** | **0.8712** | 59.8% |

The full model beats the no-skill baseline by about 17% on log loss.

**Calibration** (predicted home-win probability vs actual):

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

A leaky model is overconfident, so calibration this close is also good evidence
that no future information gets into the features.

**Elo check.** The top current ratings are Spain, Argentina, France, England,
Brazil, Colombia and Portugal, in line with eloratings.net. Home favourites by
more than 50 Elo points win 69.9% of the time (n ≈ 21k); underdogs win 26.5%
(n ≈ 19k).

---

## 3. Logistic regression vs XGBoost

Log loss by slice, same chronological backtest:

| model | all | neutral | WC finals | accuracy |
|---|---|---|---|---|
| no-skill | 1.0525 | 1.0962 | 1.1068 | n/a |
| logistic | **0.8712** | **0.9140** | 0.8530 | 59.8% |
| xgboost | 0.8739 | 0.9156 | **0.8527** | 59.9% |

The differences are noise. Elo already carries the signal in a near-linear way, so
trees find few interactions to exploit. Logistic regression is kept: simpler,
equally accurate and equally well calibrated.

---

## 4. Tournament-level calibration

The first 2022 backtest gave Brazil and Argentina about 60% combined title odds,
against roughly 35% in the betting market. The match probabilities were already
calibrated (optimal temperature close to 1 on every slice):

| slice | optimal temperature | log loss change |
|---|---|---|
| all | 1.01 | none |
| neutral | 1.00 | none |
| WC finals | 0.93 | 0.8530 to 0.8521 |

So the overconfidence came from the simulation itself: each team's rating was
treated as exact across a seven-game path. The fix gives every team one random
strength offset, drawn from N(0, sigma) in logit units, per simulation and
applies it to all of that team's matches. Sigma was tuned by champion log loss
over the 2010 to 2022 World Cups:

| sigma | mean champion log loss |
|---|---|
| 0.00 | 1.962 |
| **0.20** | **1.950** |
| 0.40 | 1.956 |
| 0.60 | 2.024 |
| 0.80 | 2.102 |
| 1.00 | 2.160 |

2022 top-two combined odds: 60.0% at sigma 0, 59.3% at 0.2, 53.3% at 0.5, 45.7% at
0.8. Matching the market would need sigma around 0.8, but the results reject it:
favourites usually do win (3 of these 4 champions were the top-rated team). The
default is a mild **sigma = 0.2**. With only four tournaments this is a sensible
default rather than a precise optimum.

---

## 5. Goal model time decay

The Dixon-Coles goal model weights each past match by `exp(-xi * age_in_days)`.
Each candidate was refit before every 2010 to 2022 World Cup and scored on World
Cup matches only:

| xi | half-life | match log loss (256 finals) | champion log loss (4 WCs) |
|---|---|---|---|
| 0.0000 | none | 1.0242 | 2.723 |
| 0.0002 | 9.5y | 1.0189 | 2.473 |
| 0.0005 | 3.8y | 1.0041 | **2.346** |
| **0.0008** | **2.4y** | **1.0024** | 2.473 |
| 0.0012 | 1.6y | 1.0032 | 2.470 |
| 0.0018 | 1.1y | 1.0269 | 2.444 |
| 0.0026 | 0.7y | 1.0270 | 2.441 |

Match log loss, pooled over every finals match, is the stronger signal. It has a
broad minimum from about 0.0005 to 0.0012, lowest at **xi = 0.0008**, which is
the default. Champion log loss uses only four data points and is much noisier.

---

## 6. Player features from goalscorer data: no gain

The goalscorer data records goals, not appearances, so goals per appearance
can't be computed without inventing a denominator. What it does support, per team
and as of each match:

| feature | meaning |
|---|---|
| `recent_goals` | goals in the team's last 20 matches |
| `top_scorer_goals` | most goals by one player in that window |
| `goal_concentration` | HHI of goals across scorers (1 = one-man team) |
| `penalty_share` | share of those goals from penalties |
| `squad_experience` | summed career international goals of the window's scorers |

Teams with no goals in the window get NaN plus a `low_data` flag, never a zero.
Low-data rate: 31.3% of all matches, 13.0% of World Cup matches.

| model | all | neutral | WC finals |
|---|---|---|---|
| backbone | 0.8712 | 0.9140 | **0.8530** |
| backbone + player | 0.8710 | 0.9137 | 0.8566 |

Neutral improves by 0.0004 (noise) and WC finals gets worse by 0.0036. A team that
scores a lot already shows strong form, so the profiles overlap with existing
features. The layer is **off by default**.

---

## 7. Confederation Elo bias: hypothesis rejected

Teams mostly play inside their own confederation, so a confederation could
inflate its own ratings. The suspicion was that CONMEBOL is overrated, which would
inflate Brazil and Argentina. Test: on inter-confederation matches since 2002,
compare each side's actual score (W/D/L = 1/0.5/0) to its Elo-expected score.

| confederation | mean residual (actual minus expected) | reading |
|---|---|---|
| OFC | -0.129 | overrated |
| AFC | -0.052 | overrated |
| CONCACAF | -0.049 | overrated |
| UEFA | +0.017 | well calibrated |
| CONMEBOL | +0.048 | slightly underrated |
| CAF | +0.054 | underrated |

CONMEBOL is not overrated, so Brazil's and Argentina's ratings are earned. The
overrated pools are the weaker ones. A correction fitted before 2018 and tested on
2018+ matches made World Cup predictions worse (Brier 0.1863 to 0.1901), so it is
kept as a diagnostic only.

---

## 8. FIFA ranking features: best single-feature gain

The only signal not derived from match results. Each match gets each team's
latest ranking published strictly before kickoff (rank, points and the
differences). About 29.5% of World Cup matches have no ranking available.

| model | all | neutral | WC finals |
|---|---|---|---|
| backbone | 0.8712 | 0.9140 | 0.8530 |
| backbone + FIFA | 0.8702 | 0.9142 | **0.8479** |

WC finals log loss improves by 0.0050, the largest single-feature gain on that
slice. It is **off by default** only because the data ends in mid-2018, so there
is no ranking for 2026.

---

## 9. Dixon-Coles goal model and the ensemble

The Dixon-Coles model fits attack and defence strengths per team, a home advantage
that is zero at neutral venues, the low-score correction and the time decay from
section 5. It produces full scoreline probabilities.

Bake-off: fit before each 2010 to 2022 World Cup, scored on those tournaments' 256
finals matches.

| model | log loss | accuracy |
|---|---|---|
| backbone (Elo + form) | 0.9961 | 53.1% |
| xgboost | 1.0063 | 52.3% |
| dixon_coles | 1.0025 | 53.5% |
| **ensemble (w_dc = 0.35)** | **0.9880** | **55.5%** |

No single model beats the backbone, but the blend beats all of them. Elo is built
from results and Dixon-Coles from goals, so their errors are different and
averaging helps. The weight has a flat optimum between 0.30 and 0.50; 0.35 is used
because heavier weights slightly hurt the non-World-Cup slices.

**2026 prediction (ensemble, 20,000 simulations):** Argentina 27.6%, Spain 17.1%,
Brazil 9.5%. Full table in the README.

---

## 10. Match-importance weighting: no gain

Weighting each match in the goal model by competition importance (a friendly
counts about a third of a World Cup match) made things worse: Dixon-Coles log loss
1.0025 to 1.0177, best ensemble 0.9878 to 0.9913. International data is sparse, so
downweighting friendlies mostly removes data, and the time decay already handles
recency. Reverted.

---

## 11. Takeaways and caveats

- XGBoost, the player layer, the confederation correction and importance weighting
  all failed to beat the Elo + form backbone. The ensemble worked because it adds
  a different view of the same matches.
- Elo dominates: the full feature set only edges out Elo alone (0.8834 to 0.8712).
- The 2016 to 2026 test set includes many lopsided qualifiers, which makes the
  headline numbers look better than on competitive matches.
- Every tuning step that uses World Cups has only four tournaments to learn from.
