"""Dixon-Coles bivariate Poisson goal model.

The baseline/XGBoost models predict the 3-way outcome (H/D/A) directly. This is a
deeper, more principled approach: model the *number of goals* each team scores,
then derive the outcome from the scoreline distribution. It is the standard
academic model for football (Dixon & Coles, 1997).

The idea
--------
Every team gets two strengths:

* ``attack``  - how many goals it tends to score
* ``defense`` - how few goals it tends to concede (higher = stingier)

Expected goals for a match are

    log(lambda_home) = gamma * at_home + attack[home] - defense[away]
    log(lambda_away) =                  attack[away] - defense[home]

where ``gamma`` is the home-field advantage and ``at_home`` is 1 for a normal
fixture and **0 at a neutral venue** (so World Cup matches get no home bonus).
Home and away goals are Poisson with those rates.

Two refinements make it a *Dixon-Coles* model rather than plain independent
Poisson:

1. **Low-score correction (rho).** Independent Poisson underrates 0-0 and 1-1
   draws and overrates 1-0 / 0-1. The ``tau`` adjustment fixes exactly those four
   cells. ``rho`` is fit from the data.
2. **Time decay (xi).** Recent matches matter more. Each match is weighted by
   ``exp(-xi * age_in_days)`` in the likelihood, so a team's strength reflects its
   current form, not its 1990s self.

Why it is a real upgrade over the flat classifier: it produces full scoreline
distributions (not just W/D/A), it is interpretable (every team has an attack and
defense number), and the eventual club/xG data plugs straight into the
attack/defense strengths.

Leakage safety: :meth:`DixonColesModel.fit` only ever sees matches strictly before
its ``cutoff``, and the time-decay age is measured relative to that cutoff.

Run the backtest (Dixon-Coles vs no-skill, by slice)::

    python -m sports_predictor.soccer.dixon_coles
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sports_predictor.core.evaluation import evaluate_probabilities, no_skill_log_loss
from sports_predictor.core.paths import PROCESSED_DIR
from sports_predictor.core.splitting import chronological_split

LABELS = ["H", "D", "A"]

# Defaults. ``xi`` is per-day; 0.0008 ~ a 2.4-year half-life, suitable for the
# sparse international calendar (club football uses faster decay). Chosen by
# ``simulation.tune_xi``: refit before each 2010-2022 World Cup and scored on the
# actual finals matches, pooled match log loss bottoms out in a broad, shallow
# basin from ~0.0005 to ~0.0012 (well below the no-decay and fast-decay extremes),
# with the minimum at 0.0008. Champion log loss, far noisier with only four data
# points, marginally prefers 0.0005; the difference is within sampling noise, so
# we take the higher-power match-LL optimum. See PROJECT_REPORT.md. ``max_age_years``
# hard-caps how far back the fit looks, bounding compute (decay handles the rest).
DEFAULT_XI = 0.0008
DEFAULT_MAX_AGE_YEARS = 12.0
MAX_GOALS = 10  # scoreline grid size for deriving outcome probabilities
RHO_BOUND = 0.25  # keep the low-score correction in a stable range


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _poisson_logpmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """log P(k; lam) for integer counts ``k`` (vectorised).

    Uses ``gammaln(k+1)`` for log(k!). Stable for the small goal counts here.
    """
    from scipy.special import gammaln

    return k * np.log(lam) - lam - gammaln(k + 1.0)


def _tau(hg: np.ndarray, ag: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles low-score dependence factor for each (home goals, away goals).

    Only the 0-0, 0-1, 1-0, 1-1 cells differ from 1; everything else is
    unaffected, so the correction is local and preserves the Poisson shape
    elsewhere.
    """
    out = np.ones_like(lam, dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho
    return out


@dataclass
class DixonColesModel:
    """A fitted Dixon-Coles goal model.

    After :meth:`fit`, holds per-team ``attack``/``defense`` dicts, the home
    advantage ``gamma`` and low-score ``rho``. Unknown teams (never seen in the
    fitting window) fall back to league-average strength and are reported via
    :attr:`low_data_teams`, so prediction degrades gracefully instead of failing.
    """

    xi: float = DEFAULT_XI
    max_age_years: float = DEFAULT_MAX_AGE_YEARS
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    gamma: float = 0.0
    rho: float = 0.0
    teams: list[str] = field(default_factory=list)
    low_data_teams: set[str] = field(default_factory=set)
    n_matches: int = 0

    # ------------------------------------------------------------------ fit
    def fit(self, matches: pd.DataFrame, cutoff) -> "DixonColesModel":
        """Fit on matches strictly before ``cutoff`` (leakage-safe).

        ``matches`` needs columns: date, home_team, away_team, home_score,
        away_score, neutral.
        """
        cutoff_ts = _utc(cutoff)
        oldest = cutoff_ts - pd.Timedelta(days=365.25 * self.max_age_years)
        dates = pd.to_datetime(matches["date"])
        window = matches[(dates < cutoff_ts) & (dates >= oldest)].copy()
        if window.empty:
            raise ValueError(f"no matches in the {self.max_age_years}y window before {cutoff_ts.date()}")

        teams = sorted(set(window["home_team"]) | set(window["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        T = len(teams)

        hi = window["home_team"].map(idx).to_numpy()
        ai = window["away_team"].map(idx).to_numpy()
        hg = window["home_score"].to_numpy().astype(int)
        ag = window["away_score"].to_numpy().astype(int)
        hf = (~window["neutral"].astype(bool)).to_numpy().astype(float)  # 1 if real home

        age_days = (cutoff_ts - _utc_series(window["date"])).dt.total_seconds().to_numpy() / 86400.0
        weight = np.exp(-self.xi * age_days)

        # Parameter vector: attack[1..T-1] (attack[0] = -sum, enforces sum=0),
        # defense[0..T-1], gamma, rho.
        n_atk_free = T - 1

        def unpack(p):
            atk_free = p[:n_atk_free]
            atk = np.empty(T)
            atk[1:] = atk_free
            atk[0] = -atk_free.sum()
            dfn = p[n_atk_free : n_atk_free + T]
            gamma = p[-2]
            rho = p[-1]
            return atk, dfn, gamma, rho

        def neg_log_likelihood(p):
            atk, dfn, gamma, rho = unpack(p)
            lam = np.exp(gamma * hf + atk[hi] - dfn[ai])
            mu = np.exp(atk[ai] - dfn[hi])
            ll = _poisson_logpmf(hg, lam) + _poisson_logpmf(ag, mu)
            tau = _tau(hg, ag, lam, mu, rho)
            ll = ll + np.log(np.clip(tau, 1e-10, None))
            return -np.sum(weight * ll)

        x0 = np.concatenate([np.zeros(n_atk_free), np.zeros(T), [0.25], [-0.05]])
        bounds = [(-3, 3)] * n_atk_free + [(-3, 3)] * T + [(-1.0, 1.0), (-RHO_BOUND, RHO_BOUND)]
        res = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)

        atk, dfn, gamma, rho = unpack(res.x)
        self.teams = teams
        self.attack = dict(zip(teams, atk))
        self.defense = dict(zip(teams, dfn))
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.n_matches = int(len(window))
        self.low_data_teams = set()
        return self

    # -------------------------------------------------------------- predict
    def _strength(self, team: str) -> tuple[float, float]:
        """(attack, defense) for ``team``; league-average fallback if unknown."""
        if team in self.attack:
            return self.attack[team], self.defense[team]
        self.low_data_teams.add(team)
        return 0.0, float(np.mean(list(self.defense.values()))) if self.defense else 0.0

    def expected_goals(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        """Expected goals (lambda_home, lambda_away) for a fixture."""
        a_h, d_h = self._strength(home)
        a_a, d_a = self._strength(away)
        hf = 0.0 if neutral else 1.0
        lam = np.exp(self.gamma * hf + a_h - d_a)
        mu = np.exp(a_a - d_h)
        return float(lam), float(mu)

    def scoreline_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """Full (MAX_GOALS+1)x(MAX_GOALS+1) scoreline probability matrix.

        Entry [x, y] = P(home scores x, away scores y), including the Dixon-Coles
        low-score correction, normalised to sum to 1.
        """
        lam, mu = self.expected_goals(home, away, neutral)
        ks = np.arange(MAX_GOALS + 1)
        ph = np.exp(_poisson_logpmf(ks, np.full_like(ks, lam, dtype=float)))
        pa = np.exp(_poisson_logpmf(ks, np.full_like(ks, mu, dtype=float)))
        mat = np.outer(ph, pa)
        # Apply tau to the four low-score cells.
        mat[0, 0] *= 1.0 - lam * mu * self.rho
        mat[0, 1] *= 1.0 + lam * self.rho
        mat[1, 0] *= 1.0 + mu * self.rho
        mat[1, 1] *= 1.0 - self.rho
        mat = np.clip(mat, 0.0, None)
        return mat / mat.sum()

    def outcome_proba(self, home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
        """(P home win, P draw, P away win) from the scoreline matrix."""
        mat = self.scoreline_matrix(home, away, neutral)
        p_home = np.tril(mat, -1).sum()  # home goals > away goals
        p_draw = np.trace(mat)
        p_away = np.triu(mat, 1).sum()
        return float(p_home), float(p_draw), float(p_away)

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        """H/D/A probabilities (in LABELS order) for a frame of fixtures.

        ``fixtures`` needs columns home_team, away_team, and optionally neutral
        (treated as True/neutral if absent).
        """
        has_neutral = "neutral" in fixtures.columns
        out = np.empty((len(fixtures), 3))
        for i, row in enumerate(fixtures.itertuples(index=False)):
            neutral = bool(getattr(row, "neutral")) if has_neutral else True
            out[i] = self.outcome_proba(row.home_team, row.away_team, neutral)
        return out


def _utc_series(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


# --------------------------------------------------------------------------- #
# Backtest: Dixon-Coles vs no-skill, by slice
# --------------------------------------------------------------------------- #
def _slice_masks(test: pd.DataFrame) -> dict[str, np.ndarray]:
    is_neutral = test["neutral"].astype(bool).to_numpy()
    is_wc = test["is_world_cup"].astype(bool).to_numpy()
    return {
        "all test": np.ones(len(test), dtype=bool),
        "neutral venue": is_neutral,
        "world cup (+qual)": is_wc,
        "WC finals~": is_wc & is_neutral,
    }


def run_backtest(test_fraction: float = 0.2, xi: float = DEFAULT_XI) -> dict:
    """Fit Dixon-Coles on the earlier matches, score the later ones by slice."""
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    matches = matches.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)

    train, test = chronological_split(matches, date_column="date", test_fraction=test_fraction)
    cutoff = _utc(test["date"].min())

    model = DixonColesModel(xi=xi).fit(train, cutoff)
    proba = model.predict_proba(test)
    y_test = test["result"].reset_index(drop=True)
    y_train = train["result"]

    print(
        f"Dixon-Coles backtest  (xi={xi}, half-life "
        f"{np.log(2) / xi / 365.25:.1f}y)\n"
        f"  fit on {model.n_matches:,} matches < {cutoff.date()}  |  test {len(test):,}\n"
        f"  home advantage gamma={model.gamma:.3f}  rho={model.rho:.3f}\n"
        f"{'-' * 60}\n"
        f"{'slice':<18}{'n':>7}{'no-skill':>11}{'model LL':>11}{'acc':>8}"
    )
    results: dict[str, dict] = {}
    for name, mask in _slice_masks(test).items():
        n = int(mask.sum())
        if n == 0:
            continue
        base = no_skill_log_loss(y_train, y_test[mask], LABELS)
        metrics = evaluate_probabilities(y_test[mask], proba[mask], LABELS)
        results[name] = {"n": n, "no_skill": base, **metrics}
        print(f"{name:<18}{n:>7,}{base:>11.4f}{metrics['log_loss']:>11.4f}{metrics['accuracy']:>8.1%}")

    if model.low_data_teams:
        print(f"\nlow-data teams (league-average fallback): {len(model.low_data_teams)}")
    return results


def _main() -> None:
    run_backtest()


if __name__ == "__main__":
    _main()
