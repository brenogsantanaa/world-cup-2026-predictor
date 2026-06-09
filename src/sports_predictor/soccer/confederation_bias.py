"""Is Elo biased across confederations? (e.g. is CONMEBOL overrated vs UEFA?)

Our Elo is a closed system: a team can only gain rating by beating other teams in
the same pool. Confederations are *near-closed* pools -- South American sides play
mostly each other, Europeans mostly each other -- linked only by the relatively
few inter-confederation games (World Cups, Confederations Cup, friendlies). If one
confederation hoards rating among itself, its teams can look strong on paper yet
under-perform when they finally meet outsiders. That would mechanically inflate
the likes of Brazil/Argentina in the tournament simulator.

The test is simple and uses only inter-confederation matches: compare each side's
**actual** score (win=1, draw=0.5, loss=0) to its **Elo-expected** score
(neutral-aware, computed pre-match). The average residual ``actual - expected``
per confederation is the bias:

    ~0   Elo is well-calibrated for that confederation against outsiders
    < 0  its teams win less than their rating predicts -> overrated
    > 0  its teams win more than predicted            -> underrated

We also fit an additive per-confederation Elo offset (UEFA = reference) that would
make those residuals vanish: the "implied Elo bias" in rating points.

This is analysis, not a model change. Run::

    python -m sports_predictor.soccer.confederation_bias
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from sports_predictor.core.paths import PROCESSED_DIR
from sports_predictor.soccer.elo import compute_elo, expected_score
from sports_predictor.soccer.teams import confederation_of

CONF_ORDER = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"]


def prepare_inter_confederation(
    matches: pd.DataFrame, since: str | None = "2002-01-01"
) -> pd.DataFrame:
    """Return inter-confederation matches with pre-match Elo and actual scores.

    Both teams must have a known confederation and the two must differ. ``since``
    restricts to a modern era (Elo pools and travel differ greatly across eras);
    pass ``None`` for all time.
    """
    elo_df, _ = compute_elo(matches)
    if since is not None:
        elo_df = elo_df[elo_df["date"] >= pd.Timestamp(since, tz="UTC")]

    df = elo_df.copy()
    df["home_conf"] = df["home_team"].map(confederation_of)
    df["away_conf"] = df["away_team"].map(confederation_of)
    df = df[
        (df["home_conf"] != "Unknown")
        & (df["away_conf"] != "Unknown")
        & (df["home_conf"] != df["away_conf"])
    ].copy()

    df["actual_home"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        [1.0, 0.0],
        default=0.5,
    )
    return df.reset_index(drop=True)


def confederation_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ``actual - expected`` per confederation over inter-confed matches.

    Expects columns ``home_conf``, ``away_conf``, ``elo_expected_home``,
    ``actual_home``. Each match contributes once per side (the away residual is the
    negative of the home residual, since both scores and expectations sum to 1).
    """
    res_home = df["actual_home"] - df["elo_expected_home"]
    long = pd.concat(
        [
            pd.DataFrame({"conf": df["home_conf"], "residual": res_home}),
            pd.DataFrame({"conf": df["away_conf"], "residual": -res_home}),
        ],
        ignore_index=True,
    )
    g = long.groupby("conf")["residual"]
    out = pd.DataFrame({"n": g.size(), "mean_residual": g.mean(), "std": g.std()})
    out["ci95"] = 1.96 * out["std"] / np.sqrt(out["n"])
    return out.sort_values("mean_residual")


def estimate_elo_offsets(df: pd.DataFrame, reference: str = "UEFA") -> pd.Series:
    """Fit additive per-confederation Elo offsets that zero out the residuals.

    Minimizes ``actual - expected_score(elo_home + off_home, elo_away + off_away)``
    in least squares, with ``reference`` pinned to 0 for identifiability. A
    negative offset means the confederation's ratings are inflated (overrated) by
    that many Elo points relative to the reference.
    """
    confs = [c for c in CONF_ORDER if c in set(df["home_conf"]) | set(df["away_conf"])]
    free = [c for c in confs if c != reference]
    idx = {c: i for i, c in enumerate(free)}

    elo_h = df["home_elo_pre"].to_numpy()
    elo_a = df["away_elo_pre"].to_numpy()
    adv = np.where(df["neutral"].to_numpy(), 0.0, 100.0)
    actual = df["actual_home"].to_numpy()
    home_i = df["home_conf"].map(lambda c: idx.get(c, -1)).to_numpy()
    away_i = df["away_conf"].map(lambda c: idx.get(c, -1)).to_numpy()

    def residuals(params):
        off_h = np.where(home_i >= 0, params[home_i], 0.0)
        off_a = np.where(away_i >= 0, params[away_i], 0.0)
        exp = 1.0 / (1.0 + 10.0 ** (-((elo_h + off_h + adv) - (elo_a + off_a)) / 400.0))
        return actual - exp

    fit = least_squares(residuals, x0=np.zeros(len(free)))
    offsets = {reference: 0.0, **{c: fit.x[idx[c]] for c in free}}
    return pd.Series(offsets, name="implied_elo_offset").reindex(confs)


def validate_correction(matches: pd.DataFrame, cutoff: str = "2018-01-01") -> dict:
    """Out-of-sample check: does a confederation Elo correction actually help?

    Fit offsets on inter-confederation matches *before* ``cutoff``, then measure
    the Brier score (mean squared ``actual - expected``) on matches *after* it,
    with raw vs corrected Elo. Lower is better. Honest because the offsets never
    see the test period.
    """
    df = prepare_inter_confederation(matches, since=None)
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    train, test = df[df["date"] < cutoff_ts], df[df["date"] >= cutoff_ts].copy()
    offsets = estimate_elo_offsets(train, reference="UEFA")

    off_h = test["home_conf"].map(offsets).fillna(0.0).to_numpy()
    off_a = test["away_conf"].map(offsets).fillna(0.0).to_numpy()
    adv = np.where(test["neutral"].to_numpy(), 0.0, 100.0)
    corrected = 1.0 / (
        1.0 + 10.0 ** (-((test["home_elo_pre"] + off_h + adv) - (test["away_elo_pre"] + off_a)) / 400.0)
    )
    actual = test["actual_home"].to_numpy()

    corrected = corrected.to_numpy()
    raw = test["elo_expected_home"].to_numpy()

    def brier(a, exp):
        return float(np.mean((a - exp) ** 2))

    raw_b, corr_b = brier(actual, raw), brier(actual, corrected)
    wc = test["is_world_cup"].astype(bool).to_numpy()
    raw_wc = brier(actual[wc], raw[wc]) if wc.any() else float("nan")
    corr_wc = brier(actual[wc], corrected[wc]) if wc.any() else float("nan")

    print(
        f"\nout-of-sample confederation correction (fit < {cutoff}, test >= {cutoff}):\n"
        f"  test inter-confed matches: {len(test):,}  (World Cup: {int(wc.sum())})\n"
        f"  Brier all:  raw {raw_b:.4f} -> corrected {corr_b:.4f}  ({(corr_b - raw_b):+.4f})\n"
        f"  Brier WC:   raw {raw_wc:.4f} -> corrected {corr_wc:.4f}  ({(corr_wc - raw_wc):+.4f})"
    )
    return {"raw": raw_b, "corrected": corr_b, "raw_wc": raw_wc, "corrected_wc": corr_wc}


def _print_report(matches: pd.DataFrame) -> dict:
    results = {}
    for label, since in [("all time", None), ("2002+", "2002-01-01")]:
        df = prepare_inter_confederation(matches, since=since)
        res = confederation_residuals(df)
        results[label] = res
        print(f"\nconfederation Elo residual (actual - expected), {label}  "
              f"[{len(df):,} inter-confed matches]")
        print(f"  {'conf':<10}{'n*':>8}{'mean':>10}{'95% CI':>12}  verdict")
        for conf, r in res.iterrows():
            sig = abs(r["mean_residual"]) > r["ci95"]
            verdict = "" if not sig else ("overrated" if r["mean_residual"] < 0 else "underrated")
            print(
                f"  {conf:<10}{int(r['n']):>8,}{r['mean_residual']:>+10.3f}"
                f"{r['ci95']:>11.3f}  {verdict}"
            )
    print("  (* n counts each match once per participating confederation)")

    # Implied Elo offsets (modern era), UEFA = reference.
    df = prepare_inter_confederation(matches, since="2002-01-01")
    offsets = estimate_elo_offsets(df, reference="UEFA")
    results["offsets_2002"] = offsets
    print("\nimplied Elo offset vs UEFA (2002+, negative = overrated by Elo):")
    for conf, off in offsets.items():
        print(f"  {conf:<10}{off:>+8.0f}")

    # Headline: CONMEBOL vs UEFA, World Cup only (the cleanest neutral sample).
    wc = df[df["is_world_cup"].astype(bool)]
    cu = wc[
        ((wc["home_conf"] == "CONMEBOL") & (wc["away_conf"] == "UEFA"))
        | ((wc["home_conf"] == "UEFA") & (wc["away_conf"] == "CONMEBOL"))
    ]
    if len(cu):
        # Express from CONMEBOL's perspective.
        conmebol_actual = np.where(
            cu["home_conf"] == "CONMEBOL", cu["actual_home"], 1 - cu["actual_home"]
        )
        conmebol_expected = np.where(
            cu["home_conf"] == "CONMEBOL", cu["elo_expected_home"], 1 - cu["elo_expected_home"]
        )
        print(
            f"\nCONMEBOL vs UEFA at World Cups (2002+, {len(cu)} matches):\n"
            f"  CONMEBOL actual score   {conmebol_actual.mean():.3f}\n"
            f"  CONMEBOL Elo-expected   {conmebol_expected.mean():.3f}\n"
            f"  residual                {conmebol_actual.mean() - conmebol_expected.mean():+.3f}"
        )
    return results


def _main() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    _print_report(matches)
    validate_correction(matches)


if __name__ == "__main__":
    _main()
