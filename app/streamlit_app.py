"""Match Outcome Predictor - a Streamlit front-end for the World Cup model.

Pick any two national teams and see what the model expects: win / draw / loss
probabilities and the most likely scorelines. It is powered by the same
Dixon-Coles goal model used in the tournament simulator, so the predictions here
are consistent with the championship odds.

Run it (from the project root, with the package installed)::

    pip install -e ".[app]"        # or just: pip install streamlit
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from sports_predictor.core.paths import PROCESSED_DIR
from sports_predictor.soccer.dixon_coles import MAX_GOALS, DixonColesModel
from sports_predictor.soccer.elo import compute_elo, expected_score
from sports_predictor.soccer.tournaments import WC_2026

# Train the model on everything before the World Cup kicked off (leakage-safe,
# and consistent with the tournament predictions).
CUTOFF = "2026-06-11"

# World Cup 2026 palette.
GREEN = "#1bb55c"
YELLOW = "#c9b400"
RED = "#e8112d"
NAVY = "#16235a"


# --------------------------------------------------------------------------- #
# Model loading (cached so it only happens once per session)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_matches() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "matches.parquet")


@st.cache_resource(show_spinner="Training the goal model (one-time, ~15s)...")
def _load_models():
    matches = _load_matches()
    dc = DixonColesModel().fit(matches, CUTOFF)
    pre = matches[matches["date"] < pd.Timestamp(CUTOFF, tz="UTC")]
    _, elo = compute_elo(pre)
    return dc, elo


def _outcome_bar(p_home: float, p_draw: float, p_away: float, home: str, away: str) -> str:
    """An HTML stacked bar: home-win (green) / draw (yellow) / away-win (red)."""
    segments = [
        (p_home, GREEN, f"{home} {p_home:.0%}"),
        (p_draw, YELLOW, f"Draw {p_draw:.0%}"),
        (p_away, RED, f"{away} {p_away:.0%}"),
    ]
    parts = "".join(
        f'<div title="{label}" style="width:{frac*100:.2f}%;background:{color};'
        f'height:34px;display:flex;align-items:center;justify-content:center;'
        f'color:white;font-weight:700;font-size:13px;overflow:hidden;white-space:nowrap;">'
        f'{label if frac > 0.12 else ""}</div>'
        for frac, color, label in segments
    )
    return (
        f'<div style="display:flex;width:100%;border-radius:8px;overflow:hidden;'
        f'box-shadow:0 1px 3px rgba(0,0,0,0.15);">{parts}</div>'
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Match Outcome Predictor", page_icon="⚽", layout="centered")

dc, elo = _load_models()
teams = sorted(WC_2026.teams)


def _default(name: str, fallback: int) -> int:
    return teams.index(name) if name in teams else fallback


st.markdown(
    f"<h1 style='color:{NAVY};margin-bottom:0;'>⚽ Match Outcome Predictor</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Pick two teams and see what my World Cup model expects. Same engine behind "
    "the tournament simulation, trained on 49,000+ internationals since 1872."
)

c1, c2 = st.columns(2)
home = c1.selectbox("Team A", teams, index=_default("Brazil", 0))
away = c2.selectbox("Team B", teams, index=_default("Argentina", 1))
neutral = st.toggle("Neutral venue (like the World Cup)", value=True)

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

# --- Predictions from the goal model ---
p_home, p_draw, p_away = dc.outcome_proba(home, away, neutral=neutral)
lam, mu = dc.expected_goals(home, away, neutral=neutral)
mat = dc.scoreline_matrix(home, away, neutral=neutral)

# Most likely scorelines.
flat = sorted(
    ((mat[x, y], x, y) for x in range(MAX_GOALS + 1) for y in range(MAX_GOALS + 1)),
    reverse=True,
)
top_prob, ti, tj = flat[0]

st.markdown("### Prediction")
st.markdown(_outcome_bar(p_home, p_draw, p_away, home, away), unsafe_allow_html=True)
st.write("")

m1, m2, m3 = st.columns(3)
m1.metric(f"{home} win", f"{p_home:.0%}")
m2.metric("Draw", f"{p_draw:.0%}")
m3.metric(f"{away} win", f"{p_away:.0%}")

st.markdown(
    f"<div style='text-align:center;font-size:30px;font-weight:800;color:{NAVY};"
    f"margin:8px 0;'>{home} &nbsp;{ti} : {tj}&nbsp; {away}</div>"
    f"<div style='text-align:center;color:#6b7280;margin-bottom:6px;'>"
    f"most likely scoreline &middot; expected goals {lam:.2f} - {mu:.2f}</div>",
    unsafe_allow_html=True,
)

# --- Top scorelines + Elo second opinion ---
with st.expander("Most likely scorelines"):
    rows = [
        {"Scoreline": f"{x} - {y}", "Chance": f"{p:.1%}"}
        for p, x, y in flat[:6]
    ]
    st.table(pd.DataFrame(rows))

with st.expander("Cross-check: the Elo model's win probability"):
    adv = 0.0 if neutral else 100.0
    e_home = expected_score(elo.get(home, 1500), elo.get(away, 1500), adv)
    st.write(
        f"Elo rates **{home}** {elo.get(home, 1500):.0f} and **{away}** "
        f"{elo.get(away, 1500):.0f}. On a {'neutral' if neutral else 'home'} ground "
        f"it gives {home} a **{e_home:.0%}** expected result (win + half of draws). "
        "The goal model above adds the scoreline detail Elo can't."
    )

st.caption(
    "Trained on international results through 11 Jun 2026. No club or player data yet "
    "(that's the next upgrade). For fun, not betting."
)
