"""Converting a 3-way match prediction into a knockout "who advances" result.

Our match model outputs three regulation (90-minute) probabilities:
P(home win), P(draw), P(away win). Group games use all three. But a World Cup
**knockout** match cannot end in a draw: a regulation draw goes to extra time and
then, if still level, a penalty shootout. So for knockout rounds we must turn the
three probabilities into a two-way advancement probability by **redistributing
the draw mass** between the two teams.

We keep the regulation win/loss mass untouched and only split the draw mass. The
question is who gets how much of it.

Strategies
----------
- ``proportional`` (default): give each team a share of the draw mass equal to
  its share of the regulation *win* probability::

      home_share = P(home win) / (P(home win) + P(away win))

  Rationale: extra time tends to favor the stronger team, while penalties are
  close to a coin flip, so the favorite should advance somewhat more often than
  in regulation -- but not dominantly. This needs no tuning parameters and uses
  only what the model already believes.

- ``even``: split the draw mass 50/50, i.e. treat extra time + penalties as a
  pure coin flip. A useful, unbiased baseline to measure the proportional
  strategy against.

Future upgrade (documented, not built)
--------------------------------------
An ``empirical`` strategy could calibrate the split from history using the CC0
``shootouts.csv`` dataset (penalty-shootout outcomes) plus extra-time results:
measure how often the stronger team actually advances as a function of the rating
gap, and fit the share from data. Build this only after the simulator runs with
``proportional``, so we can check whether the data-driven split meaningfully
changes champion probabilities.

Open hypothesis (NOT implemented)
---------------------------------
A common folk belief is that a team which equalizes *late* carries momentum into
extra time / penalties. This is unproven, and -- importantly -- the simulator
only ever sees pre-match probabilities and the final regulation result, never how
or when a draw happened, so there is no in-game momentum state to act on. Do not
hardcode any such advantage. The honest path is to first test, against historical
shootout data, whether late equalizers really win shootouts more often, before
considering any adjustment.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# A draw-split strategy maps (p_home_win, p_away_win) -> the share of the draw
# mass that goes to the home team (a number in [0, 1]).
DrawSplit = Callable[[object, object], object]


def proportional_share(p_home_win, p_away_win):
    """Home team's share of the draw mass, proportional to regulation win prob.

    Falls back to 0.5 when neither side has any regulation win probability (an
    all-draw prediction), so the result is always well defined. Works on scalars
    and on NumPy arrays.
    """
    total = p_home_win + p_away_win
    safe_total = np.where(total > 0, total, 1.0)
    return np.where(total > 0, p_home_win / safe_total, 0.5)


def even_share(p_home_win, p_away_win):
    """Split the draw mass exactly evenly (50/50 coin flip)."""
    return 0.5


STRATEGIES: dict[str, DrawSplit] = {
    "proportional": proportional_share,
    "even": even_share,
}

DEFAULT_STRATEGY = "proportional"


def advance_probabilities(
    p_home_win,
    p_draw,
    p_away_win,
    strategy: str | DrawSplit = DEFAULT_STRATEGY,
):
    """Return ``(p_home_advances, p_away_advances)`` for a knockout tie.

    The regulation win/loss mass is preserved; only the draw mass is reassigned
    according to ``strategy`` (a registered name like ``"proportional"`` /
    ``"even"``, or a custom share function). The two returned probabilities
    always sum to ``p_home_win + p_draw + p_away_win`` (i.e. 1 for a valid
    prediction). Scalars and NumPy arrays are both supported.
    """
    share_fn = _resolve_strategy(strategy)
    home_share = share_fn(p_home_win, p_away_win)
    p_home_adv = p_home_win + p_draw * home_share
    p_away_adv = p_away_win + p_draw * (1.0 - home_share)
    return p_home_adv, p_away_adv


def _resolve_strategy(strategy: str | DrawSplit) -> DrawSplit:
    if callable(strategy):
        return strategy
    try:
        return STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"unknown knockout strategy {strategy!r}; "
            f"choose one of {sorted(STRATEGIES)} or pass a callable"
        ) from None
