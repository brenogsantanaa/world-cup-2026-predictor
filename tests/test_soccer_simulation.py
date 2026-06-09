"""Tests for the tournament simulator, using a stub predictor (no model needed)."""

import numpy as np

from sports_predictor.soccer.simulation import (
    monte_carlo,
    play_tie,
    simulate_group,
    simulate_tournament,
)
from sports_predictor.soccer.tournaments import WC_2022

# Give every team a unique strength: later in this list = stronger.
STRENGTH = {team: i for i, team in enumerate(WC_2022.teams)}


def deterministic_lookup(x, y):
    """Stronger team always wins in regulation (no draws)."""
    return (1.0, 0.0, 0.0) if STRENGTH[x] > STRENGTH[y] else (0.0, 0.0, 1.0)


def coinflip_lookup(x, y):
    """Pure 50/50 with no draws."""
    return (0.5, 0.0, 0.5)


def test_group_winner_and_runner_up_are_the_two_strongest():
    rng = np.random.default_rng(0)
    group = WC_2022.groups["A"]
    winner, runner_up = simulate_group(group, deterministic_lookup, STRENGTH, rng)
    by_strength = sorted(group, key=lambda t: STRENGTH[t], reverse=True)
    assert winner == by_strength[0]
    assert runner_up == by_strength[1]


def test_play_tie_advances_the_favorite_when_certain():
    rng = np.random.default_rng(0)
    a, b = WC_2022.teams[0], WC_2022.teams[1]  # b stronger
    assert play_tie(a, b, deterministic_lookup, "proportional", rng) == b


def test_strongest_team_always_wins_a_deterministic_tournament():
    rng = np.random.default_rng(1)
    reached = simulate_tournament(WC_2022, deterministic_lookup, STRENGTH, "proportional", rng)
    champions = [t for t, r in reached.items() if r == "Champion"]
    assert len(champions) == 1
    # The globally strongest team beats everyone, so it must win.
    assert champions[0] == max(STRENGTH, key=STRENGTH.get)


def test_monte_carlo_probabilities_are_monotone_and_normalized():
    table = monte_carlo(WC_2022, coinflip_lookup, STRENGTH, n=300, seed=7)

    # Reaching a later round implies reaching every earlier one.
    assert (table["reach_R16"] >= table["reach_QF"] - 1e-9).all()
    assert (table["reach_QF"] >= table["reach_SF"] - 1e-9).all()
    assert (table["reach_SF"] >= table["reach_Final"] - 1e-9).all()
    assert (table["reach_Final"] >= table["win"] - 1e-9).all()

    # Exactly one champion and two finalists per simulation.
    assert abs(table["win"].sum() - 1.0) < 1e-9
    assert abs(table["reach_Final"].sum() - 2.0) < 1e-9
    # 16 teams reach the round of 16 every time.
    assert abs(table["reach_R16"].sum() - 16.0) < 1e-9


def test_monte_carlo_covers_all_teams():
    table = monte_carlo(WC_2022, coinflip_lookup, STRENGTH, n=50, seed=3)
    assert len(table) == len(WC_2022.teams)
