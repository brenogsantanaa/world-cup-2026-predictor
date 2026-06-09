"""Structural + probabilistic tests for the 2026 World Cup field and simulation.

These run offline (synthetic lookup), so they validate the 48-team format without
needing the processed data files.
"""

import string

from sports_predictor.soccer.simulation import monte_carlo
from sports_predictor.soccer.teams import normalize_team_name
from sports_predictor.soccer.tournaments import WC_2026


def test_field_is_48_teams_no_duplicates():
    t = WC_2026
    assert len(t.groups) == 12
    assert all(len(v) == 4 for v in t.groups.values())
    assert len(t.teams) == 48
    assert len(set(t.teams)) == 48  # no team appears twice


def test_all_team_names_are_canonical():
    # Every listed name must already be the canonical spelling (idempotent),
    # otherwise it would silently fail to join the results/Elo data.
    for team in WC_2026.teams:
        assert normalize_team_name(team) == team, team


def test_r32_uses_each_group_once_plus_eight_thirds():
    slots = [s for tie in WC_2026.r32 for s in tie]
    assert len(WC_2026.r32) == 16
    assert len(slots) == 32  # 16 ties x 2
    winners = sorted(s[1] for s in slots if s[0] == "W")
    runners = sorted(s[1] for s in slots if s[0] == "R")
    thirds = [s for s in slots if s[0] == "3"]
    assert winners == list(string.ascii_uppercase[:12])  # A..L winners, once each
    assert runners == list(string.ascii_uppercase[:12])  # A..L runners, once each
    assert len(thirds) == 8


def test_third_place_slots_match_markers():
    marker_idx = sorted(
        i for i, tie in enumerate(WC_2026.r32) if any(s[0] == "3" for s in tie)
    )
    assert marker_idx == sorted(WC_2026.third_place_slots)


def test_bracket_halving_structure():
    # R16 (8) -> QF (4) -> SF (2) -> Final (1)
    assert [len(r) for r in WC_2026.bracket] == [8, 4, 2, 1]


def test_simulation_probabilities_are_consistent():
    teams = WC_2026.teams
    strength = {t: 1500.0 for t in teams}

    def lookup(x, y):  # perfectly even matchup
        return (1 / 3, 1 / 3, 1 / 3)

    table = monte_carlo(WC_2026, lookup, strength, n=300, seed=1)
    assert len(table) == 48
    # Exactly one champion and exactly 32 R32 qualifiers per simulation.
    assert abs(table["win"].sum() - 1.0) < 1e-9
    assert abs(table["reach_R32"].sum() - 32.0) < 1e-9
    # Monotonic funnel: champion <= final <= semi <= R16 <= R32 (per team).
    for col_a, col_b in [
        ("win", "reach_Final"), ("reach_Final", "reach_SF"),
        ("reach_SF", "reach_R16"), ("reach_R16", "reach_R32"),
    ]:
        assert (table[col_a] <= table[col_b] + 1e-9).all()
