"""Tests for canonical team-name handling."""

from sports_predictor.soccer.teams import find_unmapped_teams, normalize_team_name


def test_known_aliases_map_to_canonical():
    assert normalize_team_name("USA") == "United States"
    assert normalize_team_name("Korea Republic") == "South Korea"
    assert normalize_team_name("IR Iran") == "Iran"
    assert normalize_team_name("Côte d'Ivoire") == "Ivory Coast"


def test_alias_lookup_is_case_and_whitespace_insensitive():
    assert normalize_team_name("  united   states of america ") == "United States"
    assert normalize_team_name("KOREA REPUBLIC") == "South Korea"


def test_unknown_name_is_cleaned_but_preserved():
    # A name we have no alias for should pass through (trimmed), not be dropped.
    assert normalize_team_name("  Brazil ") == "Brazil"


def test_find_unmapped_teams_reports_only_unknowns():
    known = {"Brazil", "United States"}
    names = ["Brazil", "USA", "Narnia"]
    assert find_unmapped_teams(names, known) == {"Narnia"}
