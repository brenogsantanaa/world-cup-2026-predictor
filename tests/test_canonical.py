"""Tests for the canonical reconciliation layer."""

import math

from sports_predictor.canonical.conflicts import ConflictLog, resolve
from sports_predictor.canonical.names import display_name, name_key, strip_accents
from sports_predictor.canonical.registry import CanonicalRegistry, team_id


# --------------------------------------------------------------------------- #
# Name normalization
# --------------------------------------------------------------------------- #
def test_strip_accents():
    assert strip_accents("Mbappé") == "Mbappe"
    assert strip_accents("Özil") == "Ozil"
    assert strip_accents("Héctor") == "Hector"


def test_name_key_folds_accents_and_punctuation():
    assert name_key("Kylian Mbappé") == "kylian mbappe"
    assert name_key("N'Golo Kanté") == "ngolo kante"
    assert name_key("  Luka   Modrić ") == "luka modric"


def test_name_key_handles_last_comma_first():
    assert name_key("Mbappé, Kylian") == name_key("Kylian Mbappe")


def test_name_key_handles_initials_dot():
    assert name_key("C. Ronaldo") == "c ronaldo"


def test_display_name_collapses_whitespace_keeps_accents():
    assert display_name("  Kylian   Mbappé ") == "Kylian Mbappé"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_same_player_different_accents_one_id():
    reg = CanonicalRegistry()
    a = reg.player_id("Kylian Mbappé")
    b = reg.player_id("Kylian Mbappe")
    assert a == b


def test_alias_maps_to_canonical_id():
    reg = CanonicalRegistry()
    canonical_id = reg.player_id("Cristiano Ronaldo")
    reg.add_player_alias("Cristiano Ronaldo dos Santos Aveiro", "Cristiano Ronaldo")
    assert reg.player_id("Cristiano Ronaldo dos Santos Aveiro") == canonical_id


def test_team_id_is_canonical_and_stable():
    reg = CanonicalRegistry()
    # Alias spellings resolve to the same canonical team + id.
    assert reg.team("USA")["name"] == "United States"
    assert reg.team("USA")["team_id"] == reg.team("United States")["team_id"]
    assert team_id("Korea Republic") == team_id("South Korea")


def test_first_display_name_is_kept_and_id_stable():
    reg = CanonicalRegistry()
    first = reg.player("Kylian Mbappé")
    again = reg.player("kylian mbappe")  # same key, different casing/accents
    assert again["name"] == "Kylian Mbappé"  # first display retained
    assert again["player_id"] == first["player_id"]


def test_abbreviated_name_is_not_auto_merged():
    reg = CanonicalRegistry()
    # "L. Messi" could be Lucas or Lionel; we must NOT silently merge it.
    assert reg.player_id("L. Messi") != reg.player_id("Lionel Messi")
    # ...but an explicit alias links them safely.
    reg.add_player_alias("L. Messi", "Lionel Messi")
    assert reg.player_id("L. Messi") == reg.player_id("Lionel Messi")


# --------------------------------------------------------------------------- #
# Conflict resolution
# --------------------------------------------------------------------------- #
def test_resolve_picks_priority_source():
    # FBref outranks Transfermarkt for minutes.
    value, source = resolve("minutes", {"transfermarkt": 800, "fbref": 810})
    assert (value, source) == (810, "fbref")


def test_resolve_skips_missing_values():
    value, source = resolve("xg", {"understat": None, "fbref": 4.2})
    assert (value, source) == (4.2, "fbref")


def test_resolve_logs_disagreement_but_keeps_priority():
    log = ConflictLog()
    value, source = resolve("market_value", {"transfermarkt": 1.0e8, "fbref": 9.0e7}, log=log)
    assert source == "transfermarkt"
    assert len(log) == 1
    assert log.summary() == {"market_value": 1}


def test_resolve_no_log_when_sources_agree():
    log = ConflictLog()
    resolve("goals", {"fbref": 10, "understat": 10}, log=log)
    assert len(log) == 0


def test_resolve_all_missing_returns_none():
    value, source = resolve("age", {"transfermarkt": None, "fbref": float("nan")})
    assert value is None and source is None


def test_resolve_unlisted_field_uses_stable_order():
    # No priority defined -> fall back to sorted source order.
    value, source = resolve("height", {"fbref": 180, "transfermarkt": 181})
    assert source == "fbref"  # 'fbref' < 'transfermarkt' alphabetically
    assert not math.isnan(value)
