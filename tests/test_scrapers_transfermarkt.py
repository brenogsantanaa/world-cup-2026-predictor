"""Tests for the Transfermarkt squad parser + canonical mapping (no network)."""

from pathlib import Path

import numpy as np
import pytest

from sports_predictor.canonical.registry import CanonicalRegistry
from sports_predictor.scrapers.transfermarkt import (
    parse_market_value,
    parse_squad,
    to_canonical,
)

FIXTURE = Path(__file__).parent / "fixtures" / "transfermarkt_squad_sample.html"


def _html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Market value parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("\u20ac180.00m", 180_000_000.0),
        ("\u20ac500k", 500_000.0),
        ("\u20ac1.20bn", 1_200_000_000.0),
        ("-", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_market_value(text, expected):
    assert parse_market_value(text) == expected


# --------------------------------------------------------------------------- #
# Squad parsing
# --------------------------------------------------------------------------- #
def test_parse_squad_schema_and_rows():
    df = parse_squad(_html())
    assert list(df.columns) == [
        "source", "transfermarkt_id", "player_name", "position",
        "age", "nationality", "market_value_eur",
    ]
    assert len(df) == 3
    assert (df["source"] == "transfermarkt").all()


def test_parse_squad_values():
    df = parse_squad(_html()).set_index("transfermarkt_id")
    mbappe = df.loc["342229"]
    assert mbappe["player_name"] == "Kylian Mbappé"  # accents preserved in display
    assert mbappe["position"] == "Centre-Forward"
    assert mbappe["age"] == 27
    assert mbappe["nationality"] == "France"
    assert mbappe["market_value_eur"] == 180_000_000.0


def test_missing_market_value_is_nan_not_zero():
    df = parse_squad(_html()).set_index("transfermarkt_id")
    assert np.isnan(df.loc["999999", "market_value_eur"])  # the youngster has "-"


def test_parse_squad_raises_without_items_table():
    with pytest.raises(ValueError):
        parse_squad("<html><body><p>no table</p></body></html>")


# --------------------------------------------------------------------------- #
# Canonical mapping
# --------------------------------------------------------------------------- #
def test_to_canonical_ids_and_low_data_flag():
    reg = CanonicalRegistry()
    canon = to_canonical(parse_squad(_html()), reg).set_index("transfermarkt_id")
    assert canon.loc["342229", "player_id"].startswith("plr_")
    # All three are France -> one team_id; matches the registry's canonical id.
    assert canon["team_id"].nunique() == 1
    assert canon.loc["342229", "team_id"] == reg.team("France")["team_id"]
    # Missing value -> low_data flag set, present value -> not.
    assert canon.loc["999999", "low_data"] == 1.0
    assert canon.loc["342229", "low_data"] == 0.0


def test_canonical_player_id_matches_registry_key():
    reg = CanonicalRegistry()
    canon = to_canonical(parse_squad(_html()), reg).set_index("transfermarkt_id")
    # The accented name resolves to the same id as its accent-folded form.
    assert canon.loc["342229", "player_id"] == reg.player("kylian mbappe")["player_id"]
