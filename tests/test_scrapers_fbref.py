"""Tests for the FBref stats parser + canonical mapping (no network).

Exercises the two FBref quirks: tables hidden in HTML comments, and reading cells
by ``data-stat`` while skipping repeated header rows.
"""

from pathlib import Path

import numpy as np
import pytest

from sports_predictor.canonical.registry import CanonicalRegistry
from sports_predictor.scrapers.fbref import MIN_MINUTES, parse_player_stats, to_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "fbref_stats_sample.html"


def _html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_parses_table_hidden_in_comment():
    # The fixture's table lives inside an HTML comment; it must still be found.
    df = parse_player_stats(_html())
    assert len(df) == 3  # repeated mid-table header row is skipped
    assert list(df.columns) == [
        "source", "fbref_id", "player_name", "nationality", "position",
        "minutes", "goals", "assists", "xg", "xa",
    ]


def test_values_and_numeric_coercion():
    df = parse_player_stats(_html()).set_index("fbref_id")
    mbappe = df.loc["42fd9c7f"]
    assert mbappe["player_name"] == "Kylian Mbappé"
    assert mbappe["minutes"] == 2600  # "2,600" -> 2600
    assert mbappe["goals"] == 28
    assert mbappe["xg"] == pytest.approx(25.4)
    assert mbappe["xa"] == pytest.approx(6.1)


def test_repeated_header_row_skipped():
    df = parse_player_stats(_html())
    assert "Player" not in set(df["player_name"])


def test_raises_when_table_absent():
    with pytest.raises(ValueError):
        parse_player_stats("<html><body>nothing</body></html>")


def test_to_canonical_xg_per_90_and_low_data():
    df = parse_player_stats(_html())
    canon = to_canonical(df).set_index("fbref_id")
    assert canon.loc["42fd9c7f", "xg_per_90"] == pytest.approx(25.4 / (2600 / 90))
    assert canon.loc["42fd9c7f", "player_id"].startswith("plr_")
    # 90 minutes < MIN_MINUTES -> NaN + flag, not a fabricated rate.
    assert 90 < MIN_MINUTES
    assert np.isnan(canon.loc["99999999", "xg_per_90"])
    assert canon.loc["99999999", "low_data"] == 1.0


def test_canonical_id_consistent_with_understat_and_transfermarkt():
    # Same person across sources must resolve to the SAME player_id.
    from sports_predictor.scrapers import transfermarkt, understat

    reg = CanonicalRegistry()
    fb = to_canonical(parse_player_stats(_html()), reg).set_index("fbref_id")
    tm = transfermarkt.to_canonical(
        transfermarkt.parse_squad(
            (FIXTURE.parent / "transfermarkt_squad_sample.html").read_text(encoding="utf-8")
        ),
        reg,
    ).set_index("transfermarkt_id")
    assert fb.loc["42fd9c7f", "player_id"] == tm.loc["342229", "player_id"]
