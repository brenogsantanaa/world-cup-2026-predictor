"""Tests for national-team squad assembly and the club->nation linchpin."""

from pathlib import Path

import numpy as np
import pytest

from sports_predictor.canonical.registry import CanonicalRegistry
from sports_predictor.scrapers import fbref, transfermarkt
from sports_predictor.soccer.squads import (
    assemble_squads,
    attach_nation,
    player_nation_map,
    squad_value,
)

FIX = Path(__file__).parent / "fixtures"


def _squads(reg=None):
    france = transfermarkt.parse_squad((FIX / "transfermarkt_squad_sample.html").read_text("utf-8"))
    brazil = transfermarkt.parse_squad((FIX / "transfermarkt_squad_brazil.html").read_text("utf-8"))
    return assemble_squads([france, brazil], reg)


def test_assemble_two_nations():
    squads = _squads()
    assert len(squads) == 5  # 3 France + 2 Brazil
    assert squads["team"].nunique() == 2
    assert set(squads["team"]) == {"France", "Brazil"}


def test_player_nation_map_links_player_to_nation():
    reg = CanonicalRegistry()
    squads = _squads(reg)
    mapping = player_nation_map(squads)
    mbappe_id = reg.player("Kylian Mbappé")["player_id"]
    assert mapping[mbappe_id] == reg.team("France")["team_id"]


def test_attach_nation_links_club_stats_to_country():
    # FBref club stats (keyed by player_id) gain a nation via the squad map.
    reg = CanonicalRegistry()
    squads = _squads(reg)
    fb = fbref.to_canonical(
        fbref.parse_player_stats((FIX / "fbref_stats_sample.html").read_text("utf-8")), reg
    )
    linked = attach_nation(fb, squads).set_index("fbref_id")
    assert linked.loc["42fd9c7f", "team"] == "France"  # Mbappé
    assert linked.loc["42fd9c7f", "in_squad"] == 1.0


def test_player_not_in_any_squad_gets_nan_not_fabricated():
    reg = CanonicalRegistry()
    squads = _squads(reg)
    import pandas as pd

    # A club player absent from every national squad.
    stub = pd.DataFrame(
        {"player_id": [reg.player("Totally Unknown Person")["player_id"]], "xg_per_90": [0.5]}
    )
    linked = attach_nation(stub, squads)
    assert np.isnan(linked.loc[0, "team_id"]) or linked.loc[0, "team_id"] is None
    assert linked.loc[0, "in_squad"] == 0.0


def test_squad_value_aggregates_with_coverage():
    squads = _squads()
    val = squad_value(squads).set_index("team")
    # France: 180m + 80m known, youngster has no value -> coverage 2/3.
    assert val.loc["France", "total_value_eur"] == pytest.approx(260_000_000.0)
    assert val.loc["France", "n_players"] == 3
    assert val.loc["France", "coverage"] == pytest.approx(2 / 3)
    # Brazil: both valued -> full coverage, not flagged.
    assert val.loc["Brazil", "total_value_eur"] == pytest.approx(218_000_000.0)
    assert val.loc["Brazil", "low_data"] == 0.0
