"""Tests for the Understat parser + canonical mapping, and the cache-first fetcher.

The parser is exercised against a saved sample page (no network), per the scraping
contract.
"""

from pathlib import Path

import numpy as np
import pytest

from sports_predictor.canonical.registry import CanonicalRegistry
from sports_predictor.scrapers.base import CachedFetcher, FetchConfig
from sports_predictor.scrapers.understat import MIN_MINUTES, parse_players, to_canonical

FIXTURE = Path(__file__).parent / "fixtures" / "understat_league_sample.html"


def _sample_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Parser schema + decoding
# --------------------------------------------------------------------------- #
def test_parser_schema_and_rows():
    df = parse_players(_sample_html())
    assert list(df.columns) == [
        "source", "understat_id", "player_name", "team_title",
        "games", "minutes", "goals", "assists", "xg", "xa",
    ]
    assert len(df) == 3
    assert (df["source"] == "understat").all()


def test_parser_decodes_accents_and_numbers():
    df = parse_players(_sample_html()).set_index("understat_id")
    assert df.loc["001", "player_name"] == "Kylian Mbappé"  # \xc3\xa9 decoded
    assert df.loc["001", "minutes"] == 2600
    assert df.loc["001", "xg"] == pytest.approx(25.4)


def test_parser_raises_on_missing_blob():
    with pytest.raises(ValueError):
        parse_players("<html><body>no data here</body></html>")


# --------------------------------------------------------------------------- #
# Canonical mapping + one feature
# --------------------------------------------------------------------------- #
def test_to_canonical_adds_player_id_and_xg_per_90():
    df = parse_players(_sample_html())
    canon = to_canonical(df).set_index("understat_id")
    # xg_per_90 = xg / (minutes/90); Mbappé: 25.4 / (2600/90)
    assert canon.loc["001", "xg_per_90"] == pytest.approx(25.4 / (2600 / 90))
    assert canon.loc["001", "player_id"].startswith("plr_")


def test_low_minutes_player_gets_nan_not_zero():
    df = parse_players(_sample_html())
    canon = to_canonical(df).set_index("understat_id")
    # 80 minutes < MIN_MINUTES -> per-90 is unreliable -> NaN + low_data flag.
    assert 80 < MIN_MINUTES
    assert np.isnan(canon.loc["003", "xg_per_90"])
    assert canon.loc["003", "low_data"] == 1.0
    assert canon.loc["001", "low_data"] == 0.0


def test_canonical_ids_are_consistent_across_runs():
    df = parse_players(_sample_html())
    reg = CanonicalRegistry()
    a = to_canonical(df, reg).set_index("understat_id")["player_id"]
    b = to_canonical(df, reg).set_index("understat_id")["player_id"]
    assert (a == b).all()


# --------------------------------------------------------------------------- #
# Cache-first fetcher (no network)
# --------------------------------------------------------------------------- #
def test_fetch_is_cache_first(tmp_path):
    fetcher = CachedFetcher("understat", raw_dir=tmp_path, config=FetchConfig(delay_seconds=0))
    # Pre-seed the cache; fetch must return it WITHOUT any network call.
    slug = "epl_2023"
    fetcher.cache_path(slug).write_bytes(b"<html>cached</html>")
    path = fetcher.fetch("http://unreachable.invalid/none", slug)
    assert path.read_bytes() == b"<html>cached</html>"


def test_download_retries_then_raises(tmp_path):
    fetcher = CachedFetcher(
        "understat", raw_dir=tmp_path,
        config=FetchConfig(delay_seconds=0, max_retries=2, backoff_factor=1.0),
    )
    calls = {"n": 0}

    def boom(url):
        calls["n"] += 1
        raise RuntimeError("network down")

    fetcher._download = boom  # force failures
    with pytest.raises(RuntimeError):
        fetcher.fetch("http://unreachable.invalid/x", "missing")
