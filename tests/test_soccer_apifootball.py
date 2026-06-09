"""Tests for the API-Football client (env key, cache-first, quota, resume) and
the /players parser + canonical mapping. No network is used."""

import json
from pathlib import Path

import numpy as np
import pytest

from sports_predictor.canonical.registry import CanonicalRegistry
from sports_predictor.soccer.apifootball import (
    APIFootballClient,
    APIFootballConfig,
    MissingAPIKey,
    QuotaExceeded,
    parse_players,
    to_canonical,
)

FIXTURE = Path(__file__).parent / "fixtures" / "apifootball_players_sample.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Client: auth, cache-first, quota, resume
# --------------------------------------------------------------------------- #
def test_refuses_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    client = APIFootballClient(raw_dir=tmp_path, api_key=None)
    with pytest.raises(MissingAPIKey):
        client.get("players", {"id": 276, "season": 2023})


def test_cache_first_returns_without_network(tmp_path):
    client = APIFootballClient(raw_dir=tmp_path, api_key="dummy")
    params = {"id": 276, "season": 2023}
    client.cache_path("players", params).write_text(json.dumps({"response": ["cached"]}))

    def boom(*a, **k):
        raise AssertionError("network must not be called on a cache hit")

    client._download = boom
    assert client.get("players", params) == {"response": ["cached"]}


def test_quota_guard_stops_then_resume_skips_cached(tmp_path):
    client = APIFootballClient(
        raw_dir=tmp_path, api_key="dummy",
        config=APIFootballConfig(daily_limit=2, delay_seconds=0),
    )
    # Fake a successful, error-free response for any live call.
    client._download = lambda endpoint, params: json.dumps(
        {"errors": [], "response": [{"player": {"id": params["id"]}, "statistics": []}]}
    ).encode()

    report = client.fetch_many_players([1, 2, 3, 4], season=2023)
    # Only 2 live fetches allowed; the rest are deferred, not failed.
    assert report["fetched"] == [1, 2]
    assert report["remaining"] == [3, 4]
    assert client.remaining_quota() == 0

    # A later call to an already-cached id must NOT consume quota or hit network.
    client._download = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network"))
    assert "player" in client.get("players", {"id": 1, "season": 2023})["response"][0]

    # Uncached id now raises cleanly (quota exhausted).
    with pytest.raises(QuotaExceeded):
        client.get("players", {"id": 99, "season": 2023})


def test_quota_resets_on_new_utc_day(tmp_path):
    client = APIFootballClient(
        raw_dir=tmp_path, api_key="dummy", config=APIFootballConfig(daily_limit=5)
    )
    client.state_path.write_text(json.dumps({"date": "2000-01-01", "count": 5}))
    assert client.remaining_quota() == 5  # stale day -> reset


def test_raw_cached_and_manifest_written(tmp_path):
    client = APIFootballClient(raw_dir=tmp_path, api_key="dummy",
                              config=APIFootballConfig(delay_seconds=0))
    client._download = lambda endpoint, params: json.dumps(
        {"errors": [], "response": []}
    ).encode()
    client.get("players", {"id": 7, "season": 2023})
    assert client.cache_path("players", {"id": 7, "season": 2023}).exists()
    manifest = tmp_path / "apifootball" / "players__id-7__season-2023.source.txt"
    assert manifest.exists()
    assert "sha256:" in manifest.read_text()


# --------------------------------------------------------------------------- #
# Parser + canonical mapping
# --------------------------------------------------------------------------- #
def test_parser_schema_and_values():
    df = parse_players(_fixture()).set_index("apifootball_player_id")
    assert list(df.columns) == [
        "source", "player_name", "nationality", "club", "apifootball_club_id",
        "league", "season", "appearances", "minutes", "position", "goals", "assists",
    ]
    assert df.loc[276, "player_name"] == "Kylian Mbappé"
    assert df.loc[276, "minutes"] == 2600
    assert df.loc[276, "goals"] == 28


def test_to_canonical_player_id_and_goals_per_90():
    canon = to_canonical(parse_players(_fixture())).set_index("apifootball_player_id")
    assert canon.loc[276, "player_id"].startswith("plr_")
    assert canon.loc[276, "goals_per_90"] == pytest.approx(28 / (2600 / 90))


def test_low_minutes_player_gets_nan_not_zero():
    canon = to_canonical(parse_players(_fixture())).set_index("apifootball_player_id")
    assert np.isnan(canon.loc[99999, "goals_per_90"])
    assert canon.loc[99999, "low_data"] == 1.0
    assert canon.loc[276, "low_data"] == 0.0


def test_canonical_id_matches_other_sources():
    # Same player from API-Football and FBref resolves to one player_id.
    from sports_predictor.scrapers import fbref

    reg = CanonicalRegistry()
    af = to_canonical(parse_players(_fixture()), reg).set_index("apifootball_player_id")
    fb = fbref.to_canonical(
        fbref.parse_player_stats(
            (FIXTURE.parent / "fbref_stats_sample.html").read_text("utf-8")
        ),
        reg,
    ).set_index("fbref_id")
    assert af.loc[276, "player_id"] == fb.loc["42fd9c7f", "player_id"]
