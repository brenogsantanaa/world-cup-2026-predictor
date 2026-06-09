"""API-Football (api-sports.io) client + parser for live 2026 club data.

This is a **forward-looking** source: current club form is valid input for the
2026 World Cup prediction (it exists before those matches), so it is *not*
leakage. It must NEVER feed the historical backtest -- that stays on the
international-only goalscorers layer.

Design (mirrors the project's scraper contract, adapted for a quota'd JSON API):

- **Key from env** (``API_FOOTBALL_KEY``); the client refuses to make live calls
  without it. Never hardcode the key.
- **Cache-first:** every response is saved verbatim to
  ``data/raw/apifootball/<slug>.json`` with a provenance manifest (endpoint,
  params, UTC ts, bytes, SHA-256). Re-runs read the cache and never burn quota.
- **Quota guard:** the free tier allows ~100 requests/day. A small state file
  tracks today's count; once the limit is hit the client raises
  :class:`QuotaExceeded` cleanly. Counts reset on a new UTC day.
- **Resumable:** because cache hits don't consume quota, a multi-day pull simply
  re-runs and skips everything already cached.
- **Polite:** inter-request throttle + retry-with-backoff on 429/5xx.

Parsing is separate from fetching, so parsers are tested against saved fixtures.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sports_predictor.canonical.registry import CanonicalRegistry
from sports_predictor.core.paths import RAW_DIR, ensure_dir

SOURCE = "apifootball"
MIN_MINUTES = 450  # below this, per-90 rates are unreliable -> low_data


class MissingAPIKey(RuntimeError):
    """Raised when a live call is attempted without API_FOOTBALL_KEY set."""


class QuotaExceeded(RuntimeError):
    """Raised when the daily free-tier request limit has been reached."""


class APIError(RuntimeError):
    """Raised when the API returns an error payload."""


@dataclass
class APIFootballConfig:
    base_url: str = "https://v3.football.api-sports.io"
    daily_limit: int = 100
    delay_seconds: float = 6.0
    max_retries: int = 3
    backoff_factor: float = 2.0
    timeout: int = 30


def _slug(endpoint: str, params: dict) -> str:
    parts = [endpoint.strip("/").replace("/", "_")]
    for key in sorted(params):
        parts.append(f"{key}-{params[key]}")
    raw = "__".join(parts)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
    if len(safe) > 150:  # keep filenames sane
        safe = safe[:120] + "__" + hashlib.sha1(raw.encode()).hexdigest()[:12]
    return safe


class APIFootballClient:
    """Cache-first, quota-aware client for api-sports.io football endpoints."""

    def __init__(self, raw_dir: Path = RAW_DIR, config: APIFootballConfig | None = None,
                 api_key: str | None = None):
        self.config = config or APIFootballConfig()
        self.api_key = api_key if api_key is not None else os.environ.get("API_FOOTBALL_KEY")
        self.dir = ensure_dir(Path(raw_dir) / SOURCE)
        self.state_path = self.dir / "_quota_state.json"
        self._last_request_ts = 0.0

    # ----- cache paths ----------------------------------------------------- #
    def cache_path(self, endpoint: str, params: dict) -> Path:
        return self.dir / f"{_slug(endpoint, params)}.json"

    def is_cached(self, endpoint: str, params: dict) -> bool:
        return self.cache_path(endpoint, params).exists()

    # ----- quota ----------------------------------------------------------- #
    def _today(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"date": self._today(), "count": 0}
        state = json.loads(self.state_path.read_text())
        if state.get("date") != self._today():  # new UTC day -> reset
            return {"date": self._today(), "count": 0}
        return state

    def remaining_quota(self) -> int:
        return max(0, self.config.daily_limit - self._load_state()["count"])

    def _increment_quota(self) -> None:
        state = self._load_state()
        state["count"] += 1
        self.state_path.write_text(json.dumps(state))

    # ----- fetch ----------------------------------------------------------- #
    def get(self, endpoint: str, params: dict | None = None, force: bool = False) -> dict:
        """Return the JSON for ``endpoint``+``params``, fetching only if needed."""
        params = params or {}
        path = self.cache_path(endpoint, params)
        if path.exists() and not force:
            return json.loads(path.read_text())

        if not self.api_key:
            raise MissingAPIKey(
                "API_FOOTBALL_KEY is not set; cannot make live API-Football calls. "
                "Export your key or pre-populate the cache."
            )
        if self.remaining_quota() <= 0:
            raise QuotaExceeded(
                f"daily limit of {self.config.daily_limit} requests reached; "
                f"resume tomorrow (already-cached items will be skipped)."
            )

        payload = self._download(endpoint, params)
        self._increment_quota()

        data = json.loads(payload)
        if data.get("errors"):
            raise APIError(f"API-Football returned errors for {endpoint}: {data['errors']}")

        path.write_text(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
        self._write_manifest(endpoint, params, payload)
        return data

    def _throttle(self) -> None:
        wait = self.config.delay_seconds - (time.monotonic() - self._last_request_ts)
        if wait > 0:
            time.sleep(wait)

    def _download(self, endpoint: str, params: dict) -> bytes:
        query = urllib.parse.urlencode(params)
        url = f"{self.config.base_url}/{endpoint.strip('/')}"
        if query:
            url = f"{url}?{query}"
        headers = {"x-apisports-key": self.api_key, "Accept": "application/json"}

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            self._throttle()
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    payload = response.read()
                self._last_request_ts = time.monotonic()
                return payload
            except urllib.error.HTTPError as exc:
                last_error = exc
                self._last_request_ts = time.monotonic()
                if exc.code == 429 and attempt < self.config.max_retries - 1:
                    time.sleep(self.config.backoff_factor ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                self._last_request_ts = time.monotonic()
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.backoff_factor ** attempt)
        raise RuntimeError(f"failed to fetch {url}") from last_error

    def _write_manifest(self, endpoint: str, params: dict, payload: bytes) -> None:
        slug = _slug(endpoint, params)
        manifest = self.dir / f"{slug}.source.txt"
        sha = hashlib.sha256(payload if isinstance(payload, bytes) else payload.encode()).hexdigest()
        manifest.write_text(
            f"source: {SOURCE}\n"
            f"endpoint: {endpoint}\n"
            f"params: {json.dumps(params, sort_keys=True)}\n"
            f"downloaded_at_utc: {datetime.now(timezone.utc).isoformat()}\n"
            f"bytes: {len(payload)}\n"
            f"sha256: {sha}\n"
        )

    # ----- resumable batch convenience ------------------------------------- #
    def fetch_player_statistics(self, player_id: int, season: int) -> dict:
        return self.get("players", {"id": player_id, "season": season})

    def fetch_many_players(self, player_ids, season: int) -> dict:
        """Fetch many players, skipping cached ones and stopping cleanly on quota.

        Returns a small report: which ids were cached/fetched/remaining so a
        multi-day run can pick up exactly where it left off.
        """
        cached, fetched, remaining = [], [], []
        for pid in player_ids:
            if self.is_cached("players", {"id": pid, "season": season}):
                cached.append(pid)
                continue
            try:
                self.fetch_player_statistics(pid, season)
                fetched.append(pid)
            except QuotaExceeded:
                remaining = [p for p in player_ids if p not in cached and p not in fetched]
                break
        return {"cached": cached, "fetched": fetched, "remaining": remaining}


# ===== parsing (separate from fetching) ==================================== #
def parse_players(data: dict) -> pd.DataFrame:
    """Parse a ``/players`` response into a per (player, club-season) DataFrame.

    Columns: source, apifootball_player_id, player_name, nationality, club,
    apifootball_club_id, league, season, appearances, minutes, position, goals,
    assists. Numeric fields coerced; missing values stay NaN.
    """
    rows = []
    for item in data.get("response", []):
        player = item.get("player", {})
        for stat in item.get("statistics", []):
            team = stat.get("team", {})
            league = stat.get("league", {})
            games = stat.get("games", {})
            goals = stat.get("goals", {})
            rows.append(
                {
                    "source": SOURCE,
                    "apifootball_player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "nationality": player.get("nationality"),
                    "club": team.get("name"),
                    "apifootball_club_id": team.get("id"),
                    "league": league.get("name"),
                    "season": league.get("season"),
                    "appearances": games.get("appearences"),
                    "minutes": games.get("minutes"),
                    "position": games.get("position"),
                    "goals": goals.get("total"),
                    "assists": goals.get("assists"),
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "source", "apifootball_player_id", "player_name", "nationality",
            "club", "apifootball_club_id", "league", "season",
            "appearances", "minutes", "position", "goals", "assists",
        ],
    )
    for col in ("appearances", "minutes", "goals", "assists"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def to_canonical(players: pd.DataFrame, registry: CanonicalRegistry | None = None) -> pd.DataFrame:
    """Attach canonical ``player_id`` and one feature (``goals_per_90``).

    ``goals_per_90`` is NaN with ``low_data = 1`` below :data:`MIN_MINUTES` (and
    when minutes are missing) -- never a fabricated zero.
    """
    registry = registry or CanonicalRegistry()
    out = players.copy()
    out["player_id"] = out["player_name"].map(
        lambda n: registry.player(n)["player_id"] if isinstance(n, str) and n else None
    )
    enough = out["minutes"] >= MIN_MINUTES
    out["goals_per_90"] = (out["goals"] / (out["minutes"] / 90.0)).where(enough)
    out["low_data"] = (~enough.fillna(False)).astype(float)
    return out
