"""Cache-first, polite HTTP fetching shared by all site scrapers.

Responsibilities (and *only* these -- parsing lives in each site module):

- **Cache-first:** if the page is already cached, return it without any network
  call, so pipelines are reproducible and we don't re-hit sites.
- **Polite:** a configurable inter-request delay, a real User-Agent, and
  retry-with-exponential-backoff on transient failures.
- **Auditable:** every download is saved verbatim with a provenance manifest
  (URL, UTC timestamp, byte count, SHA-256).

Note on terms of use: some sites restrict automated access. Keep the delay
conservative, cache aggressively, and only fetch what you need.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sports_predictor.core.paths import RAW_DIR, ensure_dir


@dataclass
class FetchConfig:
    delay_seconds: float = 2.0      # minimum gap between live requests
    max_retries: int = 3
    backoff_factor: float = 2.0     # sleep = backoff_factor ** attempt
    timeout: int = 30
    user_agent: str = "sports-predictor/0.1 (research project; polite, cached)"


class CachedFetcher:
    """Fetch pages for one site, caching verbatim to ``data/raw/<site>/``."""

    def __init__(self, site: str, raw_dir: Path = RAW_DIR, config: FetchConfig | None = None):
        self.site = site
        self.dir = ensure_dir(Path(raw_dir) / site)
        self.config = config or FetchConfig()
        self._last_request_ts = 0.0

    def cache_path(self, slug: str) -> Path:
        return self.dir / f"{slug}.html"

    def is_cached(self, slug: str) -> bool:
        return self.cache_path(slug).exists()

    def read(self, slug: str) -> bytes:
        return self.cache_path(slug).read_bytes()

    def fetch(self, url: str, slug: str, force: bool = False) -> Path:
        """Return the cached path for ``url``, downloading only if needed.

        ``slug`` is the cache filename stem. With ``force=True`` the page is
        re-downloaded even if cached.
        """
        path = self.cache_path(slug)
        if path.exists() and not force:
            return path

        payload = self._download(url)
        path.write_bytes(payload)
        self._write_manifest(slug, url, payload)
        return path

    # ----- internals ------------------------------------------------------- #
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        wait = self.config.delay_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

    def _download(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            self._throttle()
            request = urllib.request.Request(url, headers={"User-Agent": self.config.user_agent})
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status} from {url}")
                    payload = response.read()
                self._last_request_ts = time.monotonic()
                return payload
            except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
                last_error = exc
                self._last_request_ts = time.monotonic()
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.backoff_factor ** attempt)
        raise RuntimeError(f"failed to fetch {url} after {self.config.max_retries} attempts") from last_error

    def _write_manifest(self, slug: str, url: str, payload: bytes) -> None:
        manifest = self.dir / f"{slug}.source.txt"
        manifest.write_text(
            f"site: {self.site}\n"
            f"url: {url}\n"
            f"downloaded_at_utc: {datetime.now(timezone.utc).isoformat()}\n"
            f"bytes: {len(payload)}\n"
            f"sha256: {hashlib.sha256(payload).hexdigest()}\n"
        )
