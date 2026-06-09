"""Site scrapers.

One isolated module per site (``understat``, later ``transfermarkt``, ``fbref``)
so a change to one site's HTML can never break another. Every module follows the
same contract:

1. **Fetching is separate from parsing.** :class:`base.CachedFetcher` downloads a
   page once, verbatim, to ``data/raw/<site>/`` with a provenance manifest, and is
   cache-first (re-runs read the cache and never re-hit the site).
2. **Parsers take bytes/str and return a DataFrame** with a fixed schema, so they
   can be tested against a saved sample page with no network.

This keeps runs reproducible and respects the sites (rate-limited, cached).
"""
