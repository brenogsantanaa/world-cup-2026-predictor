"""Canonical reconciliation layer.

Every external source (martj42 CC0 data, Transfermarkt, FBref, Understat, squad
lists) spells teams and players differently and measures things slightly
differently. This package is the single place that turns all of that into one
vocabulary:

- :mod:`names` — accent/transliteration-safe name keys for matching.
- :mod:`registry` — stable ``team_id`` / ``player_id`` with alias merging.
- :mod:`conflicts` — when sources disagree on a field, resolve by a documented
  per-field source priority and *log* the disagreement instead of hiding it.

Nothing here touches the network; it is pure, deterministic, and heavily tested,
so the rest of the pipeline can rely on consistent IDs and metric definitions.
"""
