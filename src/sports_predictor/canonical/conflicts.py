"""Conflict resolution when sources disagree on a field.

Different sources measure the same thing with different rigour, so we pick a
*documented* winner per field rather than averaging blindly:

- match stats (minutes, goals, assists, xG/xA): FBref is the most detailed, then
  Understat (shot-level, top leagues only), then the CC0 backbone.
- market value / squad metadata (value, age, position): Transfermarkt.

When two sources actually disagree on a value we still take the priority source,
but we **log the disagreement** so it is auditable instead of silently dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Per-field source priority (highest trust first). Sources not listed for a field
# are still usable as a last resort, after the listed ones.
SOURCE_PRIORITY: dict[str, list[str]] = {
    "minutes": ["fbref", "understat", "transfermarkt"],
    "goals": ["fbref", "understat", "martj42", "transfermarkt"],
    "assists": ["fbref", "understat"],
    "xg": ["understat", "fbref"],
    "xa": ["understat", "fbref"],
    "market_value": ["transfermarkt", "fbref"],
    "age": ["transfermarkt", "fbref"],
    "position": ["transfermarkt", "fbref"],
    "appearances": ["fbref", "transfermarkt"],
}


@dataclass
class Disagreement:
    field: str
    chosen_source: str
    chosen_value: Any
    values: dict[str, Any]  # source -> value


@dataclass
class ConflictLog:
    """Collects field-level disagreements for auditing."""

    entries: list[Disagreement] = field(default_factory=list)

    def record(self, d: Disagreement) -> None:
        self.entries.append(d)

    def __len__(self) -> int:
        return len(self.entries)

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.field] = out.get(e.field, 0) + 1
        return out


def _values_agree(values) -> bool:
    """True if all present values are equal (numbers compared with a tolerance)."""
    present = [v for v in values if v is not None and not _is_nan(v)]
    if len(present) <= 1:
        return True
    first = present[0]
    if all(isinstance(v, (int, float)) for v in present):
        return all(math.isclose(v, first, rel_tol=1e-6, abs_tol=1e-9) for v in present)
    return all(v == first for v in present)


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def resolve(
    field_name: str,
    candidates: dict[str, Any],
    log: ConflictLog | None = None,
) -> tuple[Any, str | None]:
    """Resolve a field from ``{source: value}`` by documented priority.

    Returns ``(value, source)``. Sources with ``None``/NaN values are skipped.
    The priority list is consulted first; any remaining sources are tried in a
    stable order afterward. If present values disagree, the disagreement is logged
    (when a ``log`` is given) but the priority source still wins.
    """
    usable = {s: v for s, v in candidates.items() if v is not None and not _is_nan(v)}
    if not usable:
        return None, None

    priority = SOURCE_PRIORITY.get(field_name, [])
    ordered = [s for s in priority if s in usable] + [
        s for s in sorted(usable) if s not in priority
    ]
    chosen_source = ordered[0]
    chosen_value = usable[chosen_source]

    if log is not None and not _values_agree(list(usable.values())):
        log.record(
            Disagreement(
                field=field_name,
                chosen_source=chosen_source,
                chosen_value=chosen_value,
                values=dict(usable),
            )
        )
    return chosen_value, chosen_source
