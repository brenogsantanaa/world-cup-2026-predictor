"""Tournament structures (groups + knockout bracket).

A tournament is described by:

- ``groups``: mapping of group name -> list of four team names (canonical
  spellings, matching the results dataset).
- ``r16``: the Round-of-16 ties, as pairs of *slots*. A slot is ``(group, rank)``
  where rank 1 is the group winner and 2 the runner-up. The list is ordered so
  that pairing adjacent winners through the rounds reproduces the real bracket
  (ties 0&1 meet in a quarter-final, 2&3, etc.).

Encoded here: the 2022 FIFA World Cup (32 teams), used to validate the simulator
by backtesting a tournament whose result we already know.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tournament:
    """A World Cup structure.

    Two formats are supported:

    - **32-team (1998-2022):** eight groups, top two advance, ``r16`` holds the
      Round-of-16 slot template. ``r32``/``bracket`` stay ``None``.
    - **48-team (2026+):** twelve groups, top two **plus the eight best
      third-placed teams** advance to a Round of 32. ``r32`` lists the 16 ties as
      slot specs (see :data:`WC_2026`), ``third_place_slots`` says which r32 ties
      are filled by a best-third, and ``bracket`` gives the explicit
      winner-of-match wiring for every later round (the real 2026 bracket is *not*
      a simple adjacent-pairing, so it must be encoded literally).

    ``as_of`` records when the field/draw was accurate, since late details (e.g.
    play-off winners) can change.
    """

    name: str
    groups: dict[str, list[str]]
    r16: list[tuple[tuple[str, int], tuple[str, int]]] | None = None
    as_of: str | None = None
    # 48-team format only:
    r32: list[tuple] | None = None
    third_place_slots: tuple[int, ...] = field(default_factory=tuple)
    bracket: tuple[tuple[tuple[int, int], ...], ...] = field(default_factory=tuple)
    n_thirds: int = 8

    @property
    def teams(self) -> list[str]:
        return [team for group in self.groups.values() for team in group]

    @property
    def is_48(self) -> bool:
        return self.r32 is not None


# The 32-team Round-of-16 slot template has been the same since 1998, so all the
# 32-team World Cups below reuse it; only the group compositions differ.
_R16_32 = [
    (("A", 1), ("B", 2)),
    (("C", 1), ("D", 2)),
    (("E", 1), ("F", 2)),
    (("G", 1), ("H", 2)),
    (("B", 1), ("A", 2)),
    (("D", 1), ("C", 2)),
    (("F", 1), ("E", 2)),
    (("H", 1), ("G", 2)),
]


WC_2022 = Tournament(
    name="2022 FIFA World Cup",
    groups={
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    },
    r16=_R16_32,
)

WC_2018 = Tournament(
    name="2018 FIFA World Cup",
    groups={
        "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        "B": ["Portugal", "Spain", "Morocco", "Iran"],
        "C": ["France", "Australia", "Peru", "Denmark"],
        "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
        "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        "F": ["Germany", "Mexico", "Sweden", "South Korea"],
        "G": ["Belgium", "Panama", "Tunisia", "England"],
        "H": ["Poland", "Senegal", "Colombia", "Japan"],
    },
    r16=_R16_32,
)

WC_2014 = Tournament(
    name="2014 FIFA World Cup",
    groups={
        "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
        "B": ["Spain", "Netherlands", "Chile", "Australia"],
        "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
        "D": ["Uruguay", "Costa Rica", "England", "Italy"],
        "E": ["Switzerland", "Ecuador", "France", "Honduras"],
        "F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
        "G": ["Germany", "Portugal", "Ghana", "United States"],
        "H": ["Belgium", "Algeria", "Russia", "South Korea"],
    },
    r16=_R16_32,
)

WC_2010 = Tournament(
    name="2010 FIFA World Cup",
    groups={
        "A": ["South Africa", "Mexico", "Uruguay", "France"],
        "B": ["Argentina", "Nigeria", "South Korea", "Greece"],
        "C": ["England", "United States", "Algeria", "Slovenia"],
        "D": ["Germany", "Australia", "Serbia", "Ghana"],
        "E": ["Netherlands", "Denmark", "Japan", "Cameroon"],
        "F": ["Italy", "Paraguay", "New Zealand", "Slovakia"],
        "G": ["Brazil", "North Korea", "Ivory Coast", "Portugal"],
        "H": ["Spain", "Switzerland", "Honduras", "Chile"],
    },
    r16=_R16_32,
)


# --------------------------------------------------------------------------- #
# 2026 FIFA World Cup (48 teams, first of its kind)
# --------------------------------------------------------------------------- #
# Round-of-32 ties, exactly as published in FIFA's bracket. Each slot is one of:
#   ("W", G)  group G winner      ("R", G)  group G runner-up      ("3",)  a best-third
# The eight "3" slots are filled, in the order of ``_R32_THIRD_SLOTS`` below, from
# the best third-placed teams. NOTE: the precise FIFA combination table that maps
# *which* group's third lands in *which* slot (one of 495 cases) is approximated by
# seeding best->worst with same-group avoidance; this only affects R32 matchups and
# is a documented, easy-to-revise choice. See simulation._assign_thirds.
_R32_2026 = [
    (("R", "A"), ("R", "B")),   # M1
    (("W", "E"), ("3",)),       # M2   host E
    (("W", "F"), ("R", "C")),   # M3
    (("W", "C"), ("R", "F")),   # M4
    (("W", "I"), ("3",)),       # M5   host I
    (("R", "E"), ("R", "I")),   # M6
    (("W", "A"), ("3",)),       # M7   host A
    (("W", "L"), ("3",)),       # M8   host L
    (("W", "D"), ("3",)),       # M9   host D
    (("W", "G"), ("3",)),       # M10  host G
    (("R", "K"), ("R", "L")),   # M11
    (("W", "H"), ("R", "J")),   # M12
    (("W", "B"), ("3",)),       # M13  host B
    (("W", "J"), ("R", "H")),   # M14
    (("W", "K"), ("3",)),       # M15  host K
    (("R", "D"), ("R", "G")),   # M16
]
# r32 indices (0-based) that carry a "3" slot, in fill order.
_R32_THIRD_SLOTS = (1, 4, 6, 7, 8, 9, 12, 14)

# Explicit winner-of-match wiring for each round after the R32 (0-based indices
# into the previous round). Transcribed from FIFA's published bracket.
_BRACKET_2026 = (
    # Round of 16 (M1..M8), pairing R32 winners:
    ((1, 4), (0, 2), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)),
    # Quarter-finals, pairing R16 winners:
    ((0, 1), (4, 5), (2, 3), (6, 7)),
    # Semi-finals, pairing QF winners:
    ((0, 1), (2, 3)),
    # Final:
    ((0, 1),),
)

WC_2026 = Tournament(
    name="2026 FIFA World Cup",
    as_of="2026-04-23",  # final field confirmed after March 2026 play-offs
    groups={
        "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
        "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
        "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
        "D": ["United States", "Paraguay", "Australia", "Turkey"],
        "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
        "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
        "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
        "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
        "I": ["France", "Senegal", "Norway", "Iraq"],
        "J": ["Argentina", "Austria", "Algeria", "Jordan"],
        "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
        "L": ["England", "Croatia", "Panama", "Ghana"],
    },
    r32=_R32_2026,
    third_place_slots=_R32_THIRD_SLOTS,
    bracket=_BRACKET_2026,
    n_thirds=8,
)
