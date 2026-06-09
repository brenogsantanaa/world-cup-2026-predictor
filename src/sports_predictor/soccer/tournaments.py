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

from dataclasses import dataclass


@dataclass(frozen=True)
class Tournament:
    name: str
    groups: dict[str, list[str]]
    r16: list[tuple[tuple[str, int], tuple[str, int]]]

    @property
    def teams(self) -> list[str]:
        return [team for group in self.groups.values() for team in group]


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
