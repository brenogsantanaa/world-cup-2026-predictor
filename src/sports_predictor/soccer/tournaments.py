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
    # Ordered so sequential pairing of winners rebuilds the real bracket.
    r16=[
        (("A", 1), ("B", 2)),
        (("C", 1), ("D", 2)),
        (("E", 1), ("F", 2)),
        (("G", 1), ("H", 2)),
        (("B", 1), ("A", 2)),
        (("D", 1), ("C", 2)),
        (("F", 1), ("E", 2)),
        (("H", 1), ("G", 2)),
    ],
)
