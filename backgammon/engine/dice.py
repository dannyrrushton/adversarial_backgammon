"""Dice rolls."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Roll:
    d1: int
    d2: int

    def __post_init__(self) -> None:
        if not (1 <= self.d1 <= 6 and 1 <= self.d2 <= 6):
            raise ValueError(f"invalid dice {self.d1}-{self.d2}")

    @property
    def is_double(self) -> bool:
        return self.d1 == self.d2

    @property
    def high(self) -> int:
        return max(self.d1, self.d2)

    @property
    def low(self) -> int:
        return min(self.d1, self.d2)

    def dice_to_play(self) -> tuple[int, ...]:
        """The individual die values available: four of a kind on doubles."""
        return (self.d1,) * 4 if self.is_double else (self.d1, self.d2)

    @property
    def probability(self) -> float:
        """Chance of this roll, treating 3-1 and 1-3 as the same roll."""
        return 1 / 36 if self.is_double else 2 / 36

    def __str__(self) -> str:
        return f"{self.high}-{self.low}"


# The 21 distinct rolls; their probabilities sum to 1.
ALL_ROLLS: tuple[Roll, ...] = tuple(Roll(a, b) for a in range(1, 7) for b in range(1, a + 1))


class Dice:
    """A pair of dice with an injectable RNG so games can be replayed from a seed."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def roll(self) -> Roll:
        return Roll(self.rng.randint(1, 6), self.rng.randint(1, 6))

    def roll_one(self) -> int:
        return self.rng.randint(1, 6)
