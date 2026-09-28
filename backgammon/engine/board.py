"""Board representation.

Checkers are stored on 24 absolute points. ``points[i] > 0`` means White checkers,
``points[i] < 0`` means Black checkers. White moves from absolute index 23 down to 0 and bears
off below index 0; Black moves from 0 up to 23 and bears off above 23.

Everything the rest of the code sees is in *relative* point numbers, which is how backgammon
notation works: from the mover's point of view points are numbered 1..24 in the direction of
travel toward home (1..6 is the home board), the bar is 25 and borne-off checkers are at 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

NUM_POINTS = 24
CHECKERS_PER_SIDE = 15
BAR = 25
OFF = 0
HOME_BOARD = range(1, 7)


class Player(IntEnum):
    WHITE = 0
    BLACK = 1

    @property
    def opponent(self) -> Player:
        return Player(1 - self)

    @property
    def sign(self) -> int:
        return 1 if self is Player.WHITE else -1

    def __str__(self) -> str:
        return self.name.capitalize()


def absolute_index(player: Player, point: int) -> int:
    """Absolute array index of ``player``'s relative point 1..24."""
    if not 1 <= point <= NUM_POINTS:
        raise ValueError(f"point {point} is not on the board")
    return point - 1 if player is Player.WHITE else NUM_POINTS - point


def relative_point(player: Player, index: int) -> int:
    """Inverse of :func:`absolute_index`."""
    return index + 1 if player is Player.WHITE else NUM_POINTS - index


def opponent_point(point: int) -> int:
    """The same physical point numbered from the opponent's side (1 <-> 24, 6 <-> 19...)."""
    return NUM_POINTS + 1 - point


@dataclass(frozen=True, slots=True)
class Board:
    points: tuple[int, ...]
    bar: tuple[int, int] = (0, 0)
    off: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if len(self.points) != NUM_POINTS:
            raise ValueError("a board has exactly 24 points")
        for player in Player:
            total = self.bar[player] + self.off[player] + sum(
                abs(n) for n in self.points if n * player.sign > 0
            )
            if total != CHECKERS_PER_SIDE:
                raise ValueError(f"{player} has {total} checkers, expected {CHECKERS_PER_SIDE}")

    @classmethod
    def initial(cls) -> Board:
        return cls.from_relative({24: 2, 13: 5, 8: 3, 6: 5}, {24: 2, 13: 5, 8: 3, 6: 5})

    @classmethod
    def from_relative(
        cls,
        white: dict[int, int],
        black: dict[int, int],
        bar: tuple[int, int] = (0, 0),
        off: tuple[int, int] | None = None,
    ) -> Board:
        """Build a board from ``{relative_point: count}`` maps for each side.

        When ``off`` is omitted, every checker not placed on a point or the bar is borne off,
        which makes endgame positions short to write.
        """
        pts = [0] * NUM_POINTS
        for player, layout in ((Player.WHITE, white), (Player.BLACK, black)):
            for point, count in layout.items():
                idx = absolute_index(player, point)
                if pts[idx] != 0:
                    raise ValueError(f"absolute point {idx} is occupied by both sides")
                pts[idx] = count * player.sign
        if off is None:
            off = tuple(
                CHECKERS_PER_SIDE - bar[p] - sum((white, black)[p].values()) for p in Player
            )
        return cls(tuple(pts), tuple(bar), tuple(off))

    # --- queries ------------------------------------------------------------------------------

    def count(self, player: Player, point: int) -> int:
        """Number of ``player``'s checkers on relative ``point`` (25 = bar, 0 = off)."""
        if point == BAR:
            return self.bar[player]
        if point == OFF:
            return self.off[player]
        n = self.points[absolute_index(player, point)] * player.sign
        return n if n > 0 else 0

    def opponent_count(self, player: Player, point: int) -> int:
        """Opponent checkers on ``player``'s relative point 1..24."""
        n = self.points[absolute_index(player, point)] * player.sign
        return -n if n < 0 else 0

    def occupied_points(self, player: Player) -> list[int]:
        """Relative points holding ``player``'s checkers, highest (farthest from home) first."""
        return [p for p in range(NUM_POINTS, 0, -1) if self.count(player, p) > 0]

    def highest_point(self, player: Player) -> int:
        """Farthest relative point from home that holds a checker (25 if any on the bar, 0 if none)."""
        if self.bar[player]:
            return BAR
        occupied = self.occupied_points(player)
        return occupied[0] if occupied else 0

    def all_home(self, player: Player) -> bool:
        """True when every remaining checker is in the home board, so bearing off is allowed."""
        return self.highest_point(player) <= 6

    def pip_count(self, player: Player) -> int:
        pips = self.bar[player] * BAR
        for p in range(1, NUM_POINTS + 1):
            pips += p * self.count(player, p)
        return pips

    def is_blocked(self, player: Player, point: int) -> bool:
        """A point is blocked for ``player`` if the opponent has two or more checkers on it."""
        return self.opponent_count(player, point) >= 2

    def has_contact(self) -> bool:
        """False once the sides have passed each other and the game is a pure race."""
        white_back = self.highest_point(Player.WHITE)
        black_back = self.highest_point(Player.BLACK)
        if white_back == 0 or black_back == 0:
            return False
        # Compare White's rearmost checker with Black's rearmost checker in White's numbering
        # (a checker on Black's bar maps to 0, i.e. behind everything White has).
        return white_back > opponent_point(black_back)

    def winner(self) -> Player | None:
        for player in Player:
            if self.off[player] == CHECKERS_PER_SIDE:
                return player
        return None

    # --- mutation (returns new boards) --------------------------------------------------------

    def move_checker(self, player: Player, src: int, dst: int) -> tuple[Board, bool]:
        """Move one checker from relative ``src`` to ``dst`` without checking the dice.

        Returns the new board and whether an opposing blot was hit. Only occupancy is checked;
        rule-level legality lives in :mod:`backgammon.engine.moves`.
        """
        if self.count(player, src) == 0:
            raise ValueError(f"{player} has no checker on {src}")
        pts = list(self.points)
        bar = list(self.bar)
        off = list(self.off)
        sign = player.sign
        hit = False
        if src == BAR:
            bar[player] -= 1
        else:
            pts[absolute_index(player, src)] -= sign
        if dst == OFF:
            off[player] += 1
        else:
            idx = absolute_index(player, dst)
            opp = -pts[idx] * sign
            if opp >= 2:
                raise ValueError(f"point {dst} is blocked for {player}")
            if opp == 1:
                pts[idx] = 0
                bar[player.opponent] += 1
                hit = True
            pts[idx] += sign
        return Board(tuple(pts), tuple(bar), tuple(off)), hit

    def key(self) -> tuple:
        return (self.points, self.bar, self.off)
