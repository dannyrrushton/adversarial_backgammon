"""Game state and turn flow, including the doubling cube."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .board import Board, Player
from .dice import Dice, Roll
from .moves import Play, legal_plays

MAX_CUBE = 64


class Phase(Enum):
    OPENING = "opening roll"
    AWAIT_ROLL = "waiting for roll or double"
    AWAIT_DOUBLE_RESPONSE = "waiting for take/drop"
    AWAIT_MOVE = "waiting for move"
    GAME_OVER = "game over"


class WinKind(Enum):
    SINGLE = 1
    GAMMON = 2
    BACKGAMMON = 3


@dataclass(frozen=True)
class GameResult:
    winner: Player
    kind: WinKind
    cube: int
    dropped: bool = False

    @property
    def points(self) -> int:
        return self.cube * self.kind.value

    def __str__(self) -> str:
        how = "the double was dropped" if self.dropped else self.kind.name.lower()
        return f"{self.winner} wins {self.points} point{'s' * (self.points != 1)} ({how})"


@dataclass(frozen=True)
class TurnRecord:
    player: Player
    roll: Roll
    play: Play
    before: Board
    legal_count: int


@dataclass(frozen=True)
class CubeRecord:
    player: Player
    action: str  # "double", "take" or "drop"
    value: int


@dataclass
class GameState:
    board: Board = field(default_factory=Board.initial)
    turn: Player = Player.WHITE
    roll: Roll | None = None
    phase: Phase = Phase.OPENING
    cube_value: int = 1
    cube_owner: Player | None = None
    result: GameResult | None = None
    history: list[TurnRecord | CubeRecord] = field(default_factory=list)


def classify_win(board: Board, winner: Player) -> WinKind:
    loser = winner.opponent
    if board.off[loser] > 0:
        return WinKind.SINGLE
    # Backgammon: the loser still has a checker on the bar or in the winner's home board, which
    # is the loser's relative points 19..24.
    in_winner_home = any(board.count(loser, p) for p in range(19, 25))
    if board.bar[loser] or in_winner_home:
        return WinKind.BACKGAMMON
    return WinKind.GAMMON


class IllegalAction(Exception):
    pass


class Game:
    """Drives one game. Every action validates the phase and raises :class:`IllegalAction`."""

    def __init__(self, dice: Dice | None = None, state: GameState | None = None, use_cube: bool = True) -> None:
        self.dice = dice or Dice()
        self.state = state or GameState()
        self.use_cube = use_cube
        self._plays: list[Play] | None = None

    # --- queries ------------------------------------------------------------------------------

    @property
    def board(self) -> Board:
        return self.state.board

    @property
    def turn(self) -> Player:
        return self.state.turn

    @property
    def phase(self) -> Phase:
        return self.state.phase

    @property
    def over(self) -> bool:
        return self.state.phase is Phase.GAME_OVER

    def legal_plays(self) -> list[Play]:
        self._require(Phase.AWAIT_MOVE)
        if self._plays is None:
            self._plays = legal_plays(self.state.board, self.state.turn, self.state.roll)
        return self._plays

    def can_double(self, player: Player | None = None) -> bool:
        s = self.state
        player = s.turn if player is None else player
        return (
            self.use_cube
            and s.phase is Phase.AWAIT_ROLL
            and player is s.turn
            and s.cube_owner in (None, player)
            and s.cube_value < MAX_CUBE
        )

    # --- actions ------------------------------------------------------------------------------

    def opening_roll(self) -> Roll:
        """Each side rolls one die; the higher starts and plays both numbers. Ties re-roll."""
        self._require(Phase.OPENING)
        while True:
            white, black = self.dice.roll_one(), self.dice.roll_one()
            if white != black:
                break
        self.state.turn = Player.WHITE if white > black else Player.BLACK
        self._start_move(Roll(white, black))
        return self.state.roll

    def roll(self, roll: Roll | None = None) -> Roll:
        """Roll for the player on turn. ``roll`` forces the dice (replays, tests, physical dice)."""
        self._require(Phase.AWAIT_ROLL)
        self._start_move(roll or self.dice.roll())
        return self.state.roll

    def play(self, play: Play) -> None:
        self._require(Phase.AWAIT_MOVE)
        legal = {p.result.key(): p for p in self.legal_plays()}
        if play.result.key() not in legal or len(play.moves) != len(self.legal_plays()[0].moves):
            raise IllegalAction(f"{play.notation()} is not a legal play for {self.state.roll}")
        s = self.state
        s.history.append(TurnRecord(s.turn, s.roll, play, s.board, len(self._plays)))
        s.board = play.result
        winner = s.board.winner()
        if winner is not None:
            self._finish(GameResult(winner, classify_win(s.board, winner), s.cube_value))
            return
        s.turn = s.turn.opponent
        s.roll = None
        s.phase = Phase.AWAIT_ROLL
        self._plays = None

    def double(self) -> None:
        if not self.can_double():
            raise IllegalAction("you cannot double now")
        s = self.state
        s.history.append(CubeRecord(s.turn, "double", s.cube_value * 2))
        s.phase = Phase.AWAIT_DOUBLE_RESPONSE

    def take(self) -> None:
        self._require(Phase.AWAIT_DOUBLE_RESPONSE)
        s = self.state
        s.cube_value *= 2
        s.cube_owner = s.turn.opponent
        s.history.append(CubeRecord(s.turn.opponent, "take", s.cube_value))
        s.phase = Phase.AWAIT_ROLL

    def drop(self) -> None:
        self._require(Phase.AWAIT_DOUBLE_RESPONSE)
        s = self.state
        s.history.append(CubeRecord(s.turn.opponent, "drop", s.cube_value))
        self._finish(GameResult(s.turn, WinKind.SINGLE, s.cube_value, dropped=True))

    def resign(self, player: Player, kind: WinKind = WinKind.SINGLE) -> None:
        if self.over:
            raise IllegalAction("the game is already over")
        self._finish(GameResult(player.opponent, kind, self.state.cube_value))

    # --- internals ----------------------------------------------------------------------------

    def _start_move(self, roll: Roll) -> None:
        self.state.roll = roll
        self.state.phase = Phase.AWAIT_MOVE
        self._plays = None

    def _finish(self, result: GameResult) -> None:
        self.state.result = result
        self.state.phase = Phase.GAME_OVER
        self._plays = None

    def _require(self, phase: Phase) -> None:
        if self.state.phase is not phase:
            raise IllegalAction(f"not allowed while {self.state.phase.value}")

