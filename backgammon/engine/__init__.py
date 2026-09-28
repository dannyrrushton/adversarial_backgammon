"""Backgammon rules engine: board, dice, legal moves, bearing off, hitting, cube and game flow."""

from .board import BAR, CHECKERS_PER_SIDE, NUM_POINTS, OFF, Board, Player, absolute_index, relative_point
from .dice import ALL_ROLLS, Dice, Roll
from .game import CubeRecord, Game, GameResult, GameState, IllegalAction, Phase, TurnRecord, WinKind
from .moves import (
    Move,
    MoveKind,
    Play,
    PlayBuilder,
    Verdict,
    check_moves,
    format_moves,
    legal_plays,
    match_notation,
    move_problem,
    parse_notation,
    single_moves,
)

__all__ = [
    "ALL_ROLLS", "BAR", "CHECKERS_PER_SIDE", "NUM_POINTS", "OFF", "Board", "CubeRecord", "Dice",
    "Game", "GameResult", "GameState", "IllegalAction", "Move", "MoveKind", "Phase", "Play",
    "PlayBuilder", "Player", "Roll", "TurnRecord", "Verdict", "WinKind", "absolute_index",
    "check_moves", "format_moves", "legal_plays", "match_notation", "move_problem",
    "parse_notation", "relative_point", "single_moves",
]
