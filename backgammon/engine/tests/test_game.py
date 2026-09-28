import random

import pytest

from backgammon.engine import (
    CHECKERS_PER_SIDE,
    Board,
    Dice,
    Game,
    GameState,
    IllegalAction,
    Phase,
    Player,
    Roll,
    WinKind,
)
from backgammon.engine.game import classify_win
from backgammon.engine.text import describe, render

W, B = Player.WHITE, Player.BLACK


def test_board_invariants():
    board = Board.initial()
    assert board.pip_count(W) == board.pip_count(B) == 167
    assert board.has_contact()
    with pytest.raises(ValueError):
        Board(tuple([0] * 24))


def test_race_detection():
    race = Board.from_relative({6: 5, 5: 5}, {6: 5, 5: 5}, off=(5, 5))
    assert not race.has_contact()
    stuck = Board.from_relative({6: 5, 20: 1}, {6: 5}, off=(9, 10))
    assert stuck.has_contact()


def test_opening_roll_decides_first_player():
    game = Game(Dice(random.Random(1)))
    roll = game.opening_roll()
    assert not roll.is_double
    assert game.turn is (W if roll.d1 > roll.d2 else B)
    assert game.phase is Phase.AWAIT_MOVE


def test_turn_alternates_and_history_records():
    game = Game(Dice(random.Random(2)))
    game.opening_roll()
    first = game.turn
    game.play(game.legal_plays()[0])
    assert game.turn is first.opponent and game.phase is Phase.AWAIT_ROLL
    assert len(game.state.history) == 1


def test_illegal_play_rejected():
    game = Game(state=GameState(turn=W, phase=Phase.AWAIT_ROLL))
    game.roll(Roll(3, 1))
    other = Game(state=GameState(turn=W, phase=Phase.AWAIT_ROLL))
    other.roll(Roll(6, 6))
    with pytest.raises(IllegalAction):
        game.play(other.legal_plays()[0])


def test_actions_check_phase():
    game = Game()
    with pytest.raises(IllegalAction):
        game.roll()
    with pytest.raises(IllegalAction):
        game.take()


def test_random_games_terminate_with_consistent_scores():
    rng = random.Random(3)
    for _ in range(20):
        game = Game(Dice(rng), use_cube=False)
        game.opening_roll()
        turns = 0
        while not game.over:
            if game.phase is Phase.AWAIT_ROLL:
                game.roll()
            game.play(rng.choice(game.legal_plays()))
            turns += 1
            assert turns < 2000
        result = game.state.result
        assert game.board.off[result.winner] == CHECKERS_PER_SIDE
        assert result.points == result.kind.value


def test_win_classification():
    single = Board.from_relative({}, {6: 5}, off=(15, 10))
    gammon = Board.from_relative({}, {6: 15}, off=(15, 0))
    backgammon_home = Board.from_relative({}, {6: 14, 20: 1}, off=(15, 0))
    backgammon_bar = Board.from_relative({}, {6: 14}, bar=(0, 1), off=(15, 0))
    assert classify_win(single, W) is WinKind.SINGLE
    assert classify_win(gammon, W) is WinKind.GAMMON
    assert classify_win(backgammon_home, W) is WinKind.BACKGAMMON
    assert classify_win(backgammon_bar, W) is WinKind.BACKGAMMON


def test_bearing_off_last_checker_ends_game():
    board = Board.from_relative({1: 1}, {6: 15}, off=(14, 0))
    game = Game(state=GameState(board=board, turn=W, phase=Phase.AWAIT_ROLL))
    game.roll(Roll(2, 1))
    game.play(game.legal_plays()[0])
    assert game.over
    assert game.state.result.kind is WinKind.GAMMON and game.state.result.points == 2


def test_cube_double_take_then_owner_rules():
    game = Game(state=GameState(turn=W, phase=Phase.AWAIT_ROLL))
    assert game.can_double()
    game.double()
    assert game.phase is Phase.AWAIT_DOUBLE_RESPONSE
    game.take()
    assert game.state.cube_value == 2 and game.state.cube_owner is B
    assert not game.can_double()  # White no longer owns the cube
    game.roll(Roll(3, 1))
    game.play(game.legal_plays()[0])
    assert game.turn is B and game.can_double()


def test_cube_drop_ends_game():
    game = Game(state=GameState(turn=W, phase=Phase.AWAIT_ROLL, cube_value=2, cube_owner=W))
    game.double()
    game.drop()
    result = game.state.result
    assert result.winner is W and result.points == 2 and result.dropped


def test_cube_disabled():
    game = Game(state=GameState(turn=W, phase=Phase.AWAIT_ROLL), use_cube=False)
    assert not game.can_double()


def test_cube_multiplies_gammon():
    board = Board.from_relative({1: 1}, {6: 15}, off=(14, 0))
    game = Game(state=GameState(board=board, turn=W, phase=Phase.AWAIT_ROLL, cube_value=4, cube_owner=W))
    game.roll(Roll(1, 1))
    game.play(game.legal_plays()[0])
    assert game.state.result.points == 8


def test_text_rendering():
    text = render(Board.initial(), W)
    assert "pips: O (White)=167" in text
    assert describe(Board.initial(), B) == "Black: 24x2 13x5 8x3 6x5 | bar 0 | off 0"
