import itertools
import random

import pytest

from backgammon.engine import (
    ALL_ROLLS,
    BAR,
    OFF,
    Board,
    MoveKind,
    PlayBuilder,
    Player,
    Roll,
    check_moves,
    legal_plays,
    match_notation,
    move_problem,
    parse_notation,
)

W, B = Player.WHITE, Player.BLACK


def notations(board, player, roll):
    return {p.notation() for p in legal_plays(board, player, roll)}


def results(board, player, roll):
    return {p.result.key() for p in legal_plays(board, player, roll)}


# --- reference generator: brute force over every die order, no shortcuts ----------------------


def reference_results(board, player, roll):
    orders = set(itertools.permutations(roll.dice_to_play()))
    seqs = []

    def rec(b, dice, used):
        progressed = False
        if dice:
            sources = [BAR] if b.bar[player] else b.occupied_points(player)
            for src in sources:
                if move_problem(b, player, src, dice[0]) is None:
                    progressed = True
                    nb, _ = b.move_checker(player, src, max(src - dice[0], OFF))
                    rec(nb, dice[1:], used + [dice[0]])
        if not progressed:
            seqs.append((b, used))

    for order in orders:
        rec(board, list(order), [])
    most = max(len(u) for _, u in seqs)
    seqs = [(b, u) for b, u in seqs if len(u) == most]
    if most == 1 and not roll.is_double and any(u[0] == roll.high for _, u in seqs):
        seqs = [(b, u) for b, u in seqs if u[0] == roll.high]
    return {b.key() for b, _ in seqs}


def random_positions(n, seed=7):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board, player = Board.initial(), W
        for _ in range(rng.randint(1, 80)):
            roll = Roll(rng.randint(1, 6), rng.randint(1, 6))
            board = rng.choice(legal_plays(board, player, roll)).result
            if board.winner() is not None:
                break
            player = player.opponent
        if board.winner() is None:
            out.append((board, player))
    return out


@pytest.mark.parametrize("board,player", random_positions(40))
def test_matches_brute_force_reference(board, player):
    for roll in ALL_ROLLS:
        assert results(board, player, roll) == reference_results(board, player, roll), str(roll)


# --- basic movement ---------------------------------------------------------------------------


def test_opening_three_one_contains_making_the_five_point():
    plays = notations(Board.initial(), W, Roll(3, 1))
    assert "8/5 6/5" in plays
    assert len(plays) == 16


def test_both_sides_generate_symmetric_plays():
    for roll in ALL_ROLLS:
        assert notations(Board.initial(), W, roll) == notations(Board.initial(), B, roll)


def test_every_play_uses_both_dice_from_opening():
    for roll in ALL_ROLLS:
        for play in legal_plays(Board.initial(), W, roll):
            assert len(play.moves) == len(roll.dice_to_play())


def test_cannot_land_on_opponent_point():
    problem = move_problem(Board.initial(), W, 13, 1)  # White's 12 point is Black's midpoint
    assert problem and "blocked" in problem


def test_hitting_sends_blot_to_bar():
    board = Board.from_relative({6: 2, 8: 1}, {24 - 4 + 1: 1, 24: 1}, off=(12, 13))
    # Black blot on White's 4 point (Black's 21) and Black checker on Black's 24 (White's 1).
    play = next(p for p in legal_plays(board, W, Roll(2, 1)) if p.notation() == "6/4* 4/3")
    assert play.hits == 1
    assert play.result.bar[B] == 1


def test_must_enter_from_bar_first():
    board = Board.from_relative({24: 1, 6: 13}, {6: 15}, bar=(1, 0), off=(0, 0))
    for play in legal_plays(board, W, Roll(4, 2)):
        assert play.moves[0].src == BAR
        assert play.moves[0].kind is MoveKind.ENTER
    assert "must enter" in move_problem(board, W, 6, 2)


def test_closed_board_means_no_play():
    closed = {p: 2 for p in range(1, 7)}  # Black's home board fully made
    closed[13] = 3
    board = Board.from_relative({6: 14}, closed, bar=(1, 0), off=(0, 0))
    plays = legal_plays(board, W, Roll(6, 6))
    assert len(plays) == 1 and plays[0].is_pass
    assert plays[0].notation() == "(no play)"


def test_partial_entry_then_move():
    # Black holds White's 20 and 22 (entry points for 5 and 3); White on bar rolls 5-4.
    board = Board.from_relative({6: 14}, {5: 2, 3: 2, 6: 11}, bar=(1, 0), off=(0, 0))
    plays = notations(board, W, Roll(5, 4))
    assert all(p.startswith("bar/21") for p in plays)
    assert "bar/21 21/16" in plays


# --- the "use as many dice as possible" rules ---------------------------------------------------


def test_must_use_both_dice_when_possible():
    # One White checker on 10 (rest borne off), Black holds White's 4 point.
    board = Board.from_relative({10: 1}, {21: 2}, off=(14, 13))
    # 10/4 is blocked for the 6, so the 5 must go first and the 6 then bears off from 5.
    assert notations(board, W, Roll(6, 5)) == {"10/5 5/off"}


def test_must_play_larger_die_when_only_one_can_be_played():
    # White checker on 12; Black holds White's 2. 6-4: 12/6 or 12/8 but never both.
    board = Board.from_relative({12: 1}, {23: 2}, off=(14, 13))
    assert notations(board, W, Roll(6, 4)) == {"12/6"}


def test_smaller_die_used_when_larger_impossible():
    # 12/6 is blocked, 12/8 open, 8/2 blocked -> only the 4 can be played.
    board = Board.from_relative({12: 1}, {19: 2, 23: 2}, off=(14, 11))
    assert notations(board, W, Roll(6, 4)) == {"12/8"}


def test_doubles_play_up_to_four():
    board = Board.from_relative({13: 2}, {1: 2}, off=(13, 13))
    plays = legal_plays(board, W, Roll(3, 3))
    assert all(len(p.moves) == 4 for p in plays)
    assert "13/10(2) 10/7(2)" in {p.notation() for p in plays}


def test_doubles_partially_blocked():
    # White checker on 10; Black holds White's 4: 10/7 then 7/4 is blocked -> only one 3 plays.
    board = Board.from_relative({10: 1}, {21: 2}, off=(14, 13))
    assert notations(board, W, Roll(3, 3)) == {"10/7"}


# --- bearing off --------------------------------------------------------------------------------


def test_bear_off_exact():
    board = Board.from_relative({6: 1, 3: 1}, {12: 1}, off=(13, 14))
    assert "6/off 3/off" in notations(board, W, Roll(6, 3))


def test_bear_off_higher_die_only_from_highest_point():
    board = Board.from_relative({4: 1, 2: 1}, {12: 1}, off=(13, 14))
    plays = legal_plays(board, W, Roll(6, 5))
    assert {p.notation() for p in plays} == {"4/off 2/off"}
    assert plays[0].moves[0].kind is MoveKind.BEAR_OFF_HIGH


def test_higher_die_cannot_bear_off_while_higher_checkers_remain():
    board = Board.from_relative({6: 1, 2: 1}, {12: 1}, off=(13, 14))
    assert "no checkers sit on higher points" in move_problem(board, W, 2, 5)
    expected = {
        Board.from_relative({1: 2}, {12: 1}, off=(13, 14)).key(),
        Board.from_relative({2: 1}, {12: 1}, off=(14, 14)).key(),
    }
    assert results(board, W, Roll(5, 1)) == expected


def test_cannot_bear_off_with_checker_outside_home():
    board = Board.from_relative({7: 1, 3: 1}, {12: 1}, off=(13, 14))
    assert "all fifteen checkers" in move_problem(board, W, 3, 3)
    # After 7/1 with the 6, the 3 may bear off from 3.
    assert "7/1 3/off" in notations(board, W, Roll(6, 3))


def test_bear_off_black():
    board = Board.from_relative({12: 1}, {1: 2}, off=(14, 13))
    assert notations(board, B, Roll(2, 1)) == {"1/off(2)"}


# --- notation ---------------------------------------------------------------------------------


def test_parse_notation_forms():
    segs = parse_notation("bar/22 13/7* 6/off 8/5(2) 24/18/13")
    assert [(s.src, s.dst) for s in segs] == [(25, 22), (13, 7), (6, 0), (8, 5), (8, 5), (24, 18), (18, 13)]
    assert segs[1].hit_marked


def test_parse_notation_rejects_backwards():
    with pytest.raises(ValueError):
        parse_notation("8/13")


def test_match_multi_die_segment():
    verdict = match_notation(Board.initial(), W, Roll(6, 5), "24/13")
    assert verdict.legal and verdict.play.notation() in {"24/18 18/13", "24/19 19/13"}


def test_match_reports_rule_violation():
    verdict = match_notation(Board.initial(), W, Roll(6, 1), "13/12 8/7")
    assert not verdict.legal
    assert "blocked" in verdict.reason


def test_match_reports_unused_dice():
    verdict = match_notation(Board.initial(), W, Roll(3, 1), "8/5")
    assert not verdict.legal and "2 dice" in verdict.reason


def test_check_moves_larger_die_rule():
    board = Board.from_relative({12: 1}, {23: 2}, off=(14, 13))
    verdict = check_moves(board, W, Roll(6, 4), [(12, 4)])
    assert not verdict.legal and "larger" in verdict.reason
    assert check_moves(board, W, Roll(6, 4), [(12, 6)]).legal


# --- incremental builder --------------------------------------------------------------------------


def test_builder_completes_play():
    builder = PlayBuilder(Board.initial(), W, Roll(3, 1))
    builder.add(8, 5)
    assert builder.complete is None
    builder.add(6, 5)
    assert builder.complete.notation() == "8/5 6/5"


def test_builder_chains_combined_move():
    builder = PlayBuilder(Board.initial(), W, Roll(6, 5))
    builder.add(24, 13)
    assert builder.complete is not None and len(builder.complete.moves) == 2


def test_builder_blocks_dead_end_first_move():
    # Only 10/5 then 5/off is legal: moving the 6 first must not be offered.
    board = Board.from_relative({10: 1}, {21: 2}, off=(14, 13))
    builder = PlayBuilder(board, W, Roll(6, 5))
    assert [m.dst for m in builder.destinations(10)] == [5]
    with pytest.raises(ValueError):
        builder.add(10, 4)


def test_builder_undo():
    builder = PlayBuilder(Board.initial(), W, Roll(3, 1))
    builder.add(8, 5)
    builder.undo()
    assert builder.moves == [] and 24 in builder.sources()
