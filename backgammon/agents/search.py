"""Play ranking: static evaluation, optionally refined by a one-ply lookahead over enemy rolls."""

from __future__ import annotations

from dataclasses import dataclass

from backgammon.engine import ALL_ROLLS, Board, Play, Player, Roll, legal_plays

from .evaluator import equity


@dataclass(frozen=True)
class Candidate:
    play: Play
    equity: float  # from the mover's point of view
    static_equity: float


def reply_equity(board: Board, player: Player) -> float:
    """``player``'s equity after the opponent rolls and makes their best (static) reply."""
    opp = player.opponent
    if board.winner() is not None:
        return equity(board, player)
    total = 0.0
    for roll in ALL_ROLLS:
        best = max(equity(p.result, opp) for p in legal_plays(board, opp, roll))
        total += roll.probability * -best
    return total


def rank_plays(
    board: Board,
    player: Player,
    roll: Roll,
    depth: int = 1,
    width: int = 6,
    plays: list[Play] | None = None,
) -> list[Candidate]:
    """Plays sorted best first.

    ``depth=0`` ranks by static equity. ``depth=1`` re-scores the top ``width`` static plays by
    averaging the opponent's best reply over all 21 rolls, which catches plays that look solid but
    leave a strong return.
    """
    plays = plays if plays is not None else legal_plays(board, player, roll)
    static = sorted(
        (Candidate(p, e, e) for p in plays for e in [equity(p.result, player)]),
        key=lambda c: c.equity,
        reverse=True,
    )
    if depth <= 0 or len(static) <= 1:
        return static
    head = [Candidate(c.play, reply_equity(c.play.result, player), c.static_equity) for c in static[:width]]
    head.sort(key=lambda c: c.equity, reverse=True)
    # Plays outside the searched window keep their static score but always rank below it.
    floor = head[-1].equity
    tail = [Candidate(c.play, min(c.equity, floor) - 1e-6, c.static_equity) for c in static[width:]]
    return head + tail
