"""Plain-text board diagrams for terminals and LLM prompts."""

from __future__ import annotations

from .board import Board, Player

SYMBOL = {Player.WHITE: "O", Player.BLACK: "X"}
ROWS = 5


def render(board: Board, perspective: Player = Player.WHITE) -> str:
    """ASCII diagram with ``perspective``'s home board at the bottom right.

    Point numbers are shown from ``perspective``'s side. White is ``O``, Black is ``X``.
    """
    me = perspective
    top = list(range(13, 25))  # left to right
    bottom = list(range(12, 0, -1))

    def cell(point: int, row: int) -> str:
        n_me, n_opp = board.count(me, point), board.opponent_count(me, point)
        n, sym = (n_me, SYMBOL[me]) if n_me else (n_opp, SYMBOL[me.opponent])
        if n <= row:
            return " ."
        if row == ROWS - 1 and n > ROWS:
            return f"{n:2d}"
        return " " + sym

    def header(points: list[int]) -> str:
        left = "".join(f"{p:3d}" for p in points[:6])
        right = "".join(f"{p:3d}" for p in points[6:])
        return f" {left}  |  {right}"

    def row_line(points: list[int], row: int) -> str:
        left = "".join(" " + cell(p, row) for p in points[:6])
        right = "".join(" " + cell(p, row) for p in points[6:])
        return f" {left}  |  {right}"

    opp = me.opponent
    lines = [header(top), " " + "-" * 41]
    for r in range(ROWS):
        lines.append(row_line(top, r))
    lines.append(
        f"   bar: {SYMBOL[me]}={board.bar[me]} {SYMBOL[opp]}={board.bar[opp]}"
        f"     off: {SYMBOL[me]}={board.off[me]} {SYMBOL[opp]}={board.off[opp]}"
    )
    for r in reversed(range(ROWS)):
        lines.append(row_line(bottom, r))
    lines += [" " + "-" * 41, header(bottom)]
    lines.append(
        f"   pips: {SYMBOL[me]} ({me})={board.pip_count(me)}  "
        f"{SYMBOL[opp]} ({opp})={board.pip_count(opp)}"
    )
    return "\n".join(lines)


def describe(board: Board, player: Player) -> str:
    """Compact one-line position, e.g. ``White: 24x2 13x5 8x3 6x5 | bar 0 | off 0``."""
    pts = " ".join(f"{p}x{board.count(player, p)}" for p in board.occupied_points(player))
    return f"{player}: {pts or '-'} | bar {board.bar[player]} | off {board.off[player]}"
