"""Legal move generation.

A *move* is one checker moved by one die. A *play* is the full sequence of moves made in a turn.
The rules implemented here:

* A checker on the bar must enter (on the opponent's home board) before any other checker moves.
* A checker cannot land on a point held by two or more opposing checkers. Landing on a single
  opposing checker (a blot) hits it and sends it to the bar.
* Bearing off is only allowed once all of a side's checkers are in its home board. A die may bear
  off a checker from the matching point, or from a lower point when no checkers sit higher.
* A player must use as many dice as possible (four on doubles). If only one die of a non-double
  can be used, the larger one must be used when it can be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .board import BAR, NUM_POINTS, OFF, Board, Player
from .dice import Roll


class MoveKind(Enum):
    ENTER = "enter"
    NORMAL = "normal"
    BEAR_OFF = "bear off"
    BEAR_OFF_HIGH = "bear off with a higher die"


@dataclass(frozen=True, slots=True)
class Move:
    src: int
    dst: int
    die: int
    hit: bool = False

    def __str__(self) -> str:
        src = "bar" if self.src == BAR else str(self.src)
        dst = "off" if self.dst == OFF else str(self.dst)
        return f"{src}/{dst}{'*' if self.hit else ''}"

    @property
    def kind(self) -> MoveKind:
        if self.src == BAR:
            return MoveKind.ENTER
        if self.dst == OFF:
            return MoveKind.BEAR_OFF if self.src == self.die else MoveKind.BEAR_OFF_HIGH
        return MoveKind.NORMAL


@dataclass(frozen=True, slots=True)
class Play:
    moves: tuple[Move, ...]
    result: Board = field(compare=False)

    @property
    def hits(self) -> int:
        return sum(m.hit for m in self.moves)

    @property
    def is_pass(self) -> bool:
        return not self.moves

    def notation(self) -> str:
        return format_moves(self.moves)

    def __str__(self) -> str:
        return self.notation()


def format_moves(moves: tuple[Move, ...] | list[Move]) -> str:
    """Standard notation, e.g. ``13/8 6/5*``; ``(no play)`` when no move is possible."""
    if not moves:
        return "(no play)"
    counts: dict[str, int] = {}
    for m in moves:
        counts[str(m)] = counts.get(str(m), 0) + 1
    return " ".join(s if n == 1 else f"{s}({n})" for s, n in counts.items())


# --- single-die rules ---------------------------------------------------------------------------


def move_problem(board: Board, player: Player, src: int, die: int) -> str | None:
    """Why moving a checker from ``src`` with ``die`` is illegal, or ``None`` if it is legal.

    This checks one die in isolation; whether the whole play uses enough dice is checked by
    :func:`legal_plays`.
    """
    if board.count(player, src) == 0:
        where = "the bar" if src == BAR else f"point {src}"
        return f"you have no checker on {where}"
    if board.bar[player] and src != BAR:
        return "a checker on the bar must enter before any other checker can move"
    dst = src - die
    if dst >= 1:
        if board.is_blocked(player, dst):
            return (
                f"point {dst} is blocked: your opponent holds it with "
                f"{board.opponent_count(player, dst)} checkers"
            )
        return None
    if not board.all_home(player):
        return "you can only bear off once all fifteen checkers are in your home board"
    if dst < 0 and board.highest_point(player) > src:
        return (
            f"a {die} can only bear off from point {src} when no checkers sit on higher points "
            f"(you still have one on {board.highest_point(player)})"
        )
    return None


def single_moves(board: Board, player: Player, die: int) -> list[Move]:
    """All legal one-checker moves for a single die."""
    sources = [BAR] if board.bar[player] else board.occupied_points(player)
    moves = []
    for src in sources:
        if move_problem(board, player, src, die) is None:
            dst = max(src - die, OFF)
            hit = dst != OFF and board.opponent_count(player, dst) == 1
            moves.append(Move(src, dst, die, hit))
    return moves


def apply_move(board: Board, player: Player, move: Move) -> Board:
    return board.move_checker(player, move.src, move.dst)[0]


# --- whole-turn generation ----------------------------------------------------------------------


def legal_plays(board: Board, player: Player, roll: Roll) -> list[Play]:
    """Every distinct legal play for ``roll``, one per resulting position.

    Returns a single empty play when no checker can move.
    """
    found: dict[tuple, Play] = {}

    def extend(b: Board, remaining: tuple[int, ...], moves: tuple[Move, ...], floor: int) -> None:
        extended = False
        if remaining:
            die = remaining[0]
            for m in single_moves(b, player, die):
                # On doubles every die is the same, so moving checkers in descending source order
                # reaches every position without trying all permutations.
                if roll.is_double and m.src > floor:
                    continue
                extended = True
                extend(apply_move(b, player, m), remaining[1:], moves + (m,), m.src)
        if not extended:
            key = b.key()
            # Keep the variant with the most dice used for a given position.
            if key not in found or len(found[key].moves) < len(moves):
                found[key] = Play(moves, b)

    if roll.is_double:
        extend(board, roll.dice_to_play(), (), BAR)
    else:
        extend(board, (roll.d1, roll.d2), (), BAR)
        extend(board, (roll.d2, roll.d1), (), BAR)

    plays = list(found.values())
    most = max(len(p.moves) for p in plays)
    plays = [p for p in plays if len(p.moves) == most]
    if most == 1 and not roll.is_double:
        larger = [p for p in plays if p.moves[0].die == roll.high]
        if larger:
            plays = larger
    return plays


def max_dice_usable(board: Board, player: Player, roll: Roll) -> int:
    return len(legal_plays(board, player, roll)[0].moves)


# --- checking a proposed play -------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    legal: bool
    reason: str
    play: Play | None = None


def check_moves(board: Board, player: Player, roll: Roll, moves: list[tuple[int, int]]) -> Verdict:
    """Check a sequence of single-die ``(src, die)`` moves and explain any problem.

    Used for human input: the explanation says which rule a rejected move breaks.
    """
    dice = list(roll.dice_to_play())
    b = board
    made: list[Move] = []
    for src, die in moves:
        if die not in dice:
            return Verdict(False, f"there is no unused {die} left to play (dice left: {dice})")
        problem = move_problem(b, player, src, die)
        if problem:
            return Verdict(False, problem)
        dst = max(src - die, OFF)
        b, hit = b.move_checker(player, src, dst)
        made.append(Move(src, dst, die, hit))
        dice.remove(die)
    for play in legal_plays(board, player, roll):
        if play.result.key() == b.key() and len(play.moves) == len(made):
            return Verdict(True, "legal", play)
    plays = legal_plays(board, player, roll)
    most = len(plays[0].moves)
    if len(made) < most:
        return Verdict(False, f"you must use {most} dice this turn; only {len(made)} used")
    if most == 1 and not roll.is_double and made and made[0].die != roll.high:
        return Verdict(False, f"when only one die can be played you must play the larger ({roll.high})")
    return Verdict(False, "that sequence is not a legal play for this roll")


# --- notation parsing ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    src: int
    dst: int
    hit_marked: bool = False


def parse_notation(text: str) -> list[Segment]:
    """Parse notation such as ``13/8 6/5*``, ``bar/22``, ``6/off``, ``8/5(2)`` or ``24/18/13``."""
    segments: list[Segment] = []
    for token in text.replace(",", " ").split():
        token = token.lower()
        repeat = 1
        if token.endswith(")") and "(" in token:
            token, count = token[:-1].split("(", 1)
            repeat = int(count)
        parts = token.split("/")
        if len(parts) < 2:
            raise ValueError(f"cannot parse move '{token}'; use the form 13/8")
        spots = []
        for part in parts:
            marked = part.endswith("*")
            part = part.rstrip("*")
            if part == "bar":
                spot = BAR
            elif part == "off":
                spot = OFF
            elif part.isdigit() and 1 <= int(part) <= NUM_POINTS:
                spot = int(part)
            else:
                raise ValueError(f"unknown point '{part}'")
            spots.append((spot, marked))
        for _ in range(repeat):
            for (src, _), (dst, marked) in zip(spots, spots[1:]):
                if dst >= src:
                    raise ValueError(f"checkers move toward home: {src}/{dst} goes backwards")
                segments.append(Segment(src, dst, marked))
    return segments


def match_notation(board: Board, player: Player, roll: Roll, text: str) -> Verdict:
    """Find the legal play that ``text`` describes.

    Segments may span several dice (``24/13`` with 6-5). The play is identified by where the
    mover's checkers end up; ``*`` marks only decide between plays that differ by an
    intermediate hit.
    """
    try:
        segments = parse_notation(text)
    except ValueError as exc:
        return Verdict(False, str(exc))
    if not segments:
        return Verdict(False, "no moves given")

    target = _mover_layout(board, player)
    for seg in segments:
        if target.get(seg.src, 0) == 0:
            where = "the bar" if seg.src == BAR else f"point {seg.src}"
            return Verdict(False, f"you have no checker on {where} to move")
        target[seg.src] -= 1
        target[seg.dst] = target.get(seg.dst, 0) + 1
    target = {k: v for k, v in target.items() if v}

    plays = legal_plays(board, player, roll)
    candidates = [p for p in plays if _mover_layout(p.result, player) == target]
    if not candidates:
        return Verdict(False, _diagnose(board, player, roll, segments, plays))
    marked = {s.dst for s in segments if s.hit_marked}
    for p in candidates:
        if {m.dst for m in p.moves if m.hit} == marked:
            return Verdict(True, "legal", p)
    return Verdict(True, "legal", candidates[0])


def _mover_layout(board: Board, player: Player) -> dict[int, int]:
    layout = {p: board.count(player, p) for p in range(OFF, BAR + 1)}
    return {k: v for k, v in layout.items() if v}


def _diagnose(board: Board, player: Player, roll: Roll, segments: list[Segment], plays: list[Play]) -> str:
    """Best-effort explanation of why a notation string matched no legal play."""
    dice = sorted(roll.dice_to_play(), reverse=True)
    distances = [s.src - s.dst for s in segments]
    if board.bar[player] and segments[0].src != BAR:
        return "you have a checker on the bar and must enter it first"
    # Single-die segments can be checked rule by rule for a precise message.
    if all(d in dice for d in distances) and not any(s.dst == OFF for s in segments):
        verdict = check_moves(board, player, roll, [(s.src, s.src - s.dst) for s in segments])
        if not verdict.legal:
            return verdict.reason
    total = sum(dice)
    if sum(distances) > total and not any(s.dst == OFF for s in segments):
        return f"those moves travel {sum(distances)} pips but the roll {roll} only gives {total}"
    most = len(plays[0].moves)
    return (
        f"that is not one of the legal plays for {roll}; you must use {most} "
        f"{'die' if most == 1 else 'dice'} (e.g. {plays[0].notation()})"
    )


# --- incremental construction (click-to-move UIs) -----------------------------------------------


class PlayBuilder:
    """Builds a play one checker move at a time, only allowing prefixes of legal plays."""

    def __init__(self, board: Board, player: Player, roll: Roll) -> None:
        self.board = board
        self.player = player
        self.roll = roll
        self.plays = legal_plays(board, player, roll)
        self.moves: list[Move] = []
        self._needed = len(self.plays[0].moves)
        self._targets = {p.result.key() for p in self.plays}

    @property
    def current(self) -> Board:
        b = self.board
        for m in self.moves:
            b = apply_move(b, self.player, m)
        return b

    def _remaining(self) -> list[int]:
        remaining = list(self.roll.dice_to_play())
        for m in self.moves:
            remaining.remove(m.die)
        return remaining

    def _next_moves(self) -> list[Move]:
        """Moves that keep the partial play on the way to some legal play."""
        current = self.current
        remaining = self._remaining()
        options = []
        for die in sorted(set(remaining), reverse=True):
            rest = list(remaining)
            rest.remove(die)
            for m in single_moves(current, self.player, die):
                if self._can_complete(apply_move(current, self.player, m), rest, len(self.moves) + 1):
                    options.append(m)
        return options

    def _can_complete(self, board: Board, remaining: list[int], done: int) -> bool:
        if done == self._needed:
            return board.key() in self._targets
        for die in set(remaining):
            rest = list(remaining)
            rest.remove(die)
            for m in single_moves(board, self.player, die):
                if self._can_complete(apply_move(board, self.player, m), rest, done + 1):
                    return True
        return False

    def destinations(self, src: int) -> list[Move]:
        return [m for m in self._next_moves() if m.src == src]

    def sources(self) -> list[int]:
        return sorted({m.src for m in self._next_moves()}, reverse=True)

    def add(self, src: int, dst: int) -> Move:
        """Move a checker from ``src`` to ``dst``, combining dice if one die can't reach it."""
        for m in self.destinations(src):
            if m.dst == dst:
                self.moves.append(m)
                return m
        # A multi-die move (e.g. 24/13 with 6-5): try chaining through intermediate points.
        for m in self.destinations(src):
            if m.dst != OFF and m.dst > dst:
                self.moves.append(m)
                try:
                    self.add(m.dst, dst)
                    return m
                except ValueError:
                    self.moves.pop()
        raise ValueError(f"{src}/{'off' if dst == OFF else dst} is not legal here")

    def undo(self) -> None:
        if self.moves:
            self.moves.pop()

    @property
    def complete(self) -> Play | None:
        if len(self.moves) != self._needed:
            return None
        key = self.current.key()
        for p in self.plays:
            if p.result.key() == key:
                return Play(tuple(self.moves), p.result)
        return None
