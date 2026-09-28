"""Move annotations: what a play did, why it was legal, and the strategy behind it."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backgammon.agents.ollama_client import OllamaClient, OllamaError
from backgammon.agents.search import Candidate, rank_plays, reply_equity
from backgammon.engine import BAR, Board, Move, MoveKind, Play, Player, Roll, legal_plays, move_problem, single_moves
from backgammon.engine.moves import apply_move

from .themes import play_themes

log = logging.getLogger(__name__)

SEARCH_WIDTH = 6  # plays re-scored by lookahead; the rest keep static scores


@dataclass
class Annotation:
    player: Player
    roll: Roll
    notation: str
    summary: str
    legality: list[str]
    strategy: list[str]
    assessment: str = ""
    rank: int | None = None  # 1 = engine's top choice
    legal_count: int = 0
    equity_loss: float = 0.0
    commentary: str = ""  # the agent's own reasoning, or LLM teaching commentary
    alternatives: list[str] = field(default_factory=list)

    def to_text(self, legality: bool = True) -> str:
        lines = [f"{self.player} rolls {self.roll} and plays {self.notation}. {self.summary}"]
        if self.strategy:
            lines.append("Strategy:")
            lines += [f"  - {s}" for s in self.strategy]
        if self.commentary:
            lines.append(f"Reasoning: {self.commentary}")
        if self.assessment:
            lines.append(f"Assessment: {self.assessment}")
        if legality and self.legality:
            lines.append("Why it is legal:")
            lines += [f"  - {s}" for s in self.legality]
        return "\n".join(lines)


def describe_destination(board: Board, player: Player, dst: int) -> str:
    """Why a landing point is open, judged just before the checker arrives."""
    own = board.count(player, dst)
    opp = board.opponent_count(player, dst)
    if opp == 1:
        return f"point {dst} holds a single opposing checker (a blot), and landing on a blot is allowed: it is hit and sent to the bar"
    if own:
        return f"point {dst} already holds {own} of your checker{'s' * (own > 1)}, so you may add to it"
    return f"point {dst} is empty, so it is open"


def explain_move(board: Board, player: Player, move: Move) -> str:
    """Why one checker move is legal, given the board just before it."""
    die = move.die
    if move.kind is MoveKind.ENTER:
        opp_point = 25 - move.dst
        return (
            f"bar/{move.dst}: a checker on the bar must enter before anything else moves. A {die} "
            f"enters on your {move.dst}-point (your opponent's {opp_point}-point), which is not held "
            f"by two or more opposing checkers: {describe_destination(board, player, move.dst)}."
        )
    if move.kind is MoveKind.BEAR_OFF:
        return (
            f"{move.src}/off: every checker is in your home board, so bearing off is allowed, and the "
            f"{die} matches the {move.src}-point exactly."
        )
    if move.kind is MoveKind.BEAR_OFF_HIGH:
        return (
            f"{move.src}/off: every checker is in your home board and no checker sits above the "
            f"{move.src}-point, so the larger {die} may bear off from the highest occupied point."
        )
    return f"{move}: the {die} moves a checker from {move.src} to {move.dst}; {describe_destination(board, player, move.dst)}."


def explain_dice_usage(board: Board, player: Player, roll: Roll, play: Play) -> str:
    """Which dice-usage rule this play satisfies."""
    used = len(play.moves)
    if play.is_pass:
        if board.bar[player]:
            blocked = ", ".join(str(25 - d) for d in sorted(set(roll.dice_to_play())))
            return f"No legal move: the checker on the bar cannot enter because your {blocked}-point{'s' * (len(set(roll.dice_to_play())) > 1)} are blocked, so the turn passes."
        return "No legal move: every move with these dice lands on a blocked point, so the turn passes."
    if roll.is_double:
        if used == 4:
            return f"Doubles are played four times, and all four {roll.d1}s are used."
        return f"Doubles give four {roll.d1}s, but after {used} the rest were blocked, so only {used} could be played."
    if used == 2:
        return "Both dice are used, as the rules require whenever possible."
    die = play.moves[0].die
    if die == roll.high and single_moves(board, player, roll.low):
        return (
            f"Only one die could be played here (no sequence uses both). When either die could be "
            f"played but not both, the rules require the larger one, the {roll.high}."
        )
    other = roll.low if die == roll.high else roll.high
    return f"Only one die could be used: the {other} has no legal move before or after the {die}."


def legality_notes(board: Board, player: Player, roll: Roll, play: Play) -> list[str]:
    notes = []
    b = board
    for move in play.moves:
        notes.append(explain_move(b, player, move))
        b = apply_move(b, player, move)
    notes.append(explain_dice_usage(board, player, roll, play))
    return notes


def blocked_notes(board: Board, player: Player, roll: Roll) -> list[str]:
    """Rules that ruled out otherwise natural moves this turn (for learners)."""
    notes = []
    for die in sorted(set(roll.dice_to_play()), reverse=True):
        src = BAR if board.bar[player] else board.highest_point(player)
        if src and (problem := move_problem(board, player, src, die)):
            notes.append(f"Your rearmost checker can't move {die}: {problem}.")
    return notes


# The hand-tuned evaluator exaggerates equity gaps compared with a neural-net bot, so these bands
# are wider than the conventional 0.02 / 0.04 / 0.08 / 0.16 thresholds.
EQUITY_BANDS = [
    (0.0005, "the engine's top choice"),
    (0.05, "excellent, practically as good as the top choice"),
    (0.12, "good"),
    (0.25, "an inaccuracy"),
    (0.45, "a mistake"),
]


def grade(loss: float) -> str:
    for limit, label in EQUITY_BANDS:
        if loss < limit:
            return label
    return "a blunder"


class Annotator:
    """Builds :class:`Annotation` objects. With an Ollama client it can add teaching commentary."""

    def __init__(self, client: OllamaClient | None = None, depth: int = 1, llm_commentary: bool = False) -> None:
        self.client = client
        self.depth = depth
        self.llm_commentary = llm_commentary and client is not None

    def evaluate(self, board: Board, player: Player, roll: Roll, play: Play, candidates: list[Candidate] | None = None) -> tuple[list[Candidate], int, float]:
        """Ranked candidates, the play's rank (1-based) and its equity loss versus the best."""
        ranked = candidates or rank_plays(board, player, roll, depth=self.depth, width=SEARCH_WIDTH)
        key = play.result.key()
        idx = next((i for i, c in enumerate(ranked) if c.play.result.key() == key), None)
        best = ranked[0].equity
        if idx is None:  # not in the list: evaluate it directly
            value = reply_equity(play.result, player) if self.depth else ranked[-1].equity
            return ranked, len(ranked), max(0.0, best - value)
        chosen = ranked[idx].equity
        if self.depth and idx >= SEARCH_WIDTH:
            # Outside the searched window the list holds static scores; search it properly.
            chosen = reply_equity(play.result, player)
        return ranked, idx + 1, max(0.0, best - chosen)

    def annotate(
        self,
        board: Board,
        player: Player,
        roll: Roll,
        play: Play,
        candidates: list[Candidate] | None = None,
        reasoning: str = "",
    ) -> Annotation:
        themes = play_themes(board, player, play)
        legal = legal_plays(board, player, roll)
        pips = board.pip_count(player) - play.result.pip_count(player)
        summary = themes[0].text if themes else ""
        if not play.is_pass:
            summary = f"It moves {pips} pips. " + summary

        ann = Annotation(
            player=player,
            roll=roll,
            notation=play.notation(),
            summary=summary,
            legality=legality_notes(board, player, roll, play) + ([f"It is one of {len(legal)} distinct legal plays for this roll."] if len(legal) > 1 else []),
            strategy=[t.text for t in themes[1:]] if len(themes) > 1 else [],
            legal_count=len(legal),
            commentary=reasoning,
        )
        if len(legal) > 1:
            ranked, rank, loss = self.evaluate(board, player, roll, play, candidates)
            ann.rank, ann.equity_loss = rank, loss
            ann.assessment = f"Ranked {rank} of {len(ranked)} by the engine; {grade(loss)}"
            if rank != 1:
                best = ranked[0]
                best_theme = play_themes(board, player, best.play)[0].text
                ann.assessment += f" (costs about {loss:.2f} equity). The engine preferred {best.play.notation()}: {best_theme}"
            ann.assessment += "."
            ann.alternatives = [f"{c.play.notation()} ({c.equity:+.3f})" for c in ranked[:3]]
        if self.llm_commentary and not play.is_pass:
            ann.commentary = self._commentary(board, player, roll, ann) or ann.commentary
        return ann

    def _commentary(self, board: Board, player: Player, roll: Roll, ann: Annotation) -> str:
        from backgammon.engine.text import render

        prompt = (
            "You are a patient backgammon coach. In two to four plain sentences, explain to a "
            "beginner the idea behind this move. Use only the facts given; do not invent tactics.\n\n"
            f"{render(board, player)}\n\n{ann.to_text()}"
        )
        try:
            return self.client.chat([{"role": "user", "content": prompt}], temperature=0.3).strip()
        except OllamaError as exc:
            log.warning("LLM commentary unavailable: %s", exc)
            return ""
