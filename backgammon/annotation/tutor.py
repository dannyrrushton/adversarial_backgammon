"""Teaching helpers: hints before a move, critique after it, and rule explanations."""

from __future__ import annotations

from dataclasses import dataclass

from backgammon.agents.search import rank_plays
from backgammon.engine import Board, Play, Player, Roll, legal_plays

from .annotator import SEARCH_WIDTH, Annotation, Annotator, blocked_notes, grade
from .themes import play_themes

RULES = {
    "movement": "Each die moves one checker that many points toward your home board (from 24 toward 1). The two dice can move one checker twice or two different checkers.",
    "doubles": "Rolling doubles lets you play that number four times.",
    "blocked": "You may not land on a point held by two or more opposing checkers.",
    "hitting": "Landing on a point with a single opposing checker (a blot) hits it: it goes to the bar and must re-enter in your home board before its owner can move anything else.",
    "bar": "With a checker on the bar you must enter it first. A die of n enters on your opponent's n-point from their side (your 25-n point). If the point is blocked, that die cannot enter.",
    "bearing off": "Once all fifteen of your checkers are in your home board (points 1-6), you may bear them off: a die removes a checker from the matching point, or from the highest occupied point if the die is larger than any point you occupy.",
    "must use dice": "You must use both dice if any sequence allows it. If only one can be used, you must use the larger when possible. If nothing is legal, your turn passes.",
    "doubling cube": "Before rolling you may offer to double the stakes. Your opponent either takes (the game continues for twice the points, and they own the cube) or drops (resigns at the current stakes).",
    "gammon": "Winning before your opponent has borne off any checker is a gammon (double points); if they still have a checker on the bar or in your home board it is a backgammon (triple).",
    "pip count": "The pip count is the total number of points your checkers must travel to bear off. Lower is better in a race.",
}


def rule(topic: str) -> str:
    topic = topic.lower().strip()
    for key, text in RULES.items():
        if topic in key or key in topic:
            return text
    return "Topics: " + ", ".join(RULES)


@dataclass
class Hint:
    lines: list[str]
    best: Play

    def to_text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class Critique:
    annotation: Annotation
    grade: str
    best: Play
    text: str


class Tutor:
    """The teacher-mode companion: suggests plays and reviews the learner's choices."""

    def __init__(self, annotator: Annotator | None = None, depth: int = 1) -> None:
        self.annotator = annotator or Annotator(depth=depth)
        self.depth = depth

    def hint(self, board: Board, player: Player, roll: Roll, top: int = 3) -> Hint:
        ranked = rank_plays(board, player, roll, depth=self.depth, width=SEARCH_WIDTH)
        legal = legal_plays(board, player, roll)
        lines = [f"You rolled {roll}. There {'is' if len(legal) == 1 else 'are'} {len(legal)} distinct legal play{'s' * (len(legal) != 1)}."]
        if board.bar[player]:
            lines.append("Remember: your checker on the bar must enter first. " + rule("bar"))
        lines += blocked_notes(board, player, roll)
        for i, c in enumerate(ranked[:top], 1):
            themes = play_themes(board, player, c.play)
            lines.append(f"{i}. {c.play.notation()}: " + " ".join(t.text for t in themes[:2]))
        if len(ranked) > 1:
            lines.append(f"The engine's pick is {ranked[0].play.notation()}.")
        return Hint(lines, ranked[0].play)

    def critique(self, board: Board, player: Player, roll: Roll, play: Play) -> Critique:
        ranked = rank_plays(board, player, roll, depth=self.depth, width=SEARCH_WIDTH)
        ann = self.annotator.annotate(board, player, roll, play, candidates=ranked)
        if ann.legal_count <= 1:
            return Critique(ann, "forced", play, "That was the only legal play.")
        label = grade(ann.equity_loss)
        best = ranked[0].play
        if ann.rank == 1:
            text = f"Well played: {play.notation()} is {label}."
        else:
            mine = {t.key for t in play_themes(board, player, play)}
            theirs = play_themes(board, player, best)
            missed = [t.text for t in theirs if t.key not in mine][:2]
            text = f"{play.notation()} is {label}. A stronger play was {best.notation()}."
            if missed:
                text += " It would have: " + " ".join(missed)
        return Critique(ann, label, best, text)
