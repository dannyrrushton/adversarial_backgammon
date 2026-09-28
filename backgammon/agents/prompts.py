"""Prompt construction for LLM agents."""

from __future__ import annotations

from dataclasses import dataclass, field

from backgammon.engine import Board, Player, Roll
from backgammon.engine.text import SYMBOL, render

from .search import Candidate

PERSONAS = {
    "balanced": "You weigh safety, structure and racing chances evenly, like a strong tournament player.",
    "aggressive": "You love to attack: hit loose checkers, blitz, and play for gammons when the risk is justified.",
    "positional": "You play for structure: build primes, keep anchors, and avoid unnecessary blots.",
    "teacher": (
        "You are also coaching a beginner who is watching your moves, so explain your reasoning in "
        "plain language and name the backgammon concept involved (making points, priming, "
        "hitting, safety, racing)."
    ),
}

CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer", "minimum": 1},
        "reasoning": {"type": "string"},
    },
    "required": ["choice", "reasoning"],
}


@dataclass
class GameContext:
    """What an agent knows about the game beyond the board."""

    opponent_name: str = "your opponent"
    cube_value: int = 1
    score: tuple[int, int] = (0, 0)  # (this agent, opponent)
    recent: list[str] = field(default_factory=list)  # recent turns, oldest first


def system_prompt(name: str, persona: str) -> str:
    style = PERSONAS.get(persona, persona)
    return (
        f"You are {name}, an expert backgammon player in a head-to-head game. Your only goal is to "
        f"beat your opponent. {style}\n"
        "Each turn a search engine lists the legal candidate plays with an estimated equity "
        "(expected points for you, higher is better; about 0.1 is a real difference). The engine is "
        "good at tactics but crude at long-term planning, so use your judgment about priming, "
        "timing, hitting, safety, racing and gammon chances. Pick exactly one candidate by number "
        "and explain the decisive idea in one to three sentences."
    )


def describe_candidate(i: int, c: Candidate, themes: list[str]) -> str:
    notes = "; ".join(themes[:3]) if themes else "quiet move"
    return f"{i}. {c.play.notation()}  (equity {c.equity:+.3f}) - {notes}"


def move_prompt(
    board: Board,
    player: Player,
    roll: Roll,
    candidates: list[Candidate],
    themes: list[list[str]],
    context: GameContext,
) -> str:
    me, opp = SYMBOL[player], SYMBOL[player.opponent]
    lines = [
        f"You are {player} ({me}); {context.opponent_name} is {player.opponent} ({opp}).",
        f"You move from point 24 toward point 1 and bear off below 1. Cube value: {context.cube_value}.",
        f"Score: you {context.score[0]}, opponent {context.score[1]}.",
        "",
        render(board, player),
        "",
    ]
    if context.recent:
        lines += ["Recent turns:", *[f"  {r}" for r in context.recent[-4:]], ""]
    lines += [f"You rolled {roll}. Candidate plays:"]
    lines += [describe_candidate(i + 1, c, t) for i, (c, t) in enumerate(zip(candidates, themes))]
    lines += ["", 'Reply as JSON: {"choice": <candidate number>, "reasoning": "<why>"}']
    return "\n".join(lines)
