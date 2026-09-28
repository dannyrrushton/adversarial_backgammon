"""Backgammon agents: random, heuristic search, and LLM (Ollama) players."""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from backgammon.engine import Board, Play, Player, Roll, legal_plays

from .evaluator import features, race_win_probability, win_probability
from .ollama_client import OllamaClient, OllamaError
from .prompts import CHOICE_SCHEMA, GameContext, move_prompt, system_prompt
from .search import Candidate, rank_plays

log = logging.getLogger(__name__)


@dataclass
class Decision:
    play: Play
    candidates: list[Candidate] = field(default_factory=list)
    reasoning: str = ""
    source: str = "heuristic"  # "llm", "heuristic", "fallback", "random" or "forced"


def win_chance_on_roll(board: Board, player: Player) -> float:
    """``player``'s winning chance when ``player`` is about to roll."""
    f = features(board, player)
    if not f.contact:
        return race_win_probability(f.pips, f.opp_pips, on_roll=True)
    # The evaluator scores positions with the *other* side to roll, so look from the opponent.
    return 1 - win_probability(board, player.opponent)


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def choose_play(self, board: Board, player: Player, roll: Roll, context: GameContext | None = None) -> Decision: ...

    # Cube decisions use the evaluator's win estimate for every agent: an LLM gives no better
    # calibrated probabilities than the formula, and cube errors are expensive.
    double_window = (0.68, 0.86)
    take_point = 0.25

    def offer_double(self, board: Board, player: Player, cube_value: int) -> bool:
        low, high = self.double_window
        return low <= win_chance_on_roll(board, player) < high

    def accept_double(self, board: Board, player: Player, cube_value: int) -> bool:
        # The doubler is on roll, so our chance is 1 - theirs.
        return 1 - win_chance_on_roll(board, player.opponent) >= self.take_point


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None, name: str = "Random") -> None:
        self.rng = random.Random(seed)
        self.name = name

    def choose_play(self, board, player, roll, context=None):
        return Decision(self.rng.choice(legal_plays(board, player, roll)), source="random")

    def offer_double(self, board, player, cube_value):
        return False

    def accept_double(self, board, player, cube_value):
        return True


class HeuristicAgent(Agent):
    """Plays the evaluator's best move.

    ``noise`` > 0 makes it choose among the top plays with a softmax at that temperature (in
    equity units), which gives a gentler opponent for learners.
    """

    def __init__(self, depth: int = 1, width: int = 6, noise: float = 0.0, seed: int | None = None, name: str = "Engine") -> None:
        self.depth = depth
        self.width = width
        self.noise = noise
        self.rng = random.Random(seed)
        self.name = name

    def rank(self, board: Board, player: Player, roll: Roll) -> list[Candidate]:
        return rank_plays(board, player, roll, depth=self.depth, width=self.width)

    def choose_play(self, board, player, roll, context=None):
        ranked = self.rank(board, player, roll)
        if len(ranked) == 1:
            return Decision(ranked[0].play, ranked, source="forced")
        pick = ranked[0]
        if self.noise > 0:
            top = ranked[:4]
            weights = [math.exp((c.equity - top[0].equity) / self.noise) for c in top]
            pick = self.rng.choices(top, weights)[0]
        return Decision(pick.play, ranked, source="heuristic")


class OllamaAgent(HeuristicAgent):
    """An LLM that picks from the engine's shortlist and explains its choice.

    Falls back to the engine's top play when the model is unreachable or answers nonsense, so a
    game never stalls on the LLM.
    """

    def __init__(
        self,
        client: OllamaClient | None = None,
        name: str = "Ollama",
        persona: str = "balanced",
        shortlist: int = 5,
        depth: int = 1,
        temperature: float = 0.4,
        think: bool = False,
    ) -> None:
        super().__init__(depth=depth, width=max(shortlist, 6), name=name)
        self.client = client or OllamaClient()
        self.persona = persona
        self.shortlist = shortlist
        self.temperature = temperature
        self.think = think
        self.last_error: str | None = None

    def choose_play(self, board, player, roll, context=None):
        # Imported here: the annotation package itself builds on this package's evaluator.
        from backgammon.annotation.themes import play_themes

        ranked = self.rank(board, player, roll)
        if len(ranked) == 1:
            return Decision(ranked[0].play, ranked, source="forced")
        context = context or GameContext()
        shortlist = ranked[: self.shortlist]
        themes = [[t.text for t in play_themes(board, player, c.play)] for c in shortlist]
        messages = [
            {"role": "system", "content": system_prompt(self.name, self.persona)},
            {"role": "user", "content": move_prompt(board, player, roll, shortlist, themes, context)},
        ]
        try:
            answer = self.client.chat_json(messages, CHOICE_SCHEMA, temperature=self.temperature, think=self.think)
            choice = int(answer["choice"])
            if not 1 <= choice <= len(shortlist):
                raise ValueError(f"choice {choice} is not between 1 and {len(shortlist)}")
            self.last_error = None
            return Decision(shortlist[choice - 1].play, ranked, str(answer.get("reasoning", "")).strip(), "llm")
        except (OllamaError, KeyError, TypeError, ValueError) as exc:
            self.last_error = str(exc)
            log.warning("%s: LLM move failed (%s); using the engine's choice", self.name, exc)
            return Decision(ranked[0].play, ranked, f"(engine choice; LLM unavailable: {exc})", "fallback")
