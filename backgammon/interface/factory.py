"""Builds sessions for each play mode from command-line style options."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from backgammon.agents import DEFAULT_MODEL, Agent, HeuristicAgent, OllamaAgent, OllamaClient
from backgammon.annotation import Annotator, Tutor
from backgammon.engine import Dice, Player

from .session import GameSession, Mode, Seat, SessionConfig

log = logging.getLogger(__name__)


@dataclass
class Options:
    mode: Mode = Mode.TEACHER
    human: Player = Player.WHITE
    model: str = DEFAULT_MODEL
    model2: str | None = None  # second model for watch mode (defaults to ``model``)
    host: str | None = None
    use_llm: bool = True
    use_cube: bool = True
    llm_commentary: bool = False
    seed: int | None = None
    human_name: str = "You"


def make_agent(options: Options, name: str, persona: str, model: str, strength: str) -> tuple[Agent, str | None]:
    """An LLM agent if the model is reachable, else the heuristic engine. Returns (agent, warning)."""
    # Teacher-mode opponents search less deeply and consider more candidates, so they play well
    # but beatably; adversaries search deeper and keep a tight shortlist of the strongest plays.
    depth, shortlist, noise = (0, 4, 0.12) if strength == "gentle" else (1, 3, 0.0)
    if options.use_llm:
        client = OllamaClient(model=model, **({"host": options.host} if options.host else {}))
        if client.is_available():
            return OllamaAgent(client, name=name, persona=persona, shortlist=shortlist, depth=depth), None
        warning = f"Ollama model '{model}' is not available at {client.host}; {name} uses the built-in engine."
    else:
        warning = None
    return HeuristicAgent(depth=depth, noise=noise, seed=options.seed, name=name), warning


def build_session(options: Options) -> tuple[GameSession, list[str]]:
    warnings: list[str] = []
    rng = random.Random(options.seed)
    human = options.human
    if options.mode is Mode.WATCH:
        a, w1 = make_agent(options, "Aggressor", "aggressive", options.model, "strong")
        b, w2 = make_agent(options, "Strategist", "positional", options.model2 or options.model, "strong")
        seats = {Player.WHITE: Seat(a.name, a), Player.BLACK: Seat(b.name, b)}
        warnings += [w for w in (w1, w2) if w]
    elif options.mode is Mode.TEACHER:
        coach, w = make_agent(options, "Coach", "teacher", options.model, "gentle")
        seats = {human: Seat(options.human_name), human.opponent: Seat(coach.name, coach)}
        warnings += [w] if w else []
    else:
        rival, w = make_agent(options, "Rival", "aggressive", options.model, "strong")
        seats = {human: Seat(options.human_name), human.opponent: Seat(rival.name, rival)}
        warnings += [w] if w else []

    client = None
    if options.llm_commentary and options.use_llm:
        client = OllamaClient(model=options.model, **({"host": options.host} if options.host else {}))
    annotator = Annotator(client, depth=0, llm_commentary=options.llm_commentary)
    session = GameSession(
        seats,
        SessionConfig(mode=options.mode, use_cube=options.use_cube),
        annotator=annotator,
        tutor=Tutor(Annotator(depth=1)),
        dice=Dice(rng),
    )
    for w in warnings:
        log.warning(w)
    return session, warnings
