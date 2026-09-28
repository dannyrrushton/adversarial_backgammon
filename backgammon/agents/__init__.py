"""Backgammon agents: heuristic search and LLM players backed by a local Ollama model."""

from .agent import Agent, Decision, HeuristicAgent, OllamaAgent, RandomAgent, win_chance_on_roll
from .evaluator import Features, equity, features, win_probability
from .ollama_client import DEFAULT_MODEL, OllamaClient, OllamaError
from .prompts import PERSONAS, GameContext
from .search import Candidate, rank_plays

__all__ = [
    "DEFAULT_MODEL", "PERSONAS", "Agent", "Candidate", "Decision", "Features", "GameContext",
    "HeuristicAgent", "OllamaAgent", "OllamaClient", "OllamaError", "RandomAgent", "equity",
    "features", "rank_plays", "win_chance_on_roll", "win_probability",
]
