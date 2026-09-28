"""Refit ``CALIBRATED_WEIGHTS`` in evaluator.py from self-play outcomes.

    python -m backgammon.agents.calibrate --games 400

Plays noisy heuristic agents against each other, records the evaluator's score components for
every contact position (from the side waiting for the opponent's roll) together with who went on
to win, and fits a logistic regression by Newton's method.
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from backgammon.engine import Dice, Game, Phase

from .agent import HeuristicAgent
from .evaluator import score_components


def collect(games: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(games):
        agents = [HeuristicAgent(depth=0, noise=0.15, seed=rng.random()) for _ in range(2)]
        game = Game(Dice(rng), use_cube=False)
        game.opening_roll()
        samples = []
        while not game.over:
            if game.phase is Phase.AWAIT_ROLL:
                waiting = game.turn.opponent
                if game.board.has_contact():
                    samples.append((score_components(game.board, waiting), waiting))
                game.roll()
            game.play(agents[game.turn].choose_play(game.board, game.turn, game.state.roll).play)
        for components, player in samples:
            xs.append(components)
            ys.append(1.0 if game.state.result.winner is player else 0.0)
    return np.array(xs), np.array(ys)


def fit(x: np.ndarray, y: np.ndarray, iterations: int = 50) -> np.ndarray:
    xb = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(xb.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-xb @ w))
        hessian = -(xb * (p * (1 - p))[:, None]).T @ xb - 1e-6 * np.eye(len(w))
        w -= np.linalg.solve(hessian, xb.T @ (y - p))
    return w


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    x, y = collect(args.games, args.seed)
    w = fit(x, y)
    print(f"{len(y)} positions; CALIBRATED_WEIGHTS = {tuple(round(float(v), 4) for v in w)}")
    p = 1 / (1 + np.exp(-(np.hstack([x, np.ones((len(x), 1))]) @ w)))
    for lo in np.arange(0, 1, 0.1):
        mask = (p >= lo) & (p < lo + 0.1)
        if mask.any():
            print(f"  predicted {lo:.1f}-{lo + 0.1:.1f}: {mask.sum():5d} positions, actual win rate {y[mask].mean():.2f}")


if __name__ == "__main__":
    main()
