import json
import random

import pytest

from backgammon.agents import (
    GameContext,
    HeuristicAgent,
    OllamaAgent,
    OllamaClient,
    OllamaError,
    RandomAgent,
    equity,
    features,
    rank_plays,
    win_chance_on_roll,
)
from backgammon.agents.evaluator import longest_prime, race_win_probability, shots_at
from backgammon.agents.prompts import move_prompt
from backgammon.engine import Board, Dice, Game, Phase, Player, Roll, legal_plays

W, B = Player.WHITE, Player.BLACK


class FakeClient(OllamaClient):
    """Returns canned replies instead of calling a server; records the prompts it saw."""

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, schema=None, temperature=0.4, think=False):
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


# --- evaluator --------------------------------------------------------------------------------


def test_equity_is_roughly_symmetric_at_start():
    assert abs(equity(Board.initial(), W) - equity(Board.initial(), B)) < 1e-9


def test_terminal_equity():
    won = Board.from_relative({}, {6: 15}, off=(15, 0))
    assert equity(won, W) == 2 and equity(won, B) == -2


def test_direct_shot_count():
    # White blot on 10 (Black's 15), single Black checker 6 pips behind it on Black's 21: any
    # roll containing a 6 (11) plus 5-1, 4-2 and 3-3 = 11 + 2 + 2 + 1 = 16. 2-2 would need to
    # land on White's 6-point, which is blocked.
    board = Board.from_relative({10: 1, 6: 14}, {21: 1, 6: 14})
    assert shots_at(board, W) == 16


def test_blocked_combination_shot_not_counted():
    # Same, but White holds every intermediate landing point (White's 5..9), so only the 11
    # rolls containing a direct 6 hit.
    board = Board.from_relative({10: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 13: 4}, {21: 1, 6: 14})
    assert shots_at(board, W) == 11


def test_prime_length():
    assert longest_prime([4, 5, 6, 7, 9, 13]) == 4


def test_race_probability_monotonic():
    assert race_win_probability(60, 80, on_roll=True) > race_win_probability(70, 80, on_roll=True) > 0.5
    assert race_win_probability(80, 80, on_roll=True) > race_win_probability(80, 80, on_roll=False)


def test_features_initial():
    f = features(Board.initial(), W)
    assert f.pips == 167 and f.blots == () and f.home_points == (6,) and f.anchors == (24,)
    assert f.back_checkers == 2 and f.contact


# --- search -----------------------------------------------------------------------------------


def test_ranking_prefers_classic_openings():
    for roll, best in [(Roll(3, 1), "8/5 6/5"), (Roll(4, 2), "8/4 6/4"), (Roll(6, 1), "13/7 8/7")]:
        assert rank_plays(Board.initial(), W, roll, depth=1)[0].play.notation() == best


def test_ranking_takes_winning_bearoff():
    board = Board.from_relative({2: 1, 1: 1}, {6: 10, 20: 1}, off=(13, 4))
    assert rank_plays(board, W, Roll(2, 1), depth=1)[0].play.result.winner() is W


def test_ranking_hits_when_obvious():
    # Black blot on White's 5-point; hitting it with 8/5* 6/5 makes the golden point.
    board = Board.from_relative({8: 3, 6: 5, 13: 5, 24: 2}, {20: 1, 24: 1, 13: 5, 8: 3, 6: 5})
    top = rank_plays(board, W, Roll(3, 1), depth=0)[0].play
    assert top.hits == 1 and top.result.count(W, 5) == 2


# --- agents -----------------------------------------------------------------------------------


def play_game(white, black, seed):
    game = Game(Dice(random.Random(seed)), use_cube=False)
    game.opening_roll()
    while not game.over:
        if game.phase is Phase.AWAIT_ROLL:
            game.roll()
        agent = white if game.turn is W else black
        game.play(agent.choose_play(game.board, game.turn, game.state.roll).play)
    return game.state.result.winner


def test_heuristic_beats_random():
    wins = sum(play_game(HeuristicAgent(depth=0), RandomAgent(seed=i), seed=i) is W for i in range(12))
    assert wins >= 10


def test_noisy_heuristic_stays_among_top_plays():
    agent = HeuristicAgent(depth=0, noise=0.2, seed=1)
    ranked = [c.play.notation() for c in rank_plays(Board.initial(), W, Roll(5, 3), depth=0)[:4]]
    for _ in range(10):
        assert agent.choose_play(Board.initial(), W, Roll(5, 3)).play.notation() in ranked


def test_ollama_agent_uses_model_choice():
    client = FakeClient([json.dumps({"choice": 2, "reasoning": "Build structure."})])
    agent = OllamaAgent(client, name="Tester", shortlist=4, depth=0)
    decision = agent.choose_play(Board.initial(), W, Roll(3, 1), GameContext(opponent_name="Rival"))
    assert decision.source == "llm" and decision.reasoning == "Build structure."
    assert decision.play == decision.candidates[1].play
    prompt = client.seen[0][1]["content"]
    assert "You rolled 3-1" in prompt and "Rival" in prompt and "4." in prompt and "5." not in prompt


def test_ollama_agent_falls_back_on_bad_answers():
    for reply in [json.dumps({"choice": 99, "reasoning": "x"}), "not json", OllamaError("down")]:
        agent = OllamaAgent(FakeClient([reply]), depth=0)
        decision = agent.choose_play(Board.initial(), W, Roll(3, 1))
        assert decision.source == "fallback"
        assert decision.play == decision.candidates[0].play
        assert agent.last_error


def test_json_embedded_in_prose_is_recovered():
    client = FakeClient(['Sure! {"choice": 1, "reasoning": "ok"} hope that helps'])
    assert client.chat_json([], {})["choice"] == 1


def test_forced_play_skips_llm():
    closed = {p: 2 for p in range(1, 7)}
    closed[13] = 3
    board = Board.from_relative({6: 14}, closed, bar=(1, 0), off=(0, 0))
    client = FakeClient([])
    decision = OllamaAgent(client, depth=0).choose_play(board, W, Roll(6, 6))
    assert decision.source == "forced" and decision.play.is_pass and not client.seen


def test_prompt_contains_board_and_candidates():
    ranked = rank_plays(Board.initial(), W, Roll(6, 5), depth=0)[:2]
    text = move_prompt(Board.initial(), W, Roll(6, 5), ranked, [["escapes"], []], GameContext())
    assert "24/18 18/13" in text and "pips" in text and "escapes" in text


# --- cube -------------------------------------------------------------------------------------


def test_cube_decisions():
    agent = HeuristicAgent(depth=0)
    even = Board.initial()
    assert not agent.offer_double(even, W, 1)
    assert agent.accept_double(even, B, 1)
    # Racing lead of ~12%: a double, and still a take.
    race = Board.from_relative({6: 4, 5: 4, 4: 4, 3: 3}, {6: 5, 5: 4, 4: 3, 3: 3}, off=(0, 0))
    lead = Board.from_relative({6: 2, 5: 4, 4: 4, 3: 3, 2: 2}, {6: 5, 5: 5, 4: 5}, off=(0, 0))
    assert win_chance_on_roll(lead, W) > win_chance_on_roll(race, W)
    hopeless = Board.from_relative({1: 3}, {6: 15}, off=(12, 0))
    assert not agent.accept_double(hopeless, B, 1)
    assert not RandomAgent().offer_double(hopeless, W, 1)


# --- live server (opt-in) ---------------------------------------------------------------------


@pytest.mark.ollama
def test_live_ollama_move():
    client = OllamaClient()
    if not client.is_available():
        pytest.skip(f"Ollama model {client.model} not available")
    decision = OllamaAgent(client, depth=0).choose_play(Board.initial(), W, Roll(4, 1))
    assert decision.source == "llm"
    assert decision.play in legal_plays(Board.initial(), W, Roll(4, 1))


# --- calibration ------------------------------------------------------------------------------


def test_opening_win_chances_are_realistic():
    # The side on roll at the start wins a little over half the time.
    assert 0.5 < win_chance_on_roll(Board.initial(), W) < 0.6


def test_calibration_fit_recovers_known_weights():
    import numpy as np

    from backgammon.agents.calibrate import fit

    rng = np.random.default_rng(0)
    x = rng.normal(size=(20_000, 2))
    true = np.array([1.5, -0.7, 0.3])
    p = 1 / (1 + np.exp(-(x @ true[:2] + true[2])))
    y = (rng.random(len(p)) < p).astype(float)
    assert np.allclose(fit(x, y), true, atol=0.1)
