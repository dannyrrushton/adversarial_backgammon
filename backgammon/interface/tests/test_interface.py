import io
import os
import random
import time

import pytest

from backgammon.agents import HeuristicAgent, RandomAgent
from backgammon.engine import Dice, IllegalAction, Phase, Player, legal_plays
from backgammon.interface import GameSession, Mode, Options, Seat, SessionConfig, build_session
from backgammon.interface.cli import TextUI

W, B = Player.WHITE, Player.BLACK


def make_session(mode, human=W, seed=1, opponent=None, use_cube=True):
    opponent = opponent or HeuristicAgent(depth=0, name="Bot")
    if mode is Mode.WATCH:
        seats = {W: Seat("A", HeuristicAgent(depth=0, name="A")), B: Seat("B", RandomAgent(seed, name="B"))}
    else:
        seats = {human: Seat("You"), human.opponent: Seat(opponent.name, opponent)}
    return GameSession(seats, SessionConfig(mode=mode, use_cube=use_cube), dice=Dice(random.Random(seed)))


def play_human_game(session, choose=lambda plays: plays[0], respond=True, max_steps=5000):
    """Drive a session with a scripted human until the game ends."""
    session.start()
    for _ in range(max_steps):
        if session.game.over:
            return
        needed = session.waiting_for_human()
        if needed is None:
            assert session.step()
        elif needed == "roll":
            session.human_roll()
        elif needed == "move":
            session.human_move(choose(session.game.legal_plays()))
        else:
            session.human_respond(respond)
    raise AssertionError("game did not finish")


def kinds(session):
    return [e.kind for e in session.events]


# --- sessions -----------------------------------------------------------------------------------


def test_watch_mode_plays_full_annotated_game():
    s = make_session(Mode.WATCH)
    s.start()
    s.run_ai()
    assert s.game.over and "game_over" in kinds(s)
    moves = [e for e in s.events if e.kind == "move"]
    assert moves and all(e.annotation is not None for e in moves if "no legal move" not in e.text)
    assert sum(s.score.values()) == s.game.state.result.points


def test_teacher_mode_critiques_human_and_explains_ai():
    s = make_session(Mode.TEACHER, use_cube=False)
    play_human_game(s)
    critiques = [e for e in s.events if e.kind == "critique"]
    ai_moves = [e for e in s.events if e.kind == "move" and e.player is B and e.annotation]
    assert critiques and ai_moves
    assert any("Why it is legal" in e.annotation.to_text() for e in ai_moves)


def test_adversary_mode_hides_reasoning_and_hints():
    s = make_session(Mode.ADVERSARY, use_cube=False)
    play_human_game(s)
    assert "critique" not in kinds(s)
    assert all(e.annotation is None for e in s.events if e.kind == "move")
    assert "only available in teacher mode" in s.hint()
    review = s.review()
    assert len(review) == sum(1 for e in s.events if e.kind == "move")
    assert all(r.legality for r in review)


def test_black_human_and_review_uses_ai_reasoning():
    s = make_session(Mode.ADVERSARY, human=B, use_cube=False)
    play_human_game(s)
    assert s.game.over and s.human_player() is B


def test_hint_in_teacher_mode():
    s = make_session(Mode.TEACHER, use_cube=False)
    s.start()
    s.run_ai()
    assert s.waiting_for_human() in ("move", "roll")
    if s.waiting_for_human() == "roll":
        s.human_roll()
        s.run_ai()
    if s.waiting_for_human() == "move":
        text = s.hint()
        assert "legal play" in text and "engine's pick" in text


def test_illegal_typed_move_is_explained():
    s = make_session(Mode.TEACHER, use_cube=False)
    s.start()
    while s.waiting_for_human() != "move":
        s.human_roll() if s.waiting_for_human() == "roll" else s.step()
    verdict = s.human_move_text("24/1")
    assert not verdict.legal
    assert s.events[-1].kind == "error" and "Illegal move" in s.events[-1].text
    assert s.waiting_for_human() == "move"  # still our turn
    play = s.game.legal_plays()[0]
    assert s.human_move_text(play.notation()).legal


def test_out_of_turn_actions_raise():
    s = make_session(Mode.ADVERSARY)
    s.start()
    while s.waiting_for_human() != "move":
        s.human_roll() if s.waiting_for_human() == "roll" else s.step()
    with pytest.raises(IllegalAction):
        s.human_roll()
    with pytest.raises(IllegalAction):
        s.human_respond(True)


class AlwaysDoubles(HeuristicAgent):
    def offer_double(self, board, player, cube_value):
        return True


def test_ai_double_and_human_drop():
    s = make_session(Mode.ADVERSARY, opponent=AlwaysDoubles(depth=0, name="Doubler"))
    play_human_game(s, respond=False)
    result = s.game.state.result
    assert result.dropped and result.winner is B and s.score[B] == 1
    assert "double" in kinds(s) and "drop" in kinds(s)


def test_ai_double_and_human_take():
    s = make_session(Mode.TEACHER, opponent=AlwaysDoubles(depth=0, name="Doubler"))
    s.start()
    while s.waiting_for_human() != "double":
        needed = s.waiting_for_human()
        if needed == "roll":
            s.human_roll()
        elif needed == "move":
            s.human_move(s.game.legal_plays()[0])
        else:
            s.step()
    assert "take" in s.hint() or "drop" in s.hint()
    s.human_respond(True)
    assert s.game.state.cube_value == 2 and s.game.state.cube_owner is W


def test_human_double_is_answered_by_ai():
    s = make_session(Mode.ADVERSARY)
    s.start()
    while s.waiting_for_human() != "roll":
        if s.waiting_for_human() == "move":
            s.human_move(s.game.legal_plays()[0])
        else:
            s.step()
    s.human_double()
    assert s.game.phase is Phase.AWAIT_DOUBLE_RESPONSE
    s.step()
    assert kinds(s)[-1] in ("take", "drop", "game_over")


def test_new_game_keeps_score():
    s = make_session(Mode.WATCH)
    s.start()
    s.run_ai()
    first = dict(s.score)
    s.new_game()
    s.run_ai()
    assert s.games_played == 2 and sum(s.score.values()) > sum(first.values())


# --- factory --------------------------------------------------------------------------------------


def test_factory_without_llm():
    session, warnings = build_session(Options(mode=Mode.ADVERSARY, use_llm=False, seed=3))
    assert not warnings
    assert session.human_player() is W and isinstance(session.seat(B).agent, HeuristicAgent)


def test_factory_falls_back_when_ollama_unreachable():
    session, warnings = build_session(Options(mode=Mode.WATCH, host="http://127.0.0.1:9", seed=3))
    assert len(warnings) == 2 and "built-in engine" in warnings[0]
    assert session.human_player() is None


# --- text UI --------------------------------------------------------------------------------------


def scripted_input(session, log):
    def respond(prompt):
        log.append(prompt)
        if prompt.startswith("Game over"):
            return "q"
        if "take or drop" in prompt:
            return "take"
        if "roll or double" in prompt:
            return "roll"
        if "to play" in prompt:
            if not any(p.startswith("hint") for p in log):
                log.append("hint")
                return "hint"
            return legal_plays(session.board, session.actor(), session.game.state.roll)[0].notation()
        raise AssertionError(prompt)

    return respond


def test_text_ui_plays_a_game():
    s = make_session(Mode.TEACHER, use_cube=False)
    out = io.StringIO()
    prompts = []
    TextUI(s, input_fn=scripted_input(s, prompts), out=out).run()
    text = out.getvalue()
    assert s.game.over
    assert "Opening roll" in text and "Coach:" in text and "Why it is legal" in text and "wins" in text
    assert "engine's pick" in text  # the hint was shown


def test_text_ui_commands():
    s = make_session(Mode.ADVERSARY, use_cube=False)
    out = io.StringIO()
    commands = iter(["help", "rules bar", "board", "bogus", "hint", "q"])
    ui = TextUI(s, input_fn=lambda prompt: next(commands), out=out)
    s.start()
    ui.advance_ai()
    while ui.human_turn():
        ui.advance_ai()
    text = out.getvalue()
    assert "Commands:" in text and "must enter it first" in text and "pips" in text
    assert "Illegal move: cannot parse move 'bogus'" in text and "only available in teacher mode" in text


def test_text_ui_watch_mode():
    s = make_session(Mode.WATCH)
    out = io.StringIO()
    TextUI(s, out=out).run(games=2)
    assert s.games_played == 2 and out.getvalue().count(" wins ") >= 2


# --- GUI (offscreen Qt + GPU) -----------------------------------------------------------------------


def test_gui_click_to_move():
    from backgammon.render import available

    if not available():
        pytest.skip("OptiX renderer not built")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from backgammon.engine import BAR, OFF, absolute_index
    from backgammon.interface.gui import GameWindow
    from backgammon.render import Pick

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = make_session(Mode.TEACHER, use_cube=False, seed=5)
    window = GameWindow(s, ai_delay_ms=10)
    window.show()

    def wait(condition, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if condition():
                return
            time.sleep(0.01)
        raise AssertionError(f"timed out: busy={window.busy} waiting={s.waiting_for_human()} "
                             f"builder={window.builder and window.builder.moves} events={s.events[-3:]}")

    def to_pick(player, spot):
        if spot == BAR:
            return Pick("bar", side=player)
        if spot == OFF:
            return Pick("off", side=player)
        return Pick("point", index=absolute_index(player, spot))

    moves_made = 0
    while moves_made < 3:  # three human turns made by clicking on the board
        wait(lambda: not window.busy and s.waiting_for_human() in ("roll", "move"))
        if s.waiting_for_human() == "roll":
            window.buttons["roll"].click()
            # A roll with no legal move passes straight back to the AI.
            wait(lambda: not window.busy and s.waiting_for_human() in ("move", "roll"))
            if s.waiting_for_human() != "move":
                continue
        play = window.builder.plays[0]
        before = len(s.game.state.history)
        for move in play.moves:
            if window.selected != move.src:
                window.on_pick(to_pick(W, move.src))
            window.on_pick(to_pick(W, move.dst))
        wait(lambda: len(s.game.state.history) > before and not window.busy)
        moves_made += 1
    assert moves_made == 3
    assert "Coach" in window.log.toPlainText()
    window.close()
    app.processEvents()
