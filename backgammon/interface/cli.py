"""Terminal front end."""

from __future__ import annotations

import sys
import time
from typing import Callable, TextIO

from backgammon.annotation import rule
from backgammon.engine import IllegalAction, Player
from backgammon.engine.text import render

from .session import Event, GameSession

HELP = """Commands:
  <move>          play a move in notation, e.g. 13/8 6/5, bar/22 13/9, 6/off 5/off, 8/5(2)
  r, roll         roll the dice
  d, double       offer a double (before rolling)
  t, take         accept a double          p, drop    decline a double
  h, hint         teacher mode: suggest a play or cube action
  b, board        show the board           rules [topic]   explain a rule
  review          annotate every move of the game so far
  q, quit         leave"""


class TextUI:
    def __init__(self, session: GameSession, input_fn: Callable[[str], str] = input, out: TextIO = sys.stdout,
                 show_legality: bool = True, delay: float = 0.0) -> None:
        self.session = session
        self.input = input_fn
        self.out = out
        self.show_legality = show_legality
        self.delay = delay
        session.listeners.append(self.on_event)

    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    def on_event(self, event: Event) -> None:
        if event.kind == "hint":
            self.say(indent(event.text))
            return
        self.say(("Coach: " if event.kind == "critique" else "") + event.text)
        if event.annotation is not None:
            self.say(indent(event.annotation.to_text(legality=self.show_legality)))

    def show_board(self) -> None:
        human = self.session.human_player()
        perspective = human if human is not None else self.session.game.turn
        self.say(render(self.session.board, perspective))
        state = self.session.game.state
        owner = "centred" if state.cube_owner is None else f"owned by {self.session.name(state.cube_owner)}"
        self.say(f"   cube: {state.cube_value} ({owner})")

    def run(self, games: int = 1) -> None:
        s = self.session
        human = s.human_player()
        if human is not None:
            who = f"You play {human} against {s.name(human.opponent)}"
        else:
            who = f"{s.name(Player.WHITE)} (White) against {s.name(Player.BLACK)} (Black)"
        self.say(f"Mode: {s.mode.value}. {who}. Type 'help' for commands.")
        s.start()
        played = 0
        while True:
            self.advance_ai()
            if s.game.over:
                played += 1
                if s.human_player() is None:
                    if played >= games:
                        return
                    s.new_game()
                    continue
                if not self.after_game():
                    return
                continue
            if not self.human_turn():
                return

    def after_game(self) -> bool:
        """Offer a review and another game. Returns False to quit."""
        while True:
            answer = self.ask("Game over: [n]ew game, [r]eview the moves, or [q]uit? ").strip().lower()
            if answer.startswith("r"):
                self.review()
            elif answer.startswith("n"):
                self.session.new_game()
                return True
            elif answer.startswith("q"):
                return False

    def advance_ai(self) -> None:
        while self.session.step():
            if self.delay:
                time.sleep(self.delay)

    def ask(self, prompt: str) -> str:
        try:
            return self.input(prompt)
        except EOFError:
            return "quit"

    def human_turn(self) -> bool:
        """Handle one human command. Returns False to quit."""
        s = self.session
        needed = s.waiting_for_human()
        if needed == "move":
            self.show_board()
            prompt = f"{s.game.state.roll} to play> "
        elif needed == "roll":
            if not s.game.can_double():
                s.human_roll()  # nothing to decide: roll automatically
                return True
            prompt = "roll or double> "
        else:
            prompt = f"{s.name(s.game.turn)} doubles. take or drop> "
        command = self.ask(prompt).strip()
        low = command.lower()
        try:
            if low in ("q", "quit", "exit"):
                return False
            if low in ("help", "?"):
                self.say(HELP)
            elif low in ("b", "board"):
                self.show_board()
            elif low.startswith("rules"):
                self.say(rule(command[5:]))
            elif low in ("h", "hint"):
                s.hint()
            elif low == "review":
                self.review()
            elif low in ("r", "roll"):
                s.human_roll()
            elif low in ("d", "double"):
                s.human_double()
            elif low in ("t", "take"):
                s.human_respond(True)
            elif low in ("p", "drop", "pass"):
                s.human_respond(False)
            elif needed == "move" and command:
                s.human_move_text(command)
            elif command:
                self.say(f"Unknown command '{command}'. Type 'help'.")
        except IllegalAction as exc:
            self.say(f"Not now: {exc}.")
        return True

    def review(self) -> None:
        for i, note in enumerate(self.session.review(), 1):
            self.say(f"--- move {i} ---")
            self.say(note.to_text(legality=False))


def indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())
