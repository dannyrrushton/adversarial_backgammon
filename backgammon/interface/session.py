"""Game sessions: who plays each side, what the teacher says, and the flow of turns.

The session is UI-independent. Front ends call :meth:`GameSession.step` to let AI players act and
the ``human_*`` methods for the person at the board, then render :attr:`GameSession.events`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from backgammon.agents import Agent, Decision, GameContext, win_chance_on_roll
from backgammon.annotation import Annotation, Annotator, Tutor, rule
from backgammon.engine import (
    Board,
    Dice,
    Game,
    IllegalAction,
    Phase,
    Play,
    Player,
    TurnRecord,
    Verdict,
    match_notation,
)


class Mode(Enum):
    TEACHER = "teacher"  # the AI explains everything and coaches the human
    ADVERSARY = "adversary"  # the AI plays to win and keeps its reasoning to itself
    WATCH = "watch"  # AI against AI, fully annotated


@dataclass
class Event:
    kind: str  # opening, roll, move, double, take, drop, critique, hint, info, error, game_over
    text: str
    player: Player | None = None
    annotation: Annotation | None = None


@dataclass
class Seat:
    name: str
    agent: Agent | None = None  # None means a human sits here

    @property
    def is_human(self) -> bool:
        return self.agent is None


@dataclass
class SessionConfig:
    mode: Mode = Mode.TEACHER
    use_cube: bool = True
    critique_human: bool = True  # teacher mode: grade every human move
    annotate_ai: bool = True  # explain AI moves as they happen (teacher and watch modes)


class GameSession:
    def __init__(
        self,
        seats: dict[Player, Seat],
        config: SessionConfig | None = None,
        annotator: Annotator | None = None,
        tutor: Tutor | None = None,
        dice: Dice | None = None,
    ) -> None:
        self.seats = seats
        self.config = config or SessionConfig()
        self.annotator = annotator or Annotator(depth=0)
        self.tutor = tutor or Tutor(self.annotator)
        self.dice = dice or Dice()
        self.score = {Player.WHITE: 0, Player.BLACK: 0}
        self.events: list[Event] = []
        self.listeners: list[Callable[[Event], None]] = []
        self.annotations: list[Annotation] = []  # one per move of the current game
        self._decisions: dict[int, Decision] = {}  # history index -> the AI decision behind it
        self.games_played = 0
        self.game = Game(self.dice, use_cube=self.config.use_cube)

    # --- events -------------------------------------------------------------------------------

    def emit(self, kind: str, text: str, player: Player | None = None, annotation: Annotation | None = None) -> Event:
        event = Event(kind, text, player, annotation)
        self.events.append(event)
        for listener in self.listeners:
            listener(event)
        return event

    # --- state queries ------------------------------------------------------------------------

    @property
    def mode(self) -> Mode:
        return self.config.mode

    @property
    def board(self) -> Board:
        return self.game.board

    def seat(self, player: Player) -> Seat:
        return self.seats[player]

    def name(self, player: Player) -> str:
        return self.seats[player].name

    def verb(self, player: Player, verb: str) -> str:
        """``"rolls"`` for a named player, ``"roll"`` when the name is "You"."""
        if self.name(player) != "You":
            return verb
        return {"has": "have", "goes": "go"}.get(verb, verb[:-1] if verb.endswith("s") else verb)

    def human_player(self) -> Player | None:
        return next((p for p, s in self.seats.items() if s.is_human), None)

    def actor(self) -> Player | None:
        """Whose decision the game is waiting on."""
        phase = self.game.phase
        if phase is Phase.GAME_OVER:
            return None
        if phase is Phase.AWAIT_DOUBLE_RESPONSE:
            return self.game.turn.opponent
        return self.game.turn

    def waiting_for_human(self) -> str | None:
        """``"roll"``, ``"move"`` or ``"double"`` (respond to a double) if a human must act."""
        actor = self.actor()
        if actor is None or self.game.phase is Phase.OPENING or not self.seats[actor].is_human:
            return None
        return {Phase.AWAIT_ROLL: "roll", Phase.AWAIT_MOVE: "move", Phase.AWAIT_DOUBLE_RESPONSE: "double"}[self.game.phase]

    def context_for(self, player: Player) -> GameContext:
        recent = []
        for rec in self.game.state.history[-4:]:
            if isinstance(rec, TurnRecord):
                recent.append(f"{self.name(rec.player)} rolled {rec.roll} and played {rec.play.notation()}")
            else:
                recent.append(f"{self.name(rec.player)}: {rec.action} (cube {rec.value})")
        return GameContext(
            opponent_name=self.name(player.opponent),
            cube_value=self.game.state.cube_value,
            score=(self.score[player], self.score[player.opponent]),
            recent=recent,
        )

    # --- game lifecycle -----------------------------------------------------------------------

    def start(self) -> None:
        """Opening roll of a new game (call :meth:`new_game` first for later games)."""
        roll = self.game.opening_roll()
        first = self.game.turn
        self.emit(
            "opening",
            f"Opening roll: {self.name(Player.WHITE)} {roll.d1}, {self.name(Player.BLACK)} {roll.d2}. "
            f"{self.name(first)} {self.verb(first, 'goes')} first and {self.verb(first, 'plays')} {roll}.",
            first,
        )
        self._after_roll()

    def new_game(self) -> None:
        self.game = Game(self.dice, use_cube=self.config.use_cube)
        self.annotations = []
        self._decisions = {}
        self.start()

    def _after_roll(self) -> None:
        plays = self.game.legal_plays()
        player = self.game.turn
        if len(plays) == 1 and plays[0].is_pass:
            self.emit("info", f"{self.name(player)} rolled {self.game.state.roll} but {self.verb(player, 'has')} no legal move.", player)
            self._commit(plays[0], None)

    # --- AI turns -----------------------------------------------------------------------------

    def step(self) -> bool:
        """Let the AI take one action. Returns False when a human must act or the game is over."""
        actor = self.actor()
        if actor is None:
            return False
        if self.game.phase is Phase.OPENING:
            self.start()
            return True
        seat = self.seats[actor]
        if seat.is_human:
            return False
        agent = seat.agent
        phase = self.game.phase
        board = self.board
        if phase is Phase.AWAIT_ROLL:
            if self.game.can_double() and agent.offer_double(board, actor, self.game.state.cube_value):
                self._double()
            else:
                self._roll()
        elif phase is Phase.AWAIT_DOUBLE_RESPONSE:
            self._respond(agent.accept_double(board, actor, self.game.state.cube_value))
        elif phase is Phase.AWAIT_MOVE:
            decision = agent.choose_play(board, actor, self.game.state.roll, self.context_for(actor))
            self._commit(decision.play, decision)
        return True

    def run_ai(self, limit: int = 10_000) -> None:
        """Step until a human must act or the game ends."""
        for _ in range(limit):
            if not self.step():
                return

    # --- human actions ------------------------------------------------------------------------

    def _require_human(self, what: str) -> Player:
        needed = self.waiting_for_human()
        if needed != what:
            raise IllegalAction(f"it is not your turn to {what}" if needed is None else f"you need to {needed} now, not {what}")
        return self.actor()

    def human_roll(self) -> None:
        self._require_human("roll")
        self._roll()

    def human_double(self) -> None:
        player = self._require_human("roll")
        if not self.game.can_double(player):
            raise IllegalAction("you can't double now: " + ("the cube is disabled" if not self.config.use_cube else "your opponent owns the cube"))
        self._double()

    def human_respond(self, take: bool) -> None:
        self._require_human("double")
        self._respond(take)

    def human_move_text(self, text: str) -> Verdict:
        """Play a move typed in notation; returns the verdict (with the rule broken, if any)."""
        player = self._require_human("move")
        verdict = match_notation(self.board, player, self.game.state.roll, text)
        if verdict.legal:
            self.human_move(verdict.play)
        else:
            self.emit("error", f"Illegal move: {verdict.reason}.", player)
        return verdict

    def human_move(self, play: Play) -> None:
        player = self._require_human("move")
        before, roll = self.board, self.game.state.roll
        critique = None
        if self.mode is Mode.TEACHER and self.config.critique_human:
            critique = self.tutor.critique(before, player, roll, play)
        self._commit(play, None, annotation=critique.annotation if critique else None)
        if critique:
            self.emit("critique", critique.text, player, critique.annotation)

    def hint(self) -> str:
        if self.mode is not Mode.TEACHER:
            text = "Hints are only available in teacher mode. Your opponent is playing to win!"
            self.emit("hint", text)
            return text
        player = self.actor()
        needed = self.waiting_for_human()
        if needed == "move":
            text = self.tutor.hint(self.board, player, self.game.state.roll).to_text()
        elif needed == "roll":
            text = self.cube_advice(player)
        elif needed == "double":
            chance = 1 - win_chance_on_roll(self.board, player.opponent)
            verdict = "take" if chance >= 0.25 else "drop"
            text = f"Your winning chances are about {chance:.0%}. You need about 25% to take, so the engine says {verdict}."
        else:
            text = "Nothing to decide right now."
        self.emit("hint", text, player)
        return text

    def cube_advice(self, player: Player) -> str:
        chance = win_chance_on_roll(self.board, player)
        if not self.game.can_double(player):
            return f"Roll the dice. (Your winning chances are about {chance:.0%}.)"
        if chance >= 0.86:
            advice = "you are too good to double: play on for a gammon"
        elif chance >= 0.68:
            advice = "this is a good time to double"
        else:
            advice = "it is too early to double"
        return f"Your winning chances are about {chance:.0%}; {advice}. " + rule("doubling cube")

    # --- shared mechanics ---------------------------------------------------------------------

    def _roll(self) -> None:
        player = self.game.turn
        roll = self.game.roll()
        self.emit("roll", f"{self.name(player)} {self.verb(player, 'rolls')} {roll}.", player)
        self._after_roll()

    def _double(self) -> None:
        player = self.game.turn
        self.game.double()
        self.emit("double", f"{self.name(player)} {self.verb(player, 'doubles')} to {self.game.state.cube_value * 2}.", player)

    def _respond(self, take: bool) -> None:
        responder = self.game.turn.opponent
        if take:
            self.game.take()
            self.emit("take", f"{self.name(responder)} {self.verb(responder, 'takes')}; the cube is now {self.game.state.cube_value}.", responder)
        else:
            self.game.drop()
            self.emit("drop", f"{self.name(responder)} {self.verb(responder, 'drops')}.", responder)
            self._game_over()

    def _commit(self, play: Play, decision: Decision | None, annotation: Annotation | None = None) -> None:
        player, roll, before = self.game.turn, self.game.state.roll, self.board
        self.game.play(play)
        seat = self.seats[player]
        reasoning = decision.reasoning if decision else ""
        if annotation is None and self._should_annotate(seat):
            annotation = self.annotator.annotate(
                before, player, roll, play, candidates=decision.candidates if decision else None, reasoning=reasoning
            )
        if annotation is not None:
            self.annotations.append(annotation)
        index = len(self.game.state.history) - 1
        if decision:
            self._decisions[index] = decision
        text = f"{self.name(player)} {self.verb(player, 'plays')} {play.notation()}."
        if seat.is_human or self.mode is Mode.ADVERSARY:
            self.emit("move", text, player)
        else:
            self.emit("move", text, player, annotation)
        if self.game.over:
            self._game_over()

    def _should_annotate(self, seat: Seat) -> bool:
        if not self.config.annotate_ai or seat.is_human:
            return False
        return self.mode in (Mode.TEACHER, Mode.WATCH)

    def _game_over(self) -> None:
        result = self.game.state.result
        self.score[result.winner] += result.points
        self.games_played += 1
        self.emit(
            "game_over",
            f"{self.name(result.winner)} {self.verb(result.winner, 'wins')} {result.points} point{'s' * (result.points != 1)} "
            f"({'the double was dropped' if result.dropped else result.kind.name.lower()}). "
            f"Score: {self.name(Player.WHITE)} {self.score[Player.WHITE]}, {self.name(Player.BLACK)} {self.score[Player.BLACK]}.",
            result.winner,
        )

    def review(self) -> list[Annotation]:
        """Annotate every move of the current game (for the post-game review in adversary mode)."""
        notes = []
        for i, rec in enumerate(self.game.state.history):
            if not isinstance(rec, TurnRecord):
                continue
            decision = self._decisions.get(i)
            notes.append(
                self.annotator.annotate(
                    rec.before, rec.player, rec.roll, rec.play,
                    candidates=decision.candidates if decision else None,
                    reasoning=decision.reasoning if decision else "",
                )
            )
        return notes

