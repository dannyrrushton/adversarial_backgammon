"""Qt front end: the ray-traced 3D board plus a side panel for controls and annotations.

Moves are made by clicking a checker and then its destination on the 3D board (or typed in
notation). The board rotates with a left-drag and zooms with the wheel. Every call into the
session runs on a worker thread so the window stays responsive while the LLM thinks.
"""

from __future__ import annotations

import html
import threading
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from backgammon.engine import BAR, OFF, IllegalAction, PlayBuilder, Player, TurnRecord, absolute_index, relative_point
from backgammon.render import Pick, SceneState
from backgammon.render.optix import OptixRenderer
from backgammon.render.viewer import BoardView

from .session import Event, GameSession, Mode

PLAYER_COLOUR = {Player.WHITE: "#e8e2d0", Player.BLACK: "#e07b6a"}


class GameWindow(QtWidgets.QMainWindow):
    event_received = QtCore.Signal(object)
    task_finished = QtCore.Signal(object)  # the exception raised by the task, or None

    def __init__(self, session: GameSession, renderer: OptixRenderer | None = None, ai_delay_ms: int = 700) -> None:
        super().__init__()
        self.session = session
        self.ai_delay_ms = ai_delay_ms
        self.busy = False
        self.paused = False
        self.builder: PlayBuilder | None = None
        self.selected: int | None = None  # relative source point of the checker being moved

        self.view = BoardView(renderer)
        human = session.human_player()
        if human is not None:
            self.view.camera.face(human)
        self.view.picked.connect(self.on_pick)
        self.setWindowTitle(f"Adversarial Backgammon - {session.mode.value} mode (OptiX on {self.view.renderer.device_name})")

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.prompt = QtWidgets.QLabel()
        self.prompt.setWordWrap(True)
        self.prompt.setStyleSheet("font-weight: bold; color: #f0c040;")

        self.buttons: dict[str, QtWidgets.QPushButton] = {}
        grid = QtWidgets.QGridLayout()
        specs = [
            ("roll", "Roll", self.do_roll), ("double", "Double", self.do_double), ("take", "Take", lambda: self.do_respond(True)),
            ("drop", "Drop", lambda: self.do_respond(False)), ("hint", "Hint", self.do_hint), ("undo", "Undo", self.do_undo),
            ("new", "New game", self.do_new_game), ("review", "Review", self.do_review), ("pause", "Pause", self.do_pause),
        ]
        for i, (key, label, slot) in enumerate(specs):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(slot)
            self.buttons[key] = button
            grid.addWidget(button, i // 3, i % 3)

        self.entry = QtWidgets.QLineEdit()
        self.entry.setPlaceholderText("or type a move, e.g. 13/8 6/5")
        self.entry.returnPressed.connect(self.do_typed_move)

        self.log = QtWidgets.QTextBrowser()
        self.log.setOpenExternalLinks(False)
        self.log.setStyleSheet("QTextBrowser { background: #16181c; color: #d8d8d8; font-size: 13px; }")

        panel = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(panel)
        side.addWidget(self.status)
        side.addWidget(self.prompt)
        side.addLayout(grid)
        side.addWidget(self.entry)
        side.addWidget(self.log, 1)
        help_label = QtWidgets.QLabel("Drag to rotate - wheel to zoom - R resets the view")
        help_label.setStyleSheet("color: #888;")
        side.addWidget(help_label)
        panel.setMinimumWidth(380)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.view)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.resize(1600, 900)

        self.event_received.connect(self.show_event)
        self.task_finished.connect(self.on_task_finished)
        session.listeners.append(self.event_received.emit)

        self.run_task(session.start)

    # --- running session calls off the GUI thread -----------------------------------------------

    def run_task(self, fn: Callable[[], object]) -> None:
        if self.busy:
            return
        self.busy = True
        self.refresh()

        def work() -> None:
            error = None
            try:
                fn()
            except Exception as exc:  # reported in the log; the game state is still consistent
                error = exc
            self.task_finished.emit(error)

        threading.Thread(target=work, daemon=True).start()

    def on_task_finished(self, error: Exception | None) -> None:
        self.busy = False
        if isinstance(error, IllegalAction):
            self.append(f"<i>Not now: {html.escape(str(error))}.</i>")
        elif error is not None:
            self.append(f"<b style='color:#ff6060'>Error: {html.escape(repr(error))}</b>")
        self.prepare_human_move()
        self.refresh()
        self.schedule_ai()

    def schedule_ai(self) -> None:
        s = self.session
        if self.busy or self.paused or s.game.over or s.waiting_for_human() is not None:
            return
        QtCore.QTimer.singleShot(self.ai_delay_ms, lambda: self.run_task(self.session.step) if not self.busy else None)

    # --- human input --------------------------------------------------------------------------

    def prepare_human_move(self) -> None:
        s = self.session
        if s.waiting_for_human() == "move":
            if self.builder is None or self.builder.board is not s.board or self.builder.roll is not s.game.state.roll:
                self.builder = PlayBuilder(s.board, s.actor(), s.game.state.roll)
                self.selected = None
                sources = self.builder.sources()
                if len(sources) == 1:
                    self.selected = sources[0]
        else:
            self.builder = None
            self.selected = None

    def spot_of(self, pick: Pick, player: Player) -> int | None:
        if pick.kind == "point":
            return relative_point(player, pick.index)
        if pick.kind == "bar":
            return BAR
        if pick.kind == "off":
            return OFF
        return None

    def on_pick(self, pick: Pick) -> None:
        if self.busy or self.builder is None:
            return
        player = self.builder.player
        spot = self.spot_of(pick, player)
        if spot is None:
            return
        sources = self.builder.sources()
        if self.selected is None or spot == self.selected:
            self.selected = spot if spot in sources and spot != self.selected else None
        else:
            try:
                self.builder.add(self.selected, spot)
            except ValueError:
                # Clicking another movable checker switches the selection.
                self.selected = spot if spot in sources else None
            else:
                self.selected = None
                play = self.builder.complete
                if play is not None:
                    self.builder = None
                    self.run_task(lambda: self.session.human_move(play))
                    return
                remaining = self.builder.sources()
                if len(remaining) == 1:
                    self.selected = remaining[0]
        self.refresh()

    def do_roll(self) -> None:
        self.run_task(self.session.human_roll)

    def do_double(self) -> None:
        self.run_task(self.session.human_double)

    def do_respond(self, take: bool) -> None:
        self.run_task(lambda: self.session.human_respond(take))

    def do_hint(self) -> None:
        self.run_task(self.session.hint)

    def do_undo(self) -> None:
        if self.builder is not None:
            self.builder.undo()
            self.selected = None
            self.refresh()

    def do_typed_move(self) -> None:
        text = self.entry.text().strip()
        if text and self.session.waiting_for_human() == "move":
            self.entry.clear()
            self.builder = None
            self.run_task(lambda: self.session.human_move_text(text))

    def do_new_game(self) -> None:
        self.log.clear()
        self.run_task(self.session.new_game)

    def do_review(self) -> None:
        def review() -> None:
            notes = self.session.review()
            self.session.emit("info", f"Review of {len(notes)} moves:")
            for note in notes:
                self.session.emit("review", note.notation, note.player, note)

        self.run_task(review)

    def do_pause(self) -> None:
        self.paused = not self.paused
        self.buttons["pause"].setText("Resume" if self.paused else "Pause")
        self.schedule_ai()

    # --- display ------------------------------------------------------------------------------

    def append(self, markup: str) -> None:
        self.log.append(markup)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def show_event(self, event: Event) -> None:
        colour = PLAYER_COLOUR.get(event.player, "#9ab")
        text = html.escape(event.text)
        if event.kind == "critique":
            text = f"<b>Coach:</b> {text}"
        elif event.kind == "hint":
            text = "<b>Hint:</b><br>" + text.replace("\n", "<br>")
        elif event.kind == "game_over":
            text = f"<b>{text}</b>"
        if event.kind != "review":
            self.append(f"<span style='color:{colour}'>{text}</span>")
        if event.annotation is not None:
            self.append(annotation_html(event.annotation, colour, legality=self.session.mode is not Mode.ADVERSARY or event.kind == "review"))
        if event.kind == "game_over" and self.session.mode is Mode.ADVERSARY:
            self.append("<i>Press Review to see how every move was judged.</i>")

    def refresh(self) -> None:
        s = self.session
        state = s.game.state
        needed = None if self.busy else s.waiting_for_human()

        # Buttons
        can_double = needed == "roll" and s.game.can_double()
        enabled = {
            "roll": needed == "roll",
            "double": can_double,
            "take": needed == "double",
            "drop": needed == "double",
            "hint": needed is not None and s.mode is Mode.TEACHER,
            "undo": needed == "move" and self.builder is not None and bool(self.builder.moves),
            "new": not self.busy and (s.game.over or s.human_player() is None),
            "review": not self.busy,
            "pause": s.human_player() is None,
        }
        for key, on in enabled.items():
            self.buttons[key].setEnabled(on)
        self.buttons["pause"].setVisible(s.human_player() is None)
        self.buttons["hint"].setVisible(s.mode is Mode.TEACHER)
        self.entry.setEnabled(needed == "move")

        # Status
        board = s.board
        rows = []
        for p in Player:
            rows.append(
                f"<span style='color:{PLAYER_COLOUR[p]}'>&#9679; <b>{html.escape(s.name(p))}</b> ({p})</span>"
                f" - pips {board.pip_count(p)}, off {board.off[p]}, score {s.score[p]}"
            )
        owner = "centre" if state.cube_owner is None else s.name(state.cube_owner)
        rows.append(f"Cube: {state.cube_value} ({owner}) &nbsp; Mode: {s.mode.value}")
        if state.roll is not None and not s.game.over:
            rows.append(f"{html.escape(s.name(s.game.turn))} rolled <b>{state.roll}</b>")
        self.status.setText("<br>".join(rows))

        if s.game.over:
            prompt = "Game over. Start a new game or review the moves."
        elif self.busy:
            actor = s.actor()
            prompt = f"{s.name(actor)} is thinking..." if actor is not None and not s.seat(actor).is_human else "Working..."
        elif needed == "roll":
            prompt = "Your turn: roll the dice" + (" or offer a double." if can_double else ".")
        elif needed == "double":
            prompt = f"{s.name(s.game.turn)} offers a double to {state.cube_value * 2}. Take or drop?"
        elif needed == "move":
            used = len(self.builder.moves) if self.builder else 0
            total = len(self.builder.plays[0].moves) if self.builder else 0
            prompt = (
                f"Move {used + 1} of {total}: click a checker, then where it goes."
                if self.selected is None
                else f"Moving from {'the bar' if self.selected == BAR else self.selected}: click the destination."
            )
        elif self.paused:
            prompt = "Paused."
        else:
            prompt = ""
        self.prompt.setText(prompt)

        self.view.show_position(self.current_board(), self.scene_state())

    def current_board(self):
        return self.builder.current if self.builder is not None and self.builder.moves else self.session.board

    def scene_state(self) -> SceneState:
        s = self.session
        state = s.game.state
        scene = SceneState(roll=state.roll, roll_player=s.game.turn, cube_value=state.cube_value, cube_owner=state.cube_owner)
        history = [r for r in state.history if isinstance(r, TurnRecord)]
        if history and state.roll is None:
            last = history[-1]
            scene.last_move_points = {absolute_index(last.player, m.dst) for m in last.play.moves if m.dst != OFF}
        if self.builder is not None and not self.busy:
            player = self.builder.player
            if self.selected is None:
                for src in self.builder.sources():
                    if src == BAR:
                        scene.highlight_bar = player
                    else:
                        scene.source_points.add(absolute_index(player, src))
            else:
                if self.selected == BAR:
                    scene.highlight_bar = player
                else:
                    scene.selected = absolute_index(player, self.selected)
                    scene.source_points.add(scene.selected)
                for move in self.builder.destinations(self.selected):
                    if move.dst == OFF:
                        scene.highlight_off = player
                    else:
                        scene.target_points.add(absolute_index(player, move.dst))
        return scene


def annotation_html(ann, colour: str, legality: bool = True) -> str:
    # Plain bullet lines rather than <ul>: QTextBrowser.append() would otherwise keep every later
    # log entry inside the list.
    def bullets(items, style=""):
        return "".join(f"<br><span style='{style}'>&nbsp;&nbsp;&bull; {html.escape(i)}</span>" for i in items)

    parts = [f"<span style='color:{colour}'>{html.escape(ann.summary)}</span>"]
    if ann.strategy:
        parts.append("<br><b>Strategy</b>" + bullets(ann.strategy))
    if ann.commentary:
        parts.append(f"<br><b>Reasoning:</b> <i>{html.escape(ann.commentary)}</i>")
    if ann.assessment:
        parts.append(f"<br><b>Assessment:</b> {html.escape(ann.assessment)}")
    if legality and ann.legality:
        parts.append("<br><b>Why it is legal</b>" + bullets(ann.legality, "color:#9a9a9a"))
    return f"<p style='margin-left:12px; color:#c8c8c8'>{''.join(parts)}</p>"


def run_gui(session: GameSession, ai_delay_ms: int = 700) -> int:
    import sys

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(34, 36, 40))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(24, 26, 30))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(52, 56, 62))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(230, 230, 230))
    app.setPalette(palette)
    window = GameWindow(session, ai_delay_ms=ai_delay_ms)
    window.show()
    return app.exec()
