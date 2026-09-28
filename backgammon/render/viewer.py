"""Qt widget showing the ray-traced board. Left-drag orbits, the wheel zooms, a click picks.

Rendering is progressive: after any change a quick low-sample frame is shown, then a timer keeps
adding samples until the image converges.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from backgammon.engine import Board

from .camera import OrbitCamera, Pick, pick
from .optix import OptixRenderer
from .scene import SceneState, build_scene, static_scene

CLICK_SLOP = 4  # pixels of movement still treated as a click rather than a drag


class BoardView(QtWidgets.QWidget):
    picked = QtCore.Signal(object)  # emits a camera.Pick

    def __init__(self, renderer: OptixRenderer | None = None, parent=None, max_samples: int = 256) -> None:
        super().__init__(parent)
        self.renderer = renderer or OptixRenderer()
        self.camera = OrbitCamera()
        self.max_samples = max_samples
        self._base = static_scene()
        self._board = Board.initial()
        self._state = SceneState()
        self._image: QtGui.QImage | None = None
        self._pixels = None
        self._press: QtCore.QPointF | None = None
        self._last: QtCore.QPointF | None = None
        self._dragging = False
        self._dirty_scene = True
        self._refine = QtCore.QTimer(self)
        self._refine.setInterval(16)
        self._refine.timeout.connect(self._refine_step)
        self.setMinimumSize(640, 400)
        self.setMouseTracking(False)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setToolTip("Drag to rotate, scroll to zoom, click a point to move")

    # --- public API ---------------------------------------------------------------------------

    def show_position(self, board: Board, state: SceneState | None = None) -> None:
        self._board = board
        self._state = state or SceneState()
        self._dirty_scene = True
        self._restart()

    def reset_view(self) -> None:
        self.camera = OrbitCamera()
        self._restart()

    def snapshot(self):
        return self._pixels

    # --- rendering ----------------------------------------------------------------------------

    def _size(self) -> tuple[int, int]:
        ratio = self.devicePixelRatioF()
        return max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio))

    def _render(self, samples: int, accumulate: bool) -> None:
        if self._dirty_scene:
            self.renderer.set_scene(build_scene(self._board, self._state, base=self._base))
            self._dirty_scene = False
        w, h = self._size()
        self._pixels = self.renderer.render(self.camera, w, h, samples=samples, accumulate=accumulate)
        self._image = QtGui.QImage(self._pixels.data, w, h, w * 4, QtGui.QImage.Format.Format_RGBA8888)
        self._image.setDevicePixelRatio(self.devicePixelRatioF())
        self.update()

    def _restart(self) -> None:
        self._render(samples=2 if self._dragging else 4, accumulate=False)
        if not self._dragging:
            self._refine.start()

    def _refine_step(self) -> None:
        if self._dragging or self.renderer.accumulated_samples >= self.max_samples:
            self._refine.stop()
            return
        self._render(samples=16, accumulate=True)

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        if self._image is not None:
            painter.drawImage(0, 0, self._image)
        painter.end()

    def resizeEvent(self, event) -> None:
        self._restart()

    # --- mouse --------------------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._press = self._last = event.position()
            self._dragging = False

    def mouseMoveEvent(self, event) -> None:
        if self._press is None:
            return
        pos = event.position()
        if not self._dragging and (pos - self._press).manhattanLength() > CLICK_SLOP:
            self._dragging = True
        if self._dragging:
            delta = pos - self._last
            self.camera.orbit(delta.x(), delta.y())
            self._last = pos
            self._restart()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton or self._press is None:
            return
        was_drag = self._dragging
        self._dragging = False
        self._press = None
        if was_drag:
            self._restart()
            return
        pos = event.position()
        self.picked.emit(pick(self.camera, pos.x(), pos.y(), self.width(), self.height()))

    def wheelEvent(self, event) -> None:
        self.camera.zoom(event.angleDelta().y() / 120)
        self._restart()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == QtCore.Qt.Key.Key_R:
            self.reset_view()
        elif key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
            self.camera.orbit(-40 if key == QtCore.Qt.Key.Key_Left else 40, 0)
            self._restart()
        else:
            super().keyPressEvent(event)


__all__ = ["BoardView", "Pick"]
