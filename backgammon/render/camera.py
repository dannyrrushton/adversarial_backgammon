"""Orbit camera driven by mouse drags, and ray/board picking."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backgammon.engine import Player

from .scene import BAR_HALF, EDGE_Z, PLAY_X, POINT_WIDTH, TRAY_X0, TRAY_X1


@dataclass
class OrbitCamera:
    """Looks at ``target`` from a sphere of radius ``distance``.

    ``yaw`` 0 puts the eye on the +z side (White's chair); ``pitch`` is the elevation angle.
    """

    target: tuple[float, float, float] = (0.6, 0.0, 0.0)
    distance: float = 19.0
    yaw: float = 0.0
    pitch: float = math.radians(58)
    fov_y: float = 40.0

    MIN_PITCH = math.radians(8)
    MAX_PITCH = math.radians(89)
    MIN_DISTANCE = 7.0
    MAX_DISTANCE = 45.0

    @property
    def eye(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        offset = np.array([cp * math.sin(self.yaw), math.sin(self.pitch), cp * math.cos(self.yaw)])
        return np.asarray(self.target) + self.distance * offset

    def orbit(self, dx_pixels: float, dy_pixels: float, sensitivity: float = 0.008) -> None:
        """Drag right spins the board, drag down tilts the view toward overhead."""
        self.yaw -= dx_pixels * sensitivity
        self.pitch = min(self.MAX_PITCH, max(self.MIN_PITCH, self.pitch + dy_pixels * sensitivity))

    def zoom(self, steps: float) -> None:
        """Positive steps move closer."""
        self.distance = min(self.MAX_DISTANCE, max(self.MIN_DISTANCE, self.distance * (0.9**steps)))

    def face(self, player: Player) -> None:
        """Seat the camera behind ``player``'s side of the board."""
        self.yaw = 0.0 if player is Player.WHITE else math.pi

    def basis(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(forward, right*half_width, up*half_height), matching the device ray generation."""
        eye = self.eye
        forward = np.asarray(self.target) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        half_h = math.tan(math.radians(self.fov_y) / 2)
        half_w = half_h * width / height
        return forward, right * half_w, up * half_h

    def ray(self, px: float, py: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """World-space ray through pixel (px, py), with (0, 0) the top-left corner."""
        forward, u, v = self.basis(width, height)
        sx = 2 * (px + 0.5) / width - 1
        sy = 1 - 2 * (py + 0.5) / height
        d = forward + sx * u + sy * v
        return self.eye, d / np.linalg.norm(d)


@dataclass(frozen=True)
class Pick:
    """What a click landed on: an absolute point index, the bar, or a bear-off tray half."""

    kind: str  # "point", "bar", "off" or "none"
    index: int | None = None  # absolute point for kind == "point"
    side: Player | None = None  # which half for "bar" and "off"


def pick_board(x: float, z: float) -> Pick:
    """Map a position on the board plane to the region under it."""
    if abs(z) > EDGE_Z:
        return Pick("none")
    side = Player.WHITE if z >= 0 else Player.BLACK
    if TRAY_X0 <= x <= TRAY_X1:
        return Pick("off", side=side)
    if abs(x) <= BAR_HALF:
        return Pick("bar", side=side)
    if abs(x) > PLAY_X:
        return Pick("none")
    column = min(5, int((abs(x) - BAR_HALF) / POINT_WIDTH))  # 0 = next to the bar
    if z >= 0:
        index = 5 - column if x > 0 else 6 + column
    else:
        index = 18 + column if x > 0 else 17 - column
    return Pick("point", index=index)


def pick(camera: OrbitCamera, px: float, py: float, width: int, height: int) -> Pick:
    origin, direction = camera.ray(px, py, width, height)
    if abs(direction[1]) < 1e-6:
        return Pick("none")
    t = -origin[1] / direction[1]
    if t <= 0:
        return Pick("none")
    hit = origin + t * direction
    return pick_board(float(hit[0]), float(hit[2]))
