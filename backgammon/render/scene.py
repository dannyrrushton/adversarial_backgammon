"""Board layout and scene assembly.

World space: y is up, the playing surface is at y = 0, x runs left to right and +z points toward
the viewer sitting on White's side. White's home board (absolute points 0..5) is the near-right
quadrant, as on a real board seen from White's chair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backgammon.engine import Board, Player, Roll

from .geometry import Mesh, box, checker, die, quad, triangle
from .materials import M

POINT_WIDTH = 1.0
BAR_HALF = 0.6  # half-width of the central bar
HALF_WIDTH = 6 * POINT_WIDTH  # width of one board half
POINT_LENGTH = 4.9
EDGE_Z = 5.7  # distance from the centre line to the inner edge of the frame
FRAME = 0.45
FRAME_HEIGHT = 0.32
TRAY_WIDTH = 1.3
CHECKER_RADIUS = 0.44
CHECKER_HEIGHT = 0.17
STACK_SLOTS = 5
DIE_SIZE = 0.55

PLAY_X = BAR_HALF + HALF_WIDTH  # x of the inner frame edge
TRAY_X0 = PLAY_X + FRAME
TRAY_X1 = TRAY_X0 + TRAY_WIDTH


def point_x(index: int) -> float:
    """Centre x of absolute point ``index`` (0..23)."""
    if index < 6:
        return BAR_HALF + (5 - index + 0.5) * POINT_WIDTH
    if index < 12:
        return -BAR_HALF - (index - 6 + 0.5) * POINT_WIDTH
    if index < 18:
        return -BAR_HALF - (17 - index + 0.5) * POINT_WIDTH
    return BAR_HALF + (index - 18 + 0.5) * POINT_WIDTH


def point_side(index: int) -> int:
    """+1 for points on the near (White) side, -1 for the far side."""
    return 1 if index < 12 else -1


def checker_position(index: int, slot: int) -> tuple[float, float, float]:
    """Where the ``slot``-th checker on a point sits. Beyond five, checkers stack in layers."""
    layer, pos = divmod(slot, STACK_SLOTS)
    side = point_side(index)
    z = side * (EDGE_Z - CHECKER_RADIUS - 0.04 - pos * 2 * CHECKER_RADIUS)
    return point_x(index), layer * CHECKER_HEIGHT + 0.002, z


def bar_position(player: Player, slot: int) -> tuple[float, float, float]:
    """Checkers on the bar sit on the bar strip, White on the near half and Black on the far."""
    side = 1 if player is Player.WHITE else -1
    layer, pos = divmod(slot, 4)
    return 0.0, FRAME_HEIGHT * 0.5 + layer * CHECKER_HEIGHT, side * (1.0 + pos * 2 * CHECKER_RADIUS)


def off_position(player: Player, slot: int) -> tuple[float, float, float]:
    """Borne-off checkers pile up in the tray on the right, White's near and Black's far."""
    side = 1 if player is Player.WHITE else -1
    pile, height = divmod(slot, 5)
    x = (TRAY_X0 + TRAY_X1) / 2
    z = side * (EDGE_Z - CHECKER_RADIUS - 0.1 - pile * 2.1 * CHECKER_RADIUS)
    return x, height * CHECKER_HEIGHT, z


@dataclass
class SceneState:
    """Everything the 3D view shows besides the checkers' positions."""

    roll: Roll | None = None
    roll_player: Player | None = None
    cube_value: int = 1
    cube_owner: Player | None = None
    source_points: set[int] = field(default_factory=set)  # absolute indices to highlight
    target_points: set[int] = field(default_factory=set)
    last_move_points: set[int] = field(default_factory=set)
    highlight_bar: Player | None = None
    highlight_off: Player | None = None
    selected: int | None = None  # absolute index whose top checker is picked up


def static_scene() -> Mesh:
    """Frame, table, felt, bar and bear-off tray: everything that never changes."""
    parts = [
        box((-TRAY_X1 - 3, -0.6, -EDGE_Z - 3), (TRAY_X1 + 3, -0.3, EDGE_Z + 3), M.TABLE),
        box((-PLAY_X - FRAME, -0.3, -EDGE_Z - FRAME), (TRAY_X1 + FRAME, -0.02, EDGE_Z + FRAME), M.FRAME),
        box((-PLAY_X, -0.03, -EDGE_Z), (PLAY_X, 0.0, EDGE_Z), M.FELT),
        # Frame rails: left, right (between play area and tray), outer right, near, far.
        box((-PLAY_X - FRAME, -0.02, -EDGE_Z - FRAME), (-PLAY_X, FRAME_HEIGHT, EDGE_Z + FRAME), M.FRAME),
        box((PLAY_X, -0.02, -EDGE_Z - FRAME), (TRAY_X0, FRAME_HEIGHT, EDGE_Z + FRAME), M.FRAME),
        box((TRAY_X1, -0.02, -EDGE_Z - FRAME), (TRAY_X1 + FRAME, FRAME_HEIGHT, EDGE_Z + FRAME), M.FRAME),
        box((-PLAY_X, -0.02, EDGE_Z), (TRAY_X1, FRAME_HEIGHT, EDGE_Z + FRAME), M.FRAME),
        box((-PLAY_X, -0.02, -EDGE_Z - FRAME), (TRAY_X1, FRAME_HEIGHT, -EDGE_Z), M.FRAME),
        box((-BAR_HALF, -0.02, -EDGE_Z), (BAR_HALF, FRAME_HEIGHT * 0.5, EDGE_Z), M.BAR),
        # Divider across the middle of the bear-off tray.
        box((TRAY_X0, -0.02, -0.1), (TRAY_X1, FRAME_HEIGHT * 0.6, 0.1), M.FRAME),
    ]
    return Mesh.concat(parts)


def point_mesh(index: int, material: int) -> Mesh:
    x = point_x(index)
    side = point_side(index)
    y = 0.001
    base_z = side * EDGE_Z
    tip_z = side * (EDGE_Z - POINT_LENGTH)
    half = POINT_WIDTH * 0.46
    return triangle((x - half, y, base_z), (x + half, y, base_z), (x, y, tip_z), (0, 1, 0), material)


def marker_mesh(index: int, material: int) -> Mesh:
    """A glowing strip along the point's base, used for highlights."""
    x = point_x(index)
    side = point_side(index)
    z0 = side * EDGE_Z
    z1 = side * (EDGE_Z - 0.16)
    half = POINT_WIDTH * 0.46
    y = 0.004
    return quad([(x - half, y, z0), (x + half, y, z0), (x + half, y, z1), (x - half, y, z1)], (0, 1, 0), material)


def build_scene(board: Board, state: SceneState | None = None, base: Mesh | None = None) -> Mesh:
    state = state or SceneState()
    parts = [base if base is not None else static_scene()]

    for i in range(24):
        if i in state.source_points:
            material = M.HIGHLIGHT_SOURCE
        elif i in state.target_points:
            material = M.HIGHLIGHT_TARGET
        else:
            material = M.POINT_DARK if i % 2 == 0 else M.POINT_LIGHT
        parts.append(point_mesh(i, material))
        if i in state.last_move_points:
            parts.append(marker_mesh(i, M.LAST_MOVE))

    colours = {Player.WHITE: M.CHECKER_WHITE, Player.BLACK: M.CHECKER_BLACK}
    for i, n in enumerate(board.points):
        if n == 0:
            continue
        player = Player.WHITE if n > 0 else Player.BLACK
        for slot in range(abs(n)):
            top = slot == abs(n) - 1
            material = M.SELECTED if top and state.selected == i else colours[player]
            parts.append(checker(checker_position(i, slot), CHECKER_RADIUS, CHECKER_HEIGHT, material))

    for player in Player:
        for slot in range(board.bar[player]):
            material = M.SELECTED if state.highlight_bar is player and slot == board.bar[player] - 1 else colours[player]
            parts.append(checker(bar_position(player, slot), CHECKER_RADIUS, CHECKER_HEIGHT, material))
        for slot in range(board.off[player]):
            parts.append(checker(off_position(player, slot), CHECKER_RADIUS, CHECKER_HEIGHT * 0.8, colours[player], segments=24))
        if state.highlight_off is player:
            side = 1 if player is Player.WHITE else -1
            z0, z1 = side * 0.2, side * (EDGE_Z - 0.05)
            parts.append(quad([(TRAY_X0, 0.003, z0), (TRAY_X1, 0.003, z0), (TRAY_X1, 0.003, z1), (TRAY_X0, 0.003, z1)], (0, 1, 0), M.HIGHLIGHT_TARGET))

    if state.roll is not None:
        # Dice are thrown into the right half for White and the left half for Black, as players
        # roll on their own right-hand side.
        side = 1 if state.roll_player in (None, Player.WHITE) else -1
        cx = side * (BAR_HALF + HALF_WIDTH / 2)
        parts.append(die((cx - 0.55, 0.0, 0.05), DIE_SIZE, state.roll.d1, M.DIE, M.PIP, yaw=0.25))
        parts.append(die((cx + 0.55, 0.0, -0.08), DIE_SIZE, state.roll.d2, M.DIE, M.PIP, yaw=-0.18))

    # The doubling cube sits beside the frame: centred when nobody owns it, otherwise on the
    # owner's side. Its value is shown in the UI panel.
    cube_z = 0.0 if state.cube_owner is None else (3.5 if state.cube_owner is Player.WHITE else -3.5)
    x = -PLAY_X - FRAME - 0.9
    parts.append(box((x - 0.4, -0.3, cube_z - 0.4), (x + 0.4, 0.5, cube_z + 0.4), M.CUBE))
    return Mesh.concat(parts)


def scene_bounds() -> tuple[np.ndarray, np.ndarray]:
    return np.array([-PLAY_X - FRAME, 0, -EDGE_Z - FRAME]), np.array([TRAY_X1 + FRAME, FRAME_HEIGHT, EDGE_Z + FRAME])
