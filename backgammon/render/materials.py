"""Material table shared by the scene builder and the OptiX renderer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Pattern(IntEnum):
    NONE = 0
    WOOD = 1
    FELT = 2


@dataclass(frozen=True)
class Material:
    albedo: tuple[float, float, float]
    specular: float = 0.1
    shininess: float = 20.0
    reflectivity: float = 0.0
    emission: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pattern: Pattern = Pattern.NONE


class M(IntEnum):
    """Material ids, in the order of :data:`MATERIALS`."""

    FRAME = 0
    FELT = 1
    POINT_DARK = 2
    POINT_LIGHT = 3
    CHECKER_WHITE = 4
    CHECKER_BLACK = 5
    BAR = 6
    DIE = 7
    PIP = 8
    HIGHLIGHT_SOURCE = 9
    HIGHLIGHT_TARGET = 10
    TABLE = 11
    CUBE = 12
    SELECTED = 13
    LAST_MOVE = 14


MATERIALS: list[Material] = [
    Material((0.36, 0.20, 0.10), 0.35, 40, 0.06, pattern=Pattern.WOOD),  # FRAME
    Material((0.07, 0.22, 0.14), 0.02, 8, 0.0, pattern=Pattern.FELT),  # FELT
    Material((0.45, 0.07, 0.06), 0.05, 10, 0.0, pattern=Pattern.FELT),  # POINT_DARK
    Material((0.80, 0.72, 0.55), 0.05, 10, 0.0, pattern=Pattern.FELT),  # POINT_LIGHT
    Material((0.88, 0.86, 0.80), 0.9, 90, 0.10),  # CHECKER_WHITE
    Material((0.10, 0.08, 0.08), 1.0, 120, 0.14),  # CHECKER_BLACK
    Material((0.30, 0.16, 0.08), 0.35, 40, 0.06, pattern=Pattern.WOOD),  # BAR
    Material((0.92, 0.92, 0.90), 0.6, 60, 0.04),  # DIE
    Material((0.05, 0.05, 0.05), 0.2, 20, 0.0),  # PIP
    Material((0.85, 0.70, 0.20), 0.1, 10, 0.0, emission=(0.35, 0.28, 0.05)),  # HIGHLIGHT_SOURCE
    Material((0.20, 0.75, 0.35), 0.1, 10, 0.0, emission=(0.06, 0.30, 0.10)),  # HIGHLIGHT_TARGET
    Material((0.10, 0.10, 0.11), 0.05, 10, 0.0, pattern=Pattern.FELT),  # TABLE
    Material((0.95, 0.93, 0.85), 0.5, 50, 0.03),  # CUBE
    Material((0.95, 0.85, 0.40), 0.9, 90, 0.08, emission=(0.25, 0.18, 0.02)),  # SELECTED
    Material((0.30, 0.55, 0.95), 0.1, 10, 0.0, emission=(0.05, 0.12, 0.30)),  # LAST_MOVE
]

assert len(MATERIALS) == len(M)
