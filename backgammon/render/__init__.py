"""3D board rendering: scene geometry in Python, ray tracing in C++ via NVIDIA OptiX.

The Qt viewer (``backgammon.render.viewer``) is imported separately because PySide6 is optional.
"""

from .camera import OrbitCamera, Pick, pick, pick_board
from .geometry import Mesh
from .materials import MATERIALS, M, Material
from .optix import OptixRenderer, RenderError, available, find_library
from .scene import SceneState, build_scene, static_scene

__all__ = [
    "MATERIALS", "M", "Material", "Mesh", "OptixRenderer", "OrbitCamera", "Pick", "RenderError",
    "SceneState", "available", "build_scene", "find_library", "pick", "pick_board", "static_scene",
]
