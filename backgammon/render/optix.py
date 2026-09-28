"""ctypes bindings for the C++ OptiX renderer (libbgrender.so)."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from .camera import OrbitCamera
from .geometry import Mesh
from .materials import MATERIALS, Material

LIB_NAME = "libbgrender.so"
MODULE_NAME = "bgrender_device.optixir"
DEFAULT_BUILD_DIR = Path(__file__).parent / "cpp" / "build" / "lib"


class RenderError(RuntimeError):
    pass


class _Material(ctypes.Structure):
    _fields_ = [
        ("albedo", ctypes.c_float * 3),
        ("specular", ctypes.c_float),
        ("shininess", ctypes.c_float),
        ("reflectivity", ctypes.c_float),
        ("emission", ctypes.c_float * 3),
        ("pattern", ctypes.c_int32),
    ]


class _Camera(ctypes.Structure):
    _fields_ = [
        ("eye", ctypes.c_float * 3),
        ("look_at", ctypes.c_float * 3),
        ("up", ctypes.c_float * 3),
        ("fov_y_degrees", ctypes.c_float),
    ]


class _Light(ctypes.Structure):
    _fields_ = [("position", ctypes.c_float * 3), ("color", ctypes.c_float * 3), ("radius", ctypes.c_float)]


def find_library() -> Path | None:
    """``$BGRENDER_LIB`` if set, else the CMake build directory inside the package."""
    env = os.environ.get("BGRENDER_LIB")
    candidates = [Path(env)] if env else []
    candidates.append(DEFAULT_BUILD_DIR / LIB_NAME)
    return next((p for p in candidates if p.is_file()), None)


def available() -> bool:
    return find_library() is not None


def _vec3(values) -> ctypes.Array:
    return (ctypes.c_float * 3)(*[float(v) for v in values])


def _load(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.bgr_create.restype = ctypes.c_void_p
    lib.bgr_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
    lib.bgr_destroy.argtypes = [ctypes.c_void_p]
    lib.bgr_last_error.restype = ctypes.c_char_p
    lib.bgr_last_error.argtypes = [ctypes.c_void_p]
    lib.bgr_device_name.restype = ctypes.c_char_p
    lib.bgr_device_name.argtypes = [ctypes.c_void_p]
    lib.bgr_set_scene.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(_Material),
        ctypes.c_size_t,
    ]
    lib.bgr_set_lights.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Light), ctypes.c_size_t, ctypes.c_float * 3]
    lib.bgr_render.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Camera),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.bgr_accumulated_samples.argtypes = [ctypes.c_void_p]
    return lib


class OptixRenderer:
    """Owns one GPU renderer. Upload a scene with :meth:`set_scene`, then call :meth:`render`."""

    def __init__(self, library: Path | str | None = None, module: Path | str | None = None, device: int = 0) -> None:
        path = Path(library) if library else find_library()
        if path is None or not path.is_file():
            raise RenderError(
                "libbgrender.so not found; build it with "
                "`cmake -S backgammon/render/cpp -B backgammon/render/cpp/build && "
                "cmake --build backgammon/render/cpp/build` or set BGRENDER_LIB"
            )
        module = Path(module) if module else path.parent / MODULE_NAME
        self._lib = _load(path)
        err = ctypes.create_string_buffer(1024)
        handle = self._lib.bgr_create(str(module).encode(), device, err, len(err))
        if not handle:
            raise RenderError(f"OptiX renderer failed to start: {err.value.decode(errors='replace')}")
        self._handle = ctypes.c_void_p(handle)
        self._keep: list[np.ndarray] = []
        self.device_name = self._lib.bgr_device_name(self._handle).decode()

    def _check(self, status: int) -> None:
        if status != 0:
            raise RenderError(self._lib.bgr_last_error(self._handle).decode(errors="replace"))

    def set_scene(self, mesh: Mesh, materials: list[Material] = MATERIALS) -> None:
        pos = np.ascontiguousarray(mesh.positions, np.float32)
        nrm = np.ascontiguousarray(mesh.normals, np.float32)
        idx = np.ascontiguousarray(mesh.indices, np.uint32)
        mat = np.ascontiguousarray(mesh.material_ids, np.uint32)
        table = (_Material * len(materials))()
        for i, m in enumerate(materials):
            table[i] = _Material(_vec3(m.albedo), m.specular, m.shininess, m.reflectivity, _vec3(m.emission), int(m.pattern))
        self._check(
            self._lib.bgr_set_scene(
                self._handle,
                pos.ctypes.data,
                nrm.ctypes.data,
                len(pos),
                idx.ctypes.data,
                mat.ctypes.data,
                len(idx),
                table,
                len(materials),
            )
        )

    def set_lights(self, lights: list[tuple[tuple, tuple, float]], ambient=(0.16, 0.16, 0.18)) -> None:
        """``lights`` holds (position, colour, radius) triples."""
        arr = (_Light * len(lights))(*[_Light(_vec3(p), _vec3(c), r) for p, c, r in lights])
        self._check(self._lib.bgr_set_lights(self._handle, arr, len(lights), _vec3(ambient)))

    def render(self, camera: OrbitCamera, width: int, height: int, samples: int = 1, accumulate: bool = False) -> np.ndarray:
        """Returns an (height, width, 4) uint8 RGBA image, top row first."""
        cam = _Camera(_vec3(camera.eye), _vec3(camera.target), _vec3((0, 1, 0)), camera.fov_y)
        out = np.empty((height, width, 4), np.uint8)
        self._check(self._lib.bgr_render(self._handle, ctypes.byref(cam), width, height, samples, int(accumulate), out.ctypes.data))
        return out

    @property
    def accumulated_samples(self) -> int:
        return self._lib.bgr_accumulated_samples(self._handle)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.bgr_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> OptixRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
