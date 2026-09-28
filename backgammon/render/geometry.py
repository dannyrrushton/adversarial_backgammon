"""Triangle-mesh builders (numpy) for the board scene."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    positions: np.ndarray  # (N, 3) float32
    normals: np.ndarray  # (N, 3) float32
    indices: np.ndarray  # (M, 3) uint32
    material_ids: np.ndarray  # (M,) uint32

    @property
    def num_triangles(self) -> int:
        return len(self.indices)

    @staticmethod
    def empty() -> Mesh:
        return Mesh(
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.uint32),
            np.zeros((0,), np.uint32),
        )

    @staticmethod
    def concat(meshes: list[Mesh]) -> Mesh:
        meshes = [m for m in meshes if m.num_triangles]
        if not meshes:
            return Mesh.empty()
        offsets = np.cumsum([0] + [len(m.positions) for m in meshes[:-1]])
        return Mesh(
            np.concatenate([m.positions for m in meshes]).astype(np.float32),
            np.concatenate([m.normals for m in meshes]).astype(np.float32),
            np.concatenate([m.indices + o for m, o in zip(meshes, offsets)]).astype(np.uint32),
            np.concatenate([m.material_ids for m in meshes]).astype(np.uint32),
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions.min(axis=0), self.positions.max(axis=0)


def _mesh(positions, normals, indices, material: int) -> Mesh:
    indices = np.asarray(indices, np.uint32).reshape(-1, 3)
    return Mesh(
        np.asarray(positions, np.float32).reshape(-1, 3),
        np.asarray(normals, np.float32).reshape(-1, 3),
        indices,
        np.full(len(indices), material, np.uint32),
    )


def box(lo, hi, material: int) -> Mesh:
    """Axis-aligned box with flat-shaded faces."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    faces = [
        ((0, 1, 0), [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)]),
        ((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
        ((0, 0, -1), [(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)]),
        ((1, 0, 0), [(x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)]),
        ((-1, 0, 0), [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]),
    ]
    pos, nrm, idx = [], [], []
    for n, quad in faces:
        base = len(pos)
        pos += quad
        nrm += [n] * 4
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return _mesh(pos, nrm, idx, material)


def quad(corners, normal, material: int) -> Mesh:
    return _mesh(corners, [normal] * 4, [0, 1, 2, 0, 2, 3], material)


def triangle(a, b, c, normal, material: int) -> Mesh:
    return _mesh([a, b, c], [normal] * 3, [0, 1, 2], material)


def disc(center, radius: float, normal, material: int, segments: int = 16) -> Mesh:
    """Flat disc facing ``normal`` (used for dice pips)."""
    n = np.asarray(normal, np.float32)
    n = n / np.linalg.norm(n)
    helper = np.array([1, 0, 0], np.float32) if abs(n[0]) < 0.9 else np.array([0, 1, 0], np.float32)
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rim = np.asarray(center) + radius * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))
    pos = np.vstack([center, rim])
    idx = [(0, 1 + i, 1 + (i + 1) % segments) for i in range(segments)]
    # Wind so the geometric normal matches ``normal``.
    a, b, c = pos[0], pos[1], pos[2]
    if np.dot(np.cross(b - a, c - a), n) < 0:
        idx = [(i0, i2, i1) for i0, i1, i2 in idx]
    return _mesh(pos, np.tile(n, (len(pos), 1)), idx, material)


def checker(center, radius: float, height: float, material: int, segments: int = 40) -> Mesh:
    """A checker: a cylinder with a rounded top edge and a shallow ring groove on the face."""
    cx, cy, cz = center
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    bevel = min(0.25 * height, 0.2 * radius)
    # Profile from the bottom centre up and around to the top centre: (radius, y, normal_r, normal_y)
    profile = [
        (0.0, 0.0, 0.0, -1.0),
        (radius, 0.0, 0.0, -1.0),
        (radius, 0.0, 1.0, 0.0),
        (radius, height - bevel, 1.0, 0.0),
        (radius - 0.3 * bevel, height - 0.3 * bevel, 0.7, 0.7),
        (radius - bevel, height, 0.0, 1.0),
        (0.78 * radius, height, 0.0, 1.0),
        (0.74 * radius, height - 0.04 * height, -0.5, 0.85),  # inner groove
        (0.70 * radius, height, 0.0, 1.0),
        (0.0, height, 0.0, 1.0),
    ]
    rings = len(profile)
    pos = np.zeros((rings * segments, 3), np.float32)
    nrm = np.zeros_like(pos)
    for r, (pr, py, nr, ny) in enumerate(profile):
        sl = slice(r * segments, (r + 1) * segments)
        pos[sl, 0] = cx + pr * cos
        pos[sl, 1] = cy + py
        pos[sl, 2] = cz + pr * sin
        length = np.hypot(nr, ny)
        nrm[sl, 0] = nr / length * cos
        nrm[sl, 1] = ny / length
        nrm[sl, 2] = nr / length * sin
    idx = []
    for r in range(rings - 1):
        if profile[r][:2] == profile[r + 1][:2]:
            continue  # duplicated ring that only changes the normal (hard edge)
        for i in range(segments):
            j = (i + 1) % segments
            a, b = r * segments + i, r * segments + j
            c, d = (r + 1) * segments + i, (r + 1) * segments + j
            idx += [(a, c, b), (b, c, d)]
    return _mesh(pos, nrm, idx, material)


# Pip layouts on a unit face, coordinates in [-1, 1].
PIP_LAYOUT = {
    1: [(0, 0)],
    2: [(-1, -1), (1, 1)],
    3: [(-1, -1), (0, 0), (1, 1)],
    4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    5: [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
    6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)],
}

# For a die showing ``top``, a valid (front, right) pair: opposite faces sum to 7.
DIE_ORIENTATION = {1: (2, 3), 2: (1, 4), 3: (1, 2), 4: (1, 5), 5: (1, 3), 6: (2, 4)}


def die(center, size: float, value: int, body_material: int, pip_material: int, yaw: float = 0.0) -> Mesh:
    """A die resting on its bottom face with ``value`` on top, rotated ``yaw`` radians about y."""
    h = size / 2
    body = box((-h, 0, -h), (h, size, h), body_material)
    front, right = DIE_ORIENTATION[value]
    faces = {
        value: ((0, size + 1e-3, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)),
        7 - value: ((0, -1e-3, 0), (0, -1, 0), (1, 0, 0), (0, 0, 1)),
        front: ((0, h, h + 1e-3), (0, 0, 1), (1, 0, 0), (0, 1, 0)),
        7 - front: ((0, h, -h - 1e-3), (0, 0, -1), (1, 0, 0), (0, 1, 0)),
        right: ((h + 1e-3, h, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0)),
        7 - right: ((-h - 1e-3, h, 0), (-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    }
    pips = []
    spread = size * 0.27
    for face_value, (origin, normal, u, v) in faces.items():
        for a, b in PIP_LAYOUT[face_value]:
            c = np.asarray(origin) + spread * (a * np.asarray(u) + b * np.asarray(v))
            pips.append(disc(c, size * 0.085, normal, pip_material, segments=12))
    mesh = Mesh.concat([body, *pips])
    return transform(mesh, yaw=yaw, offset=center)


def transform(mesh: Mesh, yaw: float = 0.0, offset=(0, 0, 0)) -> Mesh:
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    return Mesh(
        mesh.positions @ rot.T + np.asarray(offset, np.float32),
        mesh.normals @ rot.T,
        mesh.indices,
        mesh.material_ids,
    )
