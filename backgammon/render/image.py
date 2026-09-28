"""Dependency-free PNG writing for snapshots."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: str | Path, rgba: np.ndarray) -> None:
    """Write an (H, W, 4) or (H, W, 3) uint8 array as a PNG."""
    img = np.ascontiguousarray(rgba, np.uint8)
    height, width, channels = img.shape
    colour_type = {3: 2, 4: 6}[channels]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)
