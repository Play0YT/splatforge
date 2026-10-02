"""COLMAP-Modelle im Binärformat lesen, ohne pycolmap.

Wird vom CPU-Trainingsbackend verwendet: Unter macOS bringen pycolmap und PyTorch jeweils eine eigene
OpenMP-Bibliothek mit, und beide im selben Prozess zu laden bricht das Programm ab. Der Trainingsprozess
liest das Modell deshalb selbst (Format: https://colmap.github.io/format.html).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

# Modell-ID → (Name, Anzahl Parameter); nur die Modelle, die nach dem Entzerren vorkommen können,
# plus die gängigen verzerrten Modelle, damit ein unerwartetes Modell eine klare Meldung ergibt.
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
}


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: NDArray[np.float64]


@dataclass
class Image:
    image_id: int
    name: str
    camera_id: int
    qvec: NDArray[np.float64]  # w, x, y, z (Welt → Kamera)
    tvec: NDArray[np.float64]

    def world_to_cam(self) -> NDArray[np.float64]:
        w, x, y, z = self.qvec / np.linalg.norm(self.qvec)
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        return np.concatenate([rot, self.tvec[:, None]], axis=1)


def _read(fh: BinaryIO, fmt: str) -> tuple[float | int, ...]:
    size = struct.calcsize(fmt)
    data = fh.read(size)
    if len(data) != size:
        raise ValueError("COLMAP-Datei ist unvollständig")
    return struct.unpack(fmt, data)


def read_cameras(path: Path) -> dict[int, Camera]:
    cameras = {}
    with path.open("rb") as fh:
        (count,) = _read(fh, "<Q")
        for _ in range(int(count)):
            camera_id, model_id, width, height = _read(fh, "<iiQQ")
            if int(model_id) not in CAMERA_MODELS:
                raise ValueError(f"Unbekanntes COLMAP-Kameramodell {model_id}")
            name, num_params = CAMERA_MODELS[int(model_id)]
            params = np.array(_read(fh, f"<{num_params}d"), dtype=np.float64)
            cameras[int(camera_id)] = Camera(int(camera_id), name, int(width), int(height), params)
    return cameras


def read_images(path: Path) -> dict[int, Image]:
    images = {}
    with path.open("rb") as fh:
        (count,) = _read(fh, "<Q")
        for _ in range(int(count)):
            values = _read(fh, "<i7di")
            image_id, camera_id = int(values[0]), int(values[8])
            name = bytearray()
            while (char := fh.read(1)) not in (b"\x00", b""):
                name += char
            (num_points,) = _read(fh, "<Q")
            fh.seek(int(num_points) * struct.calcsize("<ddq"), 1)
            images[image_id] = Image(
                image_id=image_id,
                name=name.decode("utf-8"),
                camera_id=camera_id,
                qvec=np.array(values[1:5], dtype=np.float64),
                tvec=np.array(values[5:8], dtype=np.float64),
            )
    return images


def read_points(path: Path) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Gibt Punkte (N, 3) und Farben (N, 3) in 0..1 zurück."""
    xyz, rgb = [], []
    with path.open("rb") as fh:
        (count,) = _read(fh, "<Q")
        for _ in range(int(count)):
            values = _read(fh, "<Q3d3Bd")
            xyz.append(values[1:4])
            rgb.append(values[4:7])
            (track_length,) = _read(fh, "<Q")
            fh.seek(int(track_length) * struct.calcsize("<ii"), 1)
    points = np.array(xyz, dtype=np.float32).reshape(-1, 3)
    colors = np.array(rgb, dtype=np.float32).reshape(-1, 3) / 255.0
    return points, colors


def model_dir(sparse: Path) -> Path:
    """COLMAP schreibt das Modell entweder direkt nach ``sparse`` oder nach ``sparse/0``."""
    return sparse / "0" if (sparse / "0" / "cameras.bin").is_file() else sparse
