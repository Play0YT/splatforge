"""Erzeugt kleine synthetische Testvideos mit bekannter Kamerafahrt.

Die Szene ist ein Raum mit drei texturierten Flächen (Boden, Rückwand, linke Wand). Jeder Pixel
wird per Strahlverfolgung gerendert, die Geometrie ist also exakt und COLMAP kann sie
rekonstruieren. So braucht das Repository keine Videodateien mit unklarer Lizenz.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

TEXTURE_SIZE = 1024


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), np.float32)
    for scale in (4, 16, 64):
        noise = rng.random((TEXTURE_SIZE // scale, TEXTURE_SIZE // scale, 3)).astype(np.float32)
        tex += cv2.resize(noise, (TEXTURE_SIZE, TEXTURE_SIZE), interpolation=cv2.INTER_CUBIC) / 3
    tex = np.clip(tex * 255, 0, 255).astype(np.uint8)
    for _ in range(120):
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        center = tuple(int(c) for c in rng.integers(0, TEXTURE_SIZE, 2))
        if rng.random() < 0.5:
            cv2.circle(tex, center, int(rng.integers(8, 60)), color, -1)
        else:
            size = rng.integers(10, 80, 2)
            cv2.rectangle(tex, center, (center[0] + int(size[0]), center[1] + int(size[1])), color, -1)
    return tex


# Flächen: Ursprung, Achse u, Achse v (je 4 m lang), Textur-Seed
_PLANES = [
    (np.array([0.0, 0.0, 0.0]), np.array([4.0, 0, 0]), np.array([0, 0, 4.0]), 1),  # Boden y=0
    (np.array([0.0, 0.0, 4.0]), np.array([4.0, 0, 0]), np.array([0, 3.0, 0]), 2),  # Rückwand z=4
    (np.array([0.0, 0.0, 0.0]), np.array([0, 0, 4.0]), np.array([0, 3.0, 0]), 3),  # linke Wand x=0
]


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation Welt→Kamera (OpenCV: x rechts, y unten, z vorwärts)."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def render_frame(
    eye: np.ndarray, target: np.ndarray, width: int, height: int, textures: list[np.ndarray]
) -> np.ndarray:
    rot = _look_at(eye, target)
    f = 0.8 * width
    xs, ys = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    dirs_cam = np.stack([(xs - width / 2) / f, (ys - height / 2) / f, np.ones_like(xs)], axis=-1)
    dirs = dirs_cam @ rot  # Kamera→Welt
    best_t = np.full((height, width), np.inf)
    image = np.full((height, width, 3), 200, np.uint8)
    for (origin, axis_u, axis_v, _), tex in zip(_PLANES, textures, strict=True):
        normal = np.cross(axis_u, axis_v)
        denom = dirs @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((origin - eye) @ normal) / denom
        hit = eye + dirs * t[..., None]
        rel = hit - origin
        u = rel @ axis_u / (axis_u @ axis_u)
        v = rel @ axis_v / (axis_v @ axis_v)
        valid = (t > 0) & (t < best_t) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        map_x = (u * (TEXTURE_SIZE - 1)).astype(np.float32)
        map_y = (v * (TEXTURE_SIZE - 1)).astype(np.float32)
        sampled = cv2.remap(tex, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        image[valid] = sampled[valid]
        best_t = np.where(valid, t, best_t)
    return image


def make_video(path: Path, frames: int = 120, width: int = 480, height: int = 270, fps: int = 24) -> Path:
    """Schreibt ein MP4 mit einer Kamerafahrt im Bogen durch den Raum."""
    textures = [_texture(seed) for *_, seed in _PLANES]
    frame_dir = path.parent / f"{path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    target = np.array([1.2, 0.8, 3.0])
    for i in range(frames):
        a = np.deg2rad(-35 + 70 * i / max(frames - 1, 1))
        eye = np.array([2.6 + 1.0 * np.sin(a), 1.5 + 0.2 * np.sin(2 * a), 0.9 + 0.4 * np.cos(a)])
        img = render_frame(eye, target, width, height, textures)
        ok, enc = cv2.imencode(".png", img)
        assert ok
        enc.tofile(frame_dir / f"{i:05d}.png")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
            "-i", str(frame_dir / "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path),
        ],
        check=True,
    )  # fmt: skip
    for p in frame_dir.iterdir():
        p.unlink()
    frame_dir.rmdir()
    return path
