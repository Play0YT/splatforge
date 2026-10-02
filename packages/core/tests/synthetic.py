"""Erzeugt kleine synthetische Testvideos mit bekannter Kamerafahrt.

Die Szene ist ein Raum mit drei texturierten Flächen (Boden, Rückwand, linke Wand). Jeder Pixel
wird per Strahlverfolgung gerendert, die Geometrie ist also exakt und COLMAP kann sie
rekonstruieren. So braucht das Repository keine Videodateien mit unklarer Lizenz.
"""

from __future__ import annotations

import struct
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


# --- Insta360 (.insv) -------------------------------------------------------------------------------
# Nachbau des Insta360-Formats nach telemetry-parser (MIT/Apache-2.0): MP4 plus Trailer am Dateiende.

INSV_MAGIC = b"8db42d694ccc418790edff439fe026bf"


def _pb_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _pb_field(number: int, value: int | str | bytes) -> bytes:
    if isinstance(value, int):
        return _pb_varint(number << 3) + _pb_varint(value)
    data = value.encode() if isinstance(value, str) else value
    return _pb_varint(number << 3 | 2) + _pb_varint(len(data)) + data


def insv_metadata(camera: str = "Insta360 X4", calibration: bool = True) -> bytes:
    """Protobuf-Metadaten wie im Insta360-Trailer (Datensatz 1)."""
    msg = _pb_field(1, "IXSE00TEST") + _pb_field(2, camera) + _pb_field(3, "v1.2.3")
    msg += _pb_field(19, _pb_field(1, 2880) + _pb_field(2, 2880)) + _pb_field(20, 30)
    msg += _pb_field(26, _pb_field(1, 0) + _pb_field(2, 1) + _pb_field(4, 2))
    if calibration:
        msg += _pb_field(5, "2_1440.0_1440.0_0.0_0.0_0.0_1440.0_1440.0_180.0_0.0_0.0_2880_2880_1")
        msg += _pb_field(54, "_".join(str(float(i)) for i in range(24)))
    return msg


def insv_trailer(records: dict[int, tuple[int, bytes]], with_offsets: bool = False) -> bytes:
    """Baut den Trailer. ``records``: {ID: (Format, Daten)}."""
    body = bytearray()
    table = bytearray()
    for rec_id, (fmt, data) in records.items():
        offset = len(body)
        body += data + struct.pack("<BBI", fmt, rec_id, len(data))
        table += struct.pack("<BBII", rec_id, fmt, len(data), offset)
    if with_offsets:
        body += bytes(table) + struct.pack("<BBI", 0, 0, len(table))
    extra_size = len(body) + 32 + 4 + 4 + len(INSV_MAGIC)
    return bytes(body) + bytes(32) + struct.pack("<II", extra_size, 3) + INSV_MAGIC


def make_insv(
    path: Path,
    layout: str = "two_streams",
    lens_size: int = 320,
    seconds: float = 2.0,
    trailer: bytes | None = None,
) -> Path:
    """Schreibt eine kleine .insv-Datei. ``layout``: two_streams, side_by_side oder single."""
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    src = f"testsrc2=size={lens_size}x{lens_size}:rate=24:duration={seconds}"
    src2 = f"mandelbrot=size={lens_size}x{lens_size}:rate=24"
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", str(seconds), "-f", "mp4"]
    if layout == "two_streams":
        args = [*common, "-f", "lavfi", "-i", src, "-f", "lavfi", "-i", src2, "-map", "0", "-map", "1", *enc]
    elif layout == "side_by_side":
        args = [
            *common,
            "-f",
            "lavfi",
            "-i",
            src,
            "-f",
            "lavfi",
            "-i",
            src2,
            "-filter_complex",
            "[0][1]hstack",
            *enc,
        ]
    else:
        args = [*common, "-f", "lavfi", "-i", src, *enc]
    subprocess.run([*args, str(path)], check=True)
    if trailer is None:
        trailer = insv_trailer({1: (1, insv_metadata())})
    with path.open("ab") as fh:
        fh.write(trailer)
    return path
