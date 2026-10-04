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


def render_rays(
    eye: np.ndarray, dirs: np.ndarray, textures: list[np.ndarray], planes: list | None = None
) -> np.ndarray:
    """Rendert Blickstrahlen (Weltkoordinaten, beliebige Form (..., 3)) per Strahlverfolgung."""
    planes = _PLANES if planes is None else planes
    shape = dirs.shape[:-1]
    best_t = np.full(shape, np.inf)
    image = np.full((*shape, 3), 200, np.uint8)
    for (origin, axis_u, axis_v, _), tex in zip(planes, textures, strict=True):
        normal = np.cross(axis_u, axis_v)
        denom = dirs @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((origin - eye) @ normal) / denom
        hit = eye + dirs * t[..., None]
        rel = hit - origin
        u = rel @ axis_u / (axis_u @ axis_u)
        v = rel @ axis_v / (axis_v @ axis_v)
        valid = (t > 0) & (t < best_t) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        map_x = np.nan_to_num(u * (TEXTURE_SIZE - 1)).astype(np.float32)
        map_y = np.nan_to_num(v * (TEXTURE_SIZE - 1)).astype(np.float32)
        sampled = cv2.remap(tex, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        image[valid] = sampled[valid]
        best_t = np.where(valid, t, best_t)
    return image


def render_frame(
    eye: np.ndarray, target: np.ndarray, width: int, height: int, textures: list[np.ndarray]
) -> np.ndarray:
    rot = _look_at(eye, target)
    f = 0.8 * width
    xs, ys = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    dirs_cam = np.stack([(xs - width / 2) / f, (ys - height / 2) / f, np.ones_like(xs)], axis=-1)
    return render_rays(eye, dirs_cam @ rot, textures)  # Kamera→Welt


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


# --- 360° ------------------------------------------------------------------------------------------
# Geschlossener Raum (4 × 3 × 4 m), damit in jeder Richtung Textur zu sehen ist.
_ROOM = [
    *_PLANES,
    (np.array([4.0, 0.0, 0.0]), np.array([0, 0, 4.0]), np.array([0, 3.0, 0]), 4),  # rechte Wand x=4
    (np.array([0.0, 0.0, 0.0]), np.array([4.0, 0, 0]), np.array([0, 3.0, 0]), 5),  # vordere Wand z=0
    (np.array([0.0, 3.0, 0.0]), np.array([4.0, 0, 0]), np.array([0, 0, 4.0]), 6),  # Decke y=3
]

# Werkskalibrierung einer Insta360 ONE RS (offset_v3 einer echten Datei, 6528×3264 Referenz)
ONE_RS_OFFSET_V3 = [
    2.0, 2.01493, 2992.76, 2992.32, 1605.78, 1625.37, -0.191, -0.138, 179.367, 0.0, 0.0, 0.0,
    0.2350903, -0.32572556, -0.95795417, 0.00186, 3.364e-05, 6528.0, 3264.0, 62.0,
    2.01493, 2983.37, 2982.27, 4874.88, 1625.43, 0.147, -0.468, 0.682, -0.000103, 0.000152, -0.039748,
    0.23337927, -0.29872012, -0.99052149, 0.00063585, -0.00159684, 6528.0, 3264.0, 62.0, 199680.0,
]  # fmt: skip


def rig_path(frames: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Kamera am Stick: Kreis mit 0,7 m Radius durch den Raum, aufrecht, langsam drehend.
    Gibt pro Frame (Position, Drehung Welt → Rig) zurück; Rig-Achsen wie OpenCV (y unten)."""
    path = []
    for i in range(frames):
        a = 2 * np.pi * 0.6 * i / max(frames - 1, 1)
        eye = np.array([2.0 + 0.7 * np.cos(a), 1.4 + 0.05 * np.sin(3 * a), 2.0 + 0.7 * np.sin(a)])
        heading = a + 0.3
        target = eye + np.array([np.sin(heading), 0.0, np.cos(heading)])
        path.append((eye, _look_at(eye, target)))
    return path


def _encode(frame_dir: Path, path: Path, fps: int, pattern: str = "%05d.png") -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
         "-i", str(frame_dir / pattern), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", str(path)],
        check=True,
    )  # fmt: skip


def make_equirect_video(path: Path, frames: int = 60, width: int = 1024, fps: int = 12) -> Path:
    """Equirektanguläres 360°-Video (2:1) einer Kamerafahrt durch den geschlossenen Raum."""
    from splatforge.panorama import equirect_rays

    textures = [_texture(seed) for *_, seed in _ROOM]
    rays = equirect_rays(width, width // 2)
    frame_dir = path.parent / f"{path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, (eye, rot) in enumerate(rig_path(frames)):
        img = render_rays(eye, rays @ rot, textures, _ROOM)
        cv2.imencode(".png", img)[1].tofile(frame_dir / f"{i:05d}.png")
    _encode(frame_dir, path, fps)
    for p in frame_dir.iterdir():
        p.unlink()
    frame_dir.rmdir()
    return path


def one_rs_lenses(lens_size: int) -> list:
    """Die beiden Objektive der ONE RS, umgerechnet auf ``lens_size`` Pixel Kantenlänge."""
    from splatforge.insv import InsvMetadata, lens_calibrations
    from splatforge.panorama import FisheyeLens, lens_rotation

    lenses = []
    for i, c in enumerate(lens_calibrations(InsvMetadata(offset_v3=ONE_RS_OFFSET_V3)) or []):
        lens = FisheyeLens(
            c.xi, c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.k3, c.p1, c.p2,
            int(c.width), int(c.height), lens_rotation(i, c.roll),
        )  # fmt: skip
        lenses.append(lens.scaled(lens_size, lens_size))
    return lenses


def make_dual_fisheye_insv(path: Path, frames: int = 60, lens_size: int = 640, fps: int = 12) -> Path:
    """.insv mit zwei Videospuren (je ein Objektiv) und ONE-RS-Kalibrierung im Trailer."""
    textures = [_texture(seed) for *_, seed in _ROOM]
    lenses = one_rs_lenses(lens_size)
    # Für jedes Objektivpixel den Blickstrahl im Rig finden: Strahlen dicht abtasten und vorwärts abbilden
    grids = [_inverse_lens_rays(lens, lens_size) for lens in lenses]
    frame_dir = path.parent / f"{path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, (eye, rot) in enumerate(rig_path(frames)):
        for n, (rays, valid) in enumerate(grids):
            img = render_rays(eye, rays @ rot, textures, _ROOM)
            img[~valid] = 0
            cv2.imencode(".png", img)[1].tofile(frame_dir / f"l{n}_{i:05d}.png")
    tmp = [path.with_name(f"{path.stem}_l{n}.mp4") for n in range(2)]
    for n in range(2):
        _encode(frame_dir, tmp[n], fps, f"l{n}_%05d.png")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp[0]), "-i", str(tmp[1]),
         "-map", "0", "-map", "1", "-c", "copy", "-f", "mp4", str(path)],
        check=True,
    )  # fmt: skip
    for p in [*frame_dir.iterdir(), *tmp]:
        p.unlink()
    frame_dir.rmdir()
    offset = "_".join(f"{v:g}" for v in ONE_RS_OFFSET_V3)
    meta = insv_metadata("Insta360 OneRS", calibration=False) + _pb_field(54, offset)
    with path.open("ab") as fh:
        fh.write(insv_trailer({1: (1, meta)}))
    return path


def _inverse_lens_rays(lens: object, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Blickstrahl (Rig) je Pixel eines Objektivbildes und ob das Pixel im Bildkreis liegt."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    rays, valid = lens.unproject(xx, yy)  # type: ignore[attr-defined]
    return rays, valid
