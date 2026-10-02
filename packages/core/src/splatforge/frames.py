"""Einzelbilder aus einer Eingabe exportieren, ohne einen Job zu starten (``splatforge frames``).

Gedacht zum Prüfen von Aufnahmen, vor allem von 360°-Dateien: Bei Dual-Fisheye (z. B. .insv) wird pro
Objektiv ein eigener Ordner geschrieben.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from .adapters import FfmpegAdapter
from .config import CameraType, JobConfig
from .errors import SplatForgeError, UnsupportedInputError
from .events import EventSink
from .job import JobDir
from .stages.analyze import AnalyzeStage, InputInfo
from .stages.base import StageContext, Tools

FRAME_PATTERN = "%06d.jpg"


def inspect_input(path: Path, config: JobConfig | None = None) -> InputInfo:
    """Untersucht eine einzelne Eingabe (Video, .insv oder Bildordner)."""
    config = config or JobConfig(inputs=[path])
    with tempfile.TemporaryDirectory() as tmp:
        ctx = StageContext(
            job=JobDir(Path(tmp)), config=config, events=EventSink(), tools=Tools.from_config(config)
        )
        return AnalyzeStage().inspect(ctx)[0]


def export_frames(
    source: Path, out: Path, count: int, max_edge: int, config: JobConfig | None = None
) -> list[Path]:
    """Exportiert ``count`` gleichmässig verteilte Bilder. Gibt die geschriebenen Ordner zurück."""
    if out.exists() and any(out.iterdir()):
        raise SplatForgeError(
            f"Der Ordner {out} ist nicht leer.",
            "Einen leeren oder neuen Ordner angeben, damit keine Dateien überschrieben werden.",
        )
    config = config or JobConfig(inputs=[source])
    info = inspect_input(source, config)
    if info.kind != "video":
        raise UnsupportedInputError(
            "Einzelbilder lassen sich nur aus Videos exportieren.", "Eine Videodatei angeben."
        )
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = FfmpegAdapter(config.tools.ffmpeg)
    if info.camera_type == CameraType.DUAL_FISHEYE and info.lenses:
        targets = [(out / f"objektiv_{i + 1}", lens) for i, lens in enumerate(info.lenses)]
    else:
        targets = [(out / "frames", {"path": info.path, "stream": 0, "crop": None})]
    fps = min(info.fps, count / info.duration_s)
    written = []
    for folder, lens in targets:
        folder.mkdir()
        ffmpeg.extract_frames(
            source=Path(lens["path"]),
            out_pattern=folder / FRAME_PATTERN,
            fps=fps,
            max_edge=max_edge,
            jpeg_quality=config.extract.jpeg_quality,
            tonemap=info.hdr and config.extract.tonemap_hdr,
            duration_s=info.duration_s,
            on_progress=lambda _fraction: None,
            cancel=None,
            stream_index=int(lens["stream"]),
            crop_half=lens["crop"],
        )
        written.append(folder)
    (out / "analysis.json").write_text(
        json.dumps(asdict(info), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return written
