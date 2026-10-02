"""Adapter für FFmpeg und ffprobe."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..errors import SplatForgeError, UnsupportedInputError
from .base import BinaryAdapter, CancelToken, Version

FPS_MODE_SINCE: Version = (5, 1)

_INSTALL_HINT = (
    "FFmpeg installieren (https://ffmpeg.org/download.html) oder den Pfad in den Einstellungen angeben."
)


class FfprobeAdapter(BinaryAdapter):
    name = "ffprobe"
    executable = "ffprobe"
    # Ubuntu 22.04 liefert FFmpeg 4.4
    min_version: Version = (4, 4)
    install_hint = _INSTALL_HINT

    def probe(self, path: Path, timeout: float) -> dict[str, Any]:
        """Liest Container- und Stream-Informationen als JSON."""
        result = self.run(
            ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise UnsupportedInputError(
                f"Die Datei {path.name} konnte nicht gelesen werden.",
                "Prüfen, ob es eine vollständige Videodatei ist, oder sie in ein gängiges Format "
                "(z. B. MP4) umwandeln.",
                details=result.stderr[-2000:],
            )
        data: dict[str, Any] = json.loads(result.stdout or "{}")
        return data


class FfmpegAdapter(BinaryAdapter):
    name = "FFmpeg"
    executable = "ffmpeg"
    # Ubuntu 22.04 liefert FFmpeg 4.4
    min_version: Version = (4, 4)
    install_hint = _INSTALL_HINT

    def __init__(self, configured_path: Path | None = None) -> None:
        super().__init__(configured_path)
        self._filters: set[str] | None = None

    def _passthrough_args(self) -> list[str]:
        """Frames unverändert weitergeben. Die Option heisst erst ab FFmpeg 5.1 ``-fps_mode``."""
        version = self.version()
        if version is not None and version < FPS_MODE_SINCE:
            return ["-vsync", "passthrough"]
        return ["-fps_mode", "passthrough"]

    def has_filter(self, name: str) -> bool:
        if self._filters is None:
            result = self.run(["-hide_banner", "-filters"], timeout=30, check=False)
            names: set[str] = set()
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 3 and "->" in parts[2]:
                    names.add(parts[1])
            self._filters = names
        return name in self._filters

    def extract_frames(
        self,
        source: Path,
        out_pattern: Path,
        fps: float,
        max_edge: int,
        jpeg_quality: int,
        tonemap: bool,
        duration_s: float | None,
        on_progress: Callable[[float], None],
        cancel: CancelToken | None,
        stream_index: int = 0,
    ) -> None:
        """Extrahiert Frames mit fester Rate, skaliert auf ``max_edge`` und schreibt JPEGs.

        FFmpeg dreht Smartphone-Videos anhand der Rotations-Metadaten automatisch (autorotate ist
        Standard). Skalierung und Tonemapping laufen danach.
        """
        filters = [f"fps={fps:.6f}"]
        if tonemap:
            if not self.has_filter("zscale") or not self.has_filter("tonemap"):
                raise SplatForgeError(
                    "Für HDR-Videos fehlt FFmpeg der Filter 'zscale' oder 'tonemap'.",
                    "Einen FFmpeg-Build mit libzimg verwenden oder das Video vorher als SDR exportieren.",
                )
            filters += [
                "zscale=t=linear:npl=100",
                "format=gbrpf32le",
                "zscale=p=bt709",
                "tonemap=hable:desat=0",
                "zscale=t=bt709:m=bt709:r=tv",
            ]
        # Längste Kante auf max_edge begrenzen, nie hochskalieren, gerade Pixelzahl.
        filters.append(
            f"scale=w='if(gte(iw,ih),min(iw,{max_edge}),-2)':h='if(gte(iw,ih),-2,min(ih,{max_edge}))'"
            ":flags=lanczos"
        )
        filters.append("format=yuvj420p")
        args: list[str | Path] = [
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            source,
            "-map",
            f"0:v:{stream_index}",
            "-vf",
            ",".join(filters),
            *self._passthrough_args(),
            "-q:v",
            str(jpeg_quality),
            "-progress",
            "pipe:1",
            "-nostats",
            out_pattern,
        ]

        def on_line(line: str) -> None:
            if duration_s and line.startswith("out_time_us="):
                value = line.split("=", 1)[1]
                if value.isdigit():
                    on_progress(min(int(value) / 1e6 / duration_s, 1.0))

        self.stream(args, on_line, cancel=cancel)
