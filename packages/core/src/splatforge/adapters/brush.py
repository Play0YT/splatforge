"""Adapter für Brush (https://github.com/ArthurBrussee/brush, Apache-2.0).

Brush trainiert Gaussian Splats über wgpu (Metal, Vulkan, DX12) und läuft damit auch ohne NVIDIA-GPU.
Masken liest Brush selbst aus einem Ordner ``masks`` neben ``images`` (schwarz = ignorieren),
also im selben Format wie COLMAP.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..errors import SplatForgeError, ToolMissingError
from .base import BinaryAdapter, RunResult, Version

_GPU_ERROR = re.compile(r"(no suitable adapter|adapter not found|failed to (request|find).*adapter)", re.I)


class BrushAdapter(BinaryAdapter):
    name = "Brush"
    executable = "brush"
    # Brush 0.3 liefert die Datei als brush_app bzw. brush_app.exe aus
    alt_executables = ("brush_app",)
    min_version: Version = (0, 3)
    version_args = ("--version",)
    install_hint = (
        "Brush von https://github.com/ArthurBrussee/brush/releases herunterladen und den Pfad mit "
        "--brush angeben, oder das CPU-Backend verwenden (--backend cpu)."
    )

    def __init__(self, configured_path: Path | None = None) -> None:
        super().__init__(configured_path)
        self._help: str | None = None

    def help_text(self) -> str:
        if self._help is None:
            result = self.run(["--help"], timeout=60, check=False)
            self._help = result.stdout + result.stderr
        return self._help

    def check(self) -> Version | None:
        """Vorhanden, neu genug und mit bekannter Kommandozeile."""
        found = super().check()
        self.steps_flag()
        return found

    def steps_flag(self) -> str:
        """Name der Option für die Anzahl Trainingsschritte.

        Brush 0.3 nennt sie ``--total-steps``, neuere Versionen ``--total-train-iters``.
        """
        return steps_flag_from_help(self.help_text())

    def error_from_output(self, result: RunResult) -> SplatForgeError:
        if _GPU_ERROR.search(result.stdout):
            return SplatForgeError(
                "Brush hat keine nutzbare Grafikkarte gefunden.",
                "Grafiktreiber aktualisieren oder das CPU-Backend verwenden (--backend cpu).",
                details=result.stdout[-4000:],
            )
        return super().error_from_output(result)


def steps_flag_from_help(help_text: str) -> str:
    for flag in ("--total-train-iters", "--total-steps"):
        if flag in help_text:
            return flag
    raise ToolMissingError(
        "Diese Brush-Version wird nicht unterstützt (Option für die Trainingsschritte nicht gefunden).",
        "Brush 0.3 oder neuer von https://github.com/ArthurBrussee/brush/releases verwenden.",
        details=help_text[-2000:],
    )
