"""Adapter für Brush (https://github.com/ArthurBrussee/brush, Apache-2.0).

Brush trainiert Gaussian Splats über wgpu (Metal, Vulkan, DX12) und läuft damit auch ohne NVIDIA-GPU.
Masken liest Brush selbst aus einem Ordner ``masks`` neben ``images`` (schwarz = ignorieren),
also im selben Format wie COLMAP.
"""

from __future__ import annotations

import re

from ..errors import SplatForgeError
from .base import BinaryAdapter, RunResult, Version

_GPU_ERROR = re.compile(r"(no suitable adapter|adapter not found|failed to (request|find).*adapter)", re.I)


class BrushAdapter(BinaryAdapter):
    name = "Brush"
    executable = "brush"
    min_version: Version = (0, 3)
    version_args = ("--version",)
    install_hint = (
        "Brush von https://github.com/ArthurBrussee/brush/releases herunterladen und den Pfad in den "
        "Einstellungen angeben, oder das CPU-Backend verwenden (--backend cpu)."
    )

    def error_from_output(self, result: RunResult) -> SplatForgeError:
        if _GPU_ERROR.search(result.stdout):
            return SplatForgeError(
                "Brush hat keine nutzbare Grafikkarte gefunden.",
                "Grafiktreiber aktualisieren oder das CPU-Backend verwenden (--backend cpu).",
                details=result.stdout[-4000:],
            )
        return super().error_from_output(result)
