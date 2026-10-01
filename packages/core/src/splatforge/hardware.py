"""Erkennung der verfügbaren Hardware und Trainings-Backends."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from typing import Any

from .adapters import BrushAdapter


@dataclass
class HardwareInfo:
    os: str
    machine: str
    cpu_threads: int
    ram_mb: int | None
    cuda: bool
    apple_silicon: bool
    brush: bool
    torch: bool
    recommended_backend: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ram_mb() -> int | None:
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys // (1024 * 1024))
        return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return None


def detect(brush: BrushAdapter | None = None) -> HardwareInfo:
    try:
        import torch

        has_torch = True
        cuda = bool(torch.cuda.is_available())
    except ImportError:
        has_torch = False
        cuda = shutil.which("nvidia-smi") is not None
    apple = sys.platform == "darwin" and platform.machine() == "arm64"
    has_brush = (brush or BrushAdapter()).available()
    # Reihenfolge laut Spezifikation: CUDA, Metal, Vulkan/DX12 (alle über Brush bzw. später gsplat), CPU
    recommended = "brush" if has_brush else ("cpu" if has_torch else "none")
    return HardwareInfo(
        os=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        cpu_threads=os.cpu_count() or 1,
        ram_mb=_ram_mb(),
        cuda=cuda,
        apple_silicon=apple,
        brush=has_brush,
        torch=has_torch,
        recommended_backend=recommended,
    )
