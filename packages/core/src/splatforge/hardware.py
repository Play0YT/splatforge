"""Erkennung der verfügbaren Hardware und Trainings-Backends."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from .adapters import BrushAdapter
from .adapters.base import format_version

# Ab dieser Build-Nummer ist es Windows 11 (Python meldet weiterhin "10").
WINDOWS_11_FIRST_BUILD = 22000
_PROBE_TIMEOUT_S = 15


@dataclass
class HardwareInfo:
    os: str
    machine: str
    cpu_threads: int
    ram_mb: int | None
    gpus: list[str] = field(default_factory=list)
    nvidia_gpu: bool = False
    apple_silicon: bool = False
    brush: bool = False
    brush_version: str | None = None
    brush_problem: str | None = None
    torch: bool = False
    recommended_backend: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def os_name() -> str:
    system = platform.system()
    if system == "Windows":
        build = sys.getwindowsversion().build if hasattr(sys, "getwindowsversion") else 0
        name = "Windows 11" if build >= WINDOWS_11_FIRST_BUILD else f"Windows {platform.release()}"
        return f"{name} (Build {build})"
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    if system == "Linux":
        try:
            info = platform.freedesktop_os_release()
            return str(info.get("PRETTY_NAME", "Linux"))
        except OSError:
            return f"Linux {platform.release()}"
    return f"{system} {platform.release()}"


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


def _command_lines(args: list[str]) -> list[str]:
    """Führt ein festes Systemprogramm aus (keine Nutzereingaben) und gibt die Ausgabezeilen zurück."""
    exe = shutil.which(args[0])
    if exe is None:
        return []
    try:
        result = subprocess.run(  # noqa: S603 - feste Argumentliste
            [exe, *args[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def parse_lspci(lines: list[str]) -> list[str]:
    gpus = []
    for line in lines:
        for marker in ("VGA compatible controller: ", "3D controller: ", "Display controller: "):
            if marker in line:
                gpus.append(line.split(marker, 1)[1])
    return gpus


def detect_gpus() -> list[str]:
    """Namen aller Grafikkarten, wie das Betriebssystem sie meldet."""
    if sys.platform == "win32":
        return _command_lines(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance -ClassName Win32_VideoController | ForEach-Object { $_.Name }",
            ]
        )
    if sys.platform == "darwin":
        lines = _command_lines(["system_profiler", "SPDisplaysDataType"])
        return [line.split(":", 1)[1].strip() for line in lines if line.startswith("Chipset Model:")]
    return parse_lspci(_command_lines(["lspci"]))


def _nvidia_gpu(gpus: list[str]) -> bool:
    if any("nvidia" in g.lower() for g in gpus):
        return True
    return bool(_command_lines(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]))


def detect(brush: BrushAdapter | None = None) -> HardwareInfo:
    try:
        import torch  # noqa: F401

        has_torch = True
    except ImportError:
        has_torch = False
    gpus = detect_gpus()
    brush = brush or BrushAdapter()
    brush_version: str | None = None
    brush_problem: str | None = None
    try:
        version = brush.check()
        brush_version = format_version(version) if version else "unbekannt"
    except Exception as exc:  # fehlt, zu alt oder nicht ausführbar
        brush_problem = str(exc)
    has_brush = brush_version is not None
    # Reihenfolge laut Spezifikation: CUDA, Metal, Vulkan/DX12 (alle über Brush bzw. später gsplat), CPU
    recommended = "brush" if has_brush else ("cpu" if has_torch else "none")
    return HardwareInfo(
        os=os_name(),
        machine=platform.machine(),
        cpu_threads=os.cpu_count() or 1,
        ram_mb=_ram_mb(),
        gpus=gpus,
        nvidia_gpu=_nvidia_gpu(gpus),
        apple_silicon=sys.platform == "darwin" and platform.machine() == "arm64",
        brush=has_brush,
        brush_version=brush_version,
        brush_problem=brush_problem,
        torch=has_torch,
        recommended_backend=recommended,
    )
