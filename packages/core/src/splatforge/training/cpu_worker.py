"""Eigener Prozess für das CPU-Training.

Das Training läuft getrennt vom Hauptprozess, damit PyTorch und pycolmap nie im selben Prozess geladen
werden (unter macOS bricht das wegen doppelter OpenMP-Bibliotheken ab). Nebenbei bleibt der Hauptprozess
stabil, falls das Training wegen Speichermangels abstürzt.

Aufruf: ``python -m splatforge _train-cpu <auftrag.json>``. Ausgabe: ein JSON-Objekt pro Zeile.
"""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any

from ..adapters.base import CancelToken
from ..config import TrainSettings
from ..errors import JobCancelledError, SplatForgeError

EXIT_CANCELLED = 130


def _emit(**data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False), flush=True)


def run(task_file: Path) -> int:
    from .cpu import train_cpu

    task = json.loads(task_file.read_text(encoding="utf-8"))
    cancel = CancelToken()

    def on_signal(signum: int, frame: FrameType | None) -> None:
        cancel.cancel()

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)
    try:
        result = train_cpu(
            dataset=Path(task["dataset"]),
            work_dir=Path(task["work_dir"]),
            iterations=int(task["iterations"]),
            settings=TrainSettings.model_validate(task["settings"]),
            num_threads=int(task["num_threads"]),
            on_progress=lambda step, total, loss, eta: _emit(
                type="progress", step=step, total=total, loss=loss, eta=eta
            ),
            on_preview=lambda path, step: _emit(type="preview", path=str(path), step=step),
            cancel=cancel,
        )
    except JobCancelledError:
        return EXIT_CANCELLED
    except SplatForgeError as exc:
        _emit(type="error", message=exc.message, hint=exc.hint)
        return 1
    _emit(
        type="result",
        ply=str(result.ply),
        gaussians=result.gaussians,
        psnr=result.psnr,
        ssim=result.ssim,
        eval_views=result.eval_views,
        seconds=result.seconds,
    )
    return 0


def self_command(*args: str) -> list[str]:
    """Befehl, der SplatForge selbst erneut startet (auch als gepackte Anwendung)."""
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-m", "splatforge", *args]
