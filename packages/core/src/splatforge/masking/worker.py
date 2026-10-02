"""Hintergrundprozess für die Maskierung: ``python -m splatforge _mask <auftrag.json>``."""

from __future__ import annotations

import json
import signal
from pathlib import Path
from types import FrameType

from ..adapters.base import CancelToken
from ..config import MaskMethod, MaskModel, MaskSettings
from ..errors import JobCancelledError, SplatForgeError
from ..models import default_models_dir, ensure_model
from ..worker import emit

EXIT_CANCELLED = 130
DETECTOR_MODEL = "rtdetr-r18"
# Ladebalken der Modelle zählt als erster kleiner Teil des Fortschritts
DOWNLOAD_SHARE = 0.05


def resolve(settings: MaskSettings, providers: list[str]) -> tuple[MaskMethod, str, list[str]]:
    """Wählt Verfahren und Modell. Gibt (Verfahren, Modellschlüssel, Hinweise) zurück."""
    notes: list[str] = []
    gpu = "CUDAExecutionProvider" in providers
    method = settings.method
    if method == MaskMethod.VIDEO:
        notes.append(
            "Die Video-Verfolgung (PyTorch) folgt in einer späteren Version; verwendet wird das Bildmodell."
        )
    method = MaskMethod.IMAGE
    if settings.model == MaskModel.AUTO:
        model = MaskModel.SMALL if gpu else MaskModel.TINY
    else:
        model = settings.model
    return method, str(model), notes


def run(task_file: Path) -> int:
    from .onnx_models import CLASS_GROUPS, Detector, Segmenter, choose_providers
    from .run import compute_masks

    task = json.loads(task_file.read_text(encoding="utf-8"))
    settings = MaskSettings.model_validate(task["settings"])
    cancel = CancelToken()

    def on_signal(signum: int, frame: FrameType | None) -> None:
        cancel.cancel()

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)
    try:
        providers = choose_providers(settings.device)
        method, model_key, notes = resolve(settings, providers)
        for note in notes:
            emit(type="warning", message=note)
        emit(type="log", message=f"Maskierung: {method}, Modell {model_key}, Beschleunigung {providers[0]}")
        models_dir = settings.models_dir or default_models_dir()
        folders = {}
        for i, key in enumerate((DETECTOR_MODEL, model_key)):

            def progress(done: int, total: int, i: int = i, key: str = key) -> None:
                share = DOWNLOAD_SHARE * (i + done / max(total, 1)) / 2
                emit(type="progress", fraction=share, message=f"Modell {key} herunterladen")

            folders[key] = ensure_model(key, models_dir, progress)
        unknown = [c for c in settings.classes if c not in CLASS_GROUPS]
        if unknown:
            raise SplatForgeError(
                f"Unbekannte Klassen für die Maskierung: {', '.join(unknown)}.",
                f"Möglich sind: {', '.join(CLASS_GROUPS)}.",
            )
        classes = tuple(c for group in settings.classes for c in CLASS_GROUPS[group])
        threads = int(task.get("num_threads", 0))
        summary = compute_masks(
            images_dir=Path(task["images_dir"]),
            names=list(task["names"]),
            out_dir=Path(task["out_dir"]),
            settings=settings,
            classes=classes,
            detector=Detector(folders[DETECTOR_MODEL], providers, threads),
            segmenter=Segmenter(folders[model_key], providers, threads),
            on_progress=lambda f, msg: emit(
                type="progress", fraction=DOWNLOAD_SHARE + (1 - DOWNLOAD_SHARE) * f, message=msg
            ),
            cancel=cancel,
        )
    except JobCancelledError:
        return EXIT_CANCELLED
    except SplatForgeError as exc:
        emit(type="error", message=exc.message, hint=exc.hint)
        return 1
    emit(type="result", method=str(method), model=model_key, providers=providers, **summary.to_dict())
    return 0
