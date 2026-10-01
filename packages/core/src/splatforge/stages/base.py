"""Gemeinsame Schnittstelle aller Pipeline-Stufen."""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..adapters import BrushAdapter, CancelToken, ColmapAdapter, FfmpegAdapter, FfprobeAdapter
from ..config import JobConfig
from ..events import EventSink
from ..job import DONE_FILE, JobDir


@dataclass
class Tools:
    ffmpeg: FfmpegAdapter
    ffprobe: FfprobeAdapter
    colmap: ColmapAdapter
    brush: BrushAdapter

    @classmethod
    def from_config(cls, config: JobConfig) -> Tools:
        return cls(
            ffmpeg=FfmpegAdapter(config.tools.ffmpeg),
            ffprobe=FfprobeAdapter(config.tools.ffprobe),
            colmap=ColmapAdapter(),
            brush=BrushAdapter(config.tools.brush),
        )


@dataclass
class StageContext:
    job: JobDir
    config: JobConfig
    events: EventSink
    tools: Tools
    cancel: CancelToken = field(default_factory=CancelToken)
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str, hint: str | None = None) -> None:
        self.warnings.append(message)
        self.events.warning(message, hint)


class Stage(ABC):
    """Eine Stufe schreibt nur in ihr eigenes Unterverzeichnis und liest Ergebnisse früherer Stufen."""

    #: Kurzname für Events und Logs
    name: str = ""
    #: Unterverzeichnis im Job-Ordner
    dirname: str = ""
    #: Grobes Gewicht für den Gesamtfortschritt
    weight: float = 1.0

    def out_dir(self, ctx: StageContext) -> Path:
        return ctx.job.stage_dir(self.dirname)

    def applies(self, ctx: StageContext) -> bool:
        """Ob die Stufe für diesen Job überhaupt läuft (z. B. 360°-Aufbereitung nur bei 360°)."""
        return True

    def preflight(self, ctx: StageContext) -> None:
        """Prüft vor dem Start des Jobs, ob alles Nötige vorhanden ist, damit ein fehlendes Programm
        nicht erst nach Stunden auffällt. Wirft einen SplatForgeError mit Massnahme."""
        return None

    def is_done(self, ctx: StageContext) -> bool:
        return ctx.job.is_done(self.dirname)

    def estimate_duration(self, ctx: StageContext) -> float | None:
        """Geschätzte Laufzeit in Sekunden, falls abschätzbar."""
        return None

    def estimate_disk_bytes(self, ctx: StageContext) -> int:
        """Geschätzter zusätzlicher Speicherbedarf dieser Stufe."""
        return 0

    @abstractmethod
    def run(self, ctx: StageContext) -> dict[str, Any]:
        """Führt die Stufe aus und gibt Kennzahlen für stage.done und report.json zurück."""

    def cleanup(self, ctx: StageContext) -> None:
        """Entfernt unvollständige Ergebnisse eines abgebrochenen Laufs dieser Stufe.

        Nur das eigene Unterverzeichnis wird angefasst, und nur, wenn die Stufe nicht fertig ist.
        Stufen mit Checkpoints überschreiben diese Methode, damit Checkpoints erhalten bleiben.
        """
        out = self.out_dir(ctx)
        if out.exists() and not (out / DONE_FILE).exists():
            shutil.rmtree(out)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
