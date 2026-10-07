"""Fortschritts-Events als JSON-Lines.

Desktop-App und Server-Worker lesen diese Events, um Fortschritt, Restzeit und Log anzuzeigen.
Jedes Event wird zusätzlich in ``events.jsonl`` im Job-Ordner angehängt.
"""

from __future__ import annotations

import sys
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TextIO

from pydantic import BaseModel, ConfigDict, Field

EVENT_SCHEMA_VERSION = 1


class EventType(StrEnum):
    JOB_STARTED = "job_started"
    STAGE_STARTED = "stage_started"
    STAGE_SKIPPED = "stage_skipped"
    PROGRESS = "progress"
    LOG = "log"
    WARNING = "warning"
    PREVIEW = "preview"
    STAGE_FINISHED = "stage_finished"
    JOB_FINISHED = "job_finished"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"


class ProgressEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = EVENT_SCHEMA_VERSION
    type: EventType
    time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stage: str | None = None
    stage_index: int | None = Field(default=None, ge=0)
    stage_count: int | None = Field(default=None, ge=0)
    stage_percent: float | None = Field(default=None, ge=0, le=100)
    percent: float | None = Field(default=None, ge=0, le=100, description="Gesamtfortschritt")
    eta_seconds: float | None = Field(default=None, ge=0)
    message: str = ""
    hint: str | None = None
    data: dict[str, object] | None = None


class Display(Protocol):
    """Anzeige für Menschen (z. B. ``terminal.TerminalDisplay`` mit Fortschrittsbalken)."""

    def handle(self, event: ProgressEvent) -> None: ...


class EventSink:
    """Schreibt Events als JSON-Lines in eine Datei und optional auf einen Stream oder eine Anzeige."""

    def __init__(
        self,
        log_file: Path | None = None,
        stream: TextIO | None = None,
        human: bool = False,
        display: Display | None = None,
    ):
        self._log_file = log_file
        self._stream = stream
        self._human = human
        self._display = display
        self._lock = threading.Lock()
        self.stage: str | None = None
        self.stage_index: int | None = None
        self.stage_count: int | None = None
        self.stage_weight_done = 0.0
        self.stage_weight = 0.0

    def emit(self, event: ProgressEvent) -> None:
        if event.stage is None and self.stage is not None:
            event.stage = self.stage
            event.stage_index = self.stage_index
            event.stage_count = self.stage_count
        line = event.model_dump_json(exclude_none=True)
        with self._lock:
            if self._log_file is not None:
                with self._log_file.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            if self._display is not None:
                self._display.handle(event)
            elif self._stream is not None:
                self._stream.write((human_line(event) if self._human else line) + "\n")
                self._stream.flush()

    # Komfortfunktionen für Stufen

    def log(self, message: str) -> None:
        self.emit(ProgressEvent(type=EventType.LOG, message=message))

    def warning(self, message: str, hint: str | None = None) -> None:
        self.emit(ProgressEvent(type=EventType.WARNING, message=message, hint=hint))

    def progress(self, fraction: float, eta_seconds: float | None = None, message: str = "") -> None:
        fraction = min(max(fraction, 0.0), 1.0)
        overall = self.stage_weight_done + fraction * self.stage_weight
        self.emit(
            ProgressEvent(
                type=EventType.PROGRESS,
                stage_percent=round(fraction * 100, 2),
                percent=round(min(overall, 1.0) * 100, 2),
                eta_seconds=eta_seconds,
                message=message,
            )
        )


def stderr_sink(log_file: Path | None) -> EventSink:
    return EventSink(log_file=log_file, stream=sys.stderr, human=True)


def human_line(event: ProgressEvent) -> str:
    prefix = f"[{event.stage}]" if event.stage else "[job]"
    if event.type == EventType.PROGRESS:
        eta = f", noch ca. {_fmt_duration(event.eta_seconds)}" if event.eta_seconds is not None else ""
        text = f"{event.stage_percent:5.1f} %{eta}"
        return f"{prefix} {text} {event.message}".rstrip()
    text = event.message
    if event.hint:
        text += f" → {event.hint}"
    label = {
        EventType.WARNING: "Warnung: ",
        EventType.JOB_FAILED: "Fehler: ",
    }.get(event.type, "")
    return f"{prefix} {label}{text}"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    if s < 120:
        return f"{s} s"
    if s < 7200:
        return f"{s // 60} min"
    return f"{s // 3600} h {(s % 3600) // 60} min"
