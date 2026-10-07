"""Lesbare Terminal-Anzeige: Meldungen laufen nach oben weg, unten stehen zwei Fortschrittsbalken.

Oberer Balken: die laufende Stufe. Unterer Balken: der ganze Job. Zu beiden die vergangene Zeit und eine
grobe Restzeit. Ein Hintergrund-Thread zeichnet die Balken jede Sekunde neu, damit die Zeit auch dann
weiterläuft, wenn eine Stufe (z. B. COLMAP) länger keinen Fortschritt meldet.

Nur für echte Terminals. Ohne ANSI-Unterstützung (oder bei ``--json``) bleibt es bei einfachen Zeilen.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from typing import TextIO

from .events import EventType, ProgressEvent, human_line

CLEAR_LINE = "\r\x1b[2K"
CURSOR_UP = "\x1b[1A"
REFRESH_S = 1.0
# Restzeit erst schätzen, wenn genug Fortschritt und Zeit vorliegen, sonst springt sie stark
MIN_FRACTION_FOR_ETA = 0.02
MIN_SECONDS_FOR_ETA = 5.0


def format_duration(seconds: float) -> str:
    """Dauer kompakt: 0:42, 12:05, 1:02:33."""
    s = max(0, int(round(seconds)))
    h, rest = divmod(s, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def estimate_remaining(fraction: float, elapsed: float) -> float | None:
    """Lineare Hochrechnung aus dem bisherigen Tempo; ``None``, solange sie zu unsicher ist."""
    if fraction < MIN_FRACTION_FOR_ETA or elapsed < MIN_SECONDS_FOR_ETA or fraction >= 1:
        return None
    return elapsed / fraction * (1 - fraction)


def render_bar(
    label: str, fraction: float, elapsed: float, remaining: float | None, width: int, ascii_only: bool = False
) -> str:
    """Eine Balkenzeile, genau so breit, dass sie nicht umbricht."""
    fraction = min(max(fraction, 0.0), 1.0)
    times = f" {fraction * 100:5.1f} %  {format_duration(elapsed)}"
    if fraction >= 1:
        times += "  fertig"
    else:
        times += f"  noch ~{format_duration(remaining)}" if remaining is not None else "  noch ?"
    bar_width = max(10, min(40, width - len(label) - len(times) - 3))
    filled = int(round(bar_width * fraction))
    full, empty = ("#", "-") if ascii_only else ("█", "░")
    line = f"{label} {full * filled}{empty * (bar_width - filled)}{times}"
    return line[: max(width - 1, 1)]


def supports_ansi(stream: TextIO) -> bool:
    """Ob der Stream ein Terminal ist, das Cursor-Steuerzeichen versteht (Windows: VT-Modus einschalten)."""
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if sys.platform != "win32":
        return True
    try:  # Windows 10+: virtuelle Terminal-Verarbeitung für die Konsole aktivieren
        import ctypes

        kernel32 = getattr(ctypes, "windll").kernel32  # noqa: B009 - windll gibt es nur unter Windows
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:  # alte Konsole oder kein ctypes
        return False


class TerminalDisplay:
    """Zeichnet Meldungen und die beiden Balken. Alle Aufrufe sind threadsicher."""

    def __init__(
        self,
        stream: TextIO,
        clock: Callable[[], float] = time.monotonic,
        width: Callable[[], int] | None = None,
        refresh: bool = True,
    ) -> None:
        self._stream = stream
        self._clock = clock
        self._width = width or (lambda: shutil.get_terminal_size((100, 24)).columns)
        encoding = getattr(stream, "encoding", None) or "ascii"
        try:
            "█░".encode(encoding)
            self._ascii = False
        except (UnicodeEncodeError, LookupError):
            self._ascii = True
        self._lock = threading.RLock()
        self._bars_drawn = 0
        self._job_start: float | None = None
        self._job_percent_at_start = 0.0
        self._baseline_set = False
        self._stage_start: float | None = None
        self._stage = ""
        self._stage_index: int | None = None
        self._stage_count: int | None = None
        self._stage_fraction = 0.0
        self._stage_message = ""
        self._stage_eta: float | None = None
        self._job_fraction = 0.0
        self._active = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if refresh:
            self._thread = threading.Thread(target=self._tick, name="splatforge-terminal", daemon=True)
            self._thread.start()

    def handle(self, event: ProgressEvent) -> None:
        with self._lock:
            now = self._clock()
            if event.type == EventType.JOB_STARTED:
                self._job_start = now
                self._job_percent_at_start = (event.percent or 0.0) / 100
                self._baseline_set = False
                self._active = True
            if event.stage_index is not None:
                self._stage_index, self._stage_count = event.stage_index, event.stage_count
            if event.type == EventType.STAGE_STARTED:
                # Beim Fortsetzen sind frühere Stufen schon fertig: Tempo erst ab der ersten echten Stufe
                if not self._baseline_set:
                    self._job_percent_at_start = (event.percent or 0.0) / 100
                    self._job_start = now
                    self._baseline_set = True
                self._stage = event.stage or ""
                self._stage_start = now
                self._stage_fraction = 0.0
                self._stage_message = ""
                self._stage_eta = event.eta_seconds
            if event.percent is not None:
                self._job_fraction = event.percent / 100
            if event.type == EventType.PROGRESS:
                self._stage_fraction = (event.stage_percent or 0.0) / 100
                if event.message:
                    self._stage_message = event.message
                self._stage_eta = event.eta_seconds
                self._draw_bars()
                return
            if event.type == EventType.STAGE_FINISHED:
                self._stage_fraction = 1.0
            self._clear_bars()
            self._stream.write(human_line(event) + "\n")
            if event.type in (EventType.JOB_FINISHED, EventType.JOB_FAILED, EventType.JOB_CANCELLED):
                if event.type == EventType.JOB_FINISHED:
                    self._job_fraction = self._stage_fraction = 1.0
                self._draw_bars()
                self._stream.write("\n")
                self._bars_drawn = 0
                self._active = False
                self.close()
            else:
                self._draw_bars()
            self._stream.flush()

    def close(self) -> None:
        self._stop.set()

    # Zeichnen

    def lines(self) -> tuple[str, str]:
        """Die beiden Balkenzeilen für den aktuellen Zustand."""
        now = self._clock()
        width = self._width()
        stage_elapsed = now - self._stage_start if self._stage_start is not None else 0.0
        stage_eta = self._stage_eta
        if stage_eta is None:
            stage_eta = estimate_remaining(self._stage_fraction, stage_elapsed)
        step = (
            f" {self._stage_index + 1}/{self._stage_count}"
            if self._stage_index is not None and self._stage_count
            else ""
        )
        label = f"{self._stage or 'Start'}{step}"
        if self._stage_message:
            label += f" · {self._stage_message}"
        label = label[: max(10, width // 3)].ljust(min(28, max(10, width // 3)))
        job_elapsed = now - self._job_start if self._job_start is not None else 0.0
        # Restzeit nur aus dem Fortschritt dieses Laufs (beim Fortsetzen sind Stufen schon fertig)
        done_now = max(self._job_fraction - self._job_percent_at_start, 0.0)
        todo_at_start = max(1.0 - self._job_percent_at_start, 1e-9)
        job_eta = estimate_remaining(done_now / todo_at_start, job_elapsed)
        job_label = "Gesamt".ljust(len(label))
        return (
            render_bar(label, self._stage_fraction, stage_elapsed, stage_eta, width, self._ascii),
            render_bar(job_label, self._job_fraction, job_elapsed, job_eta, width, self._ascii),
        )

    def _clear_bars(self) -> None:
        if self._bars_drawn:
            self._stream.write(CLEAR_LINE + (CURSOR_UP + CLEAR_LINE) * (self._bars_drawn - 1))
            self._bars_drawn = 0

    def _draw_bars(self) -> None:
        if not self._active:
            return
        self._clear_bars()
        top, bottom = self.lines()
        # Ohne abschliessenden Zeilenumbruch, damit der Cursor in der unteren Balkenzeile bleibt
        self._stream.write(f"{top}\n{bottom}")
        self._stream.flush()
        self._bars_drawn = 2

    def _tick(self) -> None:
        while not self._stop.wait(REFRESH_S):
            with self._lock:
                if self._active:
                    try:
                        self._draw_bars()
                    except (OSError, ValueError):  # Stream geschlossen
                        return
