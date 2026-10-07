"""Terminal-Anzeige mit zwei Fortschrittsbalken."""

from __future__ import annotations

import io
import json
from pathlib import Path

from splatforge.events import EventSink, EventType, ProgressEvent
from splatforge.terminal import (
    CLEAR_LINE,
    CURSOR_UP,
    TerminalDisplay,
    estimate_remaining,
    format_duration,
    render_bar,
)


def screen(raw: str) -> list[str]:
    """Wertet die verwendeten Steuerzeichen aus (\\n, \\r, Zeile löschen, Cursor hoch)."""
    lines = [""]
    row = 0
    i = 0
    while i < len(raw):
        if raw.startswith(CLEAR_LINE, i):
            lines[row] = ""
            i += len(CLEAR_LINE)
        elif raw.startswith(CURSOR_UP, i):
            row = max(row - 1, 0)
            i += len(CURSOR_UP)
        elif raw[i] == "\n":
            row += 1
            if row == len(lines):
                lines.append("")
            i += 1
        else:
            lines[row] += raw[i]
            i += 1
    return lines


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_format_duration() -> None:
    assert format_duration(0) == "0:00"
    assert format_duration(42.4) == "0:42"
    assert format_duration(725) == "12:05"
    assert format_duration(3753) == "1:02:33"


def test_estimate_remaining() -> None:
    assert estimate_remaining(0.01, 100) is None  # zu wenig Fortschritt
    assert estimate_remaining(0.5, 2) is None  # zu kurz gelaufen
    assert estimate_remaining(0.25, 60) == 180
    assert estimate_remaining(1.0, 60) is None


def test_render_bar_fits_width() -> None:
    for width in (40, 80, 200):
        line = render_bar("sfm 6/8 · Bilder vergleichen", 0.5, 125, 300, width)
        assert len(line) < width
    line = render_bar("Gesamt", 0.25, 60, None, 100, ascii_only=True)
    assert "#" in line and "-" in line and "25.0 %" in line and "1:00" in line and "noch ?" in line


def _event(kind: EventType, **kw: object) -> ProgressEvent:
    return ProgressEvent(type=kind, **kw)  # type: ignore[arg-type]


def test_display_keeps_bars_at_bottom() -> None:
    out = io.StringIO()
    clock = Clock()
    display = TerminalDisplay(out, clock=clock, width=lambda: 100, refresh=False)
    display.handle(_event(EventType.JOB_STARTED, message="Job in /x", percent=0))
    display.handle(_event(EventType.STAGE_STARTED, stage="sfm", stage_index=5, stage_count=8, message="sfm"))
    clock.now += 30
    display.handle(
        _event(EventType.PROGRESS, stage="sfm", stage_index=5, stage_count=8, stage_percent=25.0,
               percent=50.0, message="Bilder vergleichen")
    )  # fmt: skip
    display.handle(_event(EventType.LOG, stage="sfm", message="global: 600 von 600 Bildern"))
    lines = screen(out.getvalue())
    assert lines[0] == "[job] Job in /x"
    assert lines[1] == "[sfm] sfm"
    assert lines[2] == "[sfm] global: 600 von 600 Bildern"
    top, bottom = lines[-2], lines[-1]
    assert top.startswith("sfm 6/8 · Bilder vergleichen")
    assert "25.0 %" in top and "0:30" in top and "noch ~1:30" in top
    assert bottom.startswith("Gesamt") and "50.0 %" in bottom and "noch ~0:30" in bottom
    assert len(lines) == 5  # keine Zeile pro Fortschritts-Event

    clock.now += 30
    display.handle(_event(EventType.JOB_FINISHED, message="Fertig", percent=100))
    lines = screen(out.getvalue())
    assert lines[-4] == "[job] Fertig"
    assert "100.0 %" in lines[-2] and "1:00" in lines[-2] and "fertig" in lines[-2]
    assert lines[-1] == ""  # Cursor steht nach dem Ende in einer neuen Zeile


def test_resume_estimates_only_this_run() -> None:
    """Beim Fortsetzen zählt nur der Fortschritt dieses Laufs für die Restzeit."""
    out = io.StringIO()
    clock = Clock()
    display = TerminalDisplay(out, clock=clock, width=lambda: 100, refresh=False)
    display.handle(_event(EventType.JOB_STARTED, percent=0))
    display.handle(_event(EventType.STAGE_SKIPPED, stage="sfm", stage_index=5, stage_count=8, percent=60.0))
    display.handle(_event(EventType.STAGE_STARTED, stage="train", stage_index=6, stage_count=8, percent=60.0))
    clock.now += 100
    display.handle(_event(EventType.PROGRESS, stage="train", stage_percent=50.0, percent=80.0))
    # 20 von 40 Prozentpunkten dieses Laufs in 100 s → noch etwa 100 s
    assert "noch ~1:40" in display.lines()[1]


def test_sink_uses_display_and_still_logs(tmp_path: Path) -> None:
    out = io.StringIO()
    log = tmp_path / "events.jsonl"
    sink = EventSink(log_file=log, stream=out, human=True, display=TerminalDisplay(out, refresh=False))
    sink.emit(_event(EventType.JOB_STARTED, message="los", percent=0))
    sink.progress(0.5, message="halb")
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[-1])["type"] == "progress"
    assert "halb" in screen(out.getvalue())[-2]
