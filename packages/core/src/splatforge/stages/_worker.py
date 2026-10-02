"""Startet einen SplatForge-Hintergrundprozess und übersetzt seine Nachrichten in Events."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..adapters.base import stream_process
from ..errors import SplatForgeError
from ..worker import self_command
from .base import StageContext, write_json

# Längere Zeilen fremder Ausgaben werden im Log gekürzt
MAX_LOG_LINE = 400


def run_worker(
    ctx: StageContext,
    command: str,
    task_file: Path,
    task: dict[str, Any],
    what: str,
    on_message: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Schreibt den Auftrag, startet ``splatforge <command> <auftrag>``, gibt die Ergebnis-Nachricht zurück.

    Standard-Nachrichten: ``progress`` (mit ``fraction``), ``log``, ``warning``, ``error``, ``result``.
    Alle anderen (und ``progress`` ohne ``fraction``) gehen an ``on_message``.
    """
    write_json(task_file, task)
    result: dict[str, Any] = {}
    error: dict[str, Any] = {}

    def on_line(line: str) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            if line.strip():
                ctx.events.log(line.strip()[:MAX_LOG_LINE])
            return
        kind = msg.get("type")
        if kind == "progress" and "fraction" in msg:
            ctx.events.progress(float(msg["fraction"]), msg.get("eta"), message=str(msg.get("message", "")))
        elif kind == "log":
            ctx.events.log(str(msg.get("message", "")))
        elif kind == "warning":
            ctx.warn(str(msg.get("message", "")), msg.get("hint"))
        elif kind == "result":
            result.update(msg)
        elif kind == "error":
            error.update(msg)
        elif on_message is not None:
            on_message(msg)

    outcome = stream_process(self_command(command, str(task_file)), on_line, cancel=ctx.cancel)
    if outcome.returncode != 0 or not result:
        if error:
            raise SplatForgeError(error.get("message", f"{what} fehlgeschlagen."), error.get("hint", ""))
        raise SplatForgeError(
            f"{what} ist unerwartet abgebrochen.",
            "Den Job mit 'splatforge resume' fortsetzen. Tritt der Fehler wieder auf, weniger Frames oder "
            "eine kleinere Bildkante wählen.",
            details=f"Exit-Code {outcome.returncode}\n{outcome.stdout[-4000:]}",
        )
    return result
