"""Gemeinsame Hilfen für Hintergrundprozesse (Training, Maskierung).

Rechenintensive Stufen laufen in eigenen Prozessen: PyTorch, ONNX Runtime und pycolmap kommen sich so nie
in die Quere (unter macOS bricht das Laden von PyTorch und pycolmap im selben Prozess ab), und ein Absturz
wegen Speichermangels reisst den Hauptprozess nicht mit.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def emit(**data: Any) -> None:
    """Eine Nachricht an den Hauptprozess: ein JSON-Objekt pro Zeile auf stdout."""
    print(json.dumps(data, ensure_ascii=False), flush=True)


def self_command(*args: str) -> list[str]:
    """Befehl, der SplatForge selbst erneut startet (auch als gepackte Anwendung)."""
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-m", "splatforge", *args]
