"""Fehlerklassen mit verständlicher Meldung und konkreter Maßnahme für Laien."""

from __future__ import annotations

INSTALL_TORCH_HINT = (
    "Brush installieren (schnell, mit Grafikkarte) oder PyTorch für das CPU-Training installieren: "
    "uv pip install torch --index-url https://download.pytorch.org/whl/cpu"
)


class SplatForgeError(Exception):
    """Ein erwarteter Fehler, den der Nutzer selbst beheben kann.

    ``message`` beschreibt das Problem, ``hint`` die konkrete Maßnahme. Technische Details
    gehören in ``details`` und landen nur im Log.
    """

    def __init__(self, message: str, hint: str = "", details: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details

    def __str__(self) -> str:
        return f"{self.message} {self.hint}".strip()


class ToolMissingError(SplatForgeError):
    """Eine externe Binärdatei oder ein Python-Paket fehlt oder ist zu alt."""


class UnsupportedInputError(SplatForgeError):
    """Die Eingabe kann (noch) nicht verarbeitet werden."""


class ReconstructionError(SplatForgeError):
    """Die Kamerapositionen konnten nicht zuverlässig bestimmt werden."""


class InsufficientResourcesError(SplatForgeError):
    """Zu wenig Speicherplatz oder Arbeitsspeicher."""


class JobCancelledError(Exception):
    """Der Job wurde vom Nutzer abgebrochen (kein Fehler im eigentlichen Sinn)."""
