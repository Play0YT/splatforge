"""Migrationen für job.json.

Ein Job-Ordner kann Tage alt sein und mit einer älteren SplatForge-Version angelegt worden sein.
Damit er nach einem Update weiterläuft, wird job.json beim Laden schrittweise auf die aktuelle
``JOB_SCHEMA_VERSION`` gebracht. Die alte Datei bleibt als ``job.v<N>.json`` erhalten.

Neue Migration ergänzen:
1. ``JOB_SCHEMA_VERSION`` in config.py erhöhen.
2. Hier eine Funktion ``_v<alt>_to_v<neu>`` schreiben und in ``MIGRATIONS`` eintragen.
3. Einen Test in tests/test_migrations.py mit einer echten alten job.json ergänzen.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import JOB_SCHEMA_VERSION
from .errors import SplatForgeError

RawJob = dict[str, Any]

# Schlüssel: Ausgangsversion. Wert: Funktion, die genau eine Version weiter migriert.
MIGRATIONS: dict[int, Callable[[RawJob], RawJob]] = {}


def migrate(raw: RawJob) -> tuple[RawJob, int]:
    """Bringt eine rohe job.json auf die aktuelle Version. Gibt (Daten, Ausgangsversion) zurück."""
    original = int(raw.get("schema_version", 1))
    if original > JOB_SCHEMA_VERSION:
        raise SplatForgeError(
            f"Dieser Job wurde mit einer neueren SplatForge-Version angelegt (Format {original}).",
            "SplatForge aktualisieren, um ihn fortzusetzen.",
        )
    version = original
    data = dict(raw)
    while version < JOB_SCHEMA_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise SplatForgeError(
                f"Für das Job-Format {version} fehlt eine Migration.",
                "Bitte als Fehler melden.",
            )
        data = step(data)
        version += 1
        data["schema_version"] = version
    return data, original
