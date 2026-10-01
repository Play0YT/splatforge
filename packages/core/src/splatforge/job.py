"""Job-Ordner: Layout, job.json und stage.done-Markierungen.

Alle Zwischenstände liegen im Job-Ordner. Der Fortsetzungspunkt ergibt sich allein aus den
``stage.done``-Dateien (und später den Checkpoints), damit ein Job Absturz, Neustart und
Stromausfall übersteht.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import JobConfig
from .errors import SplatForgeError
from .migrations import migrate

JOB_FILE = "job.json"
EVENTS_FILE = "events.jsonl"
DONE_FILE = "stage.done"


class JobDir:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @property
    def job_file(self) -> Path:
        return self.root / JOB_FILE

    @property
    def events_file(self) -> Path:
        return self.root / EVENTS_FILE

    def exists(self) -> bool:
        return self.job_file.is_file()

    def stage_dir(self, dirname: str) -> Path:
        return self.root / dirname

    # job.json

    def create(self, config: JobConfig) -> None:
        if self.root.exists() and any(self.root.iterdir()) and not self.exists():
            raise SplatForgeError(
                f"Der Ausgabeordner {self.root} ist nicht leer und enthält keinen SplatForge-Job.",
                "Einen leeren oder neuen Ordner wählen, damit keine Dateien überschrieben werden.",
            )
        if self.exists():
            raise SplatForgeError(
                f"In {self.root} liegt bereits ein Job.",
                "Zum Fortsetzen 'splatforge resume' verwenden oder einen neuen Ordner wählen.",
            )
        self.root.mkdir(parents=True, exist_ok=True)
        config = config.model_copy(update={"inputs": [p.resolve() for p in config.inputs]})
        _atomic_write(self.job_file, config.model_dump_json(indent=2))

    def load(self) -> JobConfig:
        if not self.exists():
            raise SplatForgeError(
                f"In {self.root} wurde kein SplatForge-Job gefunden.",
                "Den Ordner prüfen, der beim Start mit --out angegeben wurde.",
            )
        raw = json.loads(self.job_file.read_text(encoding="utf-8"))
        data, original = migrate(raw)
        config = JobConfig.model_validate(data)
        if original != config.schema_version:
            backup = self.root / f"job.v{original}.json"
            if not backup.exists():
                backup.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            _atomic_write(self.job_file, config.model_dump_json(indent=2))
        return config

    # stage.done

    def is_done(self, dirname: str) -> bool:
        return (self.stage_dir(dirname) / DONE_FILE).is_file()

    def mark_done(self, dirname: str, info: dict[str, Any] | None = None) -> None:
        payload = {
            "finished_at": datetime.now(UTC).isoformat(),
            "core_version": __version__,
            **(info or {}),
        }
        _atomic_write(self.stage_dir(dirname) / DONE_FILE, json.dumps(payload, indent=2))

    def done_info(self, dirname: str) -> dict[str, Any]:
        path = self.stage_dir(dirname) / DONE_FILE
        if not path.is_file():
            return {}
        result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return result


def _atomic_write(path: Path, text: str) -> None:
    """Schreibt erst in eine temporäre Datei und benennt dann um, damit ein Stromausfall keine
    halb geschriebene Datei hinterlässt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
