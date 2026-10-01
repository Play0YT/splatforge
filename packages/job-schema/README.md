# @splatforge/job-schema

Schnittstelle zwischen dem Core und den Apps (Desktop, Server-Worker):

- `schemas/job-config.schema.json` – Job-Konfiguration (`job.json` im Job-Ordner)
- `schemas/progress-event.schema.json` – ein Fortschritts-Event (eine Zeile in `events.jsonl` bzw. auf stdout)

Die Schemas werden aus den Pydantic-Modellen des Cores erzeugt und nicht von Hand bearbeitet:

```bash
uv run splatforge schema --out packages/job-schema/schemas --force
```

Ein Test im Core schlägt fehl, wenn die Dateien hier nicht mehr zu den Modellen passen.

## Versionierung

Das Feld `schema_version` steht in jeder `job.json` und in jedem Event. Bei inkompatiblen Änderungen
wird es erhöht; der Core migriert ältere `job.json`-Dateien automatisch (siehe
`packages/core/src/splatforge/migrations.py`).
