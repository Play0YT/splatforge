# Architektur

## Überblick

```
apps/desktop ───────┐
                    ├──► splatforge (Core) ──► Adapter: FFmpeg, COLMAP (pycolmap), Brush
apps/server-worker ─┘     job.json rein, JSON-Lines-Events raus, Dateien in den Job-Ordner

apps/web ──► apps/server-api ──► Redis-Queue ──► apps/server-worker
packages/job-schema: JSON-Schemas der Schnittstelle, erzeugt aus den Core-Modellen
```

Der Core enthält keinen UI- und keinen HTTP-Code. Desktop und Server-Worker starten ihn als Prozess
(`splatforge run`/`resume`) oder rufen `splatforge.pipeline.Pipeline` direkt auf. Keine App importiert Code
einer anderen App.

## Job-Ordner

```
<job>/
├─ job.json            Job-Konfiguration (JobConfig, mit schema_version)
├─ job.v<N>.json       alte Fassung nach einer Migration
├─ events.jsonl        alle Events dieses Jobs
├─ 01_analyze/         analysis.json
├─ 02_extract/         frames/, frames.json
├─ 03_select/          images/ (ausgewählte Frames), selection.json
├─ 05_mask/            (Meilenstein 2) masks/<bild>.png
├─ 06_sfm/             database.db, sparse_*/, dataset/ (entzerrt: images/, sparse/), colmap.log
├─ 07_train/           backend.txt, checkpoint.pt bzw. brush_exports/, preview.ply, final.ply
└─ 08_export/          splat.ply, report.json
```

Jede Stufe schreibt nur in ihr eigenes Verzeichnis und legt am Ende `stage.done` an (atomar geschrieben).
Beim Fortsetzen werden fertige Stufen übersprungen; eine Stufe ohne `stage.done` wird vorher aufgeräumt
und neu ausgeführt. Ausnahme: das Training behält seine Checkpoints.

## Stufen

Gemeinsame Schnittstelle `Stage` (`stages/base.py`): `run`, `is_done`, `estimate_duration`,
`estimate_disk_bytes`, `cleanup`, `applies`. Vor jeder Stufe prüft die Pipeline den freien Speicherplatz.

| Nr. | Stufe | Modul | Stand |
| --- | --- | --- | --- |
| 1 | Analyse | `stages/analyze.py` | fertig |
| 2 | Frame-Extraktion | `stages/extract.py` | fertig |
| 3 | Frame-Auswahl | `stages/select.py` | fertig |
| 4 | 360°-Aufbereitung | – | Meilenstein 3 |
| 5 | Personenmaskierung | – | Meilenstein 2 |
| 6 | Kamerapositionen | `stages/sfm.py` | fertig |
| 7 | Training | `stages/train.py`, `training/cpu.py` | Brush + CPU |
| 8 | Export | `stages/export.py` | `.ply` + Report; Floater/`.spz` in Meilenstein 7 |

## Events

Ein Event ist ein JSON-Objekt pro Zeile (`events.py`, Schema in `packages/job-schema`). Typen:
`job_started`, `stage_started`, `stage_skipped`, `progress`, `log`, `warning`, `preview`,
`stage_finished`, `job_finished`, `job_failed`, `job_cancelled`. `percent` ist der Gesamtfortschritt
(gewichtet nach Stufe), `stage_percent` der Fortschritt der Stufe, `eta_seconds` die Restzeit der Stufe.
Fehler-Events tragen eine verständliche `message` und einen konkreten `hint`; technische Details stehen
als `log`-Event davor.

## Externe Programme

Jedes externe Programm hat eine Adapter-Klasse (`adapters/`) mit Suche (Einstellung, sonst `PATH`),
Versionsprüfung und verständlicher Fehlermeldung. Prozesse werden nur mit Argumentlisten gestartet, nie
über eine Shell. Abbrechen läuft über ein `CancelToken`, das laufende Prozesse beendet.

## Konfiguration und Migrationen

`config.py` enthält alle Einstellungen und Schwellenwerte als Pydantic-Modelle. Ändert sich das Format von
`job.json` inkompatibel, wird `JOB_SCHEMA_VERSION` erhöht und in `migrations.py` eine Migration ergänzt,
damit bestehende Job-Ordner weiterlaufen.

## Plattformen

Pfade immer als `pathlib.Path`. Bilder werden über `imageio.py` gelesen und geschrieben, weil OpenCV unter
Windows keine Nicht-ASCII-Pfade öffnet. Die Tests laufen mit Leerzeichen und Umlauten in den Pfaden.
