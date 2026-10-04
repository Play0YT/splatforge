# Changelog – @splatforge/job-schema

Format nach [Keep a Changelog](https://keepachangelog.com/de/1.1.0/), Versionen nach [SemVer](https://semver.org/lang/de/).

## [0.2.1] – 2026-10-04

### Hinzugefügt
- Job-Konfiguration: `sfm.orient_scene` und `sfm.min_baseline_ratio`. `schema_version` bleibt 1.

## [0.2.0] – 2026-10-02

### Hinzugefügt
- Job-Konfiguration: Abschnitt `mask` (Verfahren, Modell, Klassen, Schwellen, Sicherheitsrand,
  Qualitätskontrolle) für die Personenmaskierung; `masking` ist weiterhin standardmässig `false`.
- `masking` hat jetzt eine Beschreibung.

### Kompatibilität
- `schema_version` bleibt 1: Alle neuen Felder haben Standardwerte, bestehende `job.json` gelten weiter.

## [0.1.0] – 2026-10-01

### Hinzugefügt
- `job-config.schema.json` (Job-Format `schema_version` 1)
- `progress-event.schema.json` (Event-Format `schema_version` 1)
