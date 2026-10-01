# Plan Meilenstein 1: Core-Grundgerüst

**Ziel laut SPEC:** Monorepo, Core-CLI, Analyse, Frame-Extraktion, Frame-Auswahl, COLMAP, Training mit einem Backend,
`.ply`-Export.
**Akzeptanzkriterium:** Ein normaler Smartphone-Clip ergibt auf macOS, Windows und Ubuntu per CLI einen betrachtbaren
Splat, auch nur mit CPU.

Status: Entwurf, wartet auf Antworten zu den offenen Fragen unten.

## Umfang

Enthalten:

- Monorepo-Gerüst nach SPEC (Ordner für alle Pakete und Apps, aber nur `packages/core` und `packages/job-schema` mit Inhalt)
- `packages/job-schema`: JSON-Schema für Job-Konfiguration und Fortschritts-Events (Version 1)
- `packages/core`: Python-Paket `splatforge` mit CLI `splatforge run <input> --preset <stufe> --out <ordner>`
- Pipeline-Gerüst mit gemeinsamer Stufen-Schnittstelle (`run`, `is_done`, `estimate_duration`, `cleanup`),
  `stage.done`-Markierungen und JSON-Lines-Events
- Stufen 1, 2, 3, 6, 7 und eine einfache Stufe 8 (nur `.ply`-Export und `report.json`)
- Adapter für FFmpeg/ffprobe, COLMAP und Brush mit Versionsprüfung
- Konfiguration und Qualitätsstufen als Pydantic-Modelle
- Tests (pytest), Linting und Typprüfung (ruff, mypy), CI auf allen drei Betriebssystemen

Nicht enthalten (spätere Meilensteine): Maskierung (Stufe 5 nur als Platzhalter, der keine Masken erzeugt),
360°/INSV (Stufe 4), OOM-Behandlung und Fortsetzen mitten im Training, Floater-Entfernung, `.spz`, jede UI.

## Arbeitsschritte

1. **Gerüst:** Ordnerstruktur, `pyproject.toml` für den Core, ruff/mypy/pytest-Konfiguration, CI-Workflow `core.yml`.
2. **Job-Schema:** JSON-Schema für `JobConfig` und `ProgressEvent`. Die Pydantic-Modelle im Core sind die Quelle;
   das Schema wird daraus erzeugt und ein Test prüft, dass die eingecheckte Datei aktuell ist.
3. **Konfiguration:** `Settings` (Pfade zu Binärdateien, Threads, Speicherreserve) und `Preset` (Vorschau, Standard,
   Hoch mit den Startwerten aus der SPEC). Alle Schwellenwerte (Schärfe, Bewegung, 60 %-Grenze usw.) als Felder.
4. **Adapter:** `FfmpegAdapter`, `ColmapAdapter`, `BrushAdapter`. Jeder findet die Binärdatei (Einstellung → `PATH`),
   prüft die Mindestversion, startet Subprozesse nur mit Argumentlisten und übersetzt typische Fehler in
   verständliche Meldungen.
5. **Pipeline-Gerüst:** Basisklasse `Stage`, `Pipeline`-Runner, Job-Ordner-Layout, Event-Ausgabe als JSON-Lines
   auf stdout oder in eine Datei, Prüfung des freien Speicherplatzes vor jeder Stufe.
6. **Stufe 1 Analyse:** ffprobe-Auswertung (Container, Streams, Auflösung, Framerate, Rotation, Dauer, HDR/10 Bit).
   Kameratyp-Erkennung zunächst nur „Perspektive“ vs. „nicht unterstützt (noch)“, mit verständlicher Meldung.
7. **Stufe 2 Frame-Extraktion:** Zielanzahl oder feste Rate, Rotation beachten, Tonemapping für HDR/10 Bit,
   Skalierung auf die maximale Bildkante des Presets.
8. **Stufe 3 Frame-Auswahl:** Laplace-Varianz relativ zum Clip-Median, Bewegung per optischem Fluss,
   Auswahl nach Abdeckung der Kamerabewegung.
9. **Stufe 6 Kamerapositionen:** COLMAP Feature-Extraktion, sequenzielles Matching mit Loop-Detection, Mapping.
   Diagnose bei < 60 % registrierten Bildern. GLOMAP optional, falls verfügbar.
10. **Stufe 7 Training:** Brush als Subprozess auf dem COLMAP-Ergebnis, Fortschritt aus der Ausgabe parsen,
    Checkpoints im Job-Ordner.
11. **Stufe 8 (minimal):** `.ply` übernehmen, `report.json` mit Laufzeiten, registrierten Bildern,
    Gaussian-Anzahl und Warnungen.
12. **Tests:** Unit-Tests je Stufe mit kleinen, synthetisch erzeugten Clips; Integrationstest „Vorschau auf CPU“
    mit Prüfung auf gültige `.ply`-Datei. Pfade mit Leerzeichen und Umlauten in den Tests.
13. **Doku:** `README.md` (CLI-Nutzung), `docs/architecture.md`, `THIRD_PARTY_LICENSES.md`.
14. **Messung:** Laufzeiten der Vorschau-Stufe auf CPU messen und in `docs/` festhalten.

## Annahmen

- **Python 3.11+** für den Core, Paketverwaltung mit `uv` (wird vor dem Einsatz geprüft).
- **COLMAP über pycolmap.** Annahme: die PyPI-Pakete bringen COLMAP für alle drei Plattformen mit, aber nur mit
  CPU-Unterstützung. Für Meilenstein 1 reicht das. Ist das falsch, nutzt der Adapter eine installierte
  COLMAP-Binärdatei.
- **Brush als Standard-Backend.** Annahme: Brush hat eine Kommandozeile, die einen COLMAP-Datensatz trainiert und
  `.ply` schreibt, und es gibt fertige Binärdateien für alle drei Plattformen. Ob und wie Brush auf reiner CPU
  läuft (z. B. über einen Software-Vulkan-Treiber), ist **ungeklärt und das größte Risiko** für das
  Akzeptanzkriterium „auch nur mit CPU“. Das wird als Erstes geprüft; Ergebnis kommt als Entscheidung nach
  `docs/decisions/`.
- **FFmpeg** wird in Meilenstein 1 als installiert vorausgesetzt (Adapter meldet klar, wenn es fehlt).
  Das Mitliefern eines LGPL-Builds kommt mit dem Packaging in Meilenstein 5.
- **Testdaten** werden in Meilenstein 1 synthetisch erzeugt (gerenderte Szene mit bekannter Kamerafahrt), damit
  keine Lizenzfragen bei Videos entstehen. Ein echter Smartphone-Clip für die Abnahme wird per Skript geladen.
- **Fortsetzen** auf Stufenebene (`stage.done`) ist ab Meilenstein 1 dabei; Fortsetzen mitten im Training folgt in
  Meilenstein 4.

Alle Versionsangaben und Lizenzen werden vor dem Einbinden in der offiziellen Dokumentation geprüft und in
`THIRD_PARTY_LICENSES.md` festgehalten.

## Offene Fragen

1. **Lizenz von SplatForge selbst:** MIT, Apache 2.0 oder etwas anderes? (Beeinflusst, welche Abhängigkeiten
   möglich sind.)
2. **CPU-Training:** Falls Brush auf reiner CPU nicht oder nur unbrauchbar langsam läuft – ist ein eigenes, einfaches
   CPU-Trainings-Backend (z. B. auf Basis von PyTorch-CPU) als Ausweg in Ordnung, auch wenn es deutlich langsamer ist?
3. **Paketmanager:** Ist `uv` für Python und `pnpm` für die TypeScript-Pakete in Ordnung?
4. **Name:** Bleibt „SplatForge“ als Repository- und Paketname, oder ist das nur der Arbeitstitel?
