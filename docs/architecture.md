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
├─ 03_select/          images/ (ausgewählte Frames), images_lens2/ (Dual-Fisheye), selection.json
├─ 04_360/            nur bei 360°: images/<ansicht>/, masks/<ansicht>/ (Nadir, Bildkreis), rig.json
├─ 05_mask/            nur mit --masking: masks/<bild>.png (COLMAP-Format), overlays/, mask.json
├─ 06_sfm/             database.db, sparse_*/, dataset/ (entzerrt: images/, sparse/, masks/), colmap.log
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
| 4 | 360°-Aufbereitung | `stages/pano.py`, `panorama.py` | equirektangulär und Dual-Fisheye |
| 5 | Personenmaskierung | `stages/mask.py`, `masking/` | ONNX-Bildmodell; Video-Verfolgung folgt |
| 6 | Kamerapositionen | `stages/sfm.py` | fertig |
| 7 | Training | `stages/train.py`, `training/cpu.py` | Brush + CPU |
| 8 | Export | `stages/export.py` | `.ply` + Report; Floater/`.spz` in Meilenstein 7 |

## 360°

`panorama.py` enthält die Projektionsmathematik (Ansichten, equirektangulär, Unified-Fisheye-Modell mit
Umkehrung, Nadir-Maske, Masken-Abgleich über ein gemeinsames Kugelpanorama). Stufe 4 rendert pro
Zeitpunkt die Ansichten in je einen Ordner pro Ansicht; COLMAP legt dafür eine Kamera pro Ordner an
(`PINHOLE`, feste Werte) und fasst sie per `apply_rig_config` zu einem Rig zusammen. Beim sequenziellen
Vergleich werden alle Ansichten benachbarter Zeitpunkte verglichen (`expand_rig_images`). Nach dem Entzerren
werden die Bildnamen flach gemacht (`v00/x.jpg` → `v00_x.jpg`), weil Brush 0.3 Masken in Unterordnern nicht
findet. Die Kalibrierung von Insta360 liest `insv.lens_calibrations` aus `offset_v3`. Ob die Stufe läuft,
entscheidet die Pipeline erst nach der Analyse (`Stage.applies` wird vor jeder Stufe neu ausgewertet).

## Ausrichtung

Nach dem Mapping dreht `stages/sfm.py` das Modell so, dass oben −Y entspricht, und legt die Szenenmitte in den
Ursprung (`orient.py`). „Oben“ ist die Richtung senkrecht zu den rechten Achsen aller Kameras; das hält auch
bei nach unten geneigter oder wackelnder Kamera. Die Export-Stufe vermerkt das in der PLY
(`comment vertical axis: y`), sonst schätzt Brush die Richtung selbst aus der Kamerabahn. `orient.py` misst
ausserdem, wie weit sich die Kamera im Verhältnis zur Szene bewegt hat (`baseline_ratio`).

## Hintergrundprozesse

Bibliotheken mit eigener Laufzeit (PyTorch, onnxruntime) laufen nie im selben Prozess wie pycolmap: Unter
macOS bringen sie unverträgliche OpenMP-Versionen mit. Das CPU-Training (`splatforge _train-cpu`) und die
Maskierung (`splatforge _mask`) starten deshalb als eigener Prozess (`worker.py`, `stages/_worker.py`). Sie
lesen einen Auftrag als JSON-Datei und melden Fortschritt, Warnungen, Ergebnis oder Fehler als JSON-Zeilen.

## Maskierung

`masking/` erkennt Objekte mit RT-DETR, verbindet die Boxen über die Bilder (`tracking.py`), füllt kurze
Lücken per Interpolation und erzeugt mit SAM 2.1 pixelgenaue Masken (`onnx_models.py`). Modelle verwaltet
`models.py` (gepinnte Hugging-Face-Revision, SHA-256-Prüfung, Cache-Ordner). Die Masken gehen an COLMAP
(`mask_path`) und werden nach dem Mapping mit derselben COLMAP-Entzerrung wie die Bilder entzerrt
(`stages/sfm.py`, `undistort_masks`), damit sie pixelgenau passen. Im Datensatz heissen sie
`masks/<bildname ohne Endung>.png`, wie Brush sie sucht.

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
