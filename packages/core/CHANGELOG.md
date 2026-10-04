# Changelog – splatforge (Core)

Format nach [Keep a Changelog](https://keepachangelog.com/de/1.1.0/), Versionen nach [SemVer](https://semver.org/lang/de/).

## [0.3.1] – 2026-10-04

### Behoben
- Splats lagen schief, bei Clips mit wackliger oder sich drehender Kamera teils um rund 45°. COLMAP legt das
  Koordinatensystem beliebig fest, und Brush schätzt die Richtung „oben“ aus der Kamerabahn, was bei solchen
  Clips scheitert. Die Szene wird jetzt nach dem Mapping waagrecht ausgerichtet (oben = −Y) und die
  Szenenmitte in den Ursprung gelegt. „Oben“ kommt aus den Kameraachsen und stimmt auch, wenn die Kamera
  nach unten zeigt oder wackelt. In `splat.ply` steht `comment vertical axis: y`, damit Brush diese Richtung
  übernimmt. Abschaltbar mit `sfm.orient_scene: false`.

### Hinzugefügt
- Warnung, wenn sich die Kamera kaum von der Stelle bewegt hat (nur gedreht oder geschwenkt). Dann fehlt die
  Tiefe, und der Splat wird unbrauchbar, obwohl COLMAP alle Bilder verortet. Kennzahl `baseline_ratio` in
  `sfm.json`/`report.json`, Schwelle `sfm.min_baseline_ratio` (Standard 0,1).
- `sfm.json` enthält `oriented` und `tilt_corrected_deg`.

### Hinweise
- Keine Migration nötig: Die neuen Einstellungen haben Standardwerte. Bereits fertige Jobs bleiben, wie sie
  sind; die Ausrichtung gilt für neue Jobs (oder nach Löschen von `06_sfm` und `splatforge resume`).

## [0.3.0] – 2026-10-02

### Hinzugefügt
- Personenmaskierung (Meilenstein 2), nur auf Wunsch: `splatforge run … --masking`. Neue Stufe 5
  (`05_mask`) zwischen Frame-Auswahl und COLMAP:
  - Personen erkennen mit RT-DETR R18, Lücken einzelner Bilder per Box-Verfolgung und Interpolation
    schliessen, pixelgenaue Maske mit SAM 2.1 (beides ONNX, Apache-2.0), Sicherheitsrand 1,5 % der
    Bildbreite.
  - Qualitätskontrolle: Bilder mit mehr als 40 % maskierter Fläche werden nicht verwendet; Warnung, wenn im
    Mittel zu wenig unmaskierte Fläche bleibt.
  - Kontrollbilder mit rot markierten Bereichen in `05_mask/overlays`.
  - Läuft in einem eigenen Hintergrundprozess; nutzt CUDA, DirectML oder CoreML, sonst die CPU.
- Optionen `--mask-model` (`auto`, `sam2.1-tiny`, `sam2.1-small`; auto: Small mit NVIDIA-GPU, sonst Tiny),
  `--mask-method` (`auto`, `image`, `video`) und `--mask-classes` (`person`, `vehicle`, `animal`).
  Weitere Einstellungen im neuen Abschnitt `mask` der Job-Konfiguration.
- Befehl `splatforge models`: zeigt die Modelle an und lädt sie mit `--download` vorab herunter (z. B. für
  Rechner ohne Internet). Modelle liegen im Cache-Ordner (Windows `%LOCALAPPDATA%\SplatForge\models`,
  macOS `~/Library/Caches/SplatForge/models`, Linux `~/.cache/splatforge/models`) oder unter
  `SPLATFORGE_MODELS_DIR`. Jede Datei ist auf eine feste Version gepinnt und wird per SHA-256 geprüft.
- `report.json` enthält einen Abschnitt `masking`.

### Geändert
- COLMAP nutzt die Masken bei der Merkmalssuche; ausgeschlossene Bilder werden gar nicht verwendet.
- Die Masken werden mit derselben COLMAP-Entzerrung wie die Bilder entzerrt und liegen danach als
  `06_sfm/dataset/masks/<bildname ohne Endung>.png` im Datensatz (schwarz = ignorieren). Brush (ab 0.3) und
  das CPU-Backend blenden diese Bereiche beim Training aus.
- Das CPU-Backend liest Masken als `<name ohne Endung>.png` und weiterhin als `<name>.png`.
- Neue Abhängigkeit `onnxruntime` (MIT).

### Hinweise
- Die Video-Verfolgung mit SAM 2 (PyTorch) für NVIDIA-Grafikkarten folgt in einer späteren Version;
  `--mask-method video` verwendet bis dahin das Bildmodell und meldet das.
- Keine Migration nötig: Die neuen Felder haben Standardwerte, `masking` bleibt standardmässig aus.
  Bestehende Jobs laufen unverändert weiter.

## [0.2.2] – 2026-10-02

### Geändert
- Bei einer Datei pro Objektiv (`…_00_…`/`…_10_…`) werden fehlende Insta360-Metadaten aus der Partnerdatei
  gelesen. Bei der Insta360 ONE RS stehen sie nur in der `_00_`-Datei.
- Die Objektive haben eine feste Reihenfolge (`_00_` = Objektiv 1, `_10_` = Objektiv 2), egal welche der
  beiden Dateien angegeben wird.

## [0.2.1] – 2026-10-02

### Hinzugefügt
- Vorschaudateien von Insta360 (`LRV_….insv`, niedrige Auflösung) werden erkannt. `splatforge analyze`
  weist darauf hin, dass sie für einen Splat ungeeignet sind, und nennt die Originaldateien (`VID_…`),
  falls sie im selben Ordner liegen.

## [0.2.0] – 2026-10-02

### Hinzugefügt
- Insta360-Dateien (`.insv`) lesen: Kameramodell, Seriennummer, Firmware, Auflösung und
  Objektiv-Kalibrierung aus dem Insta360-Metadatenblock am Dateiende (eigener Leser, keine neue
  Abhängigkeit; Format nach telemetry-parser, MIT/Apache-2.0).
- Erkennung, wo die beiden Objektive liegen: zwei Videospuren, nebeneinander in einer Spur oder eine
  Datei pro Objektiv (`…_00_…` und `…_10_…`, die zweite Datei wird im selben Ordner gesucht).
- Befehl `splatforge frames <video> --out <ordner>`: exportiert Einzelbilder, bei 360°/Insta360 getrennt
  pro Objektiv (`objektiv_1`, `objektiv_2`). Schreibt nur in leere oder neue Ordner.
- `splatforge analyze` zeigt diese Angaben auch für 360°-Dateien an (bisher brach es dort ab).

### Geändert
- `splatforge run` mit einer 360°-Datei bricht weiterhin ab, nennt aber jetzt `splatforge frames` als
  Möglichkeit, die Aufnahme schon anzusehen. Die Splat-Berechnung aus 360°-Material folgt in Meilenstein 3.
- `analysis.json` enthält neue Felder (`lens_layout`, `lenses`, `insv`). Ältere Dateien ohne diese Felder
  werden weiterhin gelesen, unbekannte Felder aus neueren Versionen ignoriert.
- Der Release-Workflow lässt sich unter „Actions“ auch manuell mit einer Versionsangabe starten und legt
  den Tag dann selbst an.

## [0.1.4] – 2026-10-02

### Behoben
- Ubuntu 22.04: Die Frame-Extraktion scheiterte mit FFmpeg 4.4 (`Unrecognized option 'fps_mode'`). Für
  FFmpeg vor 5.1 wird jetzt die ältere Option `-vsync` verwendet; unterstützt wird FFmpeg ab 4.4.
- macOS: Das Programm stürzte ab, sobald PyTorch und COLMAP im selben Prozess geladen wurden (beide bringen
  eine eigene OpenMP-Bibliothek mit). Das CPU-Training läuft jetzt in einem eigenen Prozess und liest das
  COLMAP-Modell selbst ein; PyTorch wird im Hauptprozess nie geladen.
- Fehlt FFmpeg oder ist es zu alt, bricht der Job sofort beim Start mit einer klaren Meldung ab.

### Geändert
- `python -m splatforge` startet die Kommandozeile.
- CI und Release-Workflow (`core-v<Version>`-Tag erstellt ein GitHub-Release).

## [0.1.3] – 2026-10-02

### Behoben
- Brush füllte die Ausgabe mit sehr langen Debug-Zeilen (eine pro geladenem Bild, mit der ganzen
  Dateiliste). SplatForge lässt von Brush jetzt nur noch die eigenen Fortschrittsmeldungen sowie Warnungen
  und Fehler durch und kürzt überlange Zeilen. Eine selbst gesetzte Umgebungsvariable `RUST_LOG` hat
  weiterhin Vorrang.

## [0.1.2] – 2026-10-02

### Behoben
- Brush 0.3 (aktuelle veröffentlichte Version) wurde mit einer falschen Option aufgerufen
  (`--total-train-iters` statt `--total-steps`). Die passende Option wird jetzt aus `brush --help` gelesen.
- Brush wird jetzt auch unter dem Dateinamen `brush_app` bzw. `brush_app.exe` (Brush 0.3) gefunden.
- `splatforge hardware` meldete Windows 11 als „Windows 10“.
- `splatforge hardware` meldete keine NVIDIA-Grafikkarte, sobald die CPU-Variante von PyTorch installiert war.

### Hinzugefügt
- Option `--brush <Pfad>` für `run`, `resume` und `hardware`. Bei `resume` gilt sie nur für diesen Lauf.
- `splatforge hardware` listet alle Grafikkarten (`gpus`), die Brush-Version und gegebenenfalls, warum Brush
  nicht nutzbar ist (`brush_problem`).
- Brush-Ausgaben von Grafik-Bibliotheken werden auf Warnungen beschränkt, damit das Log lesbar bleibt.

### Geändert
- In der Ausgabe von `splatforge hardware` heisst das Feld `cuda` jetzt `nvidia_gpu`.

## [0.1.1] – 2026-10-01

### Geändert
- Fehlt jedes Trainings-Backend (weder Brush noch PyTorch), bricht der Job jetzt sofort beim Start ab
  statt erst nach der COLMAP-Stufe. Neue Methode `Stage.preflight` für solche Vorabprüfungen.
- Die Fehlermeldung nennt den genauen Befehl zur Installation von PyTorch (CPU-Variante).

## [0.1.0] – 2026-10-01

Erste Version (Meilenstein 1: Core-Grundgerüst).

### Hinzugefügt
- Kommandozeile `splatforge` mit `run`, `resume`, `analyze`, `hardware` und `schema`
- Pipeline mit `stage.done`-Markierungen; abgebrochene Jobs lassen sich mit `splatforge resume` fortsetzen
- Fortschritts-Events als JSON-Lines (stdout und `events.jsonl` im Job-Ordner)
- Job-Konfiguration als validierte Pydantic-Modelle, Qualitätsstufen Vorschau/Standard/Hoch
- Migrationen für `job.json`; die alte Datei bleibt als `job.v<N>.json` erhalten
- Stufe Analyse: ffprobe, Rotation, HDR/10 Bit, Erkennung von 360°/Dual-Fisheye (wird vorerst abgelehnt)
- Stufe Frame-Extraktion: FFmpeg mit Rotation, Skalierung und HDR-Tonemapping; auch Bildordner
- Stufe Frame-Auswahl: Schärfe (Laplace-Varianz) und gleichmässige Abdeckung der Kamerabewegung (optischer Fluss)
- Stufe Kamerapositionen: COLMAP über pycolmap, globales Mapping (GLOMAP) mit Rückfall auf inkrementelles
  Mapping, verständliche Diagnose bei zu wenig verorteten Bildern
- Stufe Training: Brush als Standard, CPU-Backend (PyTorch) als Ausweg ohne nutzbare GPU, mit Checkpoints
- Stufe Export: `splat.ply` und `report.json` (Laufzeiten, verortete Bilder, Gaussians, PSNR/SSIM, Warnungen)
- Eigene PLY-Lese- und Schreibfunktionen (statt des GPL-lizenzierten `plyfile`)
- Prüfung des freien Speicherplatzes vor jeder Stufe

### Bekannte Einschränkungen
- Personenmaskierung (Meilenstein 2) und 360°/INSV (Meilenstein 3) fehlen noch
- Brush-Training wird nach einem Abbruch neu begonnen; nur das CPU-Backend setzt am Checkpoint fort
- Loop-Detection nur mit selbst angegebenem Vocab-Tree
