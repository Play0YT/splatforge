# Changelog – splatforge (Core)

Format nach [Keep a Changelog](https://keepachangelog.com/de/1.1.0/), Versionen nach [SemVer](https://semver.org/lang/de/).

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
