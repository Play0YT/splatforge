# Lizenzen von Drittkomponenten

Geprüft am 2026-10-01 anhand der Paket-Metadaten (PyPI) bzw. der LICENSE-Datei im Quell-Repository.
Regel: Standard-Build nur mit MIT, BSD, Apache 2.0 oder vergleichbar permissiven Lizenzen.
GPL/AGPL nur als optionales, separat installierbares Plugin.

## Python-Abhängigkeiten des Cores

| Komponente | Version | Lizenz | Verwendung |
| --- | --- | --- | --- |
| pydantic | 2.13 | MIT | Konfiguration und Schemas |
| numpy | 2.x | BSD-3-Clause (plus 0BSD, MIT, Zlib, CC0 für Teile) | Numerik |
| opencv-python-headless | 5.0 | Apache-2.0 | Bildverarbeitung, Frame-Auswahl |
| pycolmap | 4.2 | BSD-3-Clause | Kamerapositionen (COLMAP inkl. GLOMAP). Die Wheels enthalten weitere Bibliotheken (u. a. Ceres, Eigen, SQLite); deren Lizenzen liegen den Wheels bei und werden vor dem ersten Release einzeln aufgelistet. |
| torch (optional, Zusatz `cpu-train`) | 2.x | BSD-3-Clause-Stil (Metadaten: Apache-2.0, BSD-2/3-Clause, BSL-1.0, MIT) | CPU-Trainingsbackend |

Entwicklungswerkzeuge (nicht im Produkt): pytest (MIT), ruff (MIT), mypy (MIT), jsonschema (MIT).

## Externe Programme

| Programm | Lizenz | Hinweis |
| --- | --- | --- |
| FFmpeg / ffprobe | LGPL-2.1+ (nur LGPL-Build ohne `--enable-gpl` mitliefern) | Wird ab Meilenstein 5 als LGPL-Build mitgeliefert. Bis dahin wird die installierte Version verwendet. |
| Brush | Apache-2.0 | Trainings-Backend, als Subprozess |

## Übernommenes Wissen (kein Code eingebunden)

| Quelle | Lizenz | Verwendung |
| --- | --- | --- |
| telemetry-parser (AdrianEddy) | MIT oder Apache-2.0 | Beschreibung des Insta360-Metadatenformats (.insv-Trailer). SplatForge enthält eine eigene Python-Implementierung. |

## Bewusst nicht verwendet

| Komponente | Lizenz | Grund / Ersatz |
| --- | --- | --- |
| plyfile | GPL-3.0 | Eigene PLY-Funktionen in `splatforge/ply.py` |
| OpenSplat | AGPL-3.0 | Nur als optionales Plugin denkbar |
| Ultralytics YOLO | AGPL-3.0 | Für Meilenstein 2 wird ein Detektor mit permissiver Lizenz gewählt |
