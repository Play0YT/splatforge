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
| onnxruntime | 1.30 | MIT | Personenmaskierung (Erkennung und Segmentierung) |
| protobuf, flatbuffers, packaging (Abhängigkeiten von onnxruntime) | – | BSD-3-Clause, Apache-2.0, Apache-2.0/BSD-2-Clause | – |
| torch (optional, Zusatz `cpu-train`) | 2.x | BSD-3-Clause-Stil (Metadaten: Apache-2.0, BSD-2/3-Clause, BSL-1.0, MIT) | CPU-Trainingsbackend |

Entwicklungswerkzeuge (nicht im Produkt): pytest (MIT), ruff (MIT), mypy (MIT), jsonschema (MIT).

## KI-Modelle (beim ersten Gebrauch heruntergeladen, nicht im Paket)

| Modell | Quelle | Lizenz | Verwendung |
| --- | --- | --- | --- |
| RT-DETR R18 (COCO + Objects365) | PekingU/rtdetr_r18vd_coco_o365, ONNX: onnx-community/rtdetr_r18vd_coco_o365 | Apache-2.0 | Personen-/Objekterkennung |
| SAM 2.1 Hiera Tiny und Small | facebook/sam2.1-hiera-*, ONNX: onnx-community/sam2.1-hiera-*-ONNX | Apache-2.0 | Segmentierung |

Geprüft am 2026-10-02 anhand der Modellkarten. Revisionen und Prüfsummen stehen in `splatforge/models.py`.

## Externe Programme

| Programm | Lizenz | Hinweis |
| --- | --- | --- |
| FFmpeg / ffprobe | LGPL-2.1+ (nur LGPL-Build ohne `--enable-gpl` mitliefern) | Wird ab Meilenstein 5 als LGPL-Build mitgeliefert. Bis dahin wird die installierte Version verwendet. |
| Brush | Apache-2.0 | Trainings-Backend, als Subprozess |

## Übernommenes Wissen (kein Code eingebunden)

| Quelle | Lizenz | Verwendung |
| --- | --- | --- |
| telemetry-parser (AdrianEddy) | MIT oder Apache-2.0 | Beschreibung des Insta360-Metadatenformats (.insv-Trailer) und des Aufbaus der Objektiv-Kalibrierung (`offset_v3`). SplatForge enthält eine eigene Python-Implementierung. |
| COLMAP (Beispiel `panorama_sfm.py`) | BSD-3-Clause | Vorgehen für 360°: Perspektiv-Ansichten als Kamera-Rig. Eigene Umsetzung. |
| nerfstudio (`auto_orient_and_center_poses`) | Apache-2.0 | Idee für die Richtung „oben“ aus den Kameraachsen. Eigene Umsetzung. |

## Bewusst nicht verwendet

| Komponente | Lizenz | Grund / Ersatz |
| --- | --- | --- |
| plyfile | GPL-3.0 | Eigene PLY-Funktionen in `splatforge/ply.py` |
| OpenSplat | AGPL-3.0 | Nur als optionales Plugin denkbar |
| Ultralytics YOLO | AGPL-3.0 | Ersetzt durch RT-DETR (Apache-2.0), siehe docs/decisions/0004-maskierung.md |
