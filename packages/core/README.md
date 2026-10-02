# SplatForge Core

Python-Bibliothek und Kommandozeile, die aus einem Video einen 3D Gaussian Splat berechnet.
Kein UI- und kein HTTP-Code; Desktop-App und Server-Worker nutzen den Core über die Job-Schnittstelle.

## Installation (Entwicklung)

Voraussetzungen: Python 3.11+, [uv](https://docs.astral.sh/uv/), FFmpeg im Suchpfad.

```bash
uv sync --all-packages                 # Core mit Abhängigkeiten
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-Training
```

Für schnelles Training auf der GPU [Brush](https://github.com/ArthurBrussee/brush/releases) installieren und
in den Suchpfad legen (oder `tools.brush` in der Job-Konfiguration setzen).

## Verwendung

```bash
splatforge run video.mp4 --preset preview --out ./ergebnis
splatforge resume ./ergebnis           # nach Abbruch (Strg+C) oder Absturz fortsetzen
splatforge analyze video.mp4           # Eingabe prüfen
splatforge hardware                    # erkannte Hardware und Backends
splatforge frames aufnahme.insv --out ./bilder   # Einzelbilder, bei 360° pro Objektiv
```

Wichtige Optionen von `run`: `--frames`, `--max-edge`, `--iterations`, `--backend auto|brush|cpu`,
`--threads`, `--config job.json` (alle Einstellungen aus `config.py`), `--json` (Events immer als JSON-Lines).

Ergebnis: `<out>/08_export/splat.ply` und `<out>/08_export/report.json`.

Ausgabe: Im Terminal lesbare Zeilen, sonst (oder mit `--json`) ein JSON-Event pro Zeile. Exit-Codes:
0 fertig, 1 Fehler, 2 falscher Aufruf, 130 abgebrochen.

## Tests

```bash
cd packages/core
uv run pytest -m "not integration"     # schnell
uv run pytest -m integration           # kompletter Durchlauf auf CPU (ca. 2 min)
```

Die Testvideos werden synthetisch erzeugt (`tests/synthetic.py`), es liegen keine Videodateien im Repository.
