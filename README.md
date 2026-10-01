# SplatForge

Wandelt Videodateien in 3D Gaussian Splats (`.ply`) um. Läuft auch ohne starke GPU, notfalls rein auf der CPU.

Geplant sind zwei eigenständige Anwendungen mit gemeinsamem Verarbeitungskern:

- **SplatForge Desktop:** GUI-App für macOS, Windows und Ubuntu, rechnet lokal.
- **SplatForge Server:** Web-App mit Upload im Browser und GPU-Workern.

## Stand

| Teil | Version | Stand |
| --- | --- | --- |
| [`packages/core`](packages/core) | 0.1.0 | Kommandozeile: Video → Kamerapositionen → Splat (`.ply`) |
| [`packages/job-schema`](packages/job-schema) | 0.1.0 | JSON-Schemas für Job-Konfiguration und Events |
| `packages/ui-components`, `apps/*` | – | noch nicht begonnen |

Noch nicht enthalten: Personenmaskierung, 360°/Insta360, Desktop-App, Server. Änderungen pro Version stehen
im `CHANGELOG.md` des jeweiligen Pakets.

## Schnellstart (Kommandozeile)

Voraussetzungen: Python 3.11+, [uv](https://docs.astral.sh/uv/), FFmpeg.

```bash
uv sync --all-packages
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # nur für Training ohne GPU
uv run splatforge run mein_video.mp4 --preset preview --out ./ergebnis
```

Das Ergebnis liegt danach in `ergebnis/08_export/splat.ply`. Ein abgebrochener Job läuft mit
`uv run splatforge resume ./ergebnis` weiter. Mehr dazu in [`packages/core/README.md`](packages/core/README.md).

Für schnelles Training auf der Grafikkarte [Brush](https://github.com/ArthurBrussee/brush/releases)
installieren; ohne Brush wird das deutlich langsamere CPU-Backend verwendet.

## Dokumentation

- [`docs/architecture.md`](docs/architecture.md): Aufbau für Entwickler
- [`docs/capture-guide.md`](docs/capture-guide.md): Tipps für gute Aufnahmen
- [`docs/decisions/`](docs/decisions): Architekturentscheidungen
- [`docs/measurements.md`](docs/measurements.md): Messwerte
- [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md): Lizenzen der Abhängigkeiten
