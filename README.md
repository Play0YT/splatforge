# SplatForge

Wandelt Videodateien – auch 360°-Aufnahmen und Insta360-Dateien (`.insv`) – in 3D Gaussian Splats (`.ply`, `.spz`) um.
Menschen im Bild werden automatisch maskiert. Läuft auch ohne starke GPU, notfalls rein auf der CPU.

Geplant sind zwei eigenständige Anwendungen mit gemeinsamem Verarbeitungskern:

- **SplatForge Desktop** – GUI-App für macOS, Windows und Ubuntu, rechnet lokal.
- **SplatForge Server** – Web-App mit Upload im Browser und GPU-Workern.

## Status

Planungsphase. Es gibt noch keinen lauffähigen Code.

- [`SPEC.md`](SPEC.md) – vollständige Spezifikation (verbindliche Grundlage)
- [`docs/plan-milestone-1.md`](docs/plan-milestone-1.md) – Plan, Annahmen und offene Fragen für Meilenstein 1
- [`CLAUDE.md`](CLAUDE.md) – Arbeitsregeln für Coding-Agents

## Meilensteine

1. Core-Grundgerüst (CLI, Analyse, Frames, COLMAP, Training, `.ply`-Export)
2. Personenmaskierung
3. 360° und INSV
4. Robustheit (Checkpoints, Fortsetzen, OOM-Behandlung)
5. Desktop-App
6. Server und Web
7. Feinschliff

Details und Akzeptanzkriterien stehen in [`SPEC.md`](SPEC.md#meilensteine-und-akzeptanzkriterien).
