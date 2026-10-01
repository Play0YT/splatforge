# SplatForge

Wandelt Videodateien in 3D Gaussian Splats (`.ply`) um. Läuft auch ohne starke GPU, notfalls rein auf der CPU.

Geplant sind zwei eigenständige Anwendungen mit gemeinsamem Verarbeitungskern:

- **SplatForge Desktop:** GUI-App für macOS, Windows und Ubuntu, rechnet lokal.
- **SplatForge Server:** Web-App mit Upload im Browser und GPU-Workern.

## Stand

| Teil | Version | Stand |
| --- | --- | --- |
| [`packages/core`](packages/core) | 0.1.1 | Kommandozeile: Video → Kamerapositionen → Splat (`.ply`) |
| [`packages/job-schema`](packages/job-schema) | 0.1.0 | JSON-Schemas für Job-Konfiguration und Events |
| `packages/ui-components`, `apps/*` | – | noch nicht begonnen |

Noch nicht enthalten: Personenmaskierung, 360°/Insta360, Desktop-App, Server. Änderungen pro Version stehen
im `CHANGELOG.md` des jeweiligen Pakets.

## Lokal ausprobieren

Es gibt noch keine Installationsdatei. Die Kommandozeile läuft aus dem Quellcode. Python musst du nicht selbst
installieren, das erledigt [uv](https://docs.astral.sh/uv/) beim ersten Start (Python 3.11).

Solange das Repository privat ist, fragt `git clone` nach deiner GitHub-Anmeldung.

### Windows 10/11 (PowerShell)

```powershell
# 1. Werkzeuge installieren (einmalig), danach PowerShell schliessen und neu öffnen
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id Gyan.FFmpeg -e

# 2. SplatForge holen und einrichten
git clone https://github.com/Play0YT/splatforge.git
cd splatforge
uv sync --all-packages
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Prüfen und starten
uv run splatforge hardware
uv run splatforge run "C:\Users\<name>\Videos\mein_video.mp4" --preset preview --out "$HOME\splatforge-test"
```

### macOS (Apple Silicon, M1 oder neuer)

Intel-Macs werden noch nicht unterstützt (siehe [docs/decisions/0001](docs/decisions/0001-colmap-ueber-pycolmap.md)).

```bash
# 1. Werkzeuge installieren (einmalig). Ohne Homebrew zuerst: https://brew.sh
brew install git uv ffmpeg

# 2. SplatForge holen und einrichten
git clone https://github.com/Play0YT/splatforge.git
cd splatforge
uv sync --all-packages
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Prüfen und starten
uv run splatforge hardware
uv run splatforge run ~/Movies/mein_video.mp4 --preset preview --out ~/splatforge-test
```

### Ubuntu 22.04 / 24.04

```bash
# 1. Werkzeuge installieren (einmalig)
sudo apt update && sudo apt install -y git ffmpeg curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# 2. SplatForge holen und einrichten
git clone https://github.com/Play0YT/splatforge.git
cd splatforge
uv sync --all-packages
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Prüfen und starten
uv run splatforge hardware
uv run splatforge run ~/Videos/mein_video.mp4 --preset preview --out ~/splatforge-test
```

### Was passiert, und wie lange dauert es?

- `splatforge hardware` zeigt, was erkannt wurde. Steht bei `brush` `false` und bei `torch` `true`, wird auf
  der CPU trainiert. Das funktioniert überall, ist aber langsam.
- Die Vorschau-Stufe trainiert 7 000 Iterationen, auf der CPU kann das mehrere Stunden dauern.
  Für einen ersten Test reicht deutlich weniger:
  `uv run splatforge run <video> --frames 60 --iterations 1000 --out <ordner>`
- Mit **Strg+C** brichst du ab. Mit `uv run splatforge resume <ordner>` geht es später dort weiter.
- Das Ergebnis liegt in `<ordner>/08_export/splat.ply`, ein Bericht in `<ordner>/08_export/report.json`.
  Ansehen kannst du die `.ply`-Datei z. B. per Drag-and-drop im Browser in
  [SuperSplat](https://superspl.at/editor) oder im Viewer von Brush.
- Bei einem Fehler steht in der letzten Zeile, was zu tun ist. Das vollständige Log liegt in
  `<ordner>/events.jsonl`, das COLMAP-Log in `<ordner>/06_sfm/colmap.log`.

**Gutes Testvideo:** 20 bis 60 Sekunden, langsam um einen Gegenstand mit viel Struktur herumgehen
(Pflanze, Schuh, Sofa), nicht nur schwenken. Mehr Tipps in [`docs/capture-guide.md`](docs/capture-guide.md).

**Schneller mit Grafikkarte:** [Brush](https://github.com/ArthurBrussee/brush/releases) herunterladen und die
Datei `brush` (Windows: `brush.exe`) in einen Ordner im Suchpfad legen. `splatforge hardware` zeigt dann
`"brush": true`.

**Aktualisieren:** im Ordner `splatforge` die Befehle `git pull` und `uv sync --all-packages` ausführen und
danach den `torch`-Befehl von oben wiederholen (`uv sync` entfernt PyTorch, weil es nicht fest eingetragen ist).

**Selbsttest ohne eigenes Video:** `cd packages/core` und dann `uv run pytest -m integration`. Das erzeugt ein
kleines Testvideo und rechnet es einmal komplett durch (wenige Minuten).

## Dokumentation

- [`docs/architecture.md`](docs/architecture.md): Aufbau für Entwickler
- [`docs/capture-guide.md`](docs/capture-guide.md): Tipps für gute Aufnahmen
- [`docs/decisions/`](docs/decisions): Architekturentscheidungen
- [`docs/measurements.md`](docs/measurements.md): Messwerte
- [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md): Lizenzen der Abhängigkeiten
