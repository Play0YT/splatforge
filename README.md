# SplatForge

Wandelt Videodateien in 3D Gaussian Splats (`.ply`) um. Läuft auch ohne starke GPU, notfalls rein auf der CPU.

Geplant sind zwei eigenständige Anwendungen mit gemeinsamem Verarbeitungskern:

- **SplatForge Desktop:** GUI-App für macOS, Windows und Ubuntu, rechnet lokal.
- **SplatForge Server:** Web-App mit Upload im Browser und GPU-Workern.

## Stand

| Teil | Version | Stand |
| --- | --- | --- |
| [`packages/core`](packages/core) | 0.1.2 | Kommandozeile: Video → Kamerapositionen → Splat (`.ply`) |
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
- **Das CPU-Training ist sehr langsam.** Gemessen auf einem Windows-11-PC mit 4 CPU-Threads: 1 000 Iterationen
  mit einem Stock-Video dauerten 3,7 Stunden (gut 13 Sekunden pro Iteration). Die Vorschau-Stufe mit
  7 000 Iterationen bräuchte dort über einen Tag. Für einen ersten Test ohne Grafikkarte also wenig
  Iterationen wählen: `uv run splatforge run <video> --frames 60 --iterations 1000 --out <ordner>`.
  Mit Grafikkarte über Brush (siehe unten) geht es deutlich schneller.
- Mit **Strg+C** brichst du ab. Mit `uv run splatforge resume <ordner>` geht es später dort weiter.
- Das Ergebnis liegt in `<ordner>/08_export/splat.ply`, ein Bericht in `<ordner>/08_export/report.json`.
  Ansehen kannst du die `.ply`-Datei z. B. per Drag-and-drop im Browser in
  [SuperSplat](https://superspl.at/editor) oder im Viewer von Brush.
- Bei einem Fehler steht in der letzten Zeile, was zu tun ist. Das vollständige Log liegt in
  `<ordner>/events.jsonl`, das COLMAP-Log in `<ordner>/06_sfm/colmap.log`.

**Gutes Testvideo:** 20 bis 60 Sekunden, langsam um einen Gegenstand mit viel Struktur herumgehen
(Pflanze, Schuh, Sofa), nicht nur schwenken. Mehr Tipps in [`docs/capture-guide.md`](docs/capture-guide.md).

## Training mit Grafikkarte (Brush)

[Brush](https://github.com/ArthurBrussee/brush) trainiert auf der Grafikkarte, auch auf AMD- und
Intel-Grafik (über DirectX 12, Vulkan oder Metal). Es gehört nicht zu SplatForge und wird separat
heruntergeladen. Unterstützt wird Brush 0.3 oder neuer.

1. Auf https://github.com/ArthurBrussee/brush/releases bei der neuesten Version unter „Assets“ das Archiv
   für dein System herunterladen:
   - Windows: Name endet auf `x86_64-pc-windows-msvc.zip`
   - Linux: Name endet auf `x86_64-unknown-linux-gnu.tar.xz`
   - macOS (Apple Silicon): Name endet auf `aarch64-apple-darwin.tar.xz`
2. Entpacken, z. B. nach `C:\Tools\brush` (Windows) oder `~/tools/brush` (Linux/macOS). Darin liegt die
   Programmdatei, bei Brush 0.3 heisst sie `brush_app.exe` (Windows) bzw. `brush_app`.
   Unter Linux/macOS einmal `chmod +x ~/tools/brush/brush_app` ausführen.
3. Prüfen, ob SplatForge Brush erkennt. Der Pfad muss auf die Datei zeigen, nicht auf den Ordner:
   ```
   uv run splatforge hardware --brush C:\Tools\brush\brush_app.exe
   ```
   Erwartet: `"brush": true` und eine `brush_version`. Sonst steht der Grund unter `brush_problem`.
4. Job mit Brush starten:
   ```
   uv run splatforge run <video> --preset preview --backend brush --brush C:\Tools\brush\brush_app.exe --out <neuer-ordner>
   ```

Ohne `--backend brush` (also mit `auto`) nimmt SplatForge Brush, wenn es gefunden wird, und fällt sonst auf
die CPU zurück. Liegt Brush im Suchpfad (`PATH`), kann `--brush` entfallen. Hat der Rechner mehrere
Grafikkarten, nimmt Brush automatisch die leistungsstärkste.

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
