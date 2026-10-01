# SplatForge – Projektspezifikation

## Rolle und Ziel

Du bist ein erfahrener Software-Architekt und Entwickler mit Schwerpunkt Computer Vision, 3D-Rekonstruktion und plattformübergreifende Anwendungen. Baue ein Produkt namens **SplatForge** (Arbeitstitel), das Videodateien in 3D Gaussian Splats umwandelt.

Das Produkt besteht aus zwei getrennten Anwendungen, die denselben Verarbeitungskern nutzen:

- **SplatForge Desktop:** eine GUI-App für macOS, Windows und Ubuntu Linux, die alles lokal auf dem Rechner berechnet.
- **SplatForge Server:** eine Web-App, bei der Nutzer Footage im Browser hochladen und ein Server (mit GPU-Workern) den Splat berechnet.

Kernanforderungen:

- Eingabe: gängige Videoformate (mp4, mov, mkv, avi, webm, m4v), Insta360-Dateien (`.insv`), equirektanguläre 360°-Videos sowie optional Bildordner.
- Ausgabe: ein trainierter 3D Gaussian Splat als `.ply` und `.spz`, plus eine interaktive 3D-Vorschau.
- Menschen im Bild werden automatisch erkannt und maskiert, damit sie den Splat nicht verfälschen. Das ist besonders wichtig bei 360°-Aufnahmen, wo die filmende Person fast immer im Bild ist.
- Langsame Hardware ist ausdrücklich erlaubt. Die Software muss auch ohne starke GPU funktionieren, notfalls rein auf der CPU über Stunden oder Tage. Lange Laufzeiten müssen deshalb unterbrechbar und fortsetzbar sein.

## Projektstruktur

Lege ein Monorepo an. Server und Desktop sind eigenständige Apps mit eigenem Build, eigener Versionierung und eigenem Release. Sie teilen sich ausschließlich den Verarbeitungskern und optional UI-Komponenten. Keine App darf Code aus der anderen App importieren.

```
splatforge/
├─ packages/
│  ├─ core/            # Python: komplette Pipeline, kein UI, kein HTTP
│  ├─ ui-components/   # TypeScript: geteilte UI-Bausteine + Splat-Viewer
│  └─ job-schema/      # JSON-Schema für Job-Konfiguration und Fortschritts-Events
├─ apps/
│  ├─ desktop/         # Tauri-App (Rust-Shell + Web-Frontend), Core als Sidecar
│  ├─ server-api/      # FastAPI: Uploads, Jobs, Auth, Downloads
│  ├─ server-worker/   # GPU-Worker, führt den Core für Server-Jobs aus
│  └─ web/             # Browser-Frontend für die Server-Variante
├─ deploy/             # Docker Compose, Dockerfiles, Beispiel-Konfiguration
├─ docs/
└─ tests/              # Test-Videos (klein), Integrations- und E2E-Tests
```

Der Core ist eine Python-Bibliothek mit eigener Kommandozeile (`splatforge run input.mp4 --preset standard --out ./result`). Beide Apps rufen ihn über dieselbe Schnittstelle auf: eine Job-Konfiguration als JSON rein, Fortschritts-Events als JSON-Lines raus, Ergebnisdateien in ein Arbeitsverzeichnis. Dadurch ist der Core ohne UI testbar und beide Apps verhalten sich identisch.

## Technologie-Stack und Lizenzregeln

Verwende diesen Stack als Ausgangspunkt. Weiche nur ab, wenn du einen konkreten technischen Grund nennst und ihn in `docs/decisions/` als kurze Architekturentscheidung festhältst.

| Bereich | Wahl | Begründung |
| --- | --- | --- |
| Core-Sprache | Python 3.11+ | Bestes Ökosystem für CV, SfM-Bindings und ONNX |
| Video-Dekodierung | FFmpeg / ffprobe (LGPL-Build) | Liest praktisch jedes Format, auch Multi-Stream-Container |
| Bildverarbeitung | OpenCV, NumPy | Frame-Auswahl, Projektionen, Masken-Nachbearbeitung |
| Kamerapositionen (SfM) | COLMAP über pycolmap, optional GLOMAP | Robust, unterstützt Masken und Kamera-Rigs |
| Personenmaskierung | Personendetektor + SAM 2 über ONNX Runtime | Läuft auf CPU, CUDA, CoreML und DirectML |
| Splat-Training (Standard) | Brush (Rust, wgpu) als Subprozess | Läuft auf Metal, Vulkan, DX12 und ohne NVIDIA |
| Splat-Training (schnell) | gsplat, nur wenn CUDA vorhanden | Deutlich schneller auf NVIDIA-GPUs |
| Desktop-Shell | Tauri 2 + React + TypeScript | Kleine Binärdateien, Frontend teilbar mit Web |
| Server-API | FastAPI, PostgreSQL, Redis-Queue | Asynchron, gut dokumentiert, einfach zu betreiben |
| Uploads | tus-Protokoll (fortsetzbar) | Große 360°-Dateien über instabile Verbindungen |
| Speicher | S3-kompatibel (MinIO lokal) | Gleicher Code für Selbst-Hosting und Cloud |
| Splat-Viewer | Spark oder ein vergleichbarer WebGL-Viewer | Derselbe Viewer in Web und Desktop |

Lizenzregeln:

- Bevorzuge Abhängigkeiten unter MIT, BSD oder Apache 2.0. Prüfe jede Lizenz vor dem Einbinden selbst und dokumentiere sie in `THIRD_PARTY_LICENSES.md`.
- Keine AGPL- oder GPL-Abhängigkeiten im Standard-Build. Das betrifft zum Beispiel OpenSplat und die Ultralytics-YOLO-Modelle. Solche Komponenten sind nur als optionales, separat installierbares Plugin erlaubt und müssen in der UI als solche gekennzeichnet sein.
- Modellgewichte haben eigene Lizenzen. Prüfe sie getrennt vom Code und lade sie beim ersten Start mit Prüfsumme herunter, statt sie ins Repository zu legen.
- Das Insta360-SDK ist proprietär. Es darf nicht Voraussetzung sein, sondern höchstens ein optionaler Adapter.

## Verarbeitungspipeline

Die Pipeline besteht aus acht Stufen. Jede Stufe schreibt ihre Ergebnisse in ein eigenes Unterverzeichnis des Job-Ordners und eine `stage.done`-Markierung. Ist eine Stufe bereits fertig, wird sie beim Fortsetzen übersprungen.

1. **Analyse:** `ffprobe` liest Container, Streams, Auflösung, Framerate, Rotation, Dauer und Metadaten. Erkenne daraus den Kameratyp: normale Perspektive, equirektanguläres 360°, Dual-Fisheye (zwei Streams oder beide Linsen nebeneinander in einem Stream). Lehne nicht unterstützte Dateien mit einer verständlichen Fehlermeldung ab.
2. **Frame-Extraktion:** Extrahiere Frames mit FFmpeg, wahlweise mit fester Rate oder Zielanzahl (Standard 200 bis 400 Frames). Berücksichtige die Rotations-Metadaten von Smartphones. Unterstütze HDR- und 10-Bit-Material durch Tonemapping auf 8-Bit sRGB.
3. **Frame-Auswahl:** Verwirf unscharfe Frames (Laplace-Varianz relativ zum Clip-Median) und nahezu identische Frames (zu wenig Bewegung laut optischem Fluss). Ziel ist gleichmäßige Abdeckung der Kamerabewegung, nicht gleichmäßige Zeitabstände.
4. **360°-Aufbereitung:** Nur für 360°-Material, Details im Abschnitt „360°- und INSV-Verarbeitung“.
5. **Personenmaskierung:** Details im Abschnitt „Personenmaskierung“. Ergebnis ist pro Bild eine Maske im COLMAP-Format.
6. **Kamerapositionen:** COLMAP-Feature-Extraktion mit Masken, danach Matching (sequenziell plus Loop-Detection für Videos) und Mapping. Verwende GLOMAP als schnellere Alternative, wenn es verfügbar ist. Bei 360°-Material werden die Perspektiv-Ansichten eines Frames als starres Kamera-Rig modelliert. Schlägt die Rekonstruktion fehl oder werden weniger als 60 % der Bilder registriert, gib eine konkrete Diagnose aus (zu wenig Überlappung, zu wenig Textur, zu schnelle Bewegung).
7. **Splat-Training:** Trainiere mit dem gewählten Backend. Maskierte Pixel werden vom Loss ausgeschlossen, nicht nur schwarz eingefärbt. Unterstützt ein Backend keine Masken, erweitere es oder übergib die Masken als Alpha-Kanal, falls das Backend diesen als Maske auswertet. Speichere regelmäßig Checkpoints (Standard alle 2.000 Iterationen) und liefere Zwischenstände für die Live-Vorschau.
8. **Nachbearbeitung und Export:** Entferne Floater (Gaussians mit sehr geringer Opazität oder weit außerhalb der Szene), richte die Szene am Boden aus und exportiere `.ply` und `.spz`. Optional ein verkleinerter Web-Export. Lege ein `report.json` mit Laufzeiten, registrierten Bildern, Gaussian-Anzahl und Warnungen ab.

Jede Stufe ist eine eigene Klasse mit gemeinsamer Schnittstelle (`run`, `is_done`, `estimate_duration`, `cleanup`). Die Pipeline sendet Fortschritt als JSON-Lines-Events mit Stufe, Prozent, geschätzter Restzeit und Log-Zeile.

## Personenmaskierung

Ziel ist, dass Menschen weder die Kamerapositionen noch den Splat stören. Die Maskierung läuft vollautomatisch, kann aber in der GUI kontrolliert und korrigiert werden.

- **Erkennung:** Ein Personendetektor mit permissiver Lizenz (zum Beispiel RT-DETR oder YOLOX, als ONNX exportiert) findet Personen pro Frame. Die Boxen dienen als Prompts für SAM 2, das pixelgenaue Masken erzeugt.
- **Zeitliche Konsistenz:** Nutze die Video-Propagation von SAM 2, damit Masken über aufeinanderfolgende Frames stabil bleiben und auch teilweise verdeckte oder unscharfe Personen erfasst werden. Wird eine Person in einem Frame nicht erkannt, aber davor und danach schon, interpoliere.
- **Sicherheitsrand:** Erweitere jede Maske um einen einstellbaren Rand (Standard 1,5 % der Bildbreite), um Haare, Bewegungsunschärfe und Schatten am Körper abzudecken.
- **Zusatzklassen:** Optional zuschaltbar: Selfie-Stick und Stativ (Nadir-Bereich bei 360°), Fahrzeuge, Tiere. Bei 360°-Material standardmäßig den Nadir-Bereich unterhalb eines einstellbaren Winkels maskieren.
- **Format:** Speichere Masken als PNG im COLMAP-Format: gleicher Dateiname wie das Bild plus `.png`, schwarz bedeutet ignorieren. Dieselben Masken gehen ins Training.
- **Qualitätskontrolle:** Wenn in mehr als 40 % eines Bildes Personen zu sehen sind, markiere das Bild als ungeeignet und schließe es aus. Melde, wenn insgesamt zu wenig unmaskierte Fläche übrig bleibt.
- **Manuelle Korrektur:** Die Desktop-App und die Web-App zeigen eine Galerie der Masken als halbtransparente Overlays. Nutzer können pro Bild Masken mit Pinsel und Radierer korrigieren oder per Klick ein Objekt hinzufügen, das dann mit SAM 2 über den Clip verfolgt wird.
- **Hardware:** Die Maskierung muss auf CPU laufen. Nutze CUDA, CoreML oder DirectML über ONNX Runtime, wenn verfügbar. Biete ein kleineres Modell für schwache Rechner an.

## 360°- und INSV-Verarbeitung

360°-Material wird nicht direkt trainiert, sondern in normale Perspektiv-Ansichten zerlegt. Das ist robuster für SfM und für die meisten Trainings-Backends.

- **Equirektanguläre Videos** (zum Beispiel aus Insta360 Studio exportiert): Projiziere jeden ausgewählten Frame in mehrere Perspektiv-Ansichten. Standard: 8 Ansichten mit 90° Sichtfeld rund um den Horizont plus je eine nach oben und unten (deaktivierbar). Anzahl, Sichtfeld und Auflösung sind einstellbar.
- **Rig-Modell:** Alle Ansichten eines Frames teilen denselben Kameramittelpunkt. Modelliere sie in COLMAP als starres Rig mit bekannten relativen Rotationen. Das stabilisiert die Rekonstruktion stark.
- **Masken bei 360°:** Führe die Personenerkennung auf den Perspektiv-Ansichten aus, nicht auf dem verzerrten Panorama. Ergänze eine Konsistenzprüfung über die Ansichtsgrenzen hinweg, damit eine Person am Rand zweier Ansichten in beiden maskiert wird.
- **INSV-Dateien:** Prüfe mit ffprobe, ob die Datei zwei Video-Streams (je ein Objektiv) oder einen Stream mit beiden Fisheye-Bildern nebeneinander enthält. Unterstütze beide Varianten. Weg A (Standard): Behandle die beiden Objektive direkt als Fisheye-Kameras (COLMAP-Modell `OPENCV_FISHEYE`) in einem Rig mit zwei Kameras, die in entgegengesetzte Richtungen schauen. Weg B: Stitche selbst zu equirektangulär mit geschätzter Kalibrierung. Kennzeichne Weg B als experimentell.
- **Kalibrierung:** Lies Kalibrierdaten aus den INSV-Metadaten, falls vorhanden und auswertbar. Andernfalls nutze Standardwerte für bekannte Modelle (X3, X4, X5, ONE RS) und lass COLMAP die Intrinsics verfeinern. Dokumentiere klar, dass der Export als equirektanguläres MP4 aus Insta360 Studio die zuverlässigste Variante ist.
- **Gyro-Daten:** Optional kannst du die eingebetteten Gyroskop-Daten als Startwert für die Rotation nutzen. Das ist ein Ausbauschritt, kein Muss für die erste Version.

## Desktop-App

Die Desktop-App ist eine Tauri-2-Anwendung. Der Python-Core wird mit PyInstaller oder Nuitka zu einer eigenständigen Binärdatei pro Plattform gepackt und als Sidecar mitgeliefert. Nutzer müssen weder Python noch COLMAP noch FFmpeg selbst installieren.

Funktionen:

- **Projekt anlegen:** Datei per Drag-and-drop oder Dateidialog wählen. Mehrere Clips derselben Szene können zu einem Projekt kombiniert werden.
- **Vorprüfung:** Nach dem Import zeigt die App erkannte Eigenschaften (Format, 360° ja/nein, Dauer, Auflösung), die erkannte Hardware und eine grobe Zeitschätzung pro Qualitätsstufe.
- **Einstellungen:** Qualitätsstufe, Frame-Anzahl, Maskierung an/aus, Zusatzklassen, Trainings-Backend, Geräteauswahl (GPU/CPU). Einfache Ansicht mit drei Reglern, erweiterte Ansicht mit allen Parametern.
- **Masken-Review:** Galerie mit Overlay und Korrekturwerkzeugen wie im Abschnitt Personenmaskierung beschrieben. Optional: Pipeline pausiert nach der Maskierung, bis der Nutzer bestätigt.
- **Fortschritt:** Stufenanzeige, Prozent, Restzeit, Log-Ausgabe, Live-Vorschau des Splats während des Trainings.
- **Steuerung:** Pausieren, Fortsetzen, Abbrechen. Nach Absturz oder Neustart des Rechners lässt sich ein Projekt am letzten Checkpoint fortsetzen.
- **Ergebnis:** Eingebauter 3D-Viewer (Orbit- und Flugmodus), Export als `.ply` und `.spz`, Öffnen des Projektordners.
- **Ressourcen:** Einstellbare Grenze für CPU-Threads und GPU-Speicher, Option „nur rechnen, wenn der Rechner im Leerlauf ist“, Schutz vor Standby während eines Jobs.
- **Sprache und Design:** Deutsch und Englisch, helles und dunkles Theme.

Die Desktop-App hat keinerlei Abhängigkeit vom Server. Eine optionale Funktion „auf eigenem Server rechnen lassen“ ist ein späterer Ausbauschritt und spricht dann die öffentliche Server-API an wie jeder andere Client.

## Server-App und Web-Frontend

Die Server-Variante besteht aus drei getrennten Diensten: API, Worker und Web-Frontend. Sie werden per Docker Compose gestartet und lassen sich einzeln skalieren.

**Server-API (FastAPI):**

- Konten mit E-Mail und Passwort, Sitzungen per sicherem Cookie, zusätzlich API-Schlüssel für Skripte. Rollen: Nutzer und Admin.
- Fortsetzbare Uploads über tus, konfigurierbare Maximalgröße (Standard 20 GB) und Speicherkontingent pro Nutzer.
- Endpunkte zum Anlegen, Auflisten, Abbrechen und Löschen von Jobs, zum Abrufen von Masken für das Review, zum Hochladen korrigierter Masken und zum Herunterladen der Ergebnisse über zeitlich begrenzte signierte Links.
- Fortschritt live über Server-Sent Events oder WebSocket, gespeist aus den JSON-Lines-Events des Cores.
- Vollständige OpenAPI-Dokumentation. Die API ist versioniert (`/api/v1/`).

**Worker:**

- Holt Jobs aus der Redis-Queue, lädt die Eingabe aus dem Objektspeicher, führt den Core aus und lädt Ergebnisse und Checkpoints zurück.
- Docker-Images für NVIDIA (CUDA, NVIDIA Container Toolkit) und für reine CPU-Worker. Ein Worker bearbeitet standardmäßig einen Job gleichzeitig pro GPU.
- Bricht ein Worker ab, übernimmt ein anderer den Job ab dem letzten hochgeladenen Checkpoint.
- Zeitlimits pro Stufe und pro Job, konfigurierbar.

**Web-Frontend (React, TypeScript, Vite):**

- Upload per Drag-and-drop mit Fortschrittsbalken und automatischer Fortsetzung bei Verbindungsabbruch.
- Dieselben Einstellungen, dasselbe Masken-Review und derselbe Viewer wie in der Desktop-App, über `packages/ui-components`.
- Job-Liste mit Status, Warteschlangenposition, Restzeit und E-Mail-Benachrichtigung bei Fertigstellung.
- Optional ein öffentlicher Teilen-Link für den Viewer eines fertigen Splats.

**Sicherheit und Betrieb:**

- Prüfe hochgeladene Dateien per ffprobe in einem isolierten Prozess mit Zeitlimit. Baue niemals Shell-Befehle aus Nutzereingaben zusammen.
- Rate-Limits auf Login und Uploads, CSRF-Schutz, sichere Header, HTTPS-Konfiguration für einen Reverse Proxy (Caddy oder Traefik) als Beispiel.
- Automatisches Löschen von Rohdaten und Zwischenständen nach einer einstellbaren Frist (Standard 7 Tage), Ergebnisse nach 30 Tagen. Nutzer können alles sofort löschen.
- Strukturiertes Logging, Prometheus-Metriken (Queue-Länge, Jobdauer, Fehlerrate), Health-Checks für alle Dienste.
- Admin-Ansicht: Nutzer, Kontingente, laufende Jobs, Worker-Status.

## Hardware, Qualitätsstufen und Robustheit

Die Software erkennt beim Start die verfügbare Hardware und wählt automatisch das schnellste funktionierende Backend. Reihenfolge: CUDA (NVIDIA), Metal (Apple Silicon), Vulkan oder DirectX 12 über wgpu (AMD, Intel, ältere NVIDIA), CPU. Der Nutzer kann die Wahl überschreiben.

| Qualitätsstufe | Frames | Max. Bildkante | Iterationen | Zweck |
| --- | --- | --- | --- | --- |
| Vorschau | 120 | 1.280 px | 7.000 | Schneller Test, ob die Aufnahme taugt |
| Standard | 300 | 1.600 px | 30.000 | Normaler Einsatz |
| Hoch | 600 | 2.560 px | 50.000 | Beste Qualität, sehr lange Laufzeit |

Die Zahlen sind Startwerte. Passe sie nach eigenen Tests an und dokumentiere die Messwerte.

Robustheit:

- Schätze vor dem Start den Speicherbedarf. Reicht der GPU- oder Arbeitsspeicher nicht, reduziere automatisch Auflösung oder maximale Gaussian-Anzahl und melde das, statt abzustürzen.
- Erkenne Out-of-Memory-Fehler zur Laufzeit, setze vom letzten Checkpoint mit kleineren Parametern fort.
- Ein Job überlebt Programmabsturz, Neustart und Stromausfall. Alle Zwischenstände liegen im Job-Ordner, der Fortsetzungspunkt wird aus den `stage.done`-Markierungen und Checkpoints bestimmt.
- Prüfe freien Speicherplatz vor jeder Stufe. Ein 10-minütiges 360°-Video kann mehrere zehn Gigabyte Zwischendaten erzeugen.
- Fehlermeldungen sind für Laien verständlich und nennen eine konkrete Maßnahme („Kamera langsamer bewegen“, „mehr Überlappung“, „weniger Frames wählen“). Technische Details stehen im Log.

## Tests, Packaging und CI

**Tests:**

- Unit-Tests für jede Pipeline-Stufe (pytest), inklusive Projektionsmathematik für 360° und Masken-Konvertierung.
- Ein kleines Test-Set im Repository: ein kurzer Perspektiv-Clip, ein kurzer equirektangulärer Clip mit einer Person im Bild, ein synthetisch erzeugter Dual-Fisheye-Clip. Größere Testdaten werden per Skript heruntergeladen.
- Integrationstest: kompletter Durchlauf in der Vorschau-Stufe auf CPU, Prüfung auf gültige `.ply`-Datei und Mindestanzahl registrierter Bilder.
- Qualitätsmessung: PSNR und SSIM auf zurückgehaltenen Testbildern, ausgegeben im `report.json`.
- API-Tests für den Server, E2E-Tests für Web und Desktop mit Playwright.

**Packaging:**

- macOS: signiertes und notarisiertes `.dmg`, Universal oder getrennt für Apple Silicon und Intel.
- Windows: signierter `.msi`- oder NSIS-Installer.
- Ubuntu: `.deb` und AppImage, getestet auf Ubuntu 22.04 und 24.04.
- Server: Docker-Images für API, Worker (CUDA und CPU) und Web, veröffentlicht in einer Container-Registry, plus `deploy/docker-compose.yml` mit Beispiel-`.env`.

**CI (GitHub Actions):**

- Linting und Typprüfung (ruff, mypy, eslint, tsc), Tests auf allen drei Betriebssystemen.
- Builds der Desktop-App für alle Plattformen und der Docker-Images bei jedem Release-Tag.
- Separate Workflows für Desktop und Server, damit beide unabhängig veröffentlicht werden können.

## Meilensteine und Akzeptanzkriterien

Arbeite die Meilensteine der Reihe nach ab. Ein Meilenstein ist erst fertig, wenn seine Kriterien erfüllt und die Tests grün sind. Fasse am Ende jedes Meilensteins zusammen, was funktioniert, was nicht und was du geändert hast.

1. **Core-Grundgerüst:** Monorepo, Core-CLI, Analyse, Frame-Extraktion, Frame-Auswahl, COLMAP, Training mit einem Backend, `.ply`-Export.
   - Kriterium: Ein normaler Smartphone-Clip ergibt auf macOS, Windows und Ubuntu per CLI einen betrachtbaren Splat, auch nur mit CPU.
2. **Personenmaskierung:** Detektor plus SAM 2, Masken in SfM und Training, Qualitätskontrolle.
   - Kriterium: Ein Clip mit durchs Bild laufender Person ergibt einen Splat ohne sichtbare Geister- oder Schlierenreste der Person.
3. **360° und INSV:** Equirektangulär-Zerlegung, Rig-Modell, Dual-Fisheye, Nadir-Maskierung.
   - Kriterium: Ein 360°-Clip, bei dem die filmende Person die Kamera am Stick trägt, ergibt einen sauberen Splat ohne die Person.
4. **Robustheit:** Checkpoints, Fortsetzen, Speicherprüfung, OOM-Behandlung, verständliche Fehler.
   - Kriterium: Ein Job, der mitten im Training hart beendet wird, läuft nach Neustart am letzten Checkpoint weiter.
5. **Desktop-App:** Tauri-Shell, Sidecar, alle Funktionen aus dem Abschnitt Desktop-App, Installer.
   - Kriterium: Ein Laie installiert die App ohne weitere Software und erzeugt per Drag-and-drop einen Splat.
6. **Server und Web:** API, Worker, Web-Frontend, Docker Compose, Sicherheit, Aufräumen.
   - Kriterium: `docker compose up` auf einem Ubuntu-Server mit NVIDIA-GPU liefert eine nutzbare Web-App, in der ein 5-GB-Upload nach Verbindungsabbruch fortgesetzt und erfolgreich verarbeitet wird.
7. **Feinschliff:** Floater-Entfernung, `.spz`-Export, Teilen-Links, Lokalisierung, Dokumentation für Nutzer und Betreiber.

## Arbeitsregeln

- Beginne mit einem kurzen Plan für Meilenstein 1 und den Annahmen, die du triffst. Stelle Rückfragen, wenn eine Entscheidung den Rest des Projekts stark beeinflusst, statt zu raten.
- Prüfe den aktuellen Stand jeder Bibliothek (Version, API, Lizenz, Plattformunterstützung) in der offiziellen Dokumentation, bevor du sie einsetzt. Verlass dich nicht auf dein Gedächtnis.
- Halte den Core frei von UI- und HTTP-Code. Desktop und Server sprechen ihn nur über die definierte Job-Schnittstelle an.
- Schreibe Code, der auf allen drei Betriebssystemen läuft: `pathlib` statt String-Pfade, keine Shell-Spezialitäten, Leerzeichen und Umlaute in Pfaden getestet.
- Jede externe Binärdatei (FFmpeg, COLMAP, Brush) wird über eine Adapter-Klasse angesprochen, mit Versionsprüfung und klarer Fehlermeldung, falls sie fehlt.
- Konfiguration über eine validierte Datei (Pydantic), keine magischen Zahlen im Code.
- Wenn etwas nicht wie spezifiziert umsetzbar ist, sag das offen, begründe es und schlage die beste Alternative vor.
- Dokumentiere laufend: `README.md` für Nutzer, `docs/architecture.md` für Entwickler, `docs/self-hosting.md` für Server-Betreiber, Tipps für gute Aufnahmen in `docs/capture-guide.md`.
