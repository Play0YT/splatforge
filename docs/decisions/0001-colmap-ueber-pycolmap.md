# 0001 – COLMAP und GLOMAP über pycolmap

**Kontext.** Die Spezifikation sieht COLMAP über pycolmap und optional GLOMAP vor. Nutzer sollen nichts
selbst installieren müssen.

**Entscheidung.** pycolmap ≥ 4.2 von PyPI. Seit COLMAP 4 ist GLOMAP als `pycolmap.global_mapping` enthalten.
Die Stufe versucht zuerst das globale Mapping und fällt auf das inkrementelle zurück, wenn es scheitert oder
zu wenige Bilder verortet.

**Begründung.** Ein einziges Paket statt zweier Binärdateien; gleiche Version auf allen Plattformen.

**Folgen.**
- Fertige pycolmap-Pakete gibt es für Linux x86_64, Windows x86_64 und macOS ab 14 auf Apple Silicon.
  **Intel-Macs und Linux auf ARM werden von diesen Paketen nicht abgedeckt.** Für die Desktop-App
  (Meilenstein 5) ist zu entscheiden, ob dafür COLMAP selbst gebaut wird oder diese Plattformen entfallen.
- Die PyPI-Pakete rechnen auf der CPU. GPU-Varianten (z. B. `pycolmap-cuda12`) können später für den
  Server-Worker eingesetzt werden.
- COLMAP schreibt sein Log direkt auf stderr. Der Adapter leitet es nach `06_sfm/colmap.log` um, damit die
  JSON-Events auf stdout sauber bleiben.
- Loop-Detection braucht einen Vocab-Tree. Er wird noch nicht automatisch heruntergeladen; ohne Datei
  ist Loop-Detection aus.
