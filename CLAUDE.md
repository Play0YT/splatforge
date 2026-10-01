# Hinweise für Coding-Agents

Verbindliche Grundlage ist [`SPEC.md`](SPEC.md). Bei Widersprüchen gilt `SPEC.md`.

## Arbeitsweise

- Meilenstein für Meilenstein arbeiten, nicht alles auf einmal. Ein Meilenstein ist erst fertig, wenn seine
  Akzeptanzkriterien erfüllt und die Tests grün sind. Am Ende jedes Meilensteins zusammenfassen: was funktioniert,
  was nicht, was geändert wurde.
- Vor jedem Meilenstein einen kurzen Plan mit Annahmen schreiben (`docs/plan-milestone-<n>.md`).
- Rückfragen stellen, wenn eine Entscheidung den Rest des Projekts stark beeinflusst, statt zu raten.
- Wenn etwas nicht wie spezifiziert umsetzbar ist: offen sagen, begründen, beste Alternative vorschlagen.
- Abweichungen vom Stack in `SPEC.md` nur mit konkretem technischem Grund, festgehalten als kurze
  Architekturentscheidung in `docs/decisions/`.

## Code-Regeln

- Version, API, Lizenz und Plattformunterstützung jeder Bibliothek in der offiziellen Dokumentation prüfen,
  bevor sie eingesetzt wird. Nicht auf das Gedächtnis verlassen.
- Lizenzen: MIT, BSD oder Apache 2.0 bevorzugen; jede Lizenz selbst prüfen und in `THIRD_PARTY_LICENSES.md`
  eintragen. Keine GPL/AGPL im Standard-Build (nur als optionales, gekennzeichnetes Plugin).
  Modellgewichte nie ins Repository legen, sondern beim ersten Start mit Prüfsumme herunterladen.
- Der Core (`packages/core`) enthält keinen UI- und keinen HTTP-Code. Desktop und Server sprechen ihn nur über
  die Job-Schnittstelle an (Job-Konfiguration als JSON rein, JSON-Lines-Events raus, Dateien ins Arbeitsverzeichnis).
- Keine App importiert Code aus einer anderen App. Geteilt wird nur über `packages/`.
- Plattformübergreifend: `pathlib` statt String-Pfade, keine Shell-Spezialitäten, Pfade mit Leerzeichen und
  Umlauten testen. Niemals Shell-Befehle aus Nutzereingaben zusammenbauen (Subprozesse mit Argumentliste).
- Jede externe Binärdatei (FFmpeg, COLMAP, Brush, …) über eine Adapter-Klasse mit Versionsprüfung und klarer
  Fehlermeldung, falls sie fehlt.
- Konfiguration über eine validierte Datei (Pydantic), keine magischen Zahlen im Code.
- Fehlermeldungen für Laien verständlich und mit konkreter Maßnahme; technische Details ins Log.

## Dokumentation (laufend pflegen)

- `README.md` – für Nutzer
- `docs/architecture.md` – für Entwickler
- `docs/self-hosting.md` – für Server-Betreiber
- `docs/capture-guide.md` – Tipps für gute Aufnahmen
