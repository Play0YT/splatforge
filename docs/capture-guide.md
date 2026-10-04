# Tipps für gute Aufnahmen

- **Langsam bewegen.** Die Kamera gleichmässig und ohne ruckartige Schwenks führen. Zu schnelle Bewegung ist
  die häufigste Ursache, wenn die Kamerapositionen nicht bestimmt werden können.
- **Herumgehen statt Schwenken.** Für 3D braucht es verschiedene Standpunkte. Nur auf der Stelle drehen
  reicht nicht: COLMAP verortet die Bilder dann zwar, aber die Tiefe fehlt, und der Splat wird verzerrt.
  SplatForge warnt in diesem Fall. Faustregel: Der Weg der Kamera sollte mindestens ein Zehntel des Abstands
  zum Motiv betragen, besser deutlich mehr.
- **Viel Überlappung.** Jeder Bereich sollte in vielen Bildern aus leicht unterschiedlichen Winkeln zu sehen sein.
- **Gleichmässiges Licht.** Belichtung und Weissabgleich möglichst fixieren; harte Schatten und Gegenlicht vermeiden.
- **Textur.** Einfarbige Wände, Himmel, Glas und Spiegel lassen sich schlecht rekonstruieren.
- **Bewegte Objekte vermeiden.** Personen lassen sich mit `--masking` automatisch ausblenden; andere
  bewegte Dinge (Bäume im Wind, Wasser) stören weiterhin.
- **Hohe Bildrate, kurze Belichtungszeit.** Weniger Bewegungsunschärfe heisst mehr brauchbare Frames.
