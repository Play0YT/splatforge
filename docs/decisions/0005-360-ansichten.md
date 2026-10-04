# 0005 – 360°: Perspektiv-Ansichten im Rig, auch für Dual-Fisheye

**Kontext.** Die Spezifikation sieht für equirektanguläres Material eine Zerlegung in Perspektiv-Ansichten
vor, für .insv-Dateien als Standard „Weg A“: beide Objektive direkt als Fisheye-Kameras (COLMAP-Modell
`OPENCV_FISHEYE`) in einem Rig, Weg B (selbst stitchen) nur experimentell. Freigegeben am 2026-10-04:
beide Eingabearten in einer Version.

**Entscheidung.** Beide Eingabearten laufen über denselben Weg: Jeder Zeitpunkt wird in Lochkamera-Ansichten
zerlegt, die COLMAP als starres Rig mit exakt bekannten Brennweiten und Drehungen bekommt.
- Equirektangulär: 8 Ansichten rund um den Horizont plus oben und unten (90° Sichtfeld), wie in der
  Spezifikation.
- Dual-Fisheye: pro Objektiv eine Ansicht geradeaus und vier um 50° geneigte, berechnet direkt aus dem
  Fisheye-Bild mit der Werkskalibrierung aus der Datei (`offset_v3`: Unified-Modell nach Mei mit
  Verzerrung, Sensorlage). Ohne Kalibrierung Näherungswerte und eine Warnung. Es wird nicht gestitcht:
  jede Ansicht stammt aus genau einem Objektiv. Die Lage der Objektive zueinander verfeinert COLMAP.

**Begründung (Abweichung von Weg A).**
- Brush und das CPU-Backend trainieren auf Lochkamera-Bildern. Mit Weg A müsste das Fisheye-Bild nach
  COLMAP ohnehin entzerrt werden; COLMAPs Entzerrung schneidet dabei den grössten Teil des Bildkreises ab
  (über 180° lassen sich nicht in ein Lochkamera-Bild abbilden).
- `OPENCV_FISHEYE` müsste die Linse aus den Bildern schätzen; die Werkskalibrierung von Insta360 ist
  genauer und liegt in jeder Datei vor (bei der ONE RS geprüft).
- Ein gemeinsamer Weg für beide Eingabearten halbiert Code und Tests; Masken (Nadir, Personen,
  Abgleich über Ansichtsgrenzen) funktionieren für beide gleich.
- Synthetische Tests: Positionsfehler 0,2 mm (equirektangulär) bzw. 0,4 mm (Dual-Fisheye mit
  ONE-RS-Kalibrierung) auf einem 0,7-m-Kreis.

**Folgen.** Die Genauigkeit bei .insv hängt an der Werkskalibrierung. Das Format von `offset_v3` ist nicht
offiziell dokumentiert (Aufbau nach telemetry-parser) und bisher nur an einer ONE RS geprüft; bei
unplausiblen Werten (z. B. anderes Format neuerer Modelle) greifen Näherungswerte mit Warnung. Die
Sensorlage wird nur als Drehung um die Blickachse übernommen; kleine Abweichungen gleicht COLMAP aus.
Weg A bleibt als spätere Option möglich, falls Messungen mit echten Aufnahmen dafür sprechen. Der Export
aus Insta360 Studio als equirektanguläres MP4 bleibt der zuverlässigste Weg.
