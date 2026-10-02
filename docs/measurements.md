# Messwerte

Die Werte der Qualitätsstufen in `config.py` sind Startwerte aus der Spezifikation. Hier stehen Messungen,
nach denen sie angepasst werden.

## 2026-10-01 – Core 0.1.0, CPU-Backend

Rechner: Linux-Container, 4 CPU-Threads, 16 GB RAM, keine GPU.
Eingabe: synthetisches Testvideo (480×270, 5 s, 120 Frames), `--frames 40 --iterations 300`.

| Stufe | Dauer |
| --- | --- |
| Analyse | 0,1 s |
| Frame-Extraktion | 0,4 s |
| Frame-Auswahl | 1,1 s |
| Kamerapositionen (global) | 66 s, 40/40 Bilder verortet |
| Training (CPU, 300 Iterationen) | 322 s (ca. 1 s pro Iteration, Kacheln 16 px) |

Ergebnis: 5 738 Gaussians, PSNR 24,5 dB, SSIM 0,72 auf 5 zurückgehaltenen Bildern.

Renderzeit pro Iteration (Vorwärts + Rückwärts) bei 478×268 Pixeln und 5 738 Gaussians:
Kacheln 4 px 1,21 s, 8 px 0,83 s, 16 px 0,95 s. Standard deshalb 8 px.

**Folgerung.** Die Vorschau-Stufe (7 000 Iterationen) braucht auf so einem Rechner mit dem CPU-Backend
mehrere Stunden. Das entspricht der Spezifikation („notfalls über Stunden oder Tage“). Messungen mit Brush
auf echter GPU und mit echten Smartphone-Videos stehen noch aus.

## 2026-10-02 – Core 0.1.1, CPU-Backend, echtes Video (Rückmeldung des Projektinhabers)

Rechner: Windows 11, 4 CPU-Threads, 32 GB RAM, Intel HD Graphics 630 (nicht genutzt).
Eingabe: Stock-Video (Parkanlage), `--frames 60 --iterations 1000`.

| Stufe | Dauer |
| --- | --- |
| Training (CPU, 1 000 Iterationen) | 13 466 s ≈ 3,7 h (ca. 13,5 s pro Iteration) |

Ergebnis: 25 592 Gaussians, visuell sehr gute Qualität.

**Folgerung.** Mit echten Videos ist das CPU-Backend gut zehnmal langsamer als mit dem kleinen
synthetischen Testvideo. Die Zeitangabe im README wurde korrigiert. Nächste Messung: dasselbe Video mit
Brush auf der Intel HD Graphics 630.

## 2026-10-02 – Core 0.1.2, Brush 0.3 auf Intel HD Graphics 630 (Rückmeldung des Projektinhabers)

Gleicher Rechner und gleiches Video wie oben, `--frames 60 --iterations 1000 --backend brush`.
Brush nahm 52 Bilder zum Trainieren und 8 zur Qualitätsmessung.

| Backend | Training (1 000 Iterationen) |
| --- | --- |
| CPU (PyTorch, 4 Threads) | 13 466 s ≈ 3,7 h |
| Brush (Intel HD Graphics 630, Onboard) | 559 s ≈ 9 min (rund 24-mal schneller) |

Visuell sehr ähnlich, das CPU-Ergebnis wirkt an einzelnen Stellen leicht besser. Vermutete Ursache (nicht
geprüft): Brush verteilt Lernrate und Verdichtung auf die Gesamtzahl der Schritte und ist auf 30 000 Schritte
ausgelegt; bei nur 1 000 Schritten wird kaum verdichtet. Das CPU-Backend skaliert sein Verdichtungsfenster
mit der Schrittzahl. Nächste Messung: Vorschau-Stufe (7 000 Schritte) mit Brush.

## 2026-10-02 – Core 0.1.3, Brush 0.3, Vorschau-Stufe (Rückmeldung des Projektinhabers)

Gleicher Rechner (Intel HD Graphics 630) und gleiches Video, `--preset preview --backend brush`
(120 Frames, max. 1 280 px, 7 000 Iterationen).

| Stufe | Dauer |
| --- | --- |
| Training (Brush, 7 000 Iterationen) | 3 319 s gemessen, davon einige Minuten Ruhezustand des PCs; reine Rechenzeit etwa 40 min |

Ergebnis: deutlich schärfer als mit 1 000 Iterationen, kaum noch Floater. Eine durchs Bild laufende Person
ist als halbtransparente Gestalt im Splat zu sehen (Fall für die Personenmaskierung, Meilenstein 2).

## 2026-10-02 – Core 0.2.0, erste echte .insv-Datei (Rückmeldung des Projektinhabers)

Datei aus dem Internet, umbenannt (`test_insv.insv`). Erkannt: Insta360 ONE RS (Firmware v1.6.29),
eine Videospur 3072×3072 (H.264, 24 fps, 47 s), also ein Objektiv pro Datei. Metadaten-Trailer Version 3
mit den Datensätzen 1, 2, 3, 4, 5, 9, 10, 11; `offset`, `offset_v2` und `offset_v3` gelesen, Gyrodaten
vorhanden. Damit ist der Leser erstmals mit einer echten Datei bestätigt. Die Datei des zweiten Objektivs
fehlte. Eine zweite Datei derselben Aufnahme (`LRV_20220625_140410_11_008.insv`) ist die Vorschau der
Kamera: beide Objektive nebeneinander in einer Spur, nur 384×384 pro Objektiv. Die exportierten Bilder zeigen ein Fisheye, das auf dem Kopf steht (in der Kalibrierung steht beim
ersten Objektiv ein Winkel von rund 179°).

Dritter Test mit dem Originalpaar `VID_20220625_140410_00_008.insv` und `…_10_008.insv` (je 3072×3072,
H.264, 47 s): Paar korrekt erkannt. Die `_10_`-Datei enthält keinen Insta360-Metadatenblock; Metadaten und
Kalibrierung stehen nur in der `_00_`-Datei (ab 0.2.2 werden sie von dort übernommen).
