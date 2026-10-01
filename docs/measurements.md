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
