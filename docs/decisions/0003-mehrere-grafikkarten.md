# 0003 – Umgang mit mehreren Grafikkarten

**Kontext.** Rechner können mehrere Grafikkarten haben (Onboard plus eingebaute Karte, Server mit mehreren GPUs).
Gefragt am 2026-10-02.

**Entscheidung.**
- Ein Job nutzt genau eine Grafikkarte.
- Mehrere Grafikkarten werden genutzt, indem mehrere Jobs parallel laufen: Der Server-Worker startet
  standardmässig einen Job pro Grafikkarte (wie in der Spezifikation).
- In der Desktop-App wird die Grafikkarte wählbar, soweit das Trainings-Backend es erlaubt.
- Stufen mit unabhängigen Bildern (Personenmaskierung ab Meilenstein 2) dürfen ihre Arbeit später auf
  mehrere Grafikkarten verteilen.

**Begründung.**
- Brush (0.3 und aktueller Hauptzweig) nutzt immer das Standardgerät von wgpu, also die leistungsstärkste
  Grafikkarte, und bietet keine Auswahl über die Kommandozeile. Geprüft im Brush-Quellcode
  (`brush-process`, `WgpuDevice::default()`).
- Ein einzelnes Splat-Training auf mehrere Grafikkarten zu verteilen bringt wenig, weil die Karten ständig
  Zwischenergebnisse austauschen müssen. Ob gsplat (geplantes CUDA-Backend) das sinnvoll unterstützt, wird
  geprüft, bevor es eingebaut wird.
- COLMAP (pycolmap von PyPI) rechnet auf der CPU.

**Folgen.** Keine Änderung an der bestehenden Pipeline. Für die Grafikkarten-Auswahl in Brush wäre eine
Änderung an Brush selbst nötig.
