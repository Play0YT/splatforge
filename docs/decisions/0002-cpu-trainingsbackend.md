# 0002 – Eigenes CPU-Trainingsbackend als Ausweg

**Kontext.** Standard-Backend ist Brush (wgpu: Metal, Vulkan, DX12). Ohne nutzbare Grafikkarte findet wgpu
keinen Adapter. Die Spezifikation verlangt, dass ein Splat notfalls rein auf der CPU entsteht.
Freigegeben am 2026-10-01.

**Entscheidung.** Ein einfaches Trainingsbackend in PyTorch (`splatforge/training/cpu.py`), installierbar über
den Zusatz `cpu-train`. Es wird automatisch genutzt, wenn Brush fehlt oder ohne GPU scheitert.

**Umfang und Vereinfachungen.**
- Farbe nur als Grundfarbe (Kugelflächenfunktionen Grad 0), keine blickwinkelabhängigen Effekte.
- Rasterisierung in Kacheln (Standard 8×8 Pixel) in reinem PyTorch, Gradient-Checkpointing pro Kachelgruppe.
- Verdichten und Ausdünnen nach der 3DGS-Referenz, Obergrenze `cpu_max_gaussians`.
- Masken werden aus dem Verlust ausgeschlossen.
- Checkpoints mit Optimierer-Zustand; Fortsetzen nach Abbruch am letzten Checkpoint.
- Ein Test vergleicht den Renderer Pixel für Pixel mit einer naiven Referenz.

**Folgen.** Deutlich langsamer als GPU-Backends (Messwerte in `docs/measurements.md`). Auf Linux zieht
`torch` von PyPI CUDA-Bibliotheken mit (mehrere GB); für reine CPU-Rechner genügt
`pip install torch --index-url https://download.pytorch.org/whl/cpu`. Beim Packaging der Desktop-App
(Meilenstein 5) wird die CPU-Variante eingebunden.
