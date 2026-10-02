# 0004 – Maskierung mit ONNX-Bildmodellen, Video-Verfolgung später

**Kontext.** Die Spezifikation sieht für Meilenstein 2 SAM 2 mit Video-Propagation vor. Die
Video-Propagation gibt es nur für PyTorch (Paket `sam2`); als ONNX gibt es nur das Bildmodell (Box rein,
Maske raus). Die Zielrechner reichen von Onboard-Grafik (Intel HD 630) bis zur NVIDIA-GPU. Freigegeben am
2026-10-02: „wenn NVIDIA-GPU Video-Verfolgung, sonst Bildmodell, oder einstellbar“; Modellgrösse automatisch
oder wählbar; Maskierung nur auf Wunsch.

**Entscheidung.**
- Zuerst das Bildmodell: RT-DETR R18 (Erkennung) und SAM 2.1 Hiera Tiny/Small (Segmentierung), beide als
  ONNX von `onnx-community`, Apache-2.0, ausgeführt mit onnxruntime (MIT). Lücken zwischen den Bildern
  schliesst eine einfache Box-Verfolgung mit Interpolation.
- `--mask-model auto` nimmt mit CUDA SAM 2.1 Small, sonst Tiny. `--mask-method video` ist vorgesehen und
  folgt mit der PyTorch-Variante; bis dahin wird das Bildmodell verwendet und eine Warnung ausgegeben.
- Ultralytics YOLO scheidet wegen AGPL-3.0 aus.
- Die Modelle werden beim ersten Gebrauch geladen (fest gepinnte Revision, SHA-256-Prüfung) statt
  mitgeliefert; der Download lässt sich mit `splatforge models --download` vorziehen.

**Begründung.** onnxruntime läuft auf allen Zielsystemen und nutzt CUDA, DirectML oder CoreML, falls
vorhanden. Es braucht kein PyTorch, das sonst nur für das CPU-Training optional installiert wird. Das
Tiny-Modell braucht auf der CPU etwa 3 Sekunden pro Bild.

**Folgen.** Ohne Video-Verfolgung kann eine Person, die der Detektor in mehreren Bildern hintereinander
übersieht, durchrutschen; der Sicherheitsrand, die Interpolation und die Kontrollbilder mildern das. Die
Maskierung läuft als eigener Prozess, weil onnxruntime und pycolmap unter macOS nicht im selben Prozess
liegen sollen. Für die Desktop-App (Meilenstein 5) werden die Modelle im Installer optional mitgeliefert.
