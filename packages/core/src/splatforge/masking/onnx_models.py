"""Personenerkennung (RT-DETR) und Segmentierung (SAM 2.1, Bildmodell) über ONNX Runtime.

Vorverarbeitung wie in den Hugging-Face-Konfigurationen der Modelle:
- RT-DETR: auf 640×640 skalieren, Werte 0..1, keine Normalisierung. Ausgabe: Logits für 80 COCO-Klassen
  und Boxen als (cx, cy, w, h) relativ zur Bildgrösse.
- SAM 2.1: auf 1024×1024 skalieren, mit ImageNet-Mittelwert/-Streuung normalisieren. Boxen in
  1024er-Koordinaten. Ausgabe: drei Maskenvorschläge (256×256 Logits) mit geschätzter Güte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .tracking import Detection

DETECTOR_SIZE = 640
SAM_SIZE = 1024
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# COCO-Klassen (Index im 80er-Schema des Detektors) je zuschaltbarer Gruppe
CLASS_GROUPS: dict[str, tuple[int, ...]] = {
    "person": (0,),
    "vehicle": (1, 2, 3, 5, 6, 7),
    "animal": (14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
}

# Bevorzugte Reihenfolge der Beschleuniger
PREFERRED_PROVIDERS = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)


def choose_providers(requested: str = "auto") -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    return [p for p in PREFERRED_PROVIDERS if p in available] or ["CPUExecutionProvider"]


def _session(path: Path, providers: list[str], threads: int) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=providers)


def _sigmoid(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float32)


class Detector:
    def __init__(self, folder: Path, providers: list[str], threads: int = 0) -> None:
        self.session = _session(folder / "model.onnx", providers, threads)

    def detect(self, rgb: NDArray[np.uint8], classes: tuple[int, ...], threshold: float) -> list[Detection]:
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (DETECTOR_SIZE, DETECTOR_SIZE), interpolation=cv2.INTER_LINEAR)
        x = (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        logits, boxes = self.session.run(None, {"pixel_values": x})
        scores = _sigmoid(logits[0])
        result = []
        for cls in classes:
            for i in np.flatnonzero(scores[:, cls] >= threshold):
                cx, cy, bw, bh = boxes[0][i]
                box = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                result.append(
                    Detection(box=np.clip(box, 0, [w, h, w, h]), score=float(scores[i, cls]), label=cls)
                )
        return result


class Segmenter:
    """SAM 2.1 als Bildmodell: ein Encoder-Lauf pro Bild, danach ein schneller Decoder-Lauf pro Box."""

    def __init__(self, folder: Path, providers: list[str], threads: int = 0) -> None:
        self.encoder = _session(folder / "vision_encoder.onnx", providers, threads)
        self.decoder = _session(folder / "prompt_encoder_mask_decoder.onnx", providers, threads)

    def segment(self, rgb: NDArray[np.uint8], boxes: list[NDArray[np.float64]]) -> NDArray[np.bool_]:
        """Vereinigung der Masken aller Boxen (True = Objekt)."""
        h, w = rgb.shape[:2]
        union = np.zeros((h, w), dtype=bool)
        if not boxes:
            return union
        x = cv2.resize(rgb, (SAM_SIZE, SAM_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        e0, e1, e2 = self.encoder.run(None, {"pixel_values": x})
        scale = np.array([SAM_SIZE / w, SAM_SIZE / h, SAM_SIZE / w, SAM_SIZE / h], dtype=np.float32)
        for box in boxes:
            inputs = {
                "input_points": np.zeros((1, 1, 0, 2), np.float32),
                "input_labels": np.zeros((1, 1, 0), np.int64),
                "input_boxes": (box.astype(np.float32) * scale)[None, None],
                "image_embeddings.0": e0,
                "image_embeddings.1": e1,
                "image_embeddings.2": e2,
            }
            iou_scores, masks, _ = self.decoder.run(None, inputs)
            best = masks[0, 0, int(np.argmax(iou_scores[0, 0]))]
            union |= cv2.resize(best, (w, h), interpolation=cv2.INTER_LINEAR) > 0
        return union
