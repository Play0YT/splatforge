"""Masken für alle ausgewählten Bilder berechnen (läuft im Hintergrundprozess, siehe ``masking.worker``)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from ..adapters.base import CancelToken
from ..config import MaskSettings
from ..imageio import read_image, write_image
from .postprocess import BoolMask, expand, overlay, save_colmap_mask
from .tracking import Detection, build_tracks, detections_per_frame, fill_gaps

MASKS_DIR = "masks"
OVERLAYS_DIR = "overlays"
# Anteil der Arbeit für die Erkennung; der Rest entfällt auf die Segmentierung
DETECTION_SHARE = 0.3


class DetectorLike(Protocol):
    def detect(
        self, rgb: NDArray[np.uint8], classes: tuple[int, ...], threshold: float
    ) -> list[Detection]: ...


class SegmenterLike(Protocol):
    def segment(self, rgb: NDArray[np.uint8], boxes: list[NDArray[np.float64]]) -> BoolMask: ...


@dataclass
class MaskSummary:
    frames: int
    frames_with_objects: int
    interpolated_boxes: int
    excluded: list[str] = field(default_factory=list)
    mean_masked_fraction: float = 0.0
    max_masked_fraction: float = 0.0
    fractions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "frames_with_objects": self.frames_with_objects,
            "interpolated_boxes": self.interpolated_boxes,
            "excluded": self.excluded,
            "mean_masked_fraction": round(self.mean_masked_fraction, 4),
            "max_masked_fraction": round(self.max_masked_fraction, 4),
            "fractions": {k: round(v, 4) for k, v in self.fractions.items()},
        }


def _rgb(path: Path) -> NDArray[np.uint8]:
    image = read_image(path)
    if image is None:
        raise OSError(f"Bild konnte nicht gelesen werden: {path}")
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def compute_masks(
    images_dir: Path,
    names: list[str],
    out_dir: Path,
    settings: MaskSettings,
    classes: tuple[int, ...],
    detector: DetectorLike,
    segmenter: SegmenterLike,
    on_progress: Callable[[float, str], None],
    cancel: CancelToken,
) -> MaskSummary:
    masks_dir = out_dir / MASKS_DIR
    overlays_dir = out_dir / OVERLAYS_DIR
    masks_dir.mkdir(parents=True, exist_ok=True)

    per_frame: list[list[Detection]] = []
    for i, name in enumerate(names):
        cancel.raise_if_cancelled()
        per_frame.append(detector.detect(_rgb(images_dir / name), classes, settings.detection_threshold))
        on_progress(DETECTION_SHARE * (i + 1) / len(names), "Objekte erkennen")

    tracks = build_tracks(per_frame, settings.track_min_iou, settings.max_gap_frames)
    fill_gaps(tracks, settings.max_gap_frames)
    per_frame = detections_per_frame(tracks, len(names))

    summary = MaskSummary(frames=len(names), frames_with_objects=0, interpolated_boxes=0)
    for i, name in enumerate(names):
        cancel.raise_if_cancelled()
        detections = per_frame[i]
        rgb = _rgb(images_dir / name)
        h, w = rgb.shape[:2]
        if detections:
            ignore = segmenter.segment(rgb, [d.box for d in detections])
            ignore = expand(ignore, round(settings.margin_fraction * w))
            summary.frames_with_objects += 1
            summary.interpolated_boxes += sum(d.interpolated for d in detections)
        else:
            ignore = np.zeros((h, w), dtype=bool)
        fraction = float(ignore.mean())
        summary.fractions[name] = fraction
        if fraction > settings.max_object_fraction:
            summary.excluded.append(name)
        save_colmap_mask(masks_dir / f"{name}.png", ignore)
        if settings.write_overlays and detections:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # Unterordner (360°-Ansichten) beibehalten, damit gleichnamige Frames sich nicht überschreiben
            target = overlays_dir / Path(name).with_suffix(".jpg")
            write_image(target, overlay(np.asarray(bgr, np.uint8), ignore))
        on_progress(DETECTION_SHARE + (1 - DETECTION_SHARE) * (i + 1) / len(names), "Masken berechnen")

    values = list(summary.fractions.values())
    summary.mean_masked_fraction = float(np.mean(values)) if values else 0.0
    summary.max_masked_fraction = float(np.max(values)) if values else 0.0
    return summary


def merge_summaries(parts: list[MaskSummary]) -> MaskSummary:
    """Fasst die Ergebnisse mehrerer Bildfolgen (z. B. der 360°-Ansichten) zusammen."""
    merged = MaskSummary(
        frames=sum(p.frames for p in parts),
        frames_with_objects=sum(p.frames_with_objects for p in parts),
        interpolated_boxes=sum(p.interpolated_boxes for p in parts),
    )
    for p in parts:
        merged.excluded += p.excluded
        merged.fractions.update(p.fractions)
    values = list(merged.fractions.values())
    merged.mean_masked_fraction = float(np.mean(values)) if values else 0.0
    merged.max_masked_fraction = float(np.max(values)) if values else 0.0
    return merged
