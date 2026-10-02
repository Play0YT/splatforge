"""Boxen über aufeinanderfolgende Bilder verfolgen und Lücken schliessen.

Wird eine Person in einem Bild nicht erkannt, davor und danach aber schon, wird ihre Box linear
interpoliert. Das ersetzt für das ONNX-Bildmodell die Video-Propagation von SAM 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Box = NDArray[np.float64]  # x1, y1, x2, y2 in Pixeln


@dataclass
class Detection:
    box: Box
    score: float
    label: int
    interpolated: bool = False


@dataclass
class Track:
    track_id: int
    frames: dict[int, Detection] = field(default_factory=dict)


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def build_tracks(per_frame: list[list[Detection]], min_iou: float, max_gap: int) -> list[Track]:
    """Ordnet Erkennungen über die Bilder hinweg einander zu (gierig nach Überlappung)."""
    tracks: list[Track] = []
    for frame, detections in enumerate(per_frame):
        claimed: set[int] = set()
        for det in sorted(detections, key=lambda d: -d.score):
            best, best_iou = None, min_iou
            for track in tracks:
                if track.track_id in claimed or track.frames.get(frame) is not None:
                    continue
                last_frame = max(track.frames)
                if frame - last_frame > max_gap + 1:
                    continue
                overlap = iou(track.frames[last_frame].box, det.box)
                if overlap >= best_iou and det.label == track.frames[last_frame].label:
                    best, best_iou = track, overlap
            if best is None:
                best = Track(track_id=len(tracks))
                tracks.append(best)
            best.frames[frame] = det
            claimed.add(best.track_id)
    return tracks


def fill_gaps(tracks: list[Track], max_gap: int) -> None:
    """Interpoliert fehlende Boxen innerhalb einer Spur, wenn die Lücke höchstens ``max_gap`` Bilder hat."""
    for track in tracks:
        frames = sorted(track.frames)
        for a, b in zip(frames, frames[1:], strict=False):
            gap = b - a - 1
            if gap <= 0 or gap > max_gap:
                continue
            da, db = track.frames[a], track.frames[b]
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                track.frames[a + k] = Detection(
                    box=(1 - t) * da.box + t * db.box,
                    score=min(da.score, db.score),
                    label=da.label,
                    interpolated=True,
                )


def detections_per_frame(tracks: list[Track], num_frames: int) -> list[list[Detection]]:
    result: list[list[Detection]] = [[] for _ in range(num_frames)]
    for track in tracks:
        for frame, det in track.frames.items():
            result[frame].append(det)
    return result
