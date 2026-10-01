"""Stufe 3: Frame-Auswahl.

Ziel ist eine gleichmässige Abdeckung der Kamerabewegung, nicht gleichmässige Zeitabstände:
1. Schärfe jedes Kandidaten über die Laplace-Varianz (relativ zum Median des Clips).
2. Bewegung zwischen aufeinanderfolgenden Kandidaten über optischen Fluss.
3. Auswahl gleichmässig entlang der aufsummierten Bewegung, jeweils der schärfste Frame in der
   Umgebung. Stillstand bekommt dadurch kaum Frames, schnelle Passagen mehr.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import SelectSettings
from ..errors import UnsupportedInputError
from ..imageio import Image, read_image
from .base import Stage, StageContext, read_json, write_json
from .extract import FRAMES_DIR, FRAMES_FILE, ExtractStage

IMAGES_DIR = "images"
SELECTION_FILE = "selection.json"


@dataclass
class FrameStats:
    name: str
    clip: int
    sharpness: float
    motion: float  # Bewegung zum vorherigen Kandidaten desselben Clips (Anteil der Diagonale)


def _gray_small(image: Image, edge: int) -> Image:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, edge / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return np.asarray(gray, dtype=np.uint8)


def sharpness(gray: Image) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def motion_between(prev: Image, curr: Image, max_features: int) -> float:
    """Median der Pixelverschiebung zwischen zwei Graubildern, relativ zur Bilddiagonale."""
    diag = float(np.hypot(*prev.shape))
    points = cv2.goodFeaturesToTrack(prev, maxCorners=max_features, qualityLevel=0.01, minDistance=7)
    if points is not None and len(points) >= 8:
        moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, points, None)
        ok = status.reshape(-1) == 1
        if ok.sum() >= 8:
            shift = np.linalg.norm((moved - points).reshape(-1, 2)[ok], axis=1)
            return float(np.median(shift)) / diag
    # Zu wenig Merkmale: dichter Fluss als Rückfall
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.median(np.linalg.norm(flow, axis=2))) / diag


def choose_frames(stats: list[FrameStats], count: int, settings: SelectSettings) -> list[int]:
    """Wählt ``count`` Indizes gleichmässig entlang der aufsummierten Bewegung."""
    if not stats:
        return []
    sharp = np.array([s.sharpness for s in stats])
    median = float(np.median(sharp)) or 1.0
    usable = sharp >= settings.blur_relative_threshold * median
    if usable.sum() == 0:
        usable[:] = True
    cumulative = np.cumsum([0.0 if i == 0 else s.motion for i, s in enumerate(stats)])
    total = float(cumulative[-1])
    count = min(count, int(usable.sum()))
    if total <= 0:
        positions: NDArray[np.float64] = np.linspace(0, len(stats) - 1, count)
        targets = np.interp(positions, np.arange(len(stats)), cumulative)
    else:
        targets = (np.arange(count) + 0.5) / count * total
    spacing = total / count if total > 0 else 0.0
    chosen: list[int] = []
    taken = np.zeros(len(stats), dtype=bool)
    for target in targets:
        distance = np.abs(cumulative - target)
        window = (distance <= spacing / 2) & usable & ~taken
        if window.any():
            candidates = np.flatnonzero(window)
            best = int(candidates[np.argmax(sharp[candidates])])
        else:
            free = usable & ~taken
            if not free.any():
                break
            best = int(np.flatnonzero(free)[np.argmin(distance[free])])
        taken[best] = True
        chosen.append(best)
    return sorted(chosen)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class SelectStage(Stage):
    name = "select"
    dirname = "03_select"
    weight = 1.0

    def run(self, ctx: StageContext) -> dict[str, Any]:
        settings = ctx.config.select
        extract_dir = ctx.job.stage_dir(ExtractStage.dirname)
        frames = read_json(extract_dir / FRAMES_FILE)["frames"]
        target = ctx.config.effective().frames
        stats: list[FrameStats] = []
        prev_gray: Image | None = None
        prev_clip = -1
        for n, entry in enumerate(frames):
            ctx.cancel.raise_if_cancelled()
            image = read_image(extract_dir / FRAMES_DIR / entry["name"])
            if image is None:
                ctx.warn(f"Frame {entry['name']} ist beschädigt und wird übersprungen.")
                continue
            gray = _gray_small(image, settings.analysis_edge)
            motion = 0.0
            if prev_gray is not None and entry["clip"] == prev_clip and prev_gray.shape == gray.shape:
                motion = motion_between(prev_gray, gray, settings.max_flow_features)
            stats.append(FrameStats(entry["name"], entry["clip"], sharpness(gray), motion))
            prev_gray, prev_clip = gray, entry["clip"]
            if n % 10 == 0:
                ctx.events.progress((n + 1) / len(frames), message="Schärfe und Bewegung")

        clips = sorted({s.clip for s in stats})
        clip_stats = {c: [s for s in stats if s.clip == c] for c in clips}
        motions = {c: sum(s.motion for s in clip_stats[c]) for c in clips}
        total_motion = sum(motions.values())
        if total_motion < settings.min_total_motion:
            raise UnsupportedInputError(
                "Die Kamera bewegt sich im Video kaum.",
                "Für einen 3D-Splat langsam um das Motiv herumgehen, statt still zu stehen "
                "oder nur zu schwenken.",
            )
        selected: list[FrameStats] = []
        for c in clips:
            share = motions[c] / total_motion if total_motion > 0 else 1 / len(clips)
            want = max(2, round(target * share))
            indices = choose_frames(clip_stats[c], want, settings)
            selected += [clip_stats[c][i] for i in indices]

        if len(selected) < settings.min_selected_frames:
            raise UnsupportedInputError(
                f"Nur {len(selected)} brauchbare Frames gefunden.",
                "Die Kamera langsamer bewegen, für gutes Licht sorgen oder ein längeres Video verwenden.",
            )
        blurry = sum(
            1
            for s in stats
            if s.sharpness < settings.blur_relative_threshold * np.median([x.sharpness for x in stats])
        )
        images_dir = self.out_dir(ctx) / IMAGES_DIR
        images_dir.mkdir(parents=True, exist_ok=True)
        for s in selected:
            _link_or_copy(extract_dir / FRAMES_DIR / s.name, images_dir / s.name)
        write_json(
            self.out_dir(ctx) / SELECTION_FILE,
            {
                "selected": [s.name for s in selected],
                "stats": [s.__dict__ for s in stats],
            },
        )
        ctx.events.log(f"{len(selected)} von {len(stats)} Frames ausgewählt, {blurry} unscharf")
        return {"selected_frames": len(selected), "candidate_frames": len(stats), "blurry_frames": blurry}
