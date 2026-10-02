"""Stufe 2: Frame-Extraktion mit FFmpeg."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2

from ..imageio import read_image, write_image
from .analyze import ANALYSIS_FILE, AnalyzeStage, InputInfo
from .base import Stage, StageContext, read_json, write_json

FRAMES_DIR = "frames"
FRAMES_FILE = "frames.json"
# Durchschnittliche Grösse eines JPEG-Frames pro Pixel bei hoher Qualität (Erfahrungswert)
JPEG_BYTES_PER_PIXEL = 0.35


def load_inputs(ctx: StageContext) -> list[InputInfo]:
    data = read_json(ctx.job.stage_dir(AnalyzeStage.dirname) / ANALYSIS_FILE)
    return [InputInfo.from_dict(d) for d in data["inputs"]]


def scaled_size(width: int, height: int, max_edge: int) -> tuple[int, int]:
    scale = min(1.0, max_edge / max(width, height, 1))
    return int(round(width * scale)), int(round(height * scale))


def candidate_counts(inputs: list[InputInfo], total: int) -> list[int]:
    """Verteilt die Kandidatenzahl nach Dauer (bzw. Bildanzahl) auf die Clips."""
    weights = [i.duration_s if i.kind == "video" else float(i.frame_count) for i in inputs]
    weight_sum = sum(weights) or 1.0
    counts = []
    for info, w in zip(inputs, weights, strict=True):
        wanted = max(1, math.ceil(total * w / weight_sum))
        counts.append(min(wanted, info.frame_count) if info.frame_count else wanted)
    return counts


class ExtractStage(Stage):
    name = "extract"
    dirname = "02_extract"
    weight = 1.0

    def preflight(self, ctx: StageContext) -> None:
        if any(Path(p).is_file() for p in ctx.config.inputs):
            ctx.tools.ffmpeg.check()

    def _plan(self, ctx: StageContext) -> tuple[list[InputInfo], list[int], int]:
        inputs = load_inputs(ctx)
        values = ctx.config.effective()
        total = math.ceil(values.frames * ctx.config.extract.candidate_factor)
        return inputs, candidate_counts(inputs, total), values.max_image_edge

    def estimate_disk_bytes(self, ctx: StageContext) -> int:
        inputs, counts, max_edge = self._plan(ctx)
        total = 0
        for info, count in zip(inputs, counts, strict=True):
            w, h = scaled_size(info.width, info.height, max_edge)
            total += int(count * w * h * JPEG_BYTES_PER_PIXEL)
        return total

    def run(self, ctx: StageContext) -> dict[str, Any]:
        inputs, counts, max_edge = self._plan(ctx)
        frames_dir = self.out_dir(ctx) / FRAMES_DIR
        frames_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for clip, (info, count) in enumerate(zip(inputs, counts, strict=True)):
            ctx.cancel.raise_if_cancelled()
            done_before = clip / len(inputs)

            def report(fraction: float, done_before: float = done_before) -> None:
                ctx.events.progress(done_before + fraction / len(inputs))

            if info.kind == "video":
                self._extract_video(ctx, info, count, max_edge, frames_dir, clip, report)
            else:
                self._extract_images(ctx, info, count, max_edge, frames_dir, clip, report)
            names = sorted(p.name for p in frames_dir.glob(f"c{clip:02d}_*.jpg"))
            if not names:
                raise RuntimeError(f"Aus {info.path} wurden keine Frames extrahiert.")
            entries += [{"name": n, "clip": clip, "index": i} for i, n in enumerate(names)]
            ctx.events.log(f"{Path(info.path).name}: {len(names)} Frames extrahiert")
        write_json(self.out_dir(ctx) / FRAMES_FILE, {"frames": entries})
        return {"candidate_frames": len(entries)}

    def _extract_video(
        self,
        ctx: StageContext,
        info: InputInfo,
        count: int,
        max_edge: int,
        frames_dir: Path,
        clip: int,
        report: Any,
    ) -> None:
        fps = min(info.fps, count / info.duration_s)
        tonemap = info.hdr and ctx.config.extract.tonemap_hdr
        if info.hdr and not tonemap:
            ctx.warn("HDR-Video ohne Tonemapping: Farben können flau wirken.")
        ctx.tools.ffmpeg.extract_frames(
            source=Path(info.path),
            out_pattern=frames_dir / f"c{clip:02d}_%06d.jpg",
            fps=fps,
            max_edge=max_edge,
            jpeg_quality=ctx.config.extract.jpeg_quality,
            tonemap=tonemap,
            duration_s=info.duration_s,
            on_progress=report,
            cancel=ctx.cancel,
        )

    def _extract_images(
        self,
        ctx: StageContext,
        info: InputInfo,
        count: int,
        max_edge: int,
        frames_dir: Path,
        clip: int,
        report: Any,
    ) -> None:
        step = max(1, len(info.images) // max(count, 1))
        chosen = info.images[::step]
        for n, name in enumerate(chosen):
            ctx.cancel.raise_if_cancelled()
            image = read_image(Path(info.path) / name)
            if image is None:
                ctx.warn(f"Bild {name} konnte nicht gelesen werden und wird übersprungen.")
                continue
            h, w = image.shape[:2]
            new_w, new_h = scaled_size(w, h, max_edge)
            if (new_w, new_h) != (w, h):
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            write_image(frames_dir / f"c{clip:02d}_{n + 1:06d}.jpg", image)
            report((n + 1) / len(chosen))
