from __future__ import annotations

import numpy as np

from splatforge.config import SelectSettings
from splatforge.stages.select import FrameStats, choose_frames, motion_between, sharpness


def _stats(motions: list[float], sharp: list[float] | None = None) -> list[FrameStats]:
    sharp = sharp or [100.0] * len(motions)
    return [FrameStats(f"{i}.jpg", 0, s, m) for i, (m, s) in enumerate(zip(motions, sharp, strict=True))]


def test_stationary_section_gets_few_frames() -> None:
    # 50 Frames Stillstand, dann 50 Frames gleichmässige Bewegung
    stats = _stats([0.0] * 50 + [0.01] * 50)
    chosen = choose_frames(stats, 20, SelectSettings())
    assert len(chosen) == 20
    assert sum(1 for i in chosen if i < 50) <= 1


def test_blurry_frames_are_skipped() -> None:
    sharp = [100.0] * 40
    blurry = set(range(1, 40, 2))
    for i in blurry:
        sharp[i] = 5.0
    chosen = choose_frames(_stats([0.01] * 40, sharp), 15, SelectSettings())
    assert not blurry.intersection(chosen)


def test_never_more_frames_than_usable() -> None:
    assert len(choose_frames(_stats([0.01] * 10), 50, SelectSettings())) == 10


def test_sharpness_and_motion() -> None:
    rng = np.random.default_rng(0)
    img = (rng.random((240, 320)) * 255).astype(np.uint8)
    blurred = np.asarray(np.clip(np.convolve(img.ravel(), np.ones(9) / 9, "same"), 0, 255), np.uint8).reshape(
        img.shape
    )
    assert sharpness(img) > sharpness(blurred)
    shifted = np.roll(img, 6, axis=1)
    assert abs(motion_between(img, shifted, 400) * np.hypot(240, 320) - 6.0) < 1.0
