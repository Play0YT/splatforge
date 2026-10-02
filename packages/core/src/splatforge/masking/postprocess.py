"""Nachbearbeitung von Masken: Sicherheitsrand, Qualitätskontrolle, Speichern im COLMAP-Format."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ..imageio import write_image

BoolMask = NDArray[np.bool_]


def expand(mask: BoolMask, margin_px: int) -> BoolMask:
    """Erweitert die maskierte Fläche um ``margin_px`` Pixel (Haare, Bewegungsunschärfe, Schatten)."""
    if margin_px <= 0 or not mask.any():
        return mask
    size = 2 * margin_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    return np.asarray(dilated, dtype=bool)


def save_colmap_mask(path: Path, ignore: BoolMask) -> None:
    """COLMAP-Format: gleiche Grösse wie das Bild, schwarz (0) = ignorieren, weiss (255) = verwenden."""
    write_image(path, np.where(ignore, 0, 255).astype(np.uint8))


def overlay(image: NDArray[np.uint8], ignore: BoolMask, alpha: float = 0.5) -> NDArray[np.uint8]:
    """Kontrollbild: maskierte Bereiche rot eingefärbt."""
    out = image.copy()
    red = np.array([0, 0, 255], dtype=np.float32)
    out[ignore] = ((1 - alpha) * out[ignore].astype(np.float32) + alpha * red).astype(np.uint8)
    return out
