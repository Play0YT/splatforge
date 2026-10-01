"""Bilder lesen und schreiben, auch bei Umlauten im Pfad.

``cv2.imread``/``cv2.imwrite`` scheitern unter Windows an Nicht-ASCII-Pfaden. Deshalb wird die
Datei mit NumPy gelesen und im Speicher dekodiert.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> Image | None:
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, flags)
    return None if image is None else np.asarray(image, dtype=np.uint8)


def write_image(path: Path, image: Image, jpeg_quality: int = 95) -> None:
    ext = path.suffix.lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if ext in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise OSError(f"Bild konnte nicht kodiert werden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)
