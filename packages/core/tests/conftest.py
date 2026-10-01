from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import make_video  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="FFmpeg nicht installiert"
)


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Kurzes synthetisches Video in einem Pfad mit Leerzeichen und Umlauten."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg nicht installiert")
    folder = tmp_path_factory.mktemp("daten") / "Ordner mit Ümläuten"
    folder.mkdir()
    return make_video(folder / "szene ä.mp4")
