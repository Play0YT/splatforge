from __future__ import annotations

from pathlib import Path

import pytest

from conftest import needs_ffmpeg
from splatforge.adapters import FfmpegAdapter


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ((4, 4, 2), ["-vsync", "passthrough"]),
        ((5, 1), ["-fps_mode", "passthrough"]),
        (None, ["-fps_mode", "passthrough"]),
    ],
)
def test_passthrough_option_depends_on_version(
    monkeypatch: pytest.MonkeyPatch, version: tuple[int, ...] | None, expected: list[str]
) -> None:
    adapter = FfmpegAdapter()
    monkeypatch.setattr(adapter, "version", lambda: version)
    assert adapter._passthrough_args() == expected


@needs_ffmpeg
def test_extract_with_installed_version(tmp_path: Path, synthetic_video: Path) -> None:
    """Die nach Version gewählte Option muss von der installierten FFmpeg-Version akzeptiert werden.

    Ältere Versionen kennen nur -vsync, neuere nur -fps_mode; deshalb wird hier nichts vorgetäuscht.
    """
    FfmpegAdapter().extract_frames(
        source=synthetic_video,
        out_pattern=tmp_path / "%04d.jpg",
        fps=2.0,
        max_edge=320,
        jpeg_quality=2,
        tonemap=False,
        duration_s=5.0,
        on_progress=lambda _f: None,
        cancel=None,
    )
    assert 8 <= len(list(tmp_path.glob("*.jpg"))) <= 12
