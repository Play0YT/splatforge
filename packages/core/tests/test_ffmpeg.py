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
@pytest.mark.parametrize("pretend_version", [(4, 4), None])
def test_extract_with_old_and_new_option(
    tmp_path: Path,
    synthetic_video: Path,
    monkeypatch: pytest.MonkeyPatch,
    pretend_version: tuple[int, ...] | None,
) -> None:
    """Auch die Option für FFmpeg 4.4 (-vsync) muss von der installierten Version akzeptiert werden."""
    adapter = FfmpegAdapter()
    if pretend_version is not None:
        monkeypatch.setattr(adapter, "version", lambda: pretend_version)
    adapter.extract_frames(
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
