from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from conftest import needs_ffmpeg
from splatforge.config import CameraType, JobConfig
from splatforge.errors import UnsupportedInputError
from splatforge.events import EventSink
from splatforge.job import JobDir
from splatforge.stages.analyze import ANALYSIS_FILE, AnalyzeStage, detect_camera_type
from splatforge.stages.base import StageContext, Tools, read_json


def _probe(*streams: dict[str, Any], tags: dict[str, str] | None = None) -> dict[str, Any]:
    return {"streams": [{"codec_type": "video", **s} for s in streams], "format": {"tags": tags or {}}}


@pytest.mark.parametrize(
    ("name", "probe", "expected"),
    [
        ("handy.mp4", _probe({"width": 1920, "height": 1080}), CameraType.PERSPECTIVE),
        ("pano.mp4", _probe({"width": 5760, "height": 2880}), CameraType.EQUIRECTANGULAR),
        (
            "pano.mp4",
            _probe(
                {"width": 3000, "height": 1000, "side_data_list": [{"side_data_type": "Spherical Mapping"}]}
            ),
            CameraType.EQUIRECTANGULAR,
        ),
        ("x4.insv", _probe({"width": 2880, "height": 2880}), CameraType.DUAL_FISHEYE),
        (
            "zwei.mp4",
            _probe({"width": 2880, "height": 2880}, {"width": 2880, "height": 2880}),
            CameraType.DUAL_FISHEYE,
        ),
        (
            "cover.mp4",
            _probe(
                {"width": 1280, "height": 720},
                {"width": 600, "height": 600, "disposition": {"attached_pic": 1}},
            ),
            CameraType.PERSPECTIVE,
        ),
    ],
)
def test_detect_camera_type(name: str, probe: dict[str, Any], expected: CameraType) -> None:
    camera_type, _ = detect_camera_type(Path(name), probe)
    assert camera_type == expected


def _ctx(tmp_path: Path, inputs: list[Path], **kwargs: Any) -> StageContext:
    config = JobConfig(inputs=inputs, **kwargs)
    job = JobDir(tmp_path / "job")
    job.create(config)
    return StageContext(job=job, config=job.load(), events=EventSink(), tools=Tools.from_config(config))


@needs_ffmpeg
def test_analyze_video(tmp_path: Path, synthetic_video: Path) -> None:
    ctx = _ctx(tmp_path, [synthetic_video])
    AnalyzeStage().run(ctx)
    info = read_json(ctx.job.stage_dir(AnalyzeStage.dirname) / ANALYSIS_FILE)["inputs"][0]
    assert info["kind"] == "video"
    assert info["camera_type"] == "perspective"
    assert (info["width"], info["height"]) == (480, 270)
    assert info["duration_s"] == pytest.approx(5.0, abs=0.1)


@needs_ffmpeg
def test_analyze_rotated_phone_video(tmp_path: Path, synthetic_video: Path) -> None:
    rotated = tmp_path / "hochkant.mp4"
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-display_rotation", "90",
         "-i", str(synthetic_video), "-c", "copy", str(rotated)],
        check=False,
    )  # fmt: skip
    if result.returncode != 0:
        pytest.skip("FFmpeg-Version kann keine Rotations-Metadaten setzen")
    ctx = _ctx(tmp_path, [rotated])
    AnalyzeStage().run(ctx)
    info = read_json(ctx.job.stage_dir(AnalyzeStage.dirname) / ANALYSIS_FILE)["inputs"][0]
    assert info["rotation"] in (90, 270)
    assert (info["width"], info["height"]) == (270, 480)


def test_rejects_unknown_extension(tmp_path: Path) -> None:
    bad = tmp_path / "notizen.txt"
    bad.write_text("kein Video", encoding="utf-8")
    with pytest.raises(UnsupportedInputError):
        AnalyzeStage().run(_ctx(tmp_path, [bad]))


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedInputError):
        AnalyzeStage().run(_ctx(tmp_path, [tmp_path / "fehlt.mp4"]))
