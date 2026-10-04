"""Kompletter Durchlauf mit 360°-Material: equirektangulär und Dual-Fisheye (.insv)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import needs_ffmpeg
from splatforge.cli import main
from synthetic import make_dual_fisheye_insv, make_equirect_video, rig_path

if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch nicht installiert", allow_module_level=True)

FRAMES = 20  # Mindestanzahl Zeitpunkte (select.min_selected_frames)


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg nicht installiert")
    folder = tmp_path_factory.mktemp("360 Ümlaut")
    return {
        "equirect": make_equirect_video(folder / "panorama.mp4", frames=FRAMES, width=640),
        "insv": make_dual_fisheye_insv(
            folder / "VID_20261004_120000_00_001.insv", frames=FRAMES, lens_size=400
        ),
    }


def _run(clip: Path, job: Path) -> dict[str, object]:
    args = ["run", str(clip), "--out", str(job), "--preset", "preview", "--frames", str(FRAMES),
            "--iterations", "20", "--backend", "cpu", "--json"]  # fmt: skip
    code = main(args)
    events = [json.loads(line) for line in (job / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert code == 0, [e for e in events if e["type"] in ("job_failed", "log")][-5:]
    report: dict[str, object] = json.loads((job / "08_export" / "report.json").read_text(encoding="utf-8"))
    return report


def _true_up_in_model(dataset: Path) -> np.ndarray:
    """Wo das echte Oben der synthetischen Szene im fertigen Modell liegt (soll −Y sein)."""
    import pycolmap

    rec = pycolmap.Reconstruction(dataset / "sparse")
    gt = rig_path(FRAMES)
    acc = np.zeros((3, 3))
    for image in rec.images.values():
        if image.name.startswith("v00_"):
            frame = int(image.name.split("_")[-1].split(".")[0]) - 1
            acc += gt[frame][1].T @ image.cam_from_world().rotation.matrix()
    u, _, vt = np.linalg.svd(acc)
    return np.asarray((u @ vt).T @ np.array([0.0, 1.0, 0.0]))


@needs_ffmpeg
@pytest.mark.integration
@pytest.mark.parametrize("kind", ["equirect", "insv"])
def test_360_run(tmp_path: Path, clips: dict[str, Path], kind: str) -> None:
    job = tmp_path / kind
    report = _run(clips[kind], job)
    stages = report["stages"]
    assert isinstance(stages, dict)
    pano = stages["pano"]
    assert pano["views_per_frame"] == 10
    sfm = stages["sfm"]
    assert sfm["rig_views"] == 10
    assert sfm["registered_ratio"] >= 0.9
    if kind == "insv":
        assert pano["calibrated"] is True
    dataset = job / "06_sfm" / "dataset"
    images = sorted(p.name for p in (dataset / "images").iterdir())
    assert images and all(p.is_file() for p in (dataset / "images").iterdir())  # flach, ohne Unterordner
    masks = {p.stem for p in (dataset / "masks").iterdir()}
    assert {Path(n).stem for n in images} == masks
    up = _true_up_in_model(dataset)
    assert np.degrees(np.arccos(np.clip(-up[1], -1, 1))) < 3
