"""Brush-Anbindung mit einem nachgebauten Brush-Programm (echtes Brush braucht eine GPU)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from splatforge.adapters.brush import BrushAdapter, steps_flag_from_help
from splatforge.config import JobConfig, TrainBackend
from splatforge.errors import ToolMissingError
from splatforge.events import EventSink
from splatforge.job import JobDir
from splatforge.ply import GaussianCloud, read_ply, write_ply
from splatforge.stages.base import StageContext, Tools
from splatforge.stages.train import FINAL_PLY, TrainStage

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="Nachgebautes Brush ist ein Python-Skript")

FAKE_BRUSH = """#!{python}
import sys
from pathlib import Path
args = sys.argv[1:]
if "--version" in args:
    print("brush-app {version}")
    sys.exit(0)
if "--help" in args:
    print("Usage: brush [OPTIONS] [PATH_OR_URL]\\n  {steps_flag} <TOTAL>\\n  --export-every <N>")
    sys.exit(0)
if "{fail}":
    print("Error: No suitable adapter found")
    sys.exit(1)
opts = dict(zip(args[1::2], args[2::2]))
steps = int(opts["{steps_flag}"])
out = Path(opts["--export-path"])
for it in range(100, steps + 1, 100):
    print(f"Refine iter {{it}}, 5 splats.", flush=True)
print("Eval iter {{}}: PSNR 27.5, ssim 0.88".format(steps))
(out / f"export_{{steps}}.ply").write_bytes(Path({ply!r}).read_bytes())
"""


def _fake_brush(
    tmp_path: Path, steps_flag: str = "--total-steps", version: str = "0.3.0", fail: bool = False
) -> Path:
    ply = tmp_path / "vorlage.ply"
    n = 5
    write_ply(
        ply,
        GaussianCloud(
            means=np.zeros((n, 3), np.float32),
            sh_dc=np.zeros((n, 3), np.float32),
            sh_rest=np.zeros((n, 0, 3), np.float32),
            opacity_logit=np.zeros(n, np.float32),
            log_scales=np.zeros((n, 3), np.float32),
            quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        ),
    )
    script = tmp_path / "Brush Ordner" / "brush"
    script.parent.mkdir()
    script.write_text(
        FAKE_BRUSH.format(
            python=sys.executable,
            version=version,
            steps_flag=steps_flag,
            fail="1" if fail else "",
            ply=str(ply),
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _ctx(tmp_path: Path, brush: Path, backend: TrainBackend = TrainBackend.BRUSH) -> StageContext:
    config = JobConfig.model_validate(
        {"inputs": [str(tmp_path / "v.mp4")], "iterations": 300, "tools": {"brush": str(brush)},
         "train": {"backend": backend}}
    )  # fmt: skip
    job = JobDir(tmp_path / "job")
    job.create(config)
    (job.stage_dir("06_sfm") / "dataset").mkdir(parents=True)
    return StageContext(job=job, config=job.load(), events=EventSink(), tools=Tools.from_config(config))


@pytest.mark.parametrize("flag", ["--total-steps", "--total-train-iters"])
def test_steps_flag_from_help(flag: str) -> None:
    assert steps_flag_from_help(f"Options:\n  {flag} <N>  Total number of steps") == flag


def test_unknown_brush_cli_is_rejected() -> None:
    with pytest.raises(ToolMissingError):
        steps_flag_from_help("Options:\n  --something-else")


@posix_only
@pytest.mark.parametrize("flag", ["--total-steps", "--total-train-iters"])
def test_train_with_brush(tmp_path: Path, flag: str) -> None:
    ctx = _ctx(tmp_path, _fake_brush(tmp_path, steps_flag=flag))
    stage = TrainStage()
    stage.preflight(ctx)
    info = stage.run(ctx)
    assert info["backend"] == "brush"
    assert info["psnr"] == pytest.approx(27.5)
    assert len(read_ply(stage.out_dir(ctx) / FINAL_PLY)) == 5


@posix_only
def test_brush_version_is_reported(tmp_path: Path) -> None:
    adapter = BrushAdapter(_fake_brush(tmp_path))
    assert adapter.check() == (0, 3, 0)


@posix_only
def test_too_old_brush_is_rejected(tmp_path: Path) -> None:
    adapter = BrushAdapter(_fake_brush(tmp_path, version="0.2.0"))
    assert not adapter.available()


@posix_only
def test_gpu_error_is_explained(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, _fake_brush(tmp_path, fail=True))
    with pytest.raises(Exception, match="keine nutzbare Grafikkarte"):
        TrainStage().run(ctx)


@posix_only
def test_missing_configured_brush_fails_preflight(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, tmp_path / "gibt es nicht" / "brush")
    with pytest.raises(ToolMissingError):
        TrainStage().preflight(ctx)


def test_brush_option_on_cli(tmp_path: Path) -> None:
    from splatforge.cli import _parser

    args = _parser().parse_args(["run", "v.mp4", "--out", str(tmp_path), "--brush", "C:/Tools/brush.exe"])
    assert args.brush == Path("C:/Tools/brush.exe")
    assert os.fspath(_parser().parse_args(["hardware", "--brush", "x"]).brush) == "x"


@posix_only
def test_brush_app_name_found_in_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _fake_brush(tmp_path)
    renamed = script.with_name("brush_app")
    script.rename(renamed)
    monkeypatch.setenv("PATH", str(renamed.parent))
    assert BrushAdapter().path == renamed
