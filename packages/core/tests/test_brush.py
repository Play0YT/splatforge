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
adapter = ('AdapterInfo {{ name: "{adapter_name}", vendor: 4098, device: 29663, '
           'device_type: {adapter_type}, driver: "{driver}", driver_info: "Mesa 25.0.7", backend: Vulkan }}')
print("[INFO cubecl_wgpu::runtime] Using adapter " + adapter)
print("[INFO cubecl_wgpu::runtime] Created wgpu compute server on device Device => " + adapter)
print("INFO brush_dataset::scene load_scene_img; " + "x" * 5000)
opts = dict(zip(args[1::2], args[2::2]))
steps = int(opts["{steps_flag}"])
out = Path(opts["--export-path"])
for it in range(100, steps + 1, 100):
    print(f"Refine iter {{it}}, 5 splats.", flush=True)
print("Eval iter {{}}: PSNR 27.5, ssim 0.88".format(steps))
(out / f"export_{{steps}}.ply").write_bytes(Path({ply!r}).read_bytes())
"""


def _fake_brush(
    tmp_path: Path,
    steps_flag: str = "--total-steps",
    version: str = "0.3.0",
    fail: bool = False,
    adapter: tuple[str, str, str] = ("AMD Radeon RX 6700 (RADV NAVI22)", "DiscreteGpu", "radv"),
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
            adapter_name=adapter[0],
            adapter_type=adapter[1],
            driver=adapter[2],
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


@posix_only
def test_long_brush_lines_are_shortened(tmp_path: Path) -> None:
    from splatforge.stages.train import MAX_LOG_LINE

    ctx = _ctx(tmp_path, _fake_brush(tmp_path))
    logged: list[str] = []
    ctx.events.log = logged.append  # type: ignore[method-assign]
    TrainStage().run(ctx)
    assert logged
    assert max(len(line) for line in logged) <= MAX_LOG_LINE + 2


def test_parse_adapter() -> None:
    from splatforge.stages.train import parse_adapter

    line = (
        '[2026-10-09T05:28:11Z INFO  cubecl_wgpu::runtime] Using adapter AdapterInfo { name: "llvmpipe '
        '(LLVM 19.1.1, 256 bits)", vendor: 65541, device: 0, device_type: Cpu, driver: "llvmpipe", '
        'driver_info: "Mesa 25.0.7", backend: Vulkan }'
    )
    assert parse_adapter(line) == {
        "name": "llvmpipe (LLVM 19.1.1, 256 bits)", "type": "Cpu", "driver": "llvmpipe", "backend": "Vulkan"
    }  # fmt: skip
    assert parse_adapter("Refine iter 100, 5 splats.") is None


@posix_only
def test_brush_device_is_reported(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, _fake_brush(tmp_path))
    logged: list[str] = []
    ctx.events.log = logged.append  # type: ignore[method-assign]
    info = TrainStage().run(ctx)
    assert info["device"]["type"] == "DiscreteGpu"
    device_lines = [line for line in logged if "Brush rechnet auf" in line]
    assert device_lines == [
        "Brush rechnet auf: AMD Radeon RX 6700 (RADV NAVI22) (eigene Grafikkarte, Vulkan, radv)"
    ]
    assert not any("AdapterInfo" in line for line in logged)  # Rohzeilen nicht doppelt im Log
    assert not ctx.warnings


@posix_only
def test_cpu_fallback_is_warned(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path, _fake_brush(tmp_path, adapter=("llvmpipe (LLVM 19.1.1, 256 bits)", "Cpu", "llvmpipe"))
    )
    TrainStage().run(ctx)
    assert any("nicht auf der Grafikkarte" in w for w in ctx.warnings)


def test_gpu_access_problem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from splatforge import hardware

    dri = tmp_path / "dri"
    dri.mkdir()
    (dri / "card1").touch()
    (dri / "renderD128").touch()
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.os, "access", lambda path, mode: Path(path).name != "renderD128")
    problem = hardware.gpu_access_problem(dri)
    assert problem is not None and "renderD128" in problem
    monkeypatch.setattr(hardware.os, "access", lambda path, mode: True)
    assert hardware.gpu_access_problem(dri) is None
    assert hardware.gpu_access_problem(tmp_path / "fehlt") is None
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    assert hardware.gpu_access_problem(dri) is None


def test_preflight_warns_without_gpu_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import splatforge.stages.train as train

    monkeypatch.setattr(train, "gpu_access_problem", lambda: "Kein Zugriff auf /dev/dri/renderD128")
    monkeypatch.setattr(train, "torch_available", lambda: True)
    ctx = _ctx(tmp_path, tmp_path / "brush", backend=TrainBackend.AUTO)
    TrainStage().preflight(ctx)
    assert ctx.warnings == ["Kein Zugriff auf /dev/dri/renderD128"]
