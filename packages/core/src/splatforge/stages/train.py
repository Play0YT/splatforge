"""Stufe 7: Splat-Training mit Brush (Standard) oder dem CPU-Backend (Ausweg)."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from ..config import TrainBackend
from ..errors import INSTALL_TORCH_HINT, SplatForgeError, ToolMissingError
from ..events import EventType, ProgressEvent
from ..hardware import GPU_GROUP_HINT, gpu_access_problem
from ._worker import run_worker
from .base import Stage, StageContext, write_json
from .sfm import DATASET_DIR, SfmStage

FINAL_PLY = "final.ply"
CPU_TASK_FILE = "cpu_task.json"
# Nur Brushs eigene Fortschrittsmeldungen; alles andere (z. B. die sehr langen Debug-Zeilen von
# brush_dataset beim Laden jedes Bildes) nur bei Warnungen und Fehlern.
# cubecl_wgpu=info: meldet, auf welchem Gerät (Grafikkarte oder CPU) Brush rechnet
BRUSH_LOG_FILTER = "warn,brush_cli=info,brush_process=info,cubecl_wgpu=info"
# Längere Zeilen werden im Log gekürzt
MAX_LOG_LINE = 400
TRAIN_FILE = "train.json"
_ITER = re.compile(r"iter\s+(\d+)", re.I)
_EVAL = re.compile(r"PSNR\s+([\d.]+),\s*ssim\s+([\d.]+)", re.I)
# Zeile von cubecl, z. B. "Using adapter AdapterInfo { name: \"AMD Radeon RX 6700 (RADV NAVI22)\", …,
# device_type: DiscreteGpu, driver: \"radv\", driver_info: \"Mesa 25.0\", backend: Vulkan }"
_ADAPTER = re.compile(
    r'AdapterInfo \{ name: "(?P<name>[^"]*)".*?device_type: (?P<type>\w+)'
    r'(?:.*?driver: "(?P<driver>[^"]*)")?(?:.*?backend: (?P<backend>\w+))?'
)
DEVICE_TYPES = {
    "DiscreteGpu": "eigene Grafikkarte",
    "IntegratedGpu": "integrierte Grafik",
    "VirtualGpu": "virtuelle Grafikkarte",
    "Cpu": "CPU (Software-Treiber)",
    "Other": "unbekanntes Gerät",
}


def parse_adapter(line: str) -> dict[str, str] | None:
    """Gerät aus der Brush-/cubecl-Logzeile, ``None`` wenn die Zeile keins nennt."""
    m = _ADAPTER.search(line)
    if m is None:
        return None
    return {k: v for k, v in m.groupdict().items() if v is not None}


_EXPORT = re.compile(r"export_(\d+)\.ply$")


def torch_available() -> bool:
    """Ob PyTorch installiert ist, ohne es zu laden (siehe training/cpu_worker.py)."""
    return importlib.util.find_spec("torch") is not None


class TrainStage(Stage):
    name = "train"
    dirname = "07_train"
    weight = 10.0

    def cleanup(self, ctx: StageContext) -> None:
        # Checkpoints bleiben erhalten, damit das Training dort fortgesetzt werden kann.
        return None

    def preflight(self, ctx: StageContext) -> None:
        wanted = ctx.config.train.backend
        if wanted in (TrainBackend.BRUSH, TrainBackend.AUTO) and (problem := gpu_access_problem()):
            # Früh melden: sonst fällt erst nach Stunden auf, dass Brush auf der CPU gerechnet hat
            ctx.warn(problem, GPU_GROUP_HINT)
        if wanted == TrainBackend.BRUSH:
            ctx.tools.brush.check()
        elif wanted == TrainBackend.CPU and not torch_available():
            raise ToolMissingError("Für das CPU-Training fehlt PyTorch.", INSTALL_TORCH_HINT)
        elif wanted == TrainBackend.AUTO and not ctx.tools.brush.available() and not torch_available():
            raise ToolMissingError("Es ist kein Trainings-Backend verfügbar.", INSTALL_TORCH_HINT)

    def choose_backend(self, ctx: StageContext) -> TrainBackend:
        wanted = ctx.config.train.backend
        if wanted != TrainBackend.AUTO:
            return wanted
        previous = self.out_dir(ctx) / "backend.txt"
        if previous.is_file():
            # Ein fortgesetzter Job bleibt beim Backend, mit dem er begonnen hat.
            return TrainBackend(previous.read_text(encoding="utf-8").strip())
        if ctx.tools.brush.available():
            return TrainBackend.BRUSH
        if torch_available():
            ctx.warn(
                "Brush wurde nicht gefunden, es wird das langsame CPU-Backend verwendet.",
                "Für schnelleres Training Brush installieren.",
            )
            return TrainBackend.CPU
        raise ToolMissingError(
            "Es ist kein Trainings-Backend verfügbar.",
            INSTALL_TORCH_HINT,
        )

    def run(self, ctx: StageContext) -> dict[str, Any]:
        out = self.out_dir(ctx)
        out.mkdir(parents=True, exist_ok=True)
        dataset = ctx.job.stage_dir(SfmStage.dirname) / DATASET_DIR
        backend = self.choose_backend(ctx)
        (out / "backend.txt").write_text(str(backend), encoding="utf-8")
        ctx.events.log(f"Trainings-Backend: {backend}")
        if backend == TrainBackend.BRUSH:
            try:
                info = self._run_brush(ctx, dataset, out)
            except SplatForgeError as exc:
                if ctx.config.train.backend != TrainBackend.AUTO or not torch_available():
                    raise
                ctx.warn(f"Brush ist fehlgeschlagen ({exc.message}). Weiter mit dem CPU-Backend.")
                (out / "backend.txt").write_text(str(TrainBackend.CPU), encoding="utf-8")
                info = self._run_cpu(ctx, dataset, out)
        else:
            info = self._run_cpu(ctx, dataset, out)
        write_json(out / TRAIN_FILE, info)
        return info

    def _run_cpu(self, ctx: StageContext, dataset: Path, out: Path) -> dict[str, Any]:
        """Startet das CPU-Training in einem eigenen Prozess und übersetzt dessen Ausgabe in Events."""
        iterations = ctx.config.effective().iterations

        def on_message(msg: dict[str, Any]) -> None:
            if msg.get("type") == "progress":
                step, total = int(msg["step"]), int(msg["total"])
                ctx.events.progress(
                    step / total,
                    msg.get("eta"),
                    message=f"Iteration {step}/{total}, Verlust {msg['loss']:.4f}",
                )
            elif msg.get("type") == "preview":
                ctx.events.emit(
                    ProgressEvent(
                        type=EventType.PREVIEW,
                        message=f"Zwischenstand nach {msg['step']} Iterationen",
                        data={"ply": msg["path"], "iteration": msg["step"]},
                    )
                )

        result = run_worker(
            ctx,
            "_train-cpu",
            out / CPU_TASK_FILE,
            {
                "dataset": str(dataset),
                "work_dir": str(out),
                "iterations": iterations,
                "settings": ctx.config.train.model_dump(mode="json"),
                "num_threads": ctx.config.resources.num_threads,
            },
            "Das CPU-Training",
            on_message,
        )
        return {
            "backend": "cpu",
            "iterations": iterations,
            "gaussians": result["gaussians"],
            "psnr": result["psnr"],
            "ssim": result["ssim"],
            "eval_views": result["eval_views"],
            "seconds": round(float(result["seconds"]), 1),
        }

    def _run_brush(self, ctx: StageContext, dataset: Path, out: Path) -> dict[str, Any]:
        brush = ctx.tools.brush
        brush.check()
        iterations = ctx.config.effective().iterations
        exports = out / "brush_exports"
        exports.mkdir(exist_ok=True)
        started = time.monotonic()
        metrics: dict[str, float] = {}
        device: dict[str, str] = {}

        def on_line(line: str) -> None:
            if (found := parse_adapter(line)) is not None:
                if not device:  # cubecl meldet das Gerät zweimal; einmal reicht
                    device.update(found)
                    report_device(ctx, found)
                return
            if (m := _EVAL.search(line)) is not None:
                metrics["psnr"], metrics["ssim"] = float(m.group(1)), float(m.group(2))
            if (m := _ITER.search(line)) is not None:
                step = int(m.group(1))
                elapsed = time.monotonic() - started
                eta = elapsed / step * (iterations - step) if step else None
                ctx.events.progress(step / iterations, eta, message=f"Iteration {step}/{iterations}")
            text = line.strip()
            if text:
                ctx.events.log(text if len(text) <= MAX_LOG_LINE else text[:MAX_LOG_LINE] + " …")

        env = dict(os.environ)
        env.setdefault("RUST_LOG", BRUSH_LOG_FILTER)
        brush.stream(
            [
                *brush_args(dataset, exports, iterations, ctx),
            ],
            on_line,
            cancel=ctx.cancel,
            cwd=exports,
            env=env,
        )
        candidates = sorted(
            (int(m.group(1)), p) for p in exports.glob("export_*.ply") if (m := _EXPORT.search(p.name))
        )
        if not candidates:
            raise SplatForgeError(
                "Brush hat keine Ergebnisdatei geschrieben.",
                "Details stehen im Log.",
            )
        shutil.copy2(candidates[-1][1], out / FINAL_PLY)
        return {
            "backend": "brush",
            "iterations": iterations,
            "psnr": metrics.get("psnr"),
            "ssim": metrics.get("ssim"),
            "device": device or None,
            "seconds": round(time.monotonic() - started, 1),
        }


def report_device(ctx: StageContext, device: dict[str, str]) -> None:
    """Meldet, worauf Brush rechnet, und warnt, wenn es nur der Software-Treiber auf der CPU ist."""
    kind = DEVICE_TYPES.get(device.get("type", ""), device.get("type", "?"))
    details = ", ".join(v for v in (kind, device.get("backend"), device.get("driver")) if v)
    ctx.events.log(f"Brush rechnet auf: {device.get('name', '?')} ({details})")
    if device.get("type") == "Cpu":
        problem = gpu_access_problem()
        ctx.warn(
            f"Brush rechnet nicht auf der Grafikkarte, sondern mit dem Software-Treiber "
            f"„{device.get('name', '?')}“ auf der CPU. Das Training ist dadurch sehr langsam."
            + (f" Ursache: {problem}" if problem else ""),
            GPU_GROUP_HINT
            if problem
            else "Prüfen, ob der Vulkan-Treiber der Grafikkarte installiert ist ('vulkaninfo --summary'). "
            "Mit der Umgebungsvariable CUBECL_WGPU_DEFAULT_DEVICE=DiscreteGpu(0) lässt sich die eigene "
            "Grafikkarte erzwingen.",
        )


def brush_args(dataset: Path, exports: Path, iterations: int, ctx: StageContext) -> list[str | Path]:
    train = ctx.config.train
    args: list[str | Path] = [
        dataset,
        ctx.tools.brush.steps_flag(), str(iterations),
        "--export-every", str(train.checkpoint_every),
        "--export-path", exports,
        "--export-name", "export_{iter}.ply",
        "--max-resolution", str(ctx.config.effective().max_image_edge),
        "--sh-degree", str(train.sh_degree),
    ]  # fmt: skip
    if train.eval_split_every > 0:
        args += ["--eval-split-every", str(train.eval_split_every)]
    return args
