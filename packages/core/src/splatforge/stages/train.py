"""Stufe 7: Splat-Training mit Brush (Standard) oder dem CPU-Backend (Ausweg)."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from ..adapters.base import stream_process
from ..config import TrainBackend
from ..errors import INSTALL_TORCH_HINT, SplatForgeError, ToolMissingError
from ..events import EventType, ProgressEvent
from ..training.cpu_worker import self_command
from .base import Stage, StageContext, write_json
from .sfm import DATASET_DIR, SfmStage

FINAL_PLY = "final.ply"
CPU_TASK_FILE = "cpu_task.json"
# Nur Brushs eigene Fortschrittsmeldungen; alles andere (z. B. die sehr langen Debug-Zeilen von
# brush_dataset beim Laden jedes Bildes) nur bei Warnungen und Fehlern.
BRUSH_LOG_FILTER = "warn,brush_cli=info,brush_process=info"
# Längere Zeilen werden im Log gekürzt
MAX_LOG_LINE = 400
TRAIN_FILE = "train.json"
_ITER = re.compile(r"iter\s+(\d+)", re.I)
_EVAL = re.compile(r"PSNR\s+([\d.]+),\s*ssim\s+([\d.]+)", re.I)
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
        task_file = out / CPU_TASK_FILE
        write_json(
            task_file,
            {
                "dataset": str(dataset),
                "work_dir": str(out),
                "iterations": iterations,
                "settings": ctx.config.train.model_dump(mode="json"),
                "num_threads": ctx.config.resources.num_threads,
            },
        )
        result: dict[str, Any] = {}
        error: dict[str, Any] = {}

        def on_line(line: str) -> None:
            try:
                msg = json.loads(line)
            except ValueError:
                if line.strip():
                    ctx.events.log(line.strip()[:MAX_LOG_LINE])
                return
            kind = msg.get("type")
            if kind == "progress":
                step, total = int(msg["step"]), int(msg["total"])
                ctx.events.progress(
                    step / total,
                    msg.get("eta"),
                    message=f"Iteration {step}/{total}, Verlust {msg['loss']:.4f}",
                )
            elif kind == "preview":
                ctx.events.emit(
                    ProgressEvent(
                        type=EventType.PREVIEW,
                        message=f"Zwischenstand nach {msg['step']} Iterationen",
                        data={"ply": msg["path"], "iteration": msg["step"]},
                    )
                )
            elif kind == "result":
                result.update(msg)
            elif kind == "error":
                error.update(msg)

        outcome = stream_process(self_command("_train-cpu", str(task_file)), on_line, cancel=ctx.cancel)
        if outcome.returncode != 0 or not result:
            if error:
                raise SplatForgeError(
                    error.get("message", "CPU-Training fehlgeschlagen."), error.get("hint", "")
                )
            raise SplatForgeError(
                "Das CPU-Training ist unerwartet abgebrochen.",
                "Den Job mit 'splatforge resume' fortsetzen. Tritt der Fehler wieder auf, "
                "weniger Frames oder eine kleinere Bildkante wählen.",
                details=f"Exit-Code {outcome.returncode}\n{outcome.stdout[-4000:]}",
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

        def on_line(line: str) -> None:
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
            "seconds": round(time.monotonic() - started, 1),
        }


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
