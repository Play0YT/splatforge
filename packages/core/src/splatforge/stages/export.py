"""Stufe 8: Nachbearbeitung und Export.

In Meilenstein 1 nur: Ergebnis prüfen, als ``splat.ply`` ablegen und ``report.json`` schreiben.
Floater-Entfernung, Bodenausrichtung und ``.spz`` folgen in Meilenstein 7.
"""

from __future__ import annotations

import shutil
from typing import Any

from .. import __version__
from ..ply import read_ply
from .base import Stage, StageContext, write_json
from .train import FINAL_PLY, TrainStage

SPLAT_FILE = "splat.ply"
REPORT_FILE = "report.json"


class ExportStage(Stage):
    name = "export"
    dirname = "08_export"
    weight = 0.5

    def run(self, ctx: StageContext) -> dict[str, Any]:
        out = self.out_dir(ctx)
        out.mkdir(parents=True, exist_ok=True)
        source = ctx.job.stage_dir(TrainStage.dirname) / FINAL_PLY
        cloud = read_ply(source)
        if len(cloud) == 0:
            raise RuntimeError("Das Training hat einen leeren Splat erzeugt.")
        shutil.copy2(source, out / SPLAT_FILE)
        if ctx.config.export.write_spz:
            ctx.warn("Der .spz-Export folgt in einer späteren Version; es wurde nur .ply geschrieben.")
        ctx.events.log(f"Splat mit {len(cloud)} Gaussians exportiert: {out / SPLAT_FILE}")
        return {"gaussians": len(cloud), "ply": str(out / SPLAT_FILE)}


def write_report(ctx: StageContext, stage_names: list[tuple[str, str]], total_seconds: float) -> None:
    """Fasst die Kennzahlen aller Stufen in report.json zusammen."""
    stages = {}
    warnings = list(ctx.warnings)
    for name, dirname in stage_names:
        info = ctx.job.done_info(dirname)
        stages[name] = info
        warnings += [w for w in info.get("warnings", []) if w not in warnings]
    train = stages.get("train", {})
    sfm = stages.get("sfm", {})
    report = {
        "core_version": __version__,
        "preset": str(ctx.config.preset),
        "settings": ctx.config.effective().model_dump(),
        "registered_images": sfm.get("registered_images"),
        "total_images": sfm.get("total_images"),
        "gaussians": stages.get("export", {}).get("gaussians"),
        "quality": {
            "psnr": train.get("psnr"),
            "ssim": train.get("ssim"),
            "eval_views": train.get("eval_views"),
        },
        "seconds_this_run": round(total_seconds, 1),
        "stages": stages,
        "warnings": warnings,
    }
    write_json(ctx.job.stage_dir(ExportStage.dirname) / REPORT_FILE, report)
