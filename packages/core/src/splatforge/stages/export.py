"""Stufe 8: Nachbearbeitung und Export.

In Meilenstein 1 nur: Ergebnis prüfen, als ``splat.ply`` ablegen und ``report.json`` schreiben.
Die Szene ist schon in Stufe 6 waagrecht ausgerichtet; hier wird das in der PLY vermerkt.
Floater-Entfernung und ``.spz`` folgen in Meilenstein 7.
"""

from __future__ import annotations

import shutil
from typing import Any

from .. import __version__
from ..ply import copy_with_comment, read_ply
from .base import Stage, StageContext, write_json
from .sfm import SfmStage
from .train import FINAL_PLY, TrainStage

SPLAT_FILE = "splat.ply"
REPORT_FILE = "report.json"
# Brush versteht "y" als: oben ist −Y
VERTICAL_AXIS_COMMENT = "vertical axis: y"


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
        if ctx.job.done_info(SfmStage.dirname).get("oriented"):
            # Brush liest daraus die Richtung oben (−Y) und schätzt sie dann nicht selbst
            copy_with_comment(source, out / SPLAT_FILE, VERTICAL_AXIS_COMMENT)
        else:
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
        "masking": _mask_summary(stages.get("mask")),
        "seconds_this_run": round(total_seconds, 1),
        "stages": stages,
        "warnings": warnings,
    }
    write_json(ctx.job.stage_dir(ExportStage.dirname) / REPORT_FILE, report)


def _mask_summary(info: dict[str, Any] | None) -> dict[str, Any] | None:
    """Kurzfassung der Maskierung für report.json; ``None``, wenn sie nicht lief."""
    if not info:
        return None
    keys = ("method", "model", "providers", "frames", "frames_with_objects", "interpolated_boxes")
    summary = {k: info.get(k) for k in keys}
    summary["excluded_images"] = len(info.get("excluded", []))
    summary["mean_masked_fraction"] = info.get("mean_masked_fraction")
    return summary
