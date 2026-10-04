"""Stufe 5: Personenmaskierung (läuft im Hintergrundprozess ``splatforge _mask``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..errors import ToolMissingError, UnsupportedInputError
from ..imageio import read_image, write_image
from ..panorama import share_masks
from ._worker import run_worker
from .base import Stage, StageContext, read_json, write_json
from .pano import MASKS_DIR as STATIC_MASKS_DIR
from .pano import PanoStage, images_root, read_rig, rig_views, view_image_names
from .select import IMAGES_DIR, SELECTION_FILE, SelectStage

MASK_FILE = "mask.json"
MASKS_DIR = "masks"
MASK_TASK_FILE = "mask_task.json"


def excluded_images(ctx: StageContext) -> set[str]:
    """Bilder, die wegen zu viel maskierter Fläche nicht verwendet werden."""
    path = ctx.job.stage_dir(MaskStage.dirname) / MASK_FILE
    if not path.is_file():
        return set()
    return set(read_json(path).get("excluded", []))


class MaskStage(Stage):
    name = "mask"
    dirname = "05_mask"
    weight = 2.0

    def applies(self, ctx: StageContext) -> bool:
        return ctx.config.masking

    def preflight(self, ctx: StageContext) -> None:
        # Nur suchen, nicht laden: onnxruntime läuft im eigenen Prozess.
        if importlib.util.find_spec("onnxruntime") is None:
            raise ToolMissingError(
                "Für die Maskierung fehlt onnxruntime.",
                "SplatForge neu installieren (uv sync) oder ohne --masking starten.",
            )

    def run(self, ctx: StageContext) -> dict[str, Any]:
        settings = ctx.config.mask
        out = self.out_dir(ctx)
        out.mkdir(parents=True, exist_ok=True)
        select_dir = ctx.job.stage_dir(SelectStage.dirname)
        rig = read_rig(ctx)
        task: dict[str, Any] = {
            "out_dir": str(out),
            "settings": settings.model_dump(mode="json"),
            "num_threads": ctx.config.resources.num_threads,
        }
        if rig is None:
            names = read_json(select_dir / SELECTION_FILE)["selected"]
            task |= {"images_dir": str(select_dir / IMAGES_DIR), "names": names}
        else:
            # 360°: Personen pro Ansicht über die Zeit verfolgen. Ansichten werden nie ausgeschlossen, weil
            # COLMAP pro Zeitpunkt das vollständige Rig erwartet.
            names = view_image_names(rig)
            groups = [[f"{v['name']}/{frame}" for frame in rig["frames"]] for v in rig["views"]]
            task |= {"images_dir": str(images_root(ctx)), "names": names, "groups": groups}
            task["settings"]["max_object_fraction"] = 1.0
        result = run_worker(ctx, "_mask", out / MASK_TASK_FILE, task, "Die Maskierung")
        result.pop("type", None)
        if rig is not None:
            result["shared_masks"] = combine_360_masks(ctx, rig, out / MASKS_DIR)
        write_json(out / MASK_FILE, result)

        excluded = result.get("excluded", [])
        remaining = len(names) - len(excluded)
        ctx.events.log(
            f"Objekte in {result['frames_with_objects']} von {len(names)} Bildern maskiert, "
            f"{result['interpolated_boxes']} Lücken geschlossen, {len(excluded)} Bilder ausgeschlossen"
        )
        if remaining < ctx.config.select.min_selected_frames:
            raise UnsupportedInputError(
                f"Nach der Maskierung bleiben nur {remaining} brauchbare Bilder.",
                "Eine Aufnahme mit weniger Personen im Bild verwenden oder die Grenze "
                "'mask.max_object_fraction' erhöhen.",
            )
        if excluded:
            ctx.warn(
                f"{len(excluded)} Bilder zeigen zu viel Person und werden nicht verwendet.",
                "Abstand zu Personen halten oder warten, bis sie aus dem Bild sind.",
            )
        if 1 - float(result["mean_masked_fraction"]) < settings.min_usable_fraction:
            ctx.warn(
                "Im Mittel ist ein grosser Teil der Bilder maskiert; das Ergebnis kann lückenhaft werden.",
                "Eine Aufnahme mit weniger Personen im Bild verwenden.",
            )
        return {k: v for k, v in result.items() if k != "fractions"}


def combine_360_masks(ctx: StageContext, rig: dict[str, Any], masks_dir: Path) -> int:
    """Gleicht die Personenmasken eines Zeitpunkts über die Ansichtsgrenzen ab (Person am Rand einer
    Ansicht wird auch in der Nachbaransicht maskiert) und verbindet sie mit den festen Masken aus Stufe 4.
    Gibt zurück, in wie vielen Ansichten dadurch zusätzliche Pixel maskiert wurden."""
    views = rig_views(rig)
    static_dir = ctx.job.stage_dir(PanoStage.dirname) / STATIC_MASKS_DIR
    added = 0
    for n, frame in enumerate(rig["frames"]):
        ctx.cancel.raise_if_cancelled()
        paths = [masks_dir / v.name / f"{frame}.png" for v in views]
        people = []
        for path, view in zip(paths, views, strict=True):
            stored = read_image(path, cv2.IMREAD_GRAYSCALE)
            people.append(np.zeros((view.size, view.size), bool) if stored is None else stored < 128)
        shared = share_masks(views, people) if ctx.config.pano.share_masks else people
        for path, view, own, mask in zip(paths, views, people, shared, strict=True):
            static = read_image(static_dir / view.name / f"{frame}.png", cv2.IMREAD_GRAYSCALE)
            use = ~mask if static is None else (~mask & (static > 127))
            added += int(mask.sum() > own.sum())
            write_image(path, np.where(use, 255, 0).astype(np.uint8))
        ctx.events.progress((n + 1) / len(rig["frames"]), message="Masken der Ansichten abgleichen")
    return added
