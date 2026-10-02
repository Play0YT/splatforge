"""Stufe 5: Personenmaskierung (läuft im Hintergrundprozess ``splatforge _mask``)."""

from __future__ import annotations

import importlib.util
from typing import Any

from ..errors import ToolMissingError, UnsupportedInputError
from ._worker import run_worker
from .base import Stage, StageContext, read_json, write_json
from .select import IMAGES_DIR, SELECTION_FILE, SelectStage

MASK_FILE = "mask.json"
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
        names = read_json(select_dir / SELECTION_FILE)["selected"]
        result = run_worker(
            ctx,
            "_mask",
            out / MASK_TASK_FILE,
            {
                "images_dir": str(select_dir / IMAGES_DIR),
                "names": names,
                "out_dir": str(out),
                "settings": settings.model_dump(mode="json"),
                "num_threads": ctx.config.resources.num_threads,
            },
            "Die Maskierung",
        )
        result.pop("type", None)
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
