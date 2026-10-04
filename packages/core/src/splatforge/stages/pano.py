"""Stufe 4: 360°-Aufbereitung. Zerlegt jedes ausgewählte 360°-Bild in Perspektiv-Ansichten.

Ergebnis in ``04_360``:
- ``images/<ansicht>/<frame>.jpg``: die Ansichten, ein Ordner pro Ansicht (= eine Kamera im Rig)
- ``masks/<ansicht>/<frame>.jpg.png``: feste Masken im COLMAP-Format (Nadir, Bereich ausserhalb des
  Fisheye-Bildkreises; schwarz = ignorieren)
- ``rig.json``: Ansichten, Brennweite und Art der Quelle für die SfM-Stufe
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import CameraType
from ..errors import UnsupportedInputError
from ..imageio import read_image, write_image
from ..insv import InsvMetadata, lens_calibrations
from ..panorama import (
    FisheyeLens,
    View,
    auto_view_size,
    default_lens,
    equirect_lookup,
    equirect_views,
    fisheye_views,
    lens_rotation,
    nadir_mask,
)
from .analyze import InputInfo, load_inputs, panorama_type
from .base import Stage, StageContext, read_json, write_json
from .select import IMAGES_DIR, SECOND_LENS_IMAGES_DIR, SELECTION_FILE, SelectStage

RIG_FILE = "rig.json"
MASKS_DIR = "masks"
JPEG_QUALITY = 95


def input_lenses(info: InputInfo, width: int, height: int) -> tuple[list[FisheyeLens], bool]:
    """Die beiden Objektive einer Dual-Fisheye-Eingabe für Objektivbilder von ``width`` × ``height``.
    Gibt zusätzlich zurück, ob eine Werkskalibrierung verwendet wurde."""
    calibrations = lens_calibrations(InsvMetadata(offset_v3=(info.insv or {}).get("offset_v3", [])))
    if calibrations is None or len(calibrations) < 2:
        return [default_lens(width, height, i) for i in range(2)], False
    lenses = []
    for i, c in enumerate(calibrations[:2]):
        lens = FisheyeLens(
            c.xi, c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.k3, c.p1, c.p2,
            int(round(c.width)), int(round(c.height)), lens_rotation(i, c.roll),
        )  # fmt: skip
        lenses.append(lens.scaled(width, height))
    return lenses, True


def read_rig(ctx: StageContext) -> dict[str, Any] | None:
    """Rig-Beschreibung aus Stufe 4, ``None`` bei normalen Aufnahmen."""
    path = ctx.job.stage_dir(PanoStage.dirname) / RIG_FILE
    if not path.is_file() or not ctx.job.is_done(PanoStage.dirname):
        return None
    data: dict[str, Any] = read_json(path)
    return data


def rig_views(rig: dict[str, Any]) -> list[View]:
    return [View(**v) for v in rig["views"]]


class _Maps:
    """Vorberechnete Abbildungen (Ansichtspixel → Quellpixel) und feste Masken pro Ansicht."""

    def __init__(
        self, view: View, source: tuple[int, int], lens: FisheyeLens | None, nadir_deg: float
    ) -> None:
        rays = view.rays()
        if lens is None:
            u, v = equirect_lookup(rays, source[0], source[1])
            valid: NDArray[np.bool_] = np.ones((view.size, view.size), dtype=bool)
        else:
            u, v, valid = lens.lookup(rays)
            u, v = np.where(valid, u, -1), np.where(valid, v, -1)
        self.u = u.astype(np.float32)
        self.v = v.astype(np.float32)
        self.border = cv2.BORDER_WRAP if lens is None else cv2.BORDER_CONSTANT
        self.use = valid & ~nadir_mask(view, nadir_deg, rays)

    def render(self, source: NDArray[np.uint8]) -> NDArray[np.uint8]:
        out = cv2.remap(source, self.u, self.v, cv2.INTER_LINEAR, borderMode=self.border, borderValue=0)
        return np.asarray(out, dtype=np.uint8)


class PanoStage(Stage):
    name = "pano"
    dirname = "04_360"
    weight = 1.5

    def applies(self, ctx: StageContext) -> bool:
        return panorama_type(ctx) is not None

    def run(self, ctx: StageContext) -> dict[str, Any]:
        kind = panorama_type(ctx)
        settings = ctx.config.pano
        select_dir = ctx.job.stage_dir(SelectStage.dirname)
        names: list[str] = read_json(select_dir / SELECTION_FILE)["selected"]
        inputs = load_inputs(ctx)
        first = read_image(select_dir / IMAGES_DIR / names[0])
        if first is None:
            raise UnsupportedInputError(
                "Das erste ausgewählte Bild ist nicht lesbar.", "Den Job neu starten."
            )
        height, width = first.shape[:2]
        max_edge = ctx.config.effective().max_image_edge

        lenses_per_clip: dict[int, list[FisheyeLens]] = {}
        calibrated = True
        if kind == CameraType.EQUIRECTANGULAR:
            natural = auto_view_size(width, settings.fov_deg, equirect=True)
            size = settings.view_size or min(max_edge, natural)
            views = equirect_views(settings.views, settings.fov_deg, size, settings.up_down)
        else:
            for clip, info in enumerate(inputs):
                lenses_per_clip[clip], ok = input_lenses(info, width, height)
                calibrated &= ok
            natural = auto_view_size(width, settings.fov_deg, equirect=False, lens=lenses_per_clip[0][0])
            size = settings.view_size or min(max_edge, natural)
            views = fisheye_views(settings.fisheye_tilt_deg, settings.fov_deg, size)
        ctx.events.log(
            f"{len(views)} Ansichten pro Zeitpunkt, je {size}×{size} Pixel, {settings.fov_deg:g}° Sichtfeld"
        )

        out = self.out_dir(ctx)
        maps: dict[tuple[int, str], _Maps] = {}
        for n, name in enumerate(names):
            ctx.cancel.raise_if_cancelled()
            clip = int(name[1:3])
            sources = [read_image(select_dir / IMAGES_DIR / name)]
            if kind == CameraType.DUAL_FISHEYE:
                sources.append(read_image(select_dir / SECOND_LENS_IMAGES_DIR / name))
            if any(s is None for s in sources):
                raise UnsupportedInputError(f"Bild {name} ist nicht lesbar.", "Den Job neu starten.")
            for view in views:
                key = (clip, view.name)
                if key not in maps:
                    lens = lenses_per_clip[clip][view.lens] if lenses_per_clip else None
                    maps[key] = _Maps(view, (width, height), lens, settings.nadir_mask_deg)
                source = sources[view.lens] if kind == CameraType.DUAL_FISHEYE else sources[0]
                assert source is not None
                write_image(out / IMAGES_DIR / view.name / name, maps[key].render(source), JPEG_QUALITY)
                mask = np.where(maps[key].use, 255, 0).astype(np.uint8)
                write_image(out / MASKS_DIR / view.name / f"{name}.png", mask)
            ctx.events.progress((n + 1) / len(names), message="Ansichten berechnen")

        unused = [1.0 - float(m.use.mean()) for m in maps.values()]
        rig = {
            "kind": str(kind),
            "size": size,
            "views": [v.to_dict() for v in views],
            "frames": names,
            "calibrated": calibrated,
            "refine_sensor_from_rig": kind == CameraType.DUAL_FISHEYE and settings.refine_lens_rig,
        }
        write_json(out / RIG_FILE, rig)
        return {
            "kind": str(kind),
            "views_per_frame": len(views),
            "view_size": size,
            "images": len(views) * len(names),
            "calibrated": calibrated,
            "masked_static_fraction": round(float(np.mean(unused)), 3) if unused else 0.0,
        }


def view_image_names(rig: dict[str, Any]) -> list[str]:
    """Bildnamen relativ zu ``04_360/images`` (``<ansicht>/<frame>``), sortiert nach Zeitpunkt."""
    return [f"{v['name']}/{frame}" for frame in rig["frames"] for v in rig["views"]]


def images_root(ctx: StageContext) -> Path:
    return ctx.job.stage_dir(PanoStage.dirname) / IMAGES_DIR
