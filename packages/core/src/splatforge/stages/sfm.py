"""Stufe 6: Kamerapositionen mit COLMAP (Structure from Motion)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import cv2

from ..adapters.colmap import ColmapAdapter, DatabaseStats
from ..config import Mapper, SfmSettings
from ..errors import ReconstructionError
from ..imageio import read_image, write_image
from .base import Stage, StageContext, write_json
from .mask import MaskStage, excluded_images
from .select import IMAGES_DIR, SelectStage

DATASET_DIR = "dataset"
SFM_FILE = "sfm.json"
MASKS_DIR = "masks"
# Zwischenordner für das Entzerren der Masken
MASK_UNDISTORT_DIR = "_masks_undistort"


def diagnose(stats: DatabaseStats, registered: int, settings: SfmSettings) -> tuple[str, str]:
    """Leitet aus den Datenbank-Kennzahlen eine verständliche Ursache und Massnahme ab."""
    if stats.mean_keypoints < settings.low_texture_features:
        return (
            "Die Bilder enthalten zu wenig Textur (z. B. weisse Wände, Himmel, spiegelnde Flächen).",
            "Ein Motiv mit mehr Details filmen oder strukturierte Gegenstände ins Bild bringen.",
        )
    weak = [n for n in stats.neighbour_inliers if n < settings.weak_pair_inliers]
    if stats.neighbour_inliers and len(weak) > len(stats.neighbour_inliers) * 0.3:
        return (
            "Aufeinanderfolgende Bilder überlappen zu wenig. Meist bewegt sich die Kamera zu schnell.",
            "Kamera langsamer bewegen oder mehr Frames wählen (höhere Qualitätsstufe oder --frames).",
        )
    if stats.neighbour_inliers and weak:
        return (
            "Die Aufnahme zerfällt in mehrere Teile, die nicht zusammenpassen.",
            "Mehr Überlappung zwischen den Abschnitten filmen und ruckartige Schwenks vermeiden.",
        )
    return (
        f"Nur {registered} von {stats.num_images} Bildern konnten verortet werden.",
        "Mehr Überlappung, langsamere Bewegung und gleichmässiges Licht helfen.",
    )


class SfmStage(Stage):
    name = "sfm"
    dirname = "06_sfm"
    weight = 3.0

    def run(self, ctx: StageContext) -> dict[str, Any]:
        settings = ctx.config.sfm
        threads = ctx.config.resources.num_threads
        colmap = ctx.tools.colmap
        colmap.check()
        out = self.out_dir(ctx)
        out.mkdir(parents=True, exist_ok=True)
        colmap.log_file = out / "colmap.log"
        images = ctx.job.stage_dir(SelectStage.dirname) / IMAGES_DIR
        database = out / "database.db"
        masks = self._masks_dir(ctx)
        excluded = excluded_images(ctx) if masks is not None else set()
        names = sorted(p.name for p in images.iterdir() if p.name not in excluded)
        timings: dict[str, float] = {}

        start = time.monotonic()
        ctx.events.progress(0.0, message="Merkmale suchen")
        colmap.extract_features(
            database,
            images,
            settings.camera_model,
            settings.single_camera,
            masks,
            threads,
            image_names=names if excluded else None,
        )
        timings["features_s"] = time.monotonic() - start
        ctx.cancel.raise_if_cancelled()

        start = time.monotonic()
        ctx.events.progress(0.25, message="Bilder vergleichen")
        vocab = (
            settings.vocab_tree_path
            if settings.vocab_tree_path and settings.vocab_tree_path.is_file()
            else None
        )
        if vocab is None:
            ctx.events.log("Loop-Detection aus (kein Vocab-Tree angegeben)")
        colmap.match_sequential(
            database, settings.sequential_overlap, vocab, settings.loop_detection_period, threads
        )
        timings["matching_s"] = time.monotonic() - start
        ctx.cancel.raise_if_cancelled()

        num_images = len(names)
        start = time.monotonic()
        model_dir, registered, mapper_used = self._map(ctx, database, images, out, num_images)
        timings["mapping_s"] = time.monotonic() - start
        ratio = registered / num_images if num_images else 0.0
        if model_dir is None or ratio < settings.min_registered_ratio:
            stats = colmap.database_stats(database)
            message, hint = diagnose(stats, registered, settings)
            raise ReconstructionError(
                f"Kamerapositionen konnten nicht zuverlässig bestimmt werden ({registered} von {num_images} "
                f"Bildern, mindestens {settings.min_registered_ratio:.0%} nötig). {message}",
                hint,
                details=(
                    f"mittlere Merkmale {stats.mean_keypoints:.0f}, Nachbar-Inlier {stats.neighbour_inliers}"
                ),
            )

        ctx.events.progress(0.9, message="Bilder entzerren")
        dataset = out / DATASET_DIR
        if dataset.exists():
            shutil.rmtree(dataset)
        colmap.undistort(model_dir, images, dataset, threads)
        if masks is not None:
            ctx.events.progress(0.95, message="Masken entzerren")
            undistort_masks(colmap, model_dir, masks, dataset, out / MASK_UNDISTORT_DIR, threads)
        info = {
            "registered_images": registered,
            "total_images": num_images,
            "registered_ratio": round(ratio, 3),
            "mapper": mapper_used,
            "points3d": colmap.read_model(model_dir).num_points3D(),
            "masked": masks is not None,
            "excluded_images": len(excluded),
            **{k: round(v, 1) for k, v in timings.items()},
        }
        write_json(out / SFM_FILE, info)
        ctx.events.log(f"{registered} von {num_images} Bildern verortet ({mapper_used})")
        return info

    def _masks_dir(self, ctx: StageContext) -> Path | None:
        """Masken der Stufe 5, falls die Maskierung für diesen Job an ist und fertig gelaufen ist."""
        if not ctx.config.masking or not ctx.job.is_done(MaskStage.dirname):
            return None
        candidate = ctx.job.stage_dir(MaskStage.dirname) / MASKS_DIR
        return candidate if candidate.is_dir() else None

    def _map(
        self, ctx: StageContext, database: Path, images: Path, out: Path, num_images: int
    ) -> tuple[Path | None, int, str]:
        settings = ctx.config.sfm
        colmap = ctx.tools.colmap
        threads = ctx.config.resources.num_threads
        order: list[Mapper] = []
        if settings.mapper in (Mapper.AUTO, Mapper.GLOBAL) and colmap.has_global_mapper():
            order.append(Mapper.GLOBAL)
        if settings.mapper in (Mapper.AUTO, Mapper.INCREMENTAL) or not order:
            order.append(Mapper.INCREMENTAL)

        best: tuple[Path | None, int, str] = (None, 0, "")
        for mapper in order:
            ctx.cancel.raise_if_cancelled()
            ctx.events.progress(0.5, message=f"Kamerapositionen berechnen ({mapper})")
            sparse = out / f"sparse_{mapper}"
            if sparse.exists():
                shutil.rmtree(sparse)
            sparse.mkdir(parents=True)
            try:
                if mapper == Mapper.GLOBAL:
                    models = colmap.map_global(database, images, sparse, threads)
                else:
                    models = colmap.map_incremental(database, images, sparse, threads)
            except Exception as exc:  # COLMAP meldet Fehler als RuntimeError
                ctx.events.log(f"{mapper}-Mapping fehlgeschlagen: {exc}")
                continue
            if not models:
                ctx.events.log(f"{mapper}-Mapping ohne Ergebnis")
                continue
            idx, model = max(models.items(), key=lambda kv: kv[1].num_reg_images())
            registered = int(model.num_reg_images())
            model_dir = sparse / "best"
            model_dir.mkdir(exist_ok=True)
            model.write(model_dir)
            ctx.events.log(f"{mapper}: {registered} von {num_images} Bildern in Modell {idx}")
            if registered > best[1]:
                best = (model_dir, registered, str(mapper))
            if registered / max(num_images, 1) >= settings.min_registered_ratio:
                break
        return best


def undistort_masks(
    colmap: ColmapAdapter, model_dir: Path, masks: Path, dataset: Path, work: Path, num_threads: int
) -> int:
    """Entzerrt die Masken genau wie die Bilder und legt sie als ``dataset/masks/<stem>.png`` ab.

    Trick: COLMAP entzerrt die Masken selbst, damit sie pixelgenau zu den entzerrten Bildern passen.
    Dafür bekommen sie vorübergehend die Namen der Bilder. COLMAP schreibt JPEG; danach wird wieder
    scharf in schwarz (ignorieren) und weiss (verwenden) getrennt. Der Dateiname ``<stem>.png`` ist das
    Format, das Brush (ab 0.3) und das CPU-Backend lesen. Gibt die Anzahl entzerrter Masken zurück.
    """
    if work.exists():
        shutil.rmtree(work)
    renamed = work / "input"
    renamed.mkdir(parents=True)
    names = sorted(p.name for p in (dataset / IMAGES_DIR).iterdir())
    available: list[str] = []
    for name in names:
        mask = read_image(masks / f"{name}.png", cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        # Verlustfrei als PNG speichern; COLMAP erkennt das Format am Inhalt, nicht an der Endung.
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            continue
        encoded.tofile(renamed / name)
        available.append(name)
    if not available:
        shutil.rmtree(work)
        return 0
    colmap.undistort(
        model_dir, renamed, work / "output", num_threads, image_names=available, jpeg_quality=100
    )
    target = dataset / MASKS_DIR
    target.mkdir(exist_ok=True)
    for name in available:
        mask = read_image(work / "output" / IMAGES_DIR / name, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        write_image(target / f"{Path(name).stem}.png", ((mask > 127) * 255).astype("uint8"))
    shutil.rmtree(work)
    return len(available)
