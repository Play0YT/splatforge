"""Stufe 6: Kamerapositionen mit COLMAP (Structure from Motion)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from ..adapters.colmap import DatabaseStats
from ..config import Mapper, SfmSettings
from ..errors import ReconstructionError
from .base import Stage, StageContext, write_json
from .select import IMAGES_DIR, SelectStage

DATASET_DIR = "dataset"
SFM_FILE = "sfm.json"


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
        timings: dict[str, float] = {}

        start = time.monotonic()
        ctx.events.progress(0.0, message="Merkmale suchen")
        colmap.extract_features(
            database, images, settings.camera_model, settings.single_camera, masks, threads
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

        num_images = len(list(images.iterdir()))
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
            ctx.warn("Masken werden beim Entzerren noch nicht mitgeführt (folgt mit Meilenstein 2).")
        info = {
            "registered_images": registered,
            "total_images": num_images,
            "registered_ratio": round(ratio, 3),
            "mapper": mapper_used,
            "points3d": colmap.read_model(model_dir).num_points3D(),
            **{k: round(v, 1) for k, v in timings.items()},
        }
        write_json(out / SFM_FILE, info)
        ctx.events.log(f"{registered} von {num_images} Bildern verortet ({mapper_used})")
        return info

    def _masks_dir(self, ctx: StageContext) -> Path | None:
        # Stufe 5 (Personenmaskierung) folgt in Meilenstein 2.
        candidate = ctx.job.stage_dir("05_mask") / "masks"
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
