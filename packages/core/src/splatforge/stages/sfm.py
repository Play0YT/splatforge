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
from ..orient import baseline_ratio, orientation
from .base import Stage, StageContext, write_json
from .mask import MaskStage, excluded_images
from .pano import PanoStage, images_root, read_rig, rig_views, view_image_names
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
        database = out / "database.db"
        if database.exists():
            database.unlink()  # Reste eines abgebrochenen Laufs
        rig = read_rig(ctx)
        masks = self._masks_dir(ctx, rig)
        timings: dict[str, float] = {}
        start = time.monotonic()
        ctx.events.progress(0.0, message="Merkmale suchen")
        if rig is None:
            images = ctx.job.stage_dir(SelectStage.dirname) / IMAGES_DIR
            excluded = excluded_images(ctx) if masks is not None else set()
            names = sorted(p.name for p in images.iterdir() if p.name not in excluded)
            colmap.extract_features(
                database,
                images,
                settings.camera_model,
                settings.single_camera,
                masks,
                threads,
                image_names=names if excluded else None,
            )
        else:
            # 360°: eine Lochkamera pro Ansicht mit exakt bekannten Werten, alle Ansichten als starres Rig
            images = images_root(ctx)
            excluded = set()
            names = view_image_names(rig)
            views = rig_views(rig)
            size = int(rig["size"])
            colmap.extract_features(
                database,
                images,
                "PINHOLE",
                False,
                masks,
                threads,
                image_names=names,
                camera_params=[views[0].focal, views[0].focal, size / 2, size / 2],
                per_folder=True,
                max_features=ctx.config.pano.max_features,
            )
            colmap.apply_rig(database, [(f"{v.name}/", v.cam_from_rig()) for v in views])
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
        overlap = settings.sequential_overlap if rig is None else ctx.config.pano.matching_overlap
        colmap.match_sequential(
            database, overlap, vocab, settings.loop_detection_period, threads, expand_rig=rig is not None
        )
        timings["matching_s"] = time.monotonic() - start
        ctx.cancel.raise_if_cancelled()

        num_images = len(names)
        start = time.monotonic()
        model_dir, registered, mapper_used = self._map(ctx, database, images, out, num_images, rig)
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

        geometry = self._check_geometry(ctx, model_dir)

        ctx.events.progress(0.9, message="Bilder entzerren")
        dataset = out / DATASET_DIR
        if dataset.exists():
            shutil.rmtree(dataset)
        colmap.undistort(model_dir, images, dataset, threads)
        if masks is not None:
            ctx.events.progress(0.95, message="Masken entzerren")
            undistort_masks(colmap, model_dir, masks, dataset, out / MASK_UNDISTORT_DIR, threads)
        if rig is not None:
            flatten_dataset(colmap, dataset)
        info = {
            "registered_images": registered,
            "total_images": num_images,
            "registered_ratio": round(ratio, 3),
            "mapper": mapper_used,
            "points3d": colmap.read_model(model_dir).num_points3D(),
            **geometry,
            "masked": masks is not None,
            "rig_views": len(rig["views"]) if rig is not None else 0,
            "excluded_images": len(excluded),
            **{k: round(v, 1) for k, v in timings.items()},
        }
        write_json(out / SFM_FILE, info)
        ctx.events.log(f"{registered} von {num_images} Bildern verortet ({mapper_used})")
        return info

    def _check_geometry(self, ctx: StageContext, model_dir: Path) -> dict[str, Any]:
        """Warnt, wenn sich die Kamera kaum bewegt hat, und richtet die Szene waagrecht aus."""
        settings = ctx.config.sfm
        colmap = ctx.tools.colmap
        rotations, centers, points = colmap.model_geometry(model_dir)
        ratio = baseline_ratio(centers, points)
        info: dict[str, Any] = {"baseline_ratio": round(ratio, 3), "oriented": False}
        if ratio < settings.min_baseline_ratio:
            ctx.warn(
                "Die Kamera hat sich kaum von der Stelle bewegt (nur gedreht oder geschwenkt). Ohne "
                "Bewegung fehlt die Tiefe; der Splat wird wahrscheinlich verzerrt oder unbrauchbar.",
                "Beim Filmen mit der Kamera um das Motiv herumgehen statt sich auf der Stelle zu drehen.",
            )
        if settings.orient_scene and rotations:
            o = orientation(rotations, points)
            colmap.transform_model(model_dir, o.rotation, o.translation)
            info["oriented"] = True
            info["tilt_corrected_deg"] = round(o.tilt_deg, 1)
            ctx.events.log(f"Szene waagrecht ausgerichtet (lag um {o.tilt_deg:.0f}° schief)")
        return info

    def _masks_dir(self, ctx: StageContext, rig: dict[str, Any] | None) -> Path | None:
        """Masken der Stufe 5, falls die Maskierung für diesen Job an ist und fertig gelaufen ist. Bei 360°
        sonst die festen Masken der Stufe 4 (Nadir, Rand des Fisheye-Bildkreises)."""
        if ctx.config.masking and ctx.job.is_done(MaskStage.dirname):
            candidate = ctx.job.stage_dir(MaskStage.dirname) / MASKS_DIR
            if candidate.is_dir():
                return candidate
        if rig is not None:
            static = ctx.job.stage_dir(PanoStage.dirname) / MASKS_DIR
            return static if static.is_dir() else None
        return None

    def _map(
        self,
        ctx: StageContext,
        database: Path,
        images: Path,
        out: Path,
        num_images: int,
        rig: dict[str, Any] | None = None,
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
                # 360°-Ansichten: Brennweiten exakt bekannt; Lage der Ansichten zueinander nur bei
                # Dual-Fisheye verfeinern (Objektive zueinander), bei equirektangulär ist sie exakt
                rig_options = (
                    {"fixed_intrinsics": True, "refine_rig": bool(rig["refine_sensor_from_rig"])}
                    if rig is not None
                    else {}
                )
                if mapper == Mapper.GLOBAL:
                    models = colmap.map_global(database, images, sparse, threads, **rig_options)
                else:
                    models = colmap.map_incremental(database, images, sparse, threads, **rig_options)
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
    root = dataset / IMAGES_DIR
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    available: list[str] = []
    for name in names:
        mask = read_image(masks / f"{name}.png", cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        # Verlustfrei als PNG speichern; COLMAP erkennt das Format am Inhalt, nicht an der Endung.
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            continue
        (renamed / name).parent.mkdir(parents=True, exist_ok=True)
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
        out_name = Path(name).parent / f"{Path(name).stem}.png"
        write_image(target / out_name, ((mask > 127) * 255).astype("uint8"))
    shutil.rmtree(work)
    return len(available)


def flatten_dataset(colmap: ColmapAdapter, dataset: Path) -> int:
    """Legt die Bilder der 360°-Ansichten flach ab: ``v00/c00_000001.jpg`` → ``v00_c00_000001.jpg``.

    Brush 0.3 findet Masken in Unterordnern nicht, und gleichnamige Frames verschiedener Ansichten würden
    sich bei ``masks/<stem>.png`` gegenseitig überschreiben. Gibt die Anzahl umbenannter Bilder zurück.
    """
    images = dataset / IMAGES_DIR
    masks = dataset / MASKS_DIR
    mapping = {
        p.relative_to(images).as_posix(): p.relative_to(images).as_posix().replace("/", "_")
        for p in images.rglob("*")
        if p.is_file() and p.parent != images
    }
    for old, new in mapping.items():
        (images / old).rename(images / new)
        mask = masks / Path(old).parent / f"{Path(old).stem}.png"
        if mask.is_file():
            mask.rename(masks / f"{Path(new).stem}.png")
    for folder in sorted({*images.rglob("*"), *masks.rglob("*")}, reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    if mapping:
        colmap.rename_images(dataset / "sparse", mapping)
    return len(mapping)
