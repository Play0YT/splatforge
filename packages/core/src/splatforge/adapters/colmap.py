"""Adapter für COLMAP über pycolmap.

pycolmap bringt COLMAP als Python-Erweiterung mit; eine separate Installation ist nicht nötig.
Seit COLMAP 4 ist GLOMAP als ``global_mapping`` enthalten.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from ..errors import ToolMissingError
from .base import Version, format_version, parse_version

MIN_PYCOLMAP: Version = (4, 2)


@dataclass
class DatabaseStats:
    num_images: int
    mean_keypoints: float
    # Inlier-Matches zwischen zeitlich benachbarten Bildern (nach Bildname sortiert)
    neighbour_inliers: list[int] = field(default_factory=list)


class ColmapAdapter:
    name = "COLMAP (pycolmap)"

    def __init__(self) -> None:
        self._module: ModuleType | None = None
        #: COLMAP schreibt sein Log direkt auf stderr. Ist eine Datei gesetzt, landet es dort.
        self.log_file: Path | None = None

    @contextmanager
    def _native_log(self) -> Iterator[None]:
        """Leitet stderr auf Betriebssystem-Ebene in ``log_file`` um (auch die Ausgabe von C++)."""
        if self.log_file is None:
            yield
            return
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.flush()
        saved = os.dup(2)
        with self.log_file.open("ab") as fh:
            os.dup2(fh.fileno(), 2)
            try:
                yield
            finally:
                sys.stderr.flush()
                os.dup2(saved, 2)
                os.close(saved)

    @property
    def pycolmap(self) -> ModuleType:
        if self._module is None:
            try:
                module = importlib.import_module("pycolmap")
            except ImportError as exc:
                raise ToolMissingError(
                    "COLMAP (pycolmap) ist nicht installiert.",
                    "SplatForge neu installieren; pycolmap gehört zu den Abhängigkeiten.",
                    details=str(exc),
                ) from exc
            self._module = module
        return self._module

    def check(self) -> Version | None:
        found = parse_version(getattr(self.pycolmap, "__version__", ""))
        if found is not None and found < MIN_PYCOLMAP:
            raise ToolMissingError(
                f"pycolmap {format_version(found)} ist zu alt, benötigt wird {format_version(MIN_PYCOLMAP)}.",
                "SplatForge neu installieren, um die passende Version zu erhalten.",
            )
        return found

    def has_global_mapper(self) -> bool:
        return hasattr(self.pycolmap, "global_mapping")

    def extract_features(
        self,
        database: Path,
        images: Path,
        camera_model: str,
        single_camera: bool,
        masks: Path | None,
        num_threads: int,
        image_names: list[str] | None = None,
    ) -> None:
        """``image_names`` beschränkt die Suche auf diese Bilder (leer oder ``None`` = alle)."""
        pc = self.pycolmap
        reader = pc.ImageReaderOptions()
        reader.camera_model = camera_model
        if masks is not None:
            reader.mask_path = str(masks)
        extraction = pc.FeatureExtractionOptions()
        extraction.num_threads = num_threads or -1
        mode = pc.CameraMode.SINGLE if single_camera else pc.CameraMode.AUTO
        with self._native_log():
            pc.extract_features(
                database,
                images,
                image_names=image_names or [],
                camera_mode=mode,
                reader_options=reader,
                extraction_options=extraction,
            )

    def match_sequential(
        self, database: Path, overlap: int, vocab_tree: Path | None, loop_period: int, num_threads: int
    ) -> None:
        pc = self.pycolmap
        pairing = pc.SequentialPairingOptions()
        pairing.overlap = overlap
        pairing.num_threads = num_threads or -1
        if vocab_tree is not None:
            pairing.loop_detection = True
            pairing.loop_detection_period = loop_period
            pairing.vocab_tree_path = str(vocab_tree)
        matching = pc.FeatureMatchingOptions()
        matching.num_threads = num_threads or -1
        with self._native_log():
            pc.match_sequential(database, matching_options=matching, pairing_options=pairing)

    def map_incremental(self, database: Path, images: Path, output: Path, num_threads: int) -> dict[int, Any]:
        pc = self.pycolmap
        options = pc.IncrementalPipelineOptions()
        options.num_threads = num_threads or -1
        with self._native_log():
            result: dict[int, Any] = pc.incremental_mapping(database, images, output, options=options)
        return result

    def map_global(self, database: Path, images: Path, output: Path, num_threads: int) -> dict[int, Any]:
        pc = self.pycolmap
        options = pc.GlobalPipelineOptions()
        options.num_threads = num_threads or -1
        with self._native_log():
            result: dict[int, Any] = pc.global_mapping(database, images, output, options=options)
        return result

    def undistort(
        self,
        model: Path,
        images: Path,
        output: Path,
        num_threads: int,
        image_names: list[str] | None = None,
        jpeg_quality: int = -1,
    ) -> None:
        with self._native_log():
            self.pycolmap.undistort_images(
                output,
                model,
                images,
                image_names=image_names or [],
                output_type="COLMAP",
                jpeg_quality=jpeg_quality,
                num_threads=num_threads or -1,
            )

    def read_model(self, path: Path) -> Any:
        return self.pycolmap.Reconstruction(path)

    def database_stats(self, database: Path) -> DatabaseStats:
        pc = self.pycolmap
        db = pc.Database.open(database) if hasattr(pc.Database, "open") else pc.Database(database)
        try:
            images = sorted(db.read_all_images(), key=lambda im: im.name)
            keypoints = [db.num_keypoints_for_image(im.image_id) for im in images]
            neighbours: list[int] = []
            for a, b in zip(images, images[1:], strict=False):
                try:
                    neighbours.append(int(db.read_two_view_geometry_num_inliers(a.image_id, b.image_id)))
                except Exception:  # Paar wurde nicht verifiziert
                    neighbours.append(0)
        finally:
            db.close()
        mean_kp = sum(keypoints) / len(keypoints) if keypoints else 0.0
        return DatabaseStats(num_images=len(images), mean_keypoints=mean_kp, neighbour_inliers=neighbours)
