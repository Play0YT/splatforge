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
        camera_params: list[float] | None = None,
        per_folder: bool = False,
        max_features: int | None = None,
    ) -> None:
        """``image_names`` beschränkt die Suche auf diese Bilder (leer oder ``None`` = alle).

        ``camera_params`` setzt bekannte Kameraparameter (z. B. bei 360°-Ansichten), ``per_folder`` legt
        eine Kamera pro Unterordner an (eine pro Ansicht im Rig).
        """
        pc = self.pycolmap
        reader = pc.ImageReaderOptions()
        reader.camera_model = camera_model
        if camera_params is not None:
            reader.camera_params = ",".join(f"{v:.10g}" for v in camera_params)
        if masks is not None:
            reader.mask_path = str(masks)
        extraction = pc.FeatureExtractionOptions()
        extraction.num_threads = num_threads or -1
        if max_features is not None:
            extraction.sift.max_num_features = max_features
        if per_folder:
            mode = pc.CameraMode.PER_FOLDER
        else:
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

    def apply_rig(self, database: Path, sensors: list[tuple[str, Any]]) -> None:
        """Legt ein starres Rig an. ``sensors``: (Bildpräfix, Drehung Kamera ← Rig) je Kamera; die erste
        ist die Referenz. Bilder mit gleichem Namen nach dem Präfix gehören zum selben Zeitpunkt."""
        import numpy as np

        pc = self.pycolmap
        cameras = []
        for i, (prefix, rotation) in enumerate(sensors):
            cam = pc.RigConfigCamera(ref_sensor=i == 0, image_prefix=prefix)
            if i:
                cam.cam_from_rig = pc.Rigid3d(pc.Rotation3d(np.asarray(rotation)), np.zeros(3))
            cameras.append(cam)
        db = pc.Database.open(database)
        try:
            with self._native_log():
                pc.apply_rig_config([pc.RigConfig(cameras=cameras)], db)
        finally:
            db.close()

    def match_sequential(
        self,
        database: Path,
        overlap: int,
        vocab_tree: Path | None,
        loop_period: int,
        num_threads: int,
        expand_rig: bool = False,
    ) -> None:
        pc = self.pycolmap
        pairing = pc.SequentialPairingOptions()
        pairing.overlap = overlap
        pairing.num_threads = num_threads or -1
        # Bei Rigs: alle Ansichten benachbarter Zeitpunkte miteinander vergleichen
        pairing.expand_rig_images = expand_rig
        if vocab_tree is not None:
            pairing.loop_detection = True
            pairing.loop_detection_period = loop_period
            pairing.vocab_tree_path = str(vocab_tree)
        matching = pc.FeatureMatchingOptions()
        matching.num_threads = num_threads or -1
        with self._native_log():
            pc.match_sequential(database, matching_options=matching, pairing_options=pairing)

    def map_incremental(
        self,
        database: Path,
        images: Path,
        output: Path,
        num_threads: int,
        fixed_intrinsics: bool = False,
        refine_rig: bool = True,
    ) -> dict[int, Any]:
        pc = self.pycolmap
        options = pc.IncrementalPipelineOptions()
        options.num_threads = num_threads or -1
        if fixed_intrinsics:
            options.ba_refine_focal_length = False
            options.ba_refine_principal_point = False
            options.ba_refine_extra_params = False
        options.ba_refine_sensor_from_rig = refine_rig
        with self._native_log():
            result: dict[int, Any] = pc.incremental_mapping(database, images, output, options=options)
        return result

    def map_global(
        self,
        database: Path,
        images: Path,
        output: Path,
        num_threads: int,
        fixed_intrinsics: bool = False,
        refine_rig: bool = True,
    ) -> dict[int, Any]:
        pc = self.pycolmap
        options = pc.GlobalPipelineOptions()
        options.num_threads = num_threads or -1
        ba = options.mapper.bundle_adjustment
        if fixed_intrinsics:
            # Sonst „verbessert“ COLMAP die exakt bekannten Brennweiten der 360°-Ansichten und verzieht sie
            ba.refine_focal_length = False
            ba.refine_principal_point = False
            ba.refine_extra_params = False
        ba.refine_sensor_from_rig = refine_rig
        options.mapper.refine_sensor_from_rig = refine_rig
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

    def rename_images(self, path: Path, mapping: dict[str, str]) -> None:
        """Benennt Bilder in einem Modell um (an Ort und Stelle)."""
        rec = self.read_model(path)
        for image in rec.images.values():
            if image.name in mapping:
                image.name = mapping[image.name]
        rec.write(path)

    def read_model(self, path: Path) -> Any:
        return self.pycolmap.Reconstruction(path)

    def model_geometry(self, path: Path) -> tuple[list[Any], Any, Any]:
        """Kameradrehungen (Welt → Kamera), Kamerapositionen und 3D-Punkte eines Modells."""
        import numpy as np

        rec = self.read_model(path)
        images = [im for im in rec.images.values() if im.has_pose]
        rotations = [np.asarray(im.cam_from_world().rotation.matrix()) for im in images]
        centers = np.array([im.projection_center() for im in images], dtype=np.float64).reshape(-1, 3)
        points = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64).reshape(-1, 3)
        return rotations, centers, points

    def transform_model(self, path: Path, rotation: Any, translation: Any) -> None:
        """Dreht und verschiebt ein Modell an Ort und Stelle: neue Welt = Drehung · alte + Verschiebung."""
        pc = self.pycolmap
        rec = self.read_model(path)
        rec.transform(pc.Sim3d(1.0, pc.Rotation3d(rotation), translation))
        rec.write(path)

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
