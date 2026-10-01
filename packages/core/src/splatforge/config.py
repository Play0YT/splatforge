"""Job-Konfiguration und Einstellungen (Pydantic).

Alle Schwellenwerte der Pipeline stehen hier als Felder mit Standardwert, damit im übrigen Code
keine magischen Zahlen vorkommen. Die Modelle sind gleichzeitig die Quelle für das JSON-Schema in
``packages/job-schema``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# Version des Formats von job.json. Bei jeder inkompatiblen Änderung erhöhen und in
# migrations.py eine Migration ergänzen, damit bestehende Job-Ordner weiter funktionieren.
JOB_SCHEMA_VERSION = 1


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Preset(StrEnum):
    PREVIEW = "preview"
    STANDARD = "standard"
    HIGH = "high"


class CameraType(StrEnum):
    AUTO = "auto"
    PERSPECTIVE = "perspective"
    EQUIRECTANGULAR = "equirectangular"
    DUAL_FISHEYE = "dual_fisheye"


class TrainBackend(StrEnum):
    AUTO = "auto"
    BRUSH = "brush"
    CPU = "cpu"


class Mapper(StrEnum):
    AUTO = "auto"
    GLOBAL = "global"
    INCREMENTAL = "incremental"


class PresetValues(_Model):
    frames: int = Field(gt=0, description="Zielanzahl ausgewählter Frames")
    max_image_edge: int = Field(gt=0, description="Maximale Bildkante in Pixeln")
    iterations: int = Field(gt=0, description="Trainings-Iterationen")


# Startwerte aus der Spezifikation. Messwerte und Anpassungen: docs/measurements.md
PRESETS: dict[Preset, PresetValues] = {
    Preset.PREVIEW: PresetValues(frames=120, max_image_edge=1280, iterations=7000),
    Preset.STANDARD: PresetValues(frames=300, max_image_edge=1600, iterations=30000),
    Preset.HIGH: PresetValues(frames=600, max_image_edge=2560, iterations=50000),
}


class ToolSettings(_Model):
    """Pfade zu externen Programmen. Leer bedeutet: im Suchpfad (PATH) suchen."""

    ffmpeg: Path | None = None
    ffprobe: Path | None = None
    brush: Path | None = None
    probe_timeout_s: float = Field(default=60.0, gt=0)


class ExtractSettings(_Model):
    candidate_factor: float = Field(
        default=3.0, ge=1.0, description="So viele Kandidaten pro Ziel-Frame extrahieren"
    )
    jpeg_quality: int = Field(default=2, ge=1, le=31, description="FFmpeg -q:v (1 = beste Qualität)")
    tonemap_hdr: bool = True


class SelectSettings(_Model):
    analysis_edge: int = Field(default=640, gt=0, description="Bildkante für Schärfe- und Flussanalyse")
    blur_relative_threshold: float = Field(
        default=0.35, ge=0, le=1, description="Frame gilt als unscharf unter diesem Anteil des Medians"
    )
    min_total_motion: float = Field(
        default=0.05, ge=0, description="Mindest-Gesamtbewegung (Anteil der Bilddiagonale)"
    )
    max_flow_features: int = Field(default=400, gt=0)
    min_selected_frames: int = Field(default=20, gt=1)


class SfmSettings(_Model):
    mapper: Mapper = Mapper.AUTO
    camera_model: str = "OPENCV"
    single_camera: bool = True
    sequential_overlap: int = Field(default=10, gt=0)
    vocab_tree_path: Path | None = Field(
        default=None, description="Vocab-Tree für Loop-Detection; ohne Datei ist sie aus"
    )
    loop_detection_period: int = Field(default=10, gt=0)
    min_registered_ratio: float = Field(default=0.6, gt=0, le=1)
    low_texture_features: int = Field(
        default=300, gt=0, description="Unter so vielen Merkmalen pro Bild: zu wenig Textur"
    )
    weak_pair_inliers: int = Field(
        default=30, gt=0, description="Unter so vielen Inliern zwischen Nachbarbildern: zu wenig Überlappung"
    )


class TrainSettings(_Model):
    backend: TrainBackend = TrainBackend.AUTO
    checkpoint_every: int = Field(default=2000, gt=0)
    sh_degree: int = Field(default=3, ge=0, le=3)
    eval_split_every: int = Field(
        default=8, ge=0, description="Jedes n-te Bild zur Qualitätsmessung, 0 = aus"
    )
    # Nur für das CPU-Backend
    cpu_max_image_edge: int = Field(default=800, gt=0)
    cpu_max_gaussians: int = Field(default=300_000, gt=0)
    cpu_tile_size: int = Field(default=8, gt=0)
    cpu_densify_from: int = Field(default=500, ge=0)
    cpu_densify_until_fraction: float = Field(default=0.5, gt=0, le=1)
    cpu_densify_every: int = Field(default=100, gt=0)
    cpu_densify_grad_threshold: float = Field(default=0.0002, gt=0)
    cpu_opacity_reset_every: int = Field(default=3000, gt=0)
    cpu_prune_opacity: float = Field(default=0.005, gt=0)
    cpu_ssim_weight: float = Field(default=0.2, ge=0, le=1)
    cpu_log_every: int = Field(default=50, gt=0)


class ExportSettings(_Model):
    write_spz: bool = Field(default=False, description="Ab Meilenstein 7")


class ResourceSettings(_Model):
    num_threads: int = Field(default=0, ge=0, description="0 = alle Kerne")
    min_free_disk_mb: int = Field(default=2048, ge=0)


class JobConfig(_Model):
    """Vollständige Beschreibung eines Jobs. Wird als job.json im Job-Ordner gespeichert."""

    schema_version: int = JOB_SCHEMA_VERSION
    inputs: list[Path] = Field(min_length=1, description="Videodateien oder ein Bildordner")
    preset: Preset = Preset.STANDARD
    frames: int | None = Field(default=None, gt=0, description="Überschreibt das Preset")
    max_image_edge: int | None = Field(default=None, gt=0, description="Überschreibt das Preset")
    iterations: int | None = Field(default=None, gt=0, description="Überschreibt das Preset")
    camera_type: CameraType = CameraType.AUTO
    masking: bool = Field(default=False, description="Personenmaskierung (ab Meilenstein 2)")
    tools: ToolSettings = ToolSettings()
    extract: ExtractSettings = ExtractSettings()
    select: SelectSettings = SelectSettings()
    sfm: SfmSettings = SfmSettings()
    train: TrainSettings = TrainSettings()
    export: ExportSettings = ExportSettings()
    resources: ResourceSettings = ResourceSettings()

    def effective(self) -> PresetValues:
        """Preset-Werte mit den Überschreibungen dieses Jobs."""
        base = PRESETS[self.preset]
        return PresetValues(
            frames=self.frames or base.frames,
            max_image_edge=self.max_image_edge or base.max_image_edge,
            iterations=self.iterations or base.iterations,
        )
