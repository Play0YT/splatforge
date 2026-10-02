"""Stufe 1: Analyse der Eingabe mit ffprobe und Erkennung des Kameratyps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..config import CameraType
from ..errors import UnsupportedInputError
from ..imageio import read_image
from ..insv import is_preview_file, original_files, partner_file, read_metadata
from .base import Stage, StageContext, write_json

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".insv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
ANALYSIS_FILE = "analysis.json"


@dataclass
class InputInfo:
    path: str
    kind: str  # "video" oder "images"
    camera_type: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    frame_count: int = 0
    rotation: int = 0
    hdr: bool = False
    bit_depth: int = 8
    codec: str = ""
    video_streams: int = 0
    images: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Nur bei Dual-Fisheye: wo die beiden Objektive liegen (siehe LensSource)
    lens_layout: str = ""
    lenses: list[dict[str, Any]] = field(default_factory=list)
    insv: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputInfo:
        # Unbekannte Schlüssel (z. B. aus einer neueren Version) ignorieren, fehlende haben Standardwerte
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class LensLayout(StrEnum):
    TWO_STREAMS = "two_streams"  # zwei Videospuren in einer Datei
    SIDE_BY_SIDE = "side_by_side"  # beide Objektive nebeneinander in einer Spur
    SPLIT_FILES = "split_files"  # eine Datei pro Objektiv (…_00_… und …_10_…)


LAYOUT_NOTES = {
    LensLayout.TWO_STREAMS: "Beide Objektive als getrennte Videospuren in einer Datei",
    LensLayout.SIDE_BY_SIDE: "Beide Objektive nebeneinander in einer Videospur",
    LensLayout.SPLIT_FILES: "Ein Objektiv pro Datei",
}
# Ab diesem Seitenverhältnis liegen bei einer einzigen Spur beide Objektive nebeneinander
SIDE_BY_SIDE_MIN_ASPECT = 1.8


def detect_lens_layout(path: Path, streams: list[dict[str, Any]]) -> tuple[LensLayout, Path | None]:
    main = streams[0]
    size = (main.get("width"), main.get("height"))
    if len(streams) >= 2 and (streams[1].get("width"), streams[1].get("height")) == size:
        return LensLayout.TWO_STREAMS, None
    width, height = int(main.get("width", 0)), int(main.get("height", 0))
    if height and width / height >= SIDE_BY_SIDE_MIN_ASPECT:
        return LensLayout.SIDE_BY_SIDE, None
    return LensLayout.SPLIT_FILES, partner_file(path)


def lens_sources(path: Path, layout: LensLayout, partner: Path | None) -> list[dict[str, Any]]:
    """Woher die Bilder der beiden Objektive kommen: Datei, Videospur und gegebenenfalls Bildhälfte."""
    if layout == LensLayout.TWO_STREAMS:
        return [
            {"path": str(path), "stream": 0, "crop": None},
            {"path": str(path), "stream": 1, "crop": None},
        ]
    if layout == LensLayout.SIDE_BY_SIDE:
        return [
            {"path": str(path), "stream": 0, "crop": "left"},
            {"path": str(path), "stream": 0, "crop": "right"},
        ]
    sources = [{"path": str(path), "stream": 0, "crop": None}]
    if partner is not None:
        sources.append({"path": str(partner), "stream": 0, "crop": None})
    # Feste Reihenfolge, egal welche Datei angegeben wurde: …_00_… ist Objektiv 1, …_10_… Objektiv 2
    return sorted(sources, key=lambda s: Path(s["path"]).name)


def _rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(value)


def _rotation(stream: dict[str, Any]) -> int:
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            return int(round(float(side["rotation"]))) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    return int(tag) % 360 if tag else 0


def _bit_depth(stream: dict[str, Any]) -> int:
    raw = stream.get("bits_per_raw_sample")
    if raw and str(raw).isdigit():
        return int(raw)
    pix = stream.get("pix_fmt", "")
    for depth in (16, 12, 10):
        if f"p{depth}" in pix:
            return depth
    return 8


def _is_spherical(stream: dict[str, Any], format_tags: dict[str, Any]) -> bool:
    for side in stream.get("side_data_list", []) or []:
        if "spherical" in str(side.get("side_data_type", "")).lower():
            return True
    tags = {**format_tags, **(stream.get("tags") or {})}
    return any("spherical" in str(k).lower() or "equirect" in str(v).lower() for k, v in tags.items())


def detect_camera_type(path: Path, probe: dict[str, Any]) -> tuple[CameraType, list[str]]:
    """Leitet den Kameratyp aus ffprobe-Daten ab. Gibt den Typ und Hinweise zur Erkennung zurück."""
    notes: list[str] = []
    streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    # Vorschaubilder (attached_pic) sind keine echten Videospuren
    streams = [s for s in streams if not (s.get("disposition") or {}).get("attached_pic")]
    if not streams:
        return CameraType.PERSPECTIVE, notes
    format_tags = (probe.get("format") or {}).get("tags") or {}
    main = streams[0]
    width, height = int(main.get("width", 0)), int(main.get("height", 0))

    if path.suffix.lower() == ".insv":
        notes.append("Insta360-Datei: zwei Fisheye-Objektive")
        return CameraType.DUAL_FISHEYE, notes
    if len(streams) >= 2 and all(
        (s.get("width"), s.get("height")) == (main.get("width"), main.get("height")) for s in streams[:2]
    ):
        notes.append("Zwei gleich grosse Videospuren: vermutlich Dual-Fisheye")
        return CameraType.DUAL_FISHEYE, notes
    if _is_spherical(main, format_tags):
        notes.append("Metadaten kennzeichnen das Video als 360°")
        return CameraType.EQUIRECTANGULAR, notes
    if height and width == 2 * height:
        notes.append("Seitenverhältnis 2:1 ohne 360°-Metadaten: als equirektangulär behandelt")
        return CameraType.EQUIRECTANGULAR, notes
    return CameraType.PERSPECTIVE, notes


class AnalyzeStage(Stage):
    name = "analyze"
    dirname = "01_analyze"
    weight = 0.5

    def preflight(self, ctx: StageContext) -> None:
        if any(Path(p).is_file() for p in ctx.config.inputs):
            ctx.tools.ffprobe.check()

    def inspect(self, ctx: StageContext) -> list[InputInfo]:
        """Untersucht alle Eingaben, ohne etwas abzulehnen (auch für ``splatforge analyze``)."""
        return [self._analyze_input(ctx, Path(p)) for p in ctx.config.inputs]

    def run(self, ctx: StageContext) -> dict[str, Any]:
        infos = self.inspect(ctx)
        kinds = {i.kind for i in infos}
        if len(kinds) > 1:
            raise UnsupportedInputError(
                "Videos und Bildordner können nicht im selben Job gemischt werden.",
                "Entweder nur Videos oder nur einen Bildordner wählen.",
            )
        for info in infos:
            ctx.events.log(
                f"{Path(info.path).name}: {info.kind}, {info.camera_type}, {info.width}×{info.height}"
                + (f", {info.duration_s:.1f} s, {info.fps:.2f} fps" if info.kind == "video" else "")
                + (", HDR" if info.hdr else "")
            )
            for note in info.notes:
                ctx.events.log(note)
            if info.camera_type != CameraType.PERSPECTIVE:
                raise UnsupportedInputError(
                    f"{Path(info.path).name} ist eine 360°-Aufnahme ({info.camera_type}). Die Datei wird "
                    "erkannt, die Splat-Berechnung daraus folgt aber erst in einer späteren Version.",
                    "Mit 'splatforge frames' lassen sich schon die Einzelbilder beider Objektive "
                    "ansehen. Falls die Erkennung falsch ist, den Kameratyp auf 'perspective' setzen.",
                )
        out = self.out_dir(ctx)
        write_json(out / ANALYSIS_FILE, {"inputs": [asdict(i) for i in infos]})
        return {"inputs": len(infos)}

    def _analyze_input(self, ctx: StageContext, path: Path) -> InputInfo:
        if not path.exists():
            raise UnsupportedInputError(
                f"Die Eingabe {path} wurde nicht gefunden.", "Den Pfad prüfen und erneut starten."
            )
        if path.is_dir():
            return self._analyze_folder(ctx, path)
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise UnsupportedInputError(
                f"Das Dateiformat {path.suffix or '(ohne Endung)'} wird nicht unterstützt.",
                "Unterstützt werden " + ", ".join(sorted(VIDEO_EXTENSIONS)) + " und Bildordner.",
            )
        probe = ctx.tools.ffprobe.probe(path, ctx.config.tools.probe_timeout_s)
        streams = [
            s
            for s in probe.get("streams", [])
            if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
        ]
        if not streams:
            raise UnsupportedInputError(
                f"{path.name} enthält keine Videospur.", "Eine Videodatei mit Bildinhalt wählen."
            )
        main = streams[0]
        camera_type, notes = detect_camera_type(path, probe)
        if ctx.config.camera_type != CameraType.AUTO:
            notes.append(f"Kameratyp manuell gesetzt: {ctx.config.camera_type}")
            camera_type = ctx.config.camera_type
        rotation = _rotation(main)
        width, height = int(main.get("width", 0)), int(main.get("height", 0))
        if rotation in (90, 270):
            width, height = height, width
        fps = _rate(main.get("avg_frame_rate")) or _rate(main.get("r_frame_rate"))
        duration = float(main.get("duration") or (probe.get("format") or {}).get("duration") or 0)
        frames = int(main.get("nb_frames") or 0) or int(round(duration * fps))
        if duration <= 0 or fps <= 0:
            raise UnsupportedInputError(
                f"Dauer oder Bildrate von {path.name} konnte nicht bestimmt werden.",
                "Die Datei in ein gängiges Format (z. B. MP4 mit H.264) umwandeln.",
            )
        lens_layout = ""
        lenses: list[dict[str, Any]] = []
        insv_meta: dict[str, Any] | None = None
        if path.suffix.lower() == ".insv":
            meta = read_metadata(path)
            meta_partner = partner_file(path) if meta is None else None
            if meta_partner is not None:
                # Bei einer Datei pro Objektiv stehen die Metadaten nur in einer der beiden (meist _00_)
                meta = read_metadata(meta_partner)
                if meta is not None:
                    notes.append(f"Insta360-Metadaten aus der Partnerdatei {meta_partner.name} gelesen")
            if meta is not None:
                insv_meta = meta.to_dict()
                if meta.camera_type:
                    notes.append(f"Kamera: {meta.camera_type} (Firmware {meta.fw_version or '?'})")
                notes.append(
                    "Objektiv-Kalibrierung in der Datei gefunden"
                    if meta.calibration
                    else "Keine Objektiv-Kalibrierung in der Datei, es werden Standardwerte verwendet"
                )
            else:
                notes.append("Keine Insta360-Metadaten gefunden (Datei evtl. bearbeitet oder exportiert)")
        if camera_type == CameraType.DUAL_FISHEYE:
            layout, partner = detect_lens_layout(path, streams)
            lens_layout, lenses = str(layout), lens_sources(path, layout, partner)
            notes.append(LAYOUT_NOTES[layout])
            if layout == LensLayout.SPLIT_FILES and partner is None:
                notes.append("Die Datei des zweiten Objektivs (…_00_… bzw. …_10_…) fehlt im selben Ordner")
            if layout == LensLayout.SIDE_BY_SIDE:
                width //= 2
        if is_preview_file(path):
            notes.append(
                "Das ist eine Vorschaudatei (LRV) in niedriger Auflösung, für einen Splat ungeeignet. "
                "Die Originale beginnen mit VID_."
            )
            originals = original_files(path)
            if originals:
                notes.append("Originale im selben Ordner: " + ", ".join(p.name for p in originals))
            else:
                notes.append("Die Originaldateien (VID_…) liegen nicht im selben Ordner")
        return InputInfo(
            path=str(path),
            kind="video",
            camera_type=str(camera_type),
            width=width,
            height=height,
            fps=fps,
            duration_s=duration,
            frame_count=frames,
            rotation=rotation,
            hdr=main.get("color_transfer") in HDR_TRANSFERS,
            bit_depth=_bit_depth(main),
            codec=str(main.get("codec_name", "")),
            video_streams=len(streams),
            notes=notes,
            lens_layout=lens_layout,
            lenses=lenses,
            insv=insv_meta,
        )

    def _analyze_folder(self, ctx: StageContext, path: Path) -> InputInfo:
        images = sorted(p.name for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if len(images) < ctx.config.select.min_selected_frames:
            raise UnsupportedInputError(
                f"Im Ordner {path.name} liegen nur {len(images)} Bilder.",
                f"Mindestens {ctx.config.select.min_selected_frames} Bilder mit guter Überlappung verwenden.",
            )
        first = read_image(path / images[0])
        height, width = (first.shape[:2]) if first is not None else (0, 0)
        camera_type = (
            ctx.config.camera_type if ctx.config.camera_type != CameraType.AUTO else CameraType.PERSPECTIVE
        )
        return InputInfo(
            path=str(path),
            kind="images",
            camera_type=str(camera_type),
            width=int(width),
            height=int(height),
            frame_count=len(images),
            images=images,
        )
