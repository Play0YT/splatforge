"""Personenmaskierung: Verfolgung, Masken, Qualitätskontrolle, Modell-Download und Entzerrung."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from splatforge import models
from splatforge.adapters.base import CancelToken
from splatforge.cli import _load_config, _parser
from splatforge.config import JobConfig, MaskMethod, MaskModel, MaskSettings
from splatforge.errors import JobCancelledError, SplatForgeError
from splatforge.imageio import read_image, write_image
from splatforge.masking.postprocess import expand
from splatforge.masking.run import compute_masks
from splatforge.masking.tracking import Detection, build_tracks, detections_per_frame, fill_gaps
from splatforge.masking.worker import resolve


def _det(x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(np.array([x1, y1, x2, y2], dtype=np.float64), 0.9, 0)


def test_gap_is_interpolated() -> None:
    per_frame = [[_det(0, 0, 10, 10)], [], [_det(4, 0, 14, 10)]]
    tracks = build_tracks(per_frame, min_iou=0.3, max_gap=3)
    assert len(tracks) == 1
    fill_gaps(tracks, max_gap=3)
    filled = detections_per_frame(tracks, 3)
    assert filled[1] and filled[1][0].interpolated
    np.testing.assert_allclose(filled[1][0].box, [2, 0, 12, 10])


def test_long_gap_is_not_filled() -> None:
    per_frame = [[_det(0, 0, 10, 10)], [], [], [], [_det(0, 0, 10, 10)]]
    tracks = build_tracks(per_frame, min_iou=0.3, max_gap=2)
    fill_gaps(tracks, max_gap=2)
    filled = detections_per_frame(tracks, 5)
    assert not any(filled[1:4])


def test_separate_objects_stay_separate() -> None:
    per_frame = [[_det(0, 0, 10, 10), _det(50, 50, 60, 60)], [_det(1, 0, 11, 10), _det(51, 50, 61, 60)]]
    tracks = build_tracks(per_frame, min_iou=0.3, max_gap=3)
    assert len(tracks) == 2
    assert all(len(t.frames) == 2 for t in tracks)


def test_expand_adds_margin() -> None:
    mask = np.zeros((21, 21), dtype=bool)
    mask[10, 10] = True
    grown = expand(mask, 3)
    assert grown[10, 13] and grown[7, 10]
    assert not grown[10, 15]
    assert expand(mask, 0).sum() == 1


class FakeDetector:
    """Findet ein helles Rechteck; in Bild 2 'übersieht' er es absichtlich."""

    def detect(self, rgb: np.ndarray, classes: tuple[int, ...], threshold: float) -> list[Detection]:
        ys, xs = np.nonzero(rgb[..., 0] > 200)
        if xs.size == 0 or int(rgb[0, 0, 1]) == 2:
            return []
        return [Detection(np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64), 0.9, 0)]


class FakeSegmenter:
    def segment(self, rgb: np.ndarray, boxes: list[np.ndarray]) -> np.ndarray:
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        for x1, y1, x2, y2 in (b.astype(int) for b in boxes):
            mask[y1:y2, x1:x2] = True
        return mask


def _frames(folder: Path, widths: list[int]) -> list[str]:
    """Bilder 100×60 mit einem hellen Rechteck der Breite w (0 = keins). Pixel (0,0) grün = Bildnummer."""
    names = []
    for i, w in enumerate(widths):
        img = np.zeros((60, 100, 3), dtype=np.uint8)
        img[0, 0, 1] = i
        if w:
            img[20:40, 10 : 10 + w, 2] = 255  # BGR: rot
        name = f"frame_{i:04d}.png"  # verlustfrei, damit der Kanal-Trick hält
        write_image(folder / name, img)
        names.append(name)
    return names


def test_compute_masks(tmp_path: Path) -> None:
    images = tmp_path / "images"
    names = _frames(images, [20, 20, 20, 20, 0, 90])
    settings = MaskSettings(margin_fraction=0.0, max_object_fraction=0.25)
    seen: list[float] = []
    summary = compute_masks(
        images, names, tmp_path / "out", settings, (0,), FakeDetector(), FakeSegmenter(),
        lambda f, _msg: seen.append(f), CancelToken(),
    )  # fmt: skip
    # Bild 2 wurde übersehen und per Interpolation ergänzt
    assert summary.interpolated_boxes == 1
    assert summary.frames_with_objects == 5
    assert summary.excluded == [names[5]]  # 90 % der Breite maskiert
    mask = read_image(tmp_path / "out" / "masks" / f"{names[2]}.png", 0)
    assert mask is not None and mask.shape == (60, 100)
    assert mask[30, 15] == 0 and mask[5, 5] == 255  # schwarz = ignorieren
    empty = read_image(tmp_path / "out" / "masks" / f"{names[4]}.png", 0)
    assert empty is not None and empty.min() == 255
    assert (tmp_path / "out" / "overlays" / "frame_0000.jpg").is_file()
    assert seen[-1] == pytest.approx(1.0)
    assert set(summary.to_dict()) >= {"excluded", "mean_masked_fraction", "fractions"}


def test_compute_masks_can_be_cancelled(tmp_path: Path) -> None:
    images = tmp_path / "images"
    names = _frames(images, [20, 20])
    cancel = CancelToken()
    cancel.cancel()
    with pytest.raises(JobCancelledError):
        compute_masks(
            images, names, tmp_path / "out", MaskSettings(), (0,), FakeDetector(), FakeSegmenter(),
            lambda f, m: None, cancel,
        )  # fmt: skip


def test_resolve_picks_model_by_hardware() -> None:
    cpu = ["CPUExecutionProvider"]
    gpu = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert resolve(MaskSettings(), cpu)[1] == MaskModel.TINY
    assert resolve(MaskSettings(), gpu)[1] == MaskModel.SMALL
    assert resolve(MaskSettings(model=MaskModel.SMALL), cpu)[1] == MaskModel.SMALL
    method, _, notes = resolve(MaskSettings(method=MaskMethod.VIDEO), cpu)
    assert method == MaskMethod.IMAGE and notes


def test_masking_is_off_by_default_and_cli_flags() -> None:
    assert JobConfig(inputs=[Path("a.mp4")]).masking is False
    args = _parser().parse_args(
        ["run", "a.mp4", "--out", "x", "--masking", "--mask-model", "sam2.1-small",
         "--mask-classes", "person, animal"]
    )  # fmt: skip
    config = _load_config(args)
    assert config.masking is True
    assert config.mask.model == MaskModel.SMALL
    assert config.mask.classes == ["person", "animal"]
    with pytest.raises(ValueError):
        MaskSettings.model_validate({"classes": ["ufo"]})


def test_old_job_config_without_mask_section_loads() -> None:
    config = JobConfig.model_validate({"inputs": ["a.mp4"], "masking": False})
    assert config.mask == MaskSettings()


def _fake_model(monkeypatch: pytest.MonkeyPatch, source: Path, sha256: str) -> None:
    file = models.ModelFile("model.onnx", source.as_uri(), sha256, source.stat().st_size)
    model = models.Model("test-model", "Test", "MIT", (file,))
    monkeypatch.setitem(models.MODELS, "test-model", model)


def test_model_download_checks_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.onnx"
    source.write_bytes(b"modell" * 1000)
    _fake_model(monkeypatch, source, hashlib.sha256(source.read_bytes()).hexdigest())
    progress: list[tuple[int, int]] = []
    folder = models.ensure_model("test-model", tmp_path / "cache", lambda d, t: progress.append((d, t)))
    assert (folder / "model.onnx").read_bytes() == source.read_bytes()
    assert progress[-1] == (6000, 6000)
    # Zweiter Aufruf lädt nichts mehr
    progress.clear()
    models.ensure_model("test-model", tmp_path / "cache", lambda d, t: progress.append((d, t)))
    assert progress == []


def test_model_download_rejects_wrong_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.onnx"
    source.write_bytes(b"kaputt")
    _fake_model(monkeypatch, source, "0" * 64)
    with pytest.raises(SplatForgeError, match="beschädigt"):
        models.ensure_model("test-model", tmp_path / "cache")
    assert not any((tmp_path / "cache" / "test-model").iterdir())


def test_registry_is_pinned() -> None:
    for model in models.MODELS.values():
        assert model.license
        for f in model.files:
            assert f.url.startswith("https://huggingface.co/") and "/resolve/main/" not in f.url
            assert len(f.sha256) == 64 and f.size > 0


# pycolmap nur in den Integrationstests laden: Unter macOS verträgt es sich nicht mit PyTorch im selben
# Prozess, und die übrigen Tests laden PyTorch.
@pytest.mark.integration
def test_undistort_masks(tmp_path: Path) -> None:
    pycolmap = pytest.importorskip("pycolmap")
    from splatforge.adapters.colmap import ColmapAdapter
    from splatforge.stages.sfm import undistort_masks

    w, h = 120, 80
    images, masks, dataset = tmp_path / "images", tmp_path / "masks", tmp_path / "dataset"
    name = "frame_0001.jpg"
    write_image(images / name, np.full((h, w, 3), 128, np.uint8))
    mask = np.full((h, w), 255, np.uint8)
    mask[20:60, 30:70] = 0
    write_image(masks / f"{name}.png", mask)

    rec = pycolmap.Reconstruction()
    camera = pycolmap.Camera(model="SIMPLE_RADIAL", width=w, height=h, params=[100.0, w / 2, h / 2, 0.05])
    camera.camera_id = 1
    rec.add_camera(camera)
    image = pycolmap.Image(name=name, camera_id=1, image_id=1)
    if hasattr(pycolmap, "Frame"):  # COLMAP ≥ 3.12: Bilder hängen an Frames und Rigs
        rig = pycolmap.Rig(rig_id=1)
        rig.add_ref_sensor(camera.sensor_id)
        rec.add_rig(rig)
        frame = pycolmap.Frame(frame_id=1, rig_id=1)
        frame.add_data_id(image.data_id)
        frame.rig_from_world = pycolmap.Rigid3d()
        rec.add_frame(frame)
        image.frame_id = 1
        rec.add_image(image)
        rec.register_frame(1)
    else:
        image.cam_from_world = pycolmap.Rigid3d()
        rec.add_image(image)
        rec.register_image(1)
    model = tmp_path / "model"
    model.mkdir()
    rec.write(model)

    colmap = ColmapAdapter()
    colmap.undistort(model, images, dataset, 1)
    assert undistort_masks(colmap, model, masks, dataset, tmp_path / "work", 1) == 1
    out = read_image(dataset / "masks" / "frame_0001.png", 0)
    undistorted = read_image(dataset / "images" / name)
    assert out is not None and undistorted is not None
    assert out.shape == undistorted.shape[:2]
    assert set(np.unique(out)) <= {0, 255}
    cy, cx = out.shape[0] // 2, out.shape[1] // 2
    assert out[cy, cx] == 0  # Mitte war maskiert
    assert out[cy, 2] == 255 or out[2, cx] == 255  # Rand nicht
    assert not (tmp_path / "work").exists()


# Optional: echte Modelle, nur wenn sie schon im Cache liegen (kein Download in den Tests)
_MODELS_DIR = Path(os.environ.get(models.ENV_MODELS_DIR, models.default_models_dir()))
_PERSON = os.environ.get("SPLATFORGE_TEST_PERSON_IMAGE")


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None
    or not all(models.is_installed(models.MODELS[k], _MODELS_DIR) for k in ("rtdetr-r18", "sam2.1-tiny"))
    or not _PERSON,
    reason="Modelle oder Testbild (SPLATFORGE_TEST_PERSON_IMAGE) fehlen",
)
def test_real_models_find_person() -> None:
    from splatforge.masking.onnx_models import CLASS_GROUPS, Detector, Segmenter

    assert _PERSON is not None
    bgr = read_image(Path(_PERSON))
    assert bgr is not None
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    providers = ["CPUExecutionProvider"]
    detections = Detector(_MODELS_DIR / "rtdetr-r18", providers, 0).detect(rgb, CLASS_GROUPS["person"], 0.5)
    assert detections
    mask = Segmenter(_MODELS_DIR / "sam2.1-tiny", providers, 0).segment(rgb, [d.box for d in detections])
    x1, y1, x2, y2 = detections[0].box.astype(int)
    inside = mask[y1:y2, x1:x2].mean()
    assert 0.2 < inside < 0.95
    assert mask.mean() < inside
