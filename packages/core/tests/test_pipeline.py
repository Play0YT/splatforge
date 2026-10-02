"""Kompletter Durchlauf auf der CPU mit einem kleinen synthetischen Video."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from conftest import needs_ffmpeg
from splatforge.cli import main
from splatforge.ply import read_ply

# PyTorch hier nur suchen, nicht laden: Unter macOS darf es nicht im selben Prozess wie pycolmap liegen.
if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch nicht installiert", allow_module_level=True)


def _events(job: Path) -> list[dict[str, object]]:
    lines = (job / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@needs_ffmpeg
@pytest.mark.integration
def test_full_run_and_resume(tmp_path: Path, synthetic_video: Path) -> None:
    job = tmp_path / "Ergebnis mit Ümlaut"
    code = main(
        ["run", str(synthetic_video), "--out", str(job), "--preset", "preview",
         "--frames", "30", "--iterations", "60", "--backend", "cpu", "--json"]
    )  # fmt: skip
    events = _events(job)
    assert code == 0, [e for e in events if e["type"] in ("job_failed", "log")][-5:]
    assert events[-1]["type"] == "job_finished"

    report = json.loads((job / "08_export" / "report.json").read_text(encoding="utf-8"))
    assert report["registered_images"] / report["total_images"] >= 0.6
    cloud = read_ply(job / "08_export" / "splat.ply")
    assert len(cloud) > 100
    assert report["quality"]["psnr"] is not None

    # Fortsetzen: alle Stufen sind fertig und werden übersprungen
    assert main(["resume", str(job), "--json"]) == 0
    skipped = [e for e in _events(job)[len(events) :] if e["type"] == "stage_skipped"]
    assert len(skipped) == 6


@needs_ffmpeg
@pytest.mark.integration
def test_resume_after_interrupted_stage(tmp_path: Path, synthetic_video: Path) -> None:
    """Eine Stufe ohne stage.done gilt als unvollständig und wird neu ausgeführt."""
    job = tmp_path / "job"
    args = ["run", str(synthetic_video), "--out", str(job), "--frames", "25", "--iterations", "20",
            "--backend", "cpu", "--json"]  # fmt: skip
    assert main(args) == 0
    (job / "08_export" / "stage.done").unlink()
    (job / "08_export" / "splat.ply").unlink()
    assert main(["resume", str(job), "--json"]) == 0
    assert (job / "08_export" / "splat.ply").is_file()


def test_run_refuses_foreign_folder(tmp_path: Path) -> None:
    (tmp_path / "meine_datei.txt").write_text("x", encoding="utf-8")
    assert main(["run", "video.mp4", "--out", str(tmp_path)]) == 1


def test_missing_backend_fails_before_any_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fehlt jedes Trainings-Backend, bricht der Job sofort ab statt erst nach COLMAP."""
    monkeypatch.setattr("splatforge.stages.train.torch_available", lambda: False)
    monkeypatch.setattr("splatforge.adapters.brush.BrushAdapter.available", lambda self: False)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    job = tmp_path / "job"
    assert main(["run", str(video), "--out", str(job), "--json"]) == 1
    events = _events(job)
    assert events[-1]["type"] == "job_failed"
    assert "PyTorch" in str(events[-1]["hint"])
    assert not (job / "01_analyze").exists()
