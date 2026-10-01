from __future__ import annotations

import json
from pathlib import Path

import pytest

from splatforge import migrations
from splatforge.config import JOB_SCHEMA_VERSION, JobConfig
from splatforge.errors import SplatForgeError
from splatforge.job import JobDir


def _config(tmp_path: Path) -> JobConfig:
    return JobConfig(inputs=[tmp_path / "video ä.mp4"])


def test_create_and_load_with_special_characters(tmp_path: Path) -> None:
    job = JobDir(tmp_path / "Job mit Leerzeichen und Ümlaut")
    job.create(_config(tmp_path))
    loaded = job.load()
    assert loaded.inputs[0].name == "video ä.mp4"
    assert loaded.inputs[0].is_absolute()


def test_refuses_foreign_non_empty_folder(tmp_path: Path) -> None:
    (tmp_path / "wichtig.txt").write_text("nicht überschreiben", encoding="utf-8")
    with pytest.raises(SplatForgeError):
        JobDir(tmp_path).create(_config(tmp_path))
    assert (tmp_path / "wichtig.txt").read_text(encoding="utf-8") == "nicht überschreiben"


def test_refuses_existing_job(tmp_path: Path) -> None:
    job = JobDir(tmp_path / "job")
    job.create(_config(tmp_path))
    with pytest.raises(SplatForgeError):
        job.create(_config(tmp_path))


def test_stage_done_markers(tmp_path: Path) -> None:
    job = JobDir(tmp_path / "job")
    job.create(_config(tmp_path))
    assert not job.is_done("01_analyze")
    job.mark_done("01_analyze", {"inputs": 1})
    assert job.is_done("01_analyze")
    assert job.done_info("01_analyze")["inputs"] == 1


def test_newer_job_format_is_rejected(tmp_path: Path) -> None:
    job = JobDir(tmp_path / "job")
    job.create(_config(tmp_path))
    raw = json.loads(job.job_file.read_text(encoding="utf-8"))
    raw["schema_version"] = JOB_SCHEMA_VERSION + 1
    job.job_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SplatForgeError):
        job.load()


def test_migration_keeps_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simuliert eine künftige Formatversion und prüft, dass die alte job.json erhalten bleibt."""
    job = JobDir(tmp_path / "job")
    job.create(_config(tmp_path))
    raw = json.loads(job.job_file.read_text(encoding="utf-8"))
    raw["schema_version"] = JOB_SCHEMA_VERSION
    raw["alter_name"] = "standard"
    raw.pop("preset")
    job.job_file.write_text(json.dumps(raw), encoding="utf-8")

    def rename(data: dict[str, object]) -> dict[str, object]:
        data = dict(data)
        data["preset"] = data.pop("alter_name")
        return data

    monkeypatch.setattr(migrations, "JOB_SCHEMA_VERSION", JOB_SCHEMA_VERSION + 1)
    monkeypatch.setitem(migrations.MIGRATIONS, JOB_SCHEMA_VERSION, rename)
    config = job.load()
    assert config.preset == "standard"
    assert config.schema_version == JOB_SCHEMA_VERSION + 1
    backup = job.root / f"job.v{JOB_SCHEMA_VERSION}.json"
    assert json.loads(backup.read_text(encoding="utf-8"))["alter_name"] == "standard"
