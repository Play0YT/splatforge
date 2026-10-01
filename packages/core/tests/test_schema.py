"""Die eingecheckten JSON-Schemas in packages/job-schema müssen zu den Pydantic-Modellen passen."""

from __future__ import annotations

from pathlib import Path

from splatforge.cli import main

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "job-schema" / "schemas"


def test_checked_in_schemas_are_current(tmp_path: Path) -> None:
    assert main(["schema", "--out", str(tmp_path)]) == 0
    for generated in tmp_path.iterdir():
        checked_in = SCHEMA_DIR / generated.name
        assert checked_in.is_file(), f"{checked_in} fehlt: 'splatforge schema --out {SCHEMA_DIR}' ausführen"
        assert checked_in.read_text(encoding="utf-8") == generated.read_text(encoding="utf-8"), (
            f"{checked_in.name} ist veraltet: 'splatforge schema --out {SCHEMA_DIR} --force' ausführen"
        )


def test_schema_command_does_not_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "job-config.schema.json"
    target.write_text("{}", encoding="utf-8")
    assert main(["schema", "--out", str(tmp_path)]) == 1
    assert target.read_text(encoding="utf-8") == "{}"
