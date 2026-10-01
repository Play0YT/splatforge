from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from splatforge.config import PRESETS, JobConfig, Preset


def test_presets_match_specification() -> None:
    assert PRESETS[Preset.PREVIEW].frames == 120
    assert PRESETS[Preset.STANDARD].iterations == 30000
    assert PRESETS[Preset.HIGH].max_image_edge == 2560


def test_overrides_take_precedence() -> None:
    config = JobConfig(inputs=[Path("a.mp4")], preset=Preset.PREVIEW, frames=50, iterations=10)
    values = config.effective()
    assert (values.frames, values.max_image_edge, values.iterations) == (50, 1280, 10)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        JobConfig.model_validate({"inputs": ["a.mp4"], "tippfehler": 1})


def test_inputs_required() -> None:
    with pytest.raises(ValidationError):
        JobConfig(inputs=[])
