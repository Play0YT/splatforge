from __future__ import annotations

from splatforge.adapters.colmap import DatabaseStats
from splatforge.config import SfmSettings
from splatforge.stages.sfm import diagnose


def test_low_texture() -> None:
    message, hint = diagnose(DatabaseStats(50, 80.0, [100] * 49), 10, SfmSettings())
    assert "Textur" in message


def test_fast_motion() -> None:
    message, hint = diagnose(DatabaseStats(50, 2000.0, [5] * 30 + [200] * 19), 10, SfmSettings())
    assert "zu schnell" in message
    assert "langsamer" in hint


def test_disconnected_parts() -> None:
    message, _ = diagnose(DatabaseStats(50, 2000.0, [200] * 45 + [3] * 4), 20, SfmSettings())
    assert "mehrere Teile" in message
