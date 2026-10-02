from __future__ import annotations

from splatforge.hardware import detect, parse_lspci


def test_parse_lspci() -> None:
    lines = [
        "00:02.0 VGA compatible controller: Intel Corporation HD Graphics 630 (rev 04)",
        "01:00.0 3D controller: NVIDIA Corporation AD107 [GeForce RTX 4060] (rev a1)",
        "00:1f.3 Audio device: Intel Corporation",
    ]
    assert parse_lspci(lines) == [
        "Intel Corporation HD Graphics 630 (rev 04)",
        "NVIDIA Corporation AD107 [GeForce RTX 4060] (rev a1)",
    ]


def test_detect_runs_everywhere() -> None:
    info = detect().to_dict()
    assert info["cpu_threads"] >= 1
    assert isinstance(info["gpus"], list)
    assert info["recommended_backend"] in {"brush", "cpu", "none"}
