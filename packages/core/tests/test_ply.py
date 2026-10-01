from __future__ import annotations

from pathlib import Path

import numpy as np

from splatforge.ply import GaussianCloud, read_ply, write_ply


def _cloud(n: int, rest: int) -> GaussianCloud:
    rng = np.random.default_rng(1)
    return GaussianCloud(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        sh_dc=rng.normal(size=(n, 3)).astype(np.float32),
        sh_rest=rng.normal(size=(n, rest, 3)).astype(np.float32),
        opacity_logit=rng.normal(size=n).astype(np.float32),
        log_scales=rng.normal(size=(n, 3)).astype(np.float32),
        quats=rng.normal(size=(n, 4)).astype(np.float32),
    )


def test_roundtrip_with_higher_order_sh(tmp_path: Path) -> None:
    cloud = _cloud(50, 15)
    path = tmp_path / "Ergebnis ü.ply"
    write_ply(path, cloud)
    loaded = read_ply(path)
    for field in ("means", "sh_dc", "sh_rest", "opacity_logit", "log_scales", "quats"):
        np.testing.assert_allclose(getattr(loaded, field), getattr(cloud, field))


def test_header_follows_reference_layout(tmp_path: Path) -> None:
    path = tmp_path / "a.ply"
    write_ply(path, _cloud(3, 0))
    header = path.read_bytes().split(b"end_header")[0].decode()
    assert "element vertex 3" in header
    names = [line.split()[-1] for line in header.splitlines() if line.startswith("property")]
    assert names[:9] == ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    assert names[-8:] == ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
