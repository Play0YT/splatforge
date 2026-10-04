"""Ausrichtung der Szene (oben = −Y), Bewertung der Kamerabewegung und PLY-Kommentar."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from splatforge.orient import WORLD_UP, baseline_ratio, estimate_up, orientation, rotation_between
from splatforge.ply import GaussianCloud, copy_with_comment, read_ply, write_ply
from synthetic import _look_at

UP = np.array([0.0, 1.0, 0.0])  # in den synthetischen Szenen zeigt +Y nach oben


def _roll(rot: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ rot


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q *= np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def _orbit(n: int, rng: np.random.Generator, roll_deg: float, yaw_deg: float) -> list[np.ndarray]:
    """Kameras auf einem Bogen, leicht nach unten auf ein Objekt gerichtet, mit zufälligem Wackeln."""
    target = np.array([0.0, 0.5, 0.0])
    rotations = []
    for i in range(n):
        a = np.deg2rad(-yaw_deg / 2 + yaw_deg * i / (n - 1))
        eye = np.array([2 * np.sin(a), 1.6, -2 * np.cos(a)])
        rotations.append(_roll(_look_at(eye, target), np.deg2rad(roll_deg) * rng.standard_normal()))
    return rotations


@pytest.mark.parametrize("yaw_deg", [360.0, 70.0])
def test_up_is_found_despite_pitch_and_shake(yaw_deg: float) -> None:
    rng = np.random.default_rng(0)
    rotations = _orbit(80, rng, roll_deg=4, yaw_deg=yaw_deg)
    assert _angle(estimate_up(rotations), UP) < 3
    # Das Ergebnis darf nicht davon abhängen, wie COLMAP die Welt gerade gedreht hat
    tilt = _random_rotation(rng)
    tilted = [r @ tilt.T for r in rotations]
    assert _angle(estimate_up(tilted), tilt @ UP) < 3


def test_up_without_turning_uses_camera_up() -> None:
    """Fährt die Kamera nur seitlich ohne Drehung, gilt die mittlere Oben-Achse der Kameras."""
    rotations = [_look_at(np.array([x, 1.0, 0.0]), np.array([x, 1.0, 5.0])) for x in np.linspace(0, 2, 30)]
    assert _angle(estimate_up(rotations), UP) < 0.1


def test_rotation_between() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.standard_normal(3)
        a /= np.linalg.norm(a)
        rot = rotation_between(a, WORLD_UP)
        np.testing.assert_allclose(rot @ a, WORLD_UP, atol=1e-9)
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rot) == pytest.approx(1.0)
    flipped = rotation_between(-WORLD_UP, WORLD_UP)
    np.testing.assert_allclose(flipped @ -WORLD_UP, WORLD_UP, atol=1e-9)
    assert np.linalg.det(flipped) == pytest.approx(1.0)


def test_orientation_levels_and_centers_scene() -> None:
    rng = np.random.default_rng(2)
    tilt = _random_rotation(rng)
    rotations = [r @ tilt.T for r in _orbit(40, rng, roll_deg=2, yaw_deg=360)]
    points = (tilt @ (rng.standard_normal((500, 3)) + [5.0, 1.0, 2.0]).T).T
    o = orientation(rotations, points)
    assert _angle(o.rotation @ tilt @ UP, WORLD_UP) < 3
    # Die Szenenmitte (Median der Punkte) landet im Ursprung
    np.testing.assert_allclose(o.rotation @ np.median(points, axis=0) + o.translation, 0, atol=1e-9)
    assert o.tilt_deg > 0


def test_baseline_ratio_detects_turning_on_the_spot() -> None:
    rng = np.random.default_rng(3)
    points = rng.uniform(-3, 3, (1000, 3)) + [0, 0, 6]
    standing = rng.normal(0, 0.01, (50, 3))
    walking = np.stack([np.linspace(-1.5, 1.5, 50), np.zeros(50), np.zeros(50)], axis=1)
    assert baseline_ratio(standing, points) < 0.05
    assert baseline_ratio(walking, points) > 0.3
    assert baseline_ratio(standing[:1], points) == 0.0


def test_copy_with_comment(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 10
    cloud = GaussianCloud(
        means=rng.standard_normal((n, 3)).astype(np.float32),
        sh_dc=rng.standard_normal((n, 3)).astype(np.float32),
        sh_rest=np.zeros((n, 0, 3), np.float32),
        opacity_logit=rng.standard_normal(n).astype(np.float32),
        log_scales=rng.standard_normal((n, 3)).astype(np.float32),
        quats=rng.standard_normal((n, 4)).astype(np.float32),
    )
    source, target = tmp_path / "in.ply", tmp_path / "out.ply"
    write_ply(source, cloud)
    copy_with_comment(source, target, "vertical axis: y")
    header = target.read_bytes().split(b"end_header")[0].decode("ascii").splitlines()
    assert header[:3] == ["ply", "format binary_little_endian 1.0", "comment vertical axis: y"]
    np.testing.assert_array_equal(read_ply(target).means, cloud.means)
    # Ein zweites Mal ändert nichts
    again = tmp_path / "again.ply"
    copy_with_comment(target, again, "vertical axis: y")
    assert again.read_bytes() == target.read_bytes()
