"""360°: Projektionsmathematik, Kalibrierung aus .insv, Ansichten, Masken-Abgleich."""

from __future__ import annotations

import math

import numpy as np
import pytest

from splatforge.insv import InsvMetadata, lens_calibrations
from splatforge.panorama import (
    View,
    auto_view_size,
    default_lens,
    equirect_lookup,
    equirect_rays,
    equirect_views,
    fisheye_views,
    nadir_mask,
    render_equirect_view,
    rig_from_view,
    share_masks,
)
from synthetic import _ROOM, ONE_RS_OFFSET_V3, _texture, one_rs_lenses, render_rays, rig_path


def test_view_rotation_conventions() -> None:
    # Geradeaus = +z, nach rechts gedreht = +x, nach oben = −y (OpenCV-Achsen)
    np.testing.assert_allclose(rig_from_view(0, 0), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rig_from_view(90, 0)[:, 2], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(rig_from_view(0, 90)[:, 2], [0, -1, 0], atol=1e-12)
    for yaw, pitch in [(30, 20), (200, -70), (0, -90)]:
        r = rig_from_view(yaw, pitch)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1)
        assert abs(r[1, 0]) < 1e-12  # rechte Achse bleibt waagrecht


def test_view_project_inverts_rays() -> None:
    view = View("v", 35, -20, 90, 64)
    rays = view.rays()
    x, y, inside = view.project(rays)
    grid_y, grid_x = np.mgrid[0:64, 0:64]
    assert inside.all()
    np.testing.assert_allclose(x, grid_x, atol=1e-9)
    np.testing.assert_allclose(y, grid_y, atol=1e-9)


def test_equirect_lookup_inverts_rays() -> None:
    rays = equirect_rays(64, 32)
    u, v = equirect_lookup(rays, 64, 32)
    grid_v, grid_u = np.mgrid[0:32, 0:64]
    np.testing.assert_allclose(u, grid_u, atol=1e-9)
    np.testing.assert_allclose(v, grid_v, atol=1e-9)


def test_default_layouts_cover_the_sphere() -> None:
    views = equirect_views(8, 90, 32, up_down=True)
    assert len(views) == 10
    covered = np.zeros(len(equirect_rays(128, 64).reshape(-1, 3)), dtype=bool)
    for view in views:
        covered |= view.project(equirect_rays(128, 64).reshape(-1, 3))[2]
    assert covered.all()
    fisheye = fisheye_views(50, 90, 32)
    assert [v.lens for v in fisheye] == [0] * 5 + [1] * 5
    assert fisheye[5].yaw_deg == 180


def test_equirect_view_matches_direct_rendering() -> None:
    """Eine Ansicht aus dem Panorama entspricht dem direkt gerenderten Bild derselben Blickrichtung."""
    textures = [_texture(seed) for *_, seed in _ROOM]
    eye, rot = rig_path(10)[3]
    pano = render_rays(eye, equirect_rays(2048, 1024) @ rot, textures, _ROOM)
    for view in (View("a", 0, 0, 90, 96), View("b", 135, 30, 90, 96), View("c", 0, -90, 90, 96)):
        from_pano = render_equirect_view(pano, view).astype(float)
        direct = render_rays(eye, view.rays() @ rot, textures, _ROOM).astype(float)
        assert np.abs(from_pano - direct).mean() < 6


def test_one_rs_calibration_is_read() -> None:
    lenses = lens_calibrations(InsvMetadata(offset_v3=ONE_RS_OFFSET_V3))
    assert lenses is not None and len(lenses) == 2
    assert lenses[0].roll == pytest.approx(179.367)  # Objektiv 1 steht auf dem Kopf
    assert lenses[1].cx == pytest.approx(4874.88 - 3264)  # relativ zum eigenen Objektivbild
    assert lenses[0].width == 3264


@pytest.mark.parametrize(
    "values",
    [
        [],
        [2.0, 1.0, 2.0],  # zu kurz
        [
            1.0,
            2.0,
            3000,
            3000,
            100.0,
            1600,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3264,
            3264,
            62,
            0,
        ],  # Mitte daneben
    ],
)
def test_implausible_calibration_is_rejected(values: list[float]) -> None:
    assert lens_calibrations(InsvMetadata(offset_v3=values)) is None


def test_fisheye_unproject_round_trip() -> None:
    for lens in one_rs_lenses(320):
        grid_y, grid_x = np.mgrid[0:320:4, 0:320:4].astype(np.float64)
        rays, valid = lens.unproject(grid_x, grid_y)
        u, v, ok = lens.lookup(rays)
        assert valid.mean() > 0.7  # Bildkreis füllt das quadratische Bild
        np.testing.assert_allclose(u[valid & ok], grid_x[valid & ok], atol=1e-6)
        np.testing.assert_allclose(v[valid & ok], grid_y[valid & ok], atol=1e-6)


def test_lenses_look_in_opposite_directions() -> None:
    front, back = one_rs_lenses(320)
    # Strahl nach vorne trifft Objektiv 1 nahe der Bildmitte, Objektiv 2 gar nicht
    forward = np.array([[0.0, 0.0, 1.0]])
    u, v, ok = front.lookup(forward)
    assert ok[0] and abs(u[0] - 160) < 10 and abs(v[0] - 160) < 10
    assert not back.lookup(forward)[2][0]
    assert back.lookup(-forward)[2][0]
    # Objektiv 1 ist um 180° gedreht: oben im Rig landet unten im Bild
    up = np.array([[0.0, -math.sin(math.radians(30)), math.cos(math.radians(30))]])
    assert front.lookup(up)[1][0] > 160


def test_default_lens_is_close_to_real_calibration() -> None:
    real = one_rs_lenses(320)[0]
    approx = default_lens(320, 320, 0)
    assert approx.fx == pytest.approx(real.fx, rel=0.1)


def test_auto_view_size() -> None:
    assert auto_view_size(5760, 90, equirect=True) == 1832
    lens = one_rs_lenses(3072)[0]
    assert 1700 < auto_view_size(3072, 90, equirect=False, lens=lens) < 1900


def test_nadir_mask() -> None:
    down = View("d", 0, -90, 90, 64)
    mask = nadir_mask(down, 30)
    assert mask[32, 32] and not mask[0, 0]
    assert not nadir_mask(View("f", 0, 0, 90, 64), 30).any()
    assert not nadir_mask(down, 0).any()


def test_share_masks_crosses_view_borders() -> None:
    """Eine Person an der rechten Kante von Ansicht 0 wird auch in der Nachbaransicht maskiert."""
    views = equirect_views(8, 90, 64, up_down=False)
    masks = [np.zeros((64, 64), dtype=bool) for _ in views]
    masks[0][20:40, 40:64] = True  # rechter Rand, liegt im Überlappungsbereich mit Ansicht 1 (45° rechts)
    shared = share_masks(views, masks, sphere_width=512)
    assert shared[0].sum() >= masks[0].sum()
    assert shared[1].any() and not masks[1].any()
    assert not shared[4].any()  # gegenüberliegende Ansicht bleibt frei


# Werkskalibrierung einer Insta360 X4 (offset_v3 einer echten Datei; Sensor 8000×6000 pro Objektiv, Video
# 3840×3840 pro Objektiv, Sensor um ≈90° gedreht, Video aber bereits aufrecht)
X4_OFFSET_V3 = [
    2.0, 1.94817, 4616.94, 4616.35, 4008.79, 3007.96, 0.371, 0.096, 90.399, 0.0, 0.0, 0.0,
    0.37965181, 1.41417241, -4.22088528, -0.00062194, -0.00135227, 16000.0, 6000.0, 71.0,
    1.94817, 4605.94, 4604.64, 12015.67, 2994.86, -0.362, 0.096, 90.272, -0.002026, 0.000173, -0.032125,
    0.38574281, 1.35695291, -4.07583952, -0.00042552, -0.00014615, 16000.0, 6000.0, 71.0, 197632.0,
]  # fmt: skip


def test_x4_calibration_is_cropped_and_unrotated() -> None:
    lenses = lens_calibrations(InsvMetadata(offset_v3=X4_OFFSET_V3))
    assert lenses is not None and len(lenses) == 2
    for lens in lenses:
        assert lens.width == lens.height == 6000  # mittlerer quadratischer Ausschnitt des Sensors
        assert lens.rotated_deg == 90
        assert abs(lens.roll) < 1  # nur die kleine Restabweichung bleibt
        assert abs(lens.cx - 2999.5) < 20 and abs(lens.cy - 2999.5) < 20
    # Brennweiten und Mitte werden mitgedreht
    assert lenses[0].fx == pytest.approx(4616.35) and lenses[0].fy == pytest.approx(4616.94)
    assert lenses[0].cx == pytest.approx(3007.96)


def test_x4_image_circle_fits_video_frame() -> None:
    """Mit gleichmässiger Skalierung auf 3840×3840 füllt der Bildkreis das Videobild fast aus."""
    from splatforge.stages.analyze import InputInfo
    from splatforge.stages.pano import input_lenses

    info = InputInfo(
        path="x.insv", kind="video", camera_type="dual_fisheye", insv={"offset_v3": X4_OFFSET_V3}
    )
    lenses, calibrated = input_lenses(info, 3840, 3840)
    assert calibrated
    front = lenses[0]
    u, v, ok = front.lookup(np.array([[0.0, 0.0, 1.0]]))
    assert ok[0] and abs(u[0] - 1919.5) < 15 and abs(v[0] - 1919.5) < 15
    side = np.array([[math.sin(math.radians(96)), 0.0, math.cos(math.radians(96))]])
    u, v, ok = front.lookup(side)
    assert ok[0] and 1750 < abs(u[0] - 1919.5) < 1920  # 96° liegt knapp innerhalb des Bildrands
    # Video ist aufrecht: oben im Rig landet oben im Bild
    up = np.array([[0.0, -math.sin(math.radians(30)), math.cos(math.radians(30))]])
    assert front.lookup(up)[1][0] < 1919.5
