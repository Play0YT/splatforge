"""Prüft den kachelbasierten CPU-Renderer gegen eine einfache Referenz Pixel für Pixel."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from splatforge.training.cpu import View, _quat_to_rot, render  # noqa: E402


def _view(width: int = 40, height: int = 30) -> View:
    pose = torch.zeros(3, 4)
    pose[:, :3] = torch.eye(3)
    return View(
        "a", pose, 30.0, 30.0, width / 2, height / 2, width, height, torch.zeros(height, width, 3), None
    )


def _params(n: int = 25) -> dict[str, object]:
    gen = torch.Generator().manual_seed(3)
    means = torch.rand(n, 3, generator=gen) * torch.tensor([2.0, 1.5, 2.0]) - torch.tensor([1.0, 0.75, -2.0])
    quats = torch.randn(n, 4, generator=gen)
    return {
        "means": torch.nn.Parameter(means),
        "sh_dc": torch.nn.Parameter(torch.randn(n, 3, generator=gen)),
        "opacity": torch.nn.Parameter(torch.randn(n, generator=gen)),
        "log_scales": torch.nn.Parameter(torch.rand(n, 3, generator=gen) * -2.5 - 1.0),
        "quats": torch.nn.Parameter(quats),
    }


def _reference(params: dict[str, object], view: View) -> object:
    """Naive Implementation: alle Gaussians, jeder Pixel, nach Tiefe sortiert."""
    from splatforge.ply import SH_C0

    means, scales = params["means"].detach(), torch.exp(params["log_scales"].detach())  # type: ignore[attr-defined]
    rot = _quat_to_rot(params["quats"].detach())  # type: ignore[attr-defined]
    cov = rot @ torch.diag_embed(scales**2) @ rot.transpose(1, 2)
    x, y, z = means.unbind(-1)
    jac = torch.zeros(len(z), 2, 3)
    jac[:, 0, 0], jac[:, 0, 2] = view.fx / z, -view.fx * x / z**2
    jac[:, 1, 1], jac[:, 1, 2] = view.fy / z, -view.fy * y / z**2
    cov2 = jac @ cov @ jac.transpose(1, 2) + 0.3 * torch.eye(2)
    inv = torch.linalg.inv(cov2)
    center = torch.stack([view.fx * x / z + view.cx, view.fy * y / z + view.cy], -1)
    opacity = torch.sigmoid(params["opacity"].detach())  # type: ignore[attr-defined]
    color = torch.clamp(params["sh_dc"].detach() * SH_C0 + 0.5, min=0)  # type: ignore[attr-defined]
    img = torch.zeros(view.height, view.width, 3)
    order = torch.argsort(z)
    for py in range(view.height):
        for px in range(view.width):
            t = 1.0
            for g in order.tolist():
                d = torch.tensor([px, py], dtype=torch.float32) - center[g]
                power = -0.5 * d @ inv[g] @ d
                alpha = min(0.99, float(opacity[g] * torch.exp(power)))
                if alpha < 1 / 255:
                    continue
                img[py, px] += t * alpha * color[g]
                t *= 1 - alpha
    return img


def test_matches_naive_reference() -> None:
    view, params = _view(), _params()
    for tile in (4, 8, 16):
        with torch.no_grad():
            img, _, _ = render(params, view, tile, torch.zeros(3))
        ref = _reference(params, view)
        assert torch.allclose(img, ref, atol=2e-2), f"Abweichung bei Kachelgrösse {tile}"


def test_gradients_reach_all_parameters() -> None:
    view, params = _view(), _params()
    img, means2d, visible = render(params, view, 8, torch.zeros(3))
    img.sum().backward()
    for name, p in params.items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name  # type: ignore[attr-defined]
    assert means2d.grad is not None and len(visible) == 25
