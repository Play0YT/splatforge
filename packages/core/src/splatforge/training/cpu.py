"""Ausweich-Trainingsbackend, das rein auf der CPU läuft (PyTorch).

Gedacht für Rechner, auf denen Brush keine nutzbare Grafikkarte findet. Es ist deutlich langsamer
als die GPU-Backends und bewusst einfach gehalten:

- Farbe nur als Grundfarbe (Kugelflächenfunktionen Grad 0, keine blickwinkelabhängigen Effekte)
- Rasterisierung kachelweise in reinem PyTorch, mit Gradient-Checkpointing pro Kachelgruppe,
  damit der Speicherbedarf begrenzt bleibt
- Verdichten (Klonen/Teilen) und Ausdünnen nach dem Verfahren der 3DGS-Referenz
- Masken werden aus dem Verlust ausgeschlossen (schwarz = ignorieren)

Siehe docs/decisions/0002-cpu-trainingsbackend.md.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..adapters.base import CancelToken
from ..config import TrainSettings
from ..errors import ToolMissingError
from ..imageio import read_image
from ..ply import SH_C0, GaussianCloud, write_ply

try:
    import torch
    import torch.nn.functional as F  # noqa: N812
    from torch.utils.checkpoint import checkpoint
except ImportError:  # pragma: no cover - abhängig von der Installation
    torch = None  # type: ignore[assignment]

# Feste Konstanten der 3DGS-Referenzimplementierung (keine Stellschrauben für Nutzer)
_LR_MEANS_START = 1.6e-4
_LR_MEANS_END = 1.6e-6
_LR_COLOR = 2.5e-3
_LR_OPACITY = 5e-2
_LR_SCALE = 5e-3
_LR_ROT = 1e-3
_ALPHA_MIN = 1.0 / 255.0
_ALPHA_MAX = 0.99
_COV2D_BLUR = 0.3
_NEAR = 0.01
_SPLIT_SAMPLES = 2
_SPLIT_SHRINK = 1.6
_PERCENT_DENSE = 0.01
_MAX_SCREEN_FRACTION = 0.1
_PIXELS_PER_CHUNK = 2_000_000
_INIT_OPACITY = 0.1
_OPACITY_RESET = 0.01
_MIN_INIT_POINTS = 100


def require_torch() -> None:
    if torch is None:
        raise ToolMissingError(
            "Für das CPU-Backend fehlt PyTorch.",
            "SplatForge mit dem Zusatz 'cpu-train' installieren (pip install 'splatforge[cpu-train]').",
        )


@dataclass
class View:
    name: str
    world_to_cam: Any  # torch (3, 4)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    image: Any  # torch (H, W, 3) in 0..1
    mask: Any | None  # torch (H, W, 1), 1 = verwenden


@dataclass
class TrainResult:
    ply: Path
    gaussians: int
    psnr: float | None
    ssim: float | None
    eval_views: int
    seconds: float


ProgressFn = Callable[[int, int, float, float | None], None]


def _pose(image: Any) -> Any:
    pose = image.cam_from_world
    pose = pose() if callable(pose) else pose
    return np.asarray(pose.matrix(), dtype=np.float32)


def load_views(dataset: Path, max_edge: int) -> tuple[list[View], Any, Any]:
    """Lädt das entzerrte COLMAP-Modell. Gibt Ansichten, Punkte und Punktfarben zurück."""
    import pycolmap

    sparse = dataset / "sparse"
    model = pycolmap.Reconstruction(sparse / "0" if (sparse / "0").is_dir() else sparse)
    masks_dir = dataset / "masks"
    views: list[View] = []
    for image in sorted(model.images.values(), key=lambda im: im.name):
        if hasattr(image, "has_pose") and not image.has_pose:
            continue
        cam = model.cameras[image.camera_id]
        fx, fy, cx, cy = (float(v) for v in cam.params[:4])
        if cam.model_name == "SIMPLE_PINHOLE":
            fx, cx, cy = (float(v) for v in cam.params[:3])
            fy = fx
        img = read_image(dataset / "images" / image.name)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(1.0, max_edge / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        sy, sx = img.shape[0] / h, img.shape[1] / w
        rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        mask = None
        mask_img = read_image(masks_dir / f"{image.name}.png", cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy((mask_img > 127).astype(np.float32))[..., None]
        views.append(
            View(
                name=image.name,
                world_to_cam=torch.from_numpy(_pose(image)),
                fx=fx * sx,
                fy=fy * sy,
                cx=cx * sx,
                cy=cy * sy,
                width=img.shape[1],
                height=img.shape[0],
                image=rgb,
                mask=mask,
            )
        )
    points = np.array([p.xyz for p in model.points3D.values()], dtype=np.float32).reshape(-1, 3)
    colors = np.array([p.color for p in model.points3D.values()], dtype=np.float32).reshape(-1, 3) / 255.0
    return views, torch.from_numpy(points), torch.from_numpy(colors)


def _quat_to_rot(q: Any) -> Any:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)  # fmt: skip


def _composite_chunk(
    u: Any,
    v: Any,
    conic: Any,
    opacity: Any,
    color: Any,
    tile_xy: Any,
    seg_start: Any,
    local_tile: Any,
    n_tiles: int,
    tile: int,
) -> tuple[Any, Any]:
    """Alpha-Compositing für eine Gruppe von Kacheln. Paare sind nach Kachel und Tiefe sortiert."""
    offs = torch.arange(tile, dtype=u.dtype)
    py, px = torch.meshgrid(offs, offs, indexing="ij")
    px = px.reshape(1, -1) + tile_xy[:, :1].to(u.dtype)
    py = py.reshape(1, -1) + tile_xy[:, 1:].to(u.dtype)
    dx = px - u[:, None]
    dy = py - v[:, None]
    power = -0.5 * (conic[:, :1] * dx * dx + conic[:, 2:] * dy * dy) - conic[:, 1:2] * dx * dy
    alpha = torch.clamp(opacity[:, None] * torch.exp(torch.clamp(power, max=0.0)), max=_ALPHA_MAX)
    alpha = torch.where((power <= 0) & (alpha >= _ALPHA_MIN), alpha, torch.zeros_like(alpha))
    log_t = torch.log1p(-alpha).double()
    incl = torch.cumsum(log_t, dim=0)
    # prev[i] = Summe bis Paar i-1; damit beginnt jede Kachel wieder bei voller Transmission
    prev = torch.cat([torch.zeros(1, incl.shape[1], dtype=incl.dtype), incl], dim=0)
    before_seg = prev[seg_start]
    trans = torch.exp((incl - log_t - before_seg).float())
    weight = alpha * trans
    out = torch.zeros(n_tiles, tile * tile, 3, dtype=u.dtype)
    out.index_add_(0, local_tile, weight[..., None] * color[:, None, :])
    total_log = torch.zeros(n_tiles, tile * tile, dtype=torch.float64)
    total_log.index_add_(0, local_tile, log_t)
    return out, torch.exp(total_log).float()


def render(params: dict[str, Any], view: View, tile: int, background: Any) -> tuple[Any, Any, Any]:
    """Rendert eine Ansicht.

    Gibt Bild (H, W, 3), Bildschirmpositionen und Indizes der sichtbaren Gaussians zurück.
    """
    rot, trans = view.world_to_cam[:, :3], view.world_to_cam[:, 3]
    means_cam = params["means"] @ rot.T + trans
    z = means_cam[:, 2]
    visible = torch.nonzero(z > _NEAR).squeeze(1)
    if visible.numel() == 0:
        img = background.expand(view.height, view.width, 3).clone()
        return img, torch.zeros(0, 2, requires_grad=True), visible
    pc = means_cam[visible]
    x, y, z = pc.unbind(-1)
    r_g = _quat_to_rot(params["quats"][visible])
    m = r_g * torch.exp(params["log_scales"][visible])[:, None, :]
    cov_world = m @ m.transpose(1, 2)
    cov_cam = rot @ cov_world @ rot.T
    lim_x = 1.3 * (view.width / 2) / view.fx
    lim_y = 1.3 * (view.height / 2) / view.fy
    tx = torch.clamp(x / z, -lim_x, lim_x) * z
    ty = torch.clamp(y / z, -lim_y, lim_y) * z
    zeros = torch.zeros_like(z)
    jac = torch.stack(
        [view.fx / z, zeros, -view.fx * tx / (z * z), zeros, view.fy / z, -view.fy * ty / (z * z)], dim=-1
    ).reshape(-1, 2, 3)
    cov2 = jac @ cov_cam @ jac.transpose(1, 2)
    a = cov2[:, 0, 0] + _COV2D_BLUR
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + _COV2D_BLUR
    det = a * c - b * b
    means2d = torch.stack([view.fx * x / z + view.cx, view.fy * y / z + view.cy], dim=-1)
    if means2d.requires_grad:
        means2d.retain_grad()
    u, v = means2d[:, 0], means2d[:, 1]
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    opacity = torch.sigmoid(params["opacity"][visible])
    # Radius, ab dem das Alpha unter 1/255 fällt: kleiner als 3 Sigma bei schwach deckenden Gaussians
    reach = torch.sqrt(2.0 * torch.log(torch.clamp(opacity.detach() / _ALPHA_MIN, min=1.0)))
    radius = torch.ceil(reach * torch.sqrt(lam)).detach()

    tiles_x = math.ceil(view.width / tile)
    tiles_y = math.ceil(view.height / tile)
    with torch.no_grad():
        on_screen = (
            (det > 0)
            & (radius > 0)
            & (u + radius >= 0)
            & (u - radius < view.width)
            & (v + radius >= 0)
            & (v - radius < view.height)
        )
        tx0 = torch.clamp(torch.floor((u - radius) / tile), 0, tiles_x - 1).long()
        tx1 = torch.clamp(torch.floor((u + radius) / tile), 0, tiles_x - 1).long()
        ty0 = torch.clamp(torch.floor((v - radius) / tile), 0, tiles_y - 1).long()
        ty1 = torch.clamp(torch.floor((v + radius) / tile), 0, tiles_y - 1).long()
        span_x = tx1 - tx0 + 1
        counts = torch.where(on_screen, span_x * (ty1 - ty0 + 1), torch.zeros_like(span_x))
        gauss = torch.repeat_interleave(torch.arange(len(u)), counts)
        starts = torch.cumsum(counts, 0) - counts
        offset = torch.arange(len(gauss)) - starts[gauss]
        tile_x = tx0[gauss] + offset % span_x[gauss]
        tile_y = ty0[gauss] + torch.div(offset, span_x[gauss], rounding_mode="floor")
        tile_id = tile_y * tiles_x + tile_x
        depth_rank = torch.empty_like(z, dtype=torch.long)
        depth_rank[torch.argsort(z.detach())] = torch.arange(len(z))
        order = torch.argsort(tile_id * len(z) + depth_rank[gauss])
        gauss, tile_id = gauss[order], tile_id[order]

    inv_det = 1.0 / det
    conic = torch.stack([c * inv_det, -b * inv_det, a * inv_det], dim=-1)
    color = torch.clamp(params["sh_dc"][visible] * SH_C0 + 0.5, min=0.0)

    n_all_tiles = tiles_x * tiles_y
    canvas = torch.zeros(n_all_tiles, tile * tile, 3)
    final_t = torch.ones(n_all_tiles, tile * tile)
    if len(gauss) > 0:
        tiles_present, pair_counts = torch.unique_consecutive(tile_id, return_counts=True)
        pair_offsets = torch.cumsum(pair_counts, 0) - pair_counts
        max_pairs = max(1, _PIXELS_PER_CHUNK // (tile * tile))
        start_tile = 0
        while start_tile < len(tiles_present):
            end_tile = start_tile
            used = 0
            while end_tile < len(tiles_present) and (used == 0 or used + pair_counts[end_tile] <= max_pairs):
                used += int(pair_counts[end_tile])
                end_tile += 1
            p0 = int(pair_offsets[start_tile])
            p1 = p0 + used
            chunk_tiles = tiles_present[start_tile:end_tile]
            chunk_gauss = gauss[p0:p1]
            local_tile = torch.repeat_interleave(
                torch.arange(len(chunk_tiles)), pair_counts[start_tile:end_tile]
            )
            seg_first = (pair_offsets[start_tile:end_tile] - p0)[local_tile]
            tile_xy = torch.stack(
                [
                    (chunk_tiles % tiles_x) * tile,
                    torch.div(chunk_tiles, tiles_x, rounding_mode="floor") * tile,
                ],
                dim=-1,
            )[local_tile]
            args = (
                u[chunk_gauss], v[chunk_gauss], conic[chunk_gauss], opacity[chunk_gauss], color[chunk_gauss],
                tile_xy, seg_first, local_tile, len(chunk_tiles), tile,
            )  # fmt: skip
            if torch.is_grad_enabled():
                rgb, t_final = checkpoint(_composite_chunk, *args, use_reentrant=False)
            else:
                rgb, t_final = _composite_chunk(*args)
            canvas = canvas.index_copy(0, chunk_tiles, rgb)
            final_t = final_t.index_copy(0, chunk_tiles, t_final)
            start_tile = end_tile

    canvas = canvas + final_t[..., None] * background
    img = (
        canvas.reshape(tiles_y, tiles_x, tile, tile, 3)
        .permute(0, 2, 1, 3, 4)
        .reshape(tiles_y * tile, tiles_x * tile, 3)[: view.height, : view.width]
    )
    return img, means2d, visible


def _ssim(img1: Any, img2: Any) -> Any:
    window_size, sigma = 11, 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    window = g.expand(3, 1, window_size, window_size).contiguous()
    x = img1.permute(2, 0, 1)[None]
    y = img2.permute(2, 0, 1)[None]
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=3)
    mu_y = F.conv2d(y, window, padding=pad, groups=3)
    sxx = F.conv2d(x * x, window, padding=pad, groups=3) - mu_x**2
    syy = F.conv2d(y * y, window, padding=pad, groups=3) - mu_y**2
    sxy = F.conv2d(x * y, window, padding=pad, groups=3) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return ssim_map.mean()


def _knn_scale(points: Any, k: int = 3) -> Any:
    """Mittlerer Abstand zu den k nächsten Nachbarn (blockweise, um Speicher zu sparen)."""
    result = torch.empty(len(points))
    block = 2048
    for i in range(0, len(points), block):
        d = torch.cdist(points[i : i + block], points)
        d, _ = torch.topk(d, min(k + 1, len(points)), largest=False)
        result[i : i + block] = d[:, 1:].mean(dim=1) if d.shape[1] > 1 else d[:, 0]
    return torch.clamp(result, min=1e-7)


class CpuTrainer:
    PARAM_NAMES = ("means", "sh_dc", "opacity", "log_scales", "quats")

    def __init__(
        self,
        dataset: Path,
        work_dir: Path,
        iterations: int,
        settings: TrainSettings,
        num_threads: int,
        seed: int = 42,
    ) -> None:
        require_torch()
        if num_threads > 0:
            torch.set_num_threads(num_threads)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.work_dir = work_dir
        self.iterations = iterations
        self.settings = settings
        views, points, colors = load_views(dataset, settings.cpu_max_image_edge)
        if not views:
            raise RuntimeError("Keine Ansichten im Datensatz gefunden")
        every = settings.eval_split_every
        if every > 0 and len(views) >= 3 * every:
            self.eval_views = [v for i, v in enumerate(views) if i % every == 0]
            self.train_views = [v for i, v in enumerate(views) if i % every != 0]
        else:
            self.eval_views, self.train_views = [], views
        centers = torch.stack([-(v.world_to_cam[:, :3].T @ v.world_to_cam[:, 3]) for v in views])
        self.extent = float(torch.linalg.norm(centers - centers.mean(0), dim=1).max()) * 1.1 or 1.0
        if len(points) < _MIN_INIT_POINTS:
            points = (torch.rand(_MIN_INIT_POINTS * 10, 3) * 2 - 1) * self.extent + centers.mean(0)
            colors = torch.full_like(points, 0.5)
        self.params = self._init_params(points, colors)
        self.optimizer = self._make_optimizer()
        self.grad_accum = torch.zeros(len(points))
        self.grad_count = torch.zeros(len(points))
        self.max_radius_fraction = torch.zeros(len(points))
        self.start_iter = 0
        self.background = torch.zeros(3)

    # Parameter und Optimierer

    def _init_params(self, points: Any, colors: Any) -> dict[str, Any]:
        n = len(points)
        quats = torch.zeros(n, 4)
        quats[:, 0] = 1.0
        init = {
            "means": points.clone(),
            "sh_dc": (colors - 0.5) / SH_C0,
            "opacity": torch.logit(torch.full((n,), _INIT_OPACITY)),
            "log_scales": torch.log(_knn_scale(points))[:, None].repeat(1, 3),
            "quats": quats,
        }
        return {k: torch.nn.Parameter(v.contiguous()) for k, v in init.items()}

    def _make_optimizer(self) -> Any:
        lrs = {
            "means": _LR_MEANS_START * self.extent,
            "sh_dc": _LR_COLOR,
            "opacity": _LR_OPACITY,
            "log_scales": _LR_SCALE,
            "quats": _LR_ROT,
        }
        groups = [{"params": [self.params[k]], "lr": lrs[k], "name": k} for k in self.PARAM_NAMES]
        return torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def _replace(self, keep: Any, extra: dict[str, Any] | None) -> None:
        """Behält die Gaussians in ``keep`` und hängt ``extra`` an. Optimierer-Zustand wird mitgeführt."""
        for group in self.optimizer.param_groups:
            name = group["name"]
            old = group["params"][0]
            new_value = old.detach()[keep]
            if extra is not None:
                new_value = torch.cat([new_value, extra[name]])
            new_param = torch.nn.Parameter(new_value.contiguous())
            state = self.optimizer.state.pop(old, None)
            if state:
                for key in ("exp_avg", "exp_avg_sq"):
                    kept = state[key][keep]
                    if extra is not None:
                        kept = torch.cat([kept, torch.zeros_like(extra[name])])
                    state[key] = kept
                self.optimizer.state[new_param] = state
            group["params"][0] = new_param
            self.params[name] = new_param
        n_new = 0 if extra is None else len(extra["means"])

        def grow(t: Any) -> Any:
            return torch.cat([t[keep], torch.zeros(n_new)])

        self.grad_accum = grow(self.grad_accum)
        self.grad_count = grow(self.grad_count)
        self.max_radius_fraction = grow(self.max_radius_fraction)

    def _densify(self, step: int) -> None:
        s = self.settings
        with torch.no_grad():
            avg = self.grad_accum / self.grad_count.clamp(min=1)
            high = avg >= s.cpu_densify_grad_threshold
            scales = torch.exp(self.params["log_scales"])
            max_scale = scales.max(dim=1).values
            room = s.cpu_max_gaussians - len(max_scale)
            clone = high & (max_scale <= _PERCENT_DENSE * self.extent)
            split = high & (max_scale > _PERCENT_DENSE * self.extent)
            if room <= 0:
                clone[:] = False
                split[:] = False
            elif int(clone.sum()) + int(split.sum()) * (_SPLIT_SAMPLES - 1) > room:
                # Nur die stärksten Kandidaten, bis die Obergrenze erreicht ist
                order = torch.argsort(avg, descending=True)
                budget = room
                allowed = torch.zeros_like(high)
                for idx in order[: int(high.sum())].tolist():
                    cost = 1 if clone[idx] else _SPLIT_SAMPLES - 1
                    if cost > budget:
                        break
                    allowed[idx] = True
                    budget -= cost
                clone &= allowed
                split &= allowed

            extras: dict[str, list[Any]] = {k: [] for k in self.PARAM_NAMES}
            for k in self.PARAM_NAMES:
                extras[k].append(self.params[k].detach()[clone])
            if split.any():
                idx = torch.nonzero(split).squeeze(1)
                rot = _quat_to_rot(self.params["quats"][idx]).repeat(_SPLIT_SAMPLES, 1, 1)
                std = scales[idx].repeat(_SPLIT_SAMPLES, 1)
                offsets = (rot @ (torch.randn_like(std) * std)[..., None]).squeeze(-1)
                extras["means"].append(self.params["means"][idx].repeat(_SPLIT_SAMPLES, 1) + offsets)
                extras["log_scales"].append(torch.log(std / _SPLIT_SHRINK))
                for k in ("sh_dc", "opacity", "quats"):
                    extras[k].append(
                        self.params[k][idx].detach().repeat(_SPLIT_SAMPLES, *[1] * (self.params[k].dim() - 1))
                    )
            opacity = torch.sigmoid(self.params["opacity"])
            prune = opacity < s.cpu_prune_opacity
            if step > s.cpu_opacity_reset_every:
                prune |= self.max_radius_fraction > _MAX_SCREEN_FRACTION
                prune |= max_scale > _MAX_SCREEN_FRACTION * self.extent
            keep = ~(prune | split)
            # Es muss immer mindestens ein Gaussian übrig bleiben
            if not keep.any() and not any(len(e) for v in extras.values() for e in v):
                keep[torch.argmax(opacity)] = True
            extra = {k: torch.cat(v) for k, v in extras.items()}
        self._replace(keep, extra)
        self.grad_accum.zero_()
        self.grad_count.zero_()
        self.max_radius_fraction.zero_()

    def _reset_opacity(self) -> None:
        with torch.no_grad():
            limit = torch.logit(torch.tensor(_OPACITY_RESET))
            self.params["opacity"].data = torch.minimum(self.params["opacity"].data, limit)
        state = self.optimizer.state.get(self.params["opacity"])
        if state:
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()

    # Checkpoints

    def save_checkpoint(self, path: Path, step: int) -> None:
        tmp = path.with_name(path.name + ".tmp")
        torch.save(
            {
                "step": step,
                "params": {k: v.detach() for k, v in self.params.items()},
                "optimizer": self.optimizer.state_dict(),
                "rng": self.rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
            },
            tmp,
        )
        tmp.replace(path)

    def load_checkpoint(self, path: Path) -> None:
        data = torch.load(path, weights_only=False)
        self.params = {k: torch.nn.Parameter(v) for k, v in data["params"].items()}
        self.optimizer = self._make_optimizer()
        self.optimizer.load_state_dict(data["optimizer"])
        self.rng.bit_generator.state = data["rng"]
        torch.set_rng_state(data["torch_rng"])
        n = len(self.params["means"])
        self.grad_accum = torch.zeros(n)
        self.grad_count = torch.zeros(n)
        self.max_radius_fraction = torch.zeros(n)
        self.start_iter = int(data["step"])

    def cloud(self) -> GaussianCloud:
        p = {k: v.detach().numpy().astype(np.float32) for k, v in self.params.items()}
        quats = p["quats"] / np.linalg.norm(p["quats"], axis=1, keepdims=True)
        return GaussianCloud(
            means=p["means"],
            sh_dc=p["sh_dc"],
            sh_rest=np.zeros((len(p["means"]), 0, 3), dtype=np.float32),
            opacity_logit=p["opacity"],
            log_scales=p["log_scales"],
            quats=quats.astype(np.float32),
        )

    # Training

    def _set_lr(self, step: int) -> None:
        t = min(step / max(self.iterations, 1), 1.0)
        lr = math.exp((1 - t) * math.log(_LR_MEANS_START) + t * math.log(_LR_MEANS_END)) * self.extent
        for group in self.optimizer.param_groups:
            if group["name"] == "means":
                group["lr"] = lr

    def train(
        self,
        checkpoint_path: Path,
        on_progress: ProgressFn,
        on_checkpoint: Callable[[int], None],
        cancel: CancelToken,
    ) -> None:
        s = self.settings
        densify_until = int(self.iterations * s.cpu_densify_until_fraction)
        tile = s.cpu_tile_size
        started = time.monotonic()
        ema_loss = 0.0
        for step in range(self.start_iter + 1, self.iterations + 1):
            cancel.raise_if_cancelled()
            self._set_lr(step)
            view = self.train_views[int(self.rng.integers(len(self.train_views)))]
            img, means2d, visible = render(self.params, view, tile, self.background)
            target = view.image
            if view.mask is not None:
                img = img * view.mask
                target = target * view.mask
            l1 = torch.abs(img - target).mean()
            loss = (1 - s.cpu_ssim_weight) * l1 + s.cpu_ssim_weight * (1 - _ssim(img, target))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                if means2d.grad is not None and len(visible) > 0:
                    g = means2d.grad * torch.tensor([view.width / 2, view.height / 2])
                    self.grad_accum[visible] += torch.linalg.norm(g, dim=1)
                    self.grad_count[visible] += 1
            self.optimizer.step()
            ema_loss = 0.9 * ema_loss + 0.1 * loss.item() if step > self.start_iter + 1 else loss.item()

            if s.cpu_densify_from <= step <= densify_until and step % s.cpu_densify_every == 0:
                self._densify(step)
            if step % s.cpu_opacity_reset_every == 0 and step <= densify_until:
                self._reset_opacity()
            if step % s.cpu_log_every == 0 or step == self.iterations:
                done = step - self.start_iter
                elapsed = time.monotonic() - started
                eta = elapsed / done * (self.iterations - step) if done else None
                on_progress(step, self.iterations, ema_loss, eta)
            if step % s.checkpoint_every == 0 or step == self.iterations:
                self.save_checkpoint(checkpoint_path, step)
                on_checkpoint(step)

    def evaluate(self) -> tuple[float | None, float | None]:
        if not self.eval_views:
            return None, None
        psnrs, ssims = [], []
        with torch.no_grad():
            for view in self.eval_views:
                img, _, _ = render(self.params, view, self.settings.cpu_tile_size, self.background)
                img = img.clamp(0, 1)
                mse = torch.mean((img - view.image) ** 2)
                psnrs.append(float(10 * torch.log10(1.0 / mse.clamp(min=1e-10))))
                ssims.append(float(_ssim(img, view.image)))
        return float(np.mean(psnrs)), float(np.mean(ssims))


def train_cpu(
    dataset: Path,
    work_dir: Path,
    iterations: int,
    settings: TrainSettings,
    num_threads: int,
    on_progress: ProgressFn,
    on_preview: Callable[[Path, int], None],
    cancel: CancelToken,
) -> TrainResult:
    started = time.monotonic()
    work_dir.mkdir(parents=True, exist_ok=True)
    trainer = CpuTrainer(dataset, work_dir, iterations, settings, num_threads)
    ckpt = work_dir / "checkpoint.pt"
    if ckpt.is_file():
        trainer.load_checkpoint(ckpt)

    def checkpoint_done(step: int) -> None:
        preview = work_dir / "preview.ply"
        write_ply(preview, trainer.cloud())
        on_preview(preview, step)

    if trainer.start_iter < iterations:
        trainer.train(ckpt, on_progress, checkpoint_done, cancel)
    final = work_dir / "final.ply"
    cloud = trainer.cloud()
    write_ply(final, cloud)
    psnr, ssim = trainer.evaluate()
    return TrainResult(
        ply=final,
        gaussians=len(cloud),
        psnr=psnr,
        ssim=ssim,
        eval_views=len(trainer.eval_views),
        seconds=time.monotonic() - started,
    )
