"""360°-Material in normale Perspektiv-Ansichten zerlegen (Projektionsmathematik, nur NumPy und OpenCV).

Koordinaten: Das Rig-System ist das der Kamera bei 360°-Aufnahmen, mit den OpenCV-Achsen
(x rechts, y unten, z vorwärts). Jede Ansicht ist eine Lochkamera im selben Mittelpunkt, gedreht um
``yaw`` (nach rechts positiv) und ``pitch`` (nach oben positiv). COLMAP bekommt die Ansichten eines
Zeitpunkts als starres Rig (``cam_from_rig`` = Drehung der Ansicht).

Quellen:
- Equirektangulär: ein Panoramabild, Bildmitte = vorne.
- Dual-Fisheye (Insta360): zwei Objektivbilder mit dem Unified-Kameramodell (Mei) aus der
  Werkskalibrierung. Objektiv 2 schaut nach hinten (180° um die Hochachse).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Image = NDArray[np.uint8]


def rig_from_view(yaw_deg: float, pitch_deg: float) -> Array:
    """Drehung Ansicht → Rig. Spalten: rechts, unten, vorne der Ansicht in Rig-Koordinaten."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    forward = np.array([math.sin(yaw) * math.cos(pitch), -math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])  # bleibt immer waagrecht
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


@dataclass
class View:
    name: str
    yaw_deg: float
    pitch_deg: float
    fov_deg: float
    size: int
    lens: int = 0  # bei Dual-Fisheye: aus welchem Objektiv die Ansicht stammt

    @property
    def focal(self) -> float:
        return self.size / 2 / math.tan(math.radians(self.fov_deg) / 2)

    def cam_from_rig(self) -> Array:
        return rig_from_view(self.yaw_deg, self.pitch_deg).T

    def rays(self) -> Array:
        """Blickstrahlen aller Pixel in Rig-Koordinaten, Form (H, W, 3), normiert."""
        c = (self.size - 1) / 2
        xs, ys = np.meshgrid(np.arange(self.size) - c, np.arange(self.size) - c)
        dirs = np.stack([xs / self.focal, ys / self.focal, np.ones_like(xs, dtype=np.float64)], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        return np.asarray(dirs @ rig_from_view(self.yaw_deg, self.pitch_deg).T)

    def project(self, rays: Array) -> tuple[Array, Array, NDArray[np.bool_]]:
        """Rig-Strahlen → Pixel dieser Ansicht. Gibt x, y und „liegt im Bild“ zurück."""
        cam = rays @ self.cam_from_rig().T
        z = cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = self.focal * cam[..., 0] / z + (self.size - 1) / 2
            y = self.focal * cam[..., 1] / z + (self.size - 1) / 2
        inside = (z > 1e-6) & (x >= -0.5) & (x <= self.size - 0.5) & (y >= -0.5) & (y <= self.size - 0.5)
        return x, y, inside

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def equirect_views(count: int, fov_deg: float, size: int, up_down: bool) -> list[View]:
    """Standard laut Spezifikation: ``count`` Ansichten rund um den Horizont, dazu oben und unten."""
    views = [View(f"v{i:02d}", 360.0 * i / count, 0.0, fov_deg, size) for i in range(count)]
    if up_down:
        views += [
            View(f"v{count:02d}", 0.0, 90.0, fov_deg, size),
            View(f"v{count + 1:02d}", 0.0, -90.0, fov_deg, size),
        ]
    return views


def fisheye_views(tilt_deg: float, fov_deg: float, size: int, lenses: int = 2) -> list[View]:
    """Pro Objektiv eine Ansicht geradeaus und vier um ``tilt_deg`` nach links, rechts, oben, unten."""
    views: list[View] = []
    offsets = [(0.0, 0.0), (-tilt_deg, 0.0), (tilt_deg, 0.0), (0.0, tilt_deg), (0.0, -tilt_deg)]
    for lens in range(lenses):
        for yaw, pitch in offsets:
            views.append(View(f"v{len(views):02d}", 180.0 * lens + yaw, pitch, fov_deg, size, lens))
    return views


# --- Quellen --------------------------------------------------------------------------------------


def equirect_lookup(rays: Array, width: int, height: int) -> tuple[Array, Array]:
    """Rig-Strahlen → Pixel im equirektangulären Bild (Mitte = vorne, oben = −y)."""
    lon = np.arctan2(rays[..., 0], rays[..., 2])
    lat = np.arctan2(-rays[..., 1], np.hypot(rays[..., 0], rays[..., 2]))
    u = (lon / (2 * math.pi) + 0.5) * width - 0.5
    v = (0.5 - lat / math.pi) * height - 0.5
    return u, v


def equirect_rays(width: int, height: int) -> Array:
    """Blickstrahl jedes Pixels eines equirektangulären Bildes (Umkehrung von ``equirect_lookup``)."""
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    lon = (u / width - 0.5) * 2 * math.pi
    lat = (0.5 - v / height) * math.pi
    return np.stack([np.sin(lon) * np.cos(lat), -np.sin(lat), np.cos(lon) * np.cos(lat)], axis=-1)


@dataclass
class FisheyeLens:
    """Unified-Kameramodell (Mei) mit radialer und tangentialer Verzerrung, wie Insta360 es speichert.

    ``lens_from_rig`` dreht Rig-Strahlen in das Objektivsystem (Sensorlage inklusive, z. B. ein um 180°
    gedrehter Sensor). Pixelangaben beziehen sich auf das Bild eines einzelnen Objektivs.
    """

    xi: float
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    width: int = 0
    height: int = 0
    lens_from_rig: Array | None = None
    max_angle_deg: float = 97.0

    def lookup(self, rays: Array) -> tuple[Array, Array, NDArray[np.bool_]]:
        rot = self.lens_from_rig if self.lens_from_rig is not None else np.eye(3)
        d = rays @ rot.T
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        denom = d[..., 2] + self.xi
        valid = d[..., 2] > math.cos(math.radians(self.max_angle_deg))
        with np.errstate(divide="ignore", invalid="ignore"):
            x = d[..., 0] / denom
            y = d[..., 1] / denom
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3
        xd = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yd = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy
        valid &= (u >= 0) & (u <= self.width - 1) & (v >= 0) & (v <= self.height - 1) & (denom > 0)
        return u, v, valid

    def unproject(self, u: Array, v: Array, iterations: int = 20) -> tuple[Array, NDArray[np.bool_]]:
        """Pixel → Blickstrahl im Rig (Umkehrung von ``lookup``). Verzerrung iterativ entfernt."""
        xd = (u - self.cx) / self.fx
        yd = (v - self.cy) / self.fy
        x, y = xd.copy(), yd.copy()
        for _ in range(iterations):
            r2 = x * x + y * y
            radial = 1 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3
            dx = 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
            dy = self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
            x = (xd - dx) / radial
            y = (yd - dy) / radial
        r2 = x * x + y * y
        disc = 1 + (1 - self.xi * self.xi) * r2
        valid = disc >= 0
        lam = (self.xi + np.sqrt(np.maximum(disc, 0))) / (1 + r2)
        d = np.stack([lam * x, lam * y, lam - self.xi], axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        rot = self.lens_from_rig if self.lens_from_rig is not None else np.eye(3)
        rays = np.asarray(d @ rot)
        # Ausserhalb des Bildkreises findet die Iteration keine echte Lösung: per Rundreise aussortieren
        back_u, back_v, ok = self.lookup(rays)
        valid &= ok & (np.abs(back_u - u) < 0.5) & (np.abs(back_v - v) < 0.5)
        return rays, valid

    def scaled(self, width: int, height: int) -> FisheyeLens:
        """Dieselbe Linse für ein Bild anderer Auflösung (z. B. heruntergerechnet)."""
        sx, sy = width / self.width, height / self.height
        return FisheyeLens(
            self.xi, self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5,
            self.k1, self.k2, self.k3, self.p1, self.p2, width, height, self.lens_from_rig,
            self.max_angle_deg,
        )  # fmt: skip

    def center_pixels_per_radian(self) -> float:
        return self.fx / (1 + self.xi)


def rotation_z(deg: float) -> Array:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def rotation_y(deg: float) -> Array:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def lens_rotation(lens: int, roll_deg: float) -> Array:
    """Objektiv 1 schaut nach vorne, Objektiv 2 nach hinten; ``roll_deg`` ist die Sensorlage aus der
    Kalibrierung (z. B. 180° bei einem kopfstehenden Bild)."""
    return rotation_z(roll_deg) @ (rotation_y(180.0) if lens == 1 else np.eye(3))


def default_lens(width: int, height: int, lens: int) -> FisheyeLens:
    """Ersatzwerte ohne Kalibrierung: Bildkreis füllt das Bild, Sichtfeld etwa 195°.

    Hergeleitet aus der Werkskalibrierung einer Insta360 ONE RS (xi ≈ 2, Brennweite ≈ 0,92 × Bildbreite),
    für andere Modelle nur eine Näherung.
    """
    return FisheyeLens(
        xi=2.0, fx=0.92 * width, fy=0.92 * width, cx=(width - 1) / 2, cy=(height - 1) / 2,
        width=width, height=height, lens_from_rig=lens_rotation(lens, 0.0),
    )  # fmt: skip


# --- Rendern --------------------------------------------------------------------------------------


def render_equirect_view(pano: Image, view: View, rays: Array | None = None) -> Image:
    h, w = pano.shape[:2]
    u, v = equirect_lookup(view.rays() if rays is None else rays, w, h)
    out = cv2.remap(
        pano, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    )
    return np.asarray(out, dtype=np.uint8)


def render_fisheye_view(
    image: Image, lens: FisheyeLens, view: View, rays: Array | None = None
) -> tuple[Image, NDArray[np.bool_]]:
    """Gibt die Ansicht und die Pixel zurück, die das Objektiv tatsächlich gesehen hat."""
    u, v, valid = lens.lookup(view.rays() if rays is None else rays)
    u = np.where(valid, u, -1).astype(np.float32)
    v = np.where(valid, v, -1).astype(np.float32)
    out = cv2.remap(image, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.asarray(out, dtype=np.uint8), valid


def nadir_mask(view: View, radius_deg: float, rays: Array | None = None) -> NDArray[np.bool_]:
    """Pixel innerhalb von ``radius_deg`` um den Nadir (senkrecht nach unten, +y im Rig)."""
    if radius_deg <= 0:
        return np.zeros((view.size, view.size), dtype=bool)
    r = view.rays() if rays is None else rays
    return np.asarray(r[..., 1] > math.cos(math.radians(radius_deg)))


def share_masks(
    views: list[View], masks: list[NDArray[np.bool_]], sphere_width: int = 1024
) -> list[NDArray[np.bool_]]:
    """Gleicht Masken über die Ansichtsgrenzen ab: Was in einer Ansicht maskiert ist, wird in allen
    überlappenden Ansichten ebenfalls maskiert (über ein gemeinsames Kugelpanorama)."""
    sphere_height = sphere_width // 2
    sphere_rays = equirect_rays(sphere_width, sphere_height)
    sphere = np.zeros((sphere_height, sphere_width), dtype=bool)
    for view, mask in zip(views, masks, strict=True):
        if not mask.any():
            continue
        x, y, inside = view.project(sphere_rays)
        xi = np.clip(np.round(np.nan_to_num(x)), 0, view.size - 1).astype(np.int64)
        yi = np.clip(np.round(np.nan_to_num(y)), 0, view.size - 1).astype(np.int64)
        sphere |= inside & mask[yi, xi]
    if not sphere.any():
        return [m.copy() for m in masks]
    sphere_img = sphere.astype(np.uint8) * 255
    shared = []
    for view, mask in zip(views, masks, strict=True):
        u, v = equirect_lookup(view.rays(), sphere_width, sphere_height)
        sampled = cv2.remap(
            sphere_img,
            u.astype(np.float32),
            v.astype(np.float32),
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_WRAP,
        )
        shared.append(mask | (sampled > 127))
    return shared


def auto_view_size(source_width: int, fov_deg: float, equirect: bool, lens: FisheyeLens | None = None) -> int:
    """Kantenlänge, bei der die Ansicht etwa die Auflösung der Quelle in der Bildmitte erreicht."""
    if equirect or lens is None:
        pixels_per_radian = source_width / (2 * math.pi)
    else:
        pixels_per_radian = lens.center_pixels_per_radian()
    size = 2 * pixels_per_radian * math.tan(math.radians(fov_deg) / 2)
    return int(size) // 2 * 2
