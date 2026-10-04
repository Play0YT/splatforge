"""Szene ausrichten und Kamerabewegung bewerten (nur NumPy, ohne pycolmap).

COLMAP legt das Koordinatensystem beliebig fest. Viewer zeigen den Splat dann schief an; Brush schätzt die
Richtung „oben“ zwar selbst, aber aus der Kamerabahn, was beim Drehen auf der Stelle oder bei wackliger
Kamera scheitert. Deshalb wird die Szene hier so gedreht, dass oben der Richtung −Y entspricht
(Konvention von COLMAP/OpenCV und Brush: ``comment vertical axis: y`` in der PLY).

Die Richtung „oben“ kommt aus den Kameraachsen: Wer filmt, hält die Kamera meist ohne Schräglage. Die
rechte Achse jeder Kamera liegt dann waagrecht, auch wenn die Kamera nach unten oder oben zeigt. „Oben“
ist die Richtung, die zu allen rechten Achsen senkrecht steht. Idee wie in nerfstudio
(``auto_orient_and_center_poses``, Methode „vertical“, Apache-2.0); eigene Umsetzung.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
WORLD_UP: Array = np.array([0.0, -1.0, 0.0])
# Haben sich die rechten Achsen kaum gedreht (unter etwa 3°), ist die Richtung senkrecht zu ihnen nicht
# eindeutig. Ausserdem muss die Drehung um die Hochachse deutlich grösser sein als das Kippen durch
# Wackeln (zweiter gegen dritten Singulärwert). Sonst gilt die mittlere Oben-Achse der Kameras.
MIN_SPREAD = 0.05
MIN_SPREAD_OVER_SHAKE = 1.5


@dataclass
class Orientation:
    rotation: Array  # neue Welt aus alter Welt (3×3)
    translation: Array  # nach der Drehung: Szenenmitte in den Ursprung
    tilt_deg: float  # wie schief die Szene vorher lag


def estimate_up(cam_from_world: list[Array]) -> Array:
    """Richtung „oben“ in Weltkoordinaten aus den Kameradrehungen (Welt → Kamera, OpenCV-Achsen)."""
    rights = np.array([r[0] for r in cam_from_world])  # Kamera-x (rechts) in Weltkoordinaten
    ups = np.array([-r[1] for r in cam_from_world])  # Kamera −y (oben) in Weltkoordinaten
    mean_up = ups.mean(axis=0)
    mean_up /= np.linalg.norm(mean_up)
    _, s, vt = np.linalg.svd(rights, full_matrices=False)
    if s[1] > MIN_SPREAD * np.sqrt(len(rights)) and s[1] > MIN_SPREAD_OVER_SHAKE * s[2]:
        up = vt[2]
    else:
        # Kaum Drehung: mittleres Oben, ohne Anteil der (gemeinsamen) rechten Achse
        up = mean_up - vt[0] * float(mean_up @ vt[0])
    up = np.asarray(up / np.linalg.norm(up), dtype=np.float64)
    return up if float(up @ mean_up) >= 0 else -up


def rotation_between(a: Array, b: Array) -> Array:
    """Kleinste Drehung, die den Einheitsvektor ``a`` auf ``b`` abbildet."""
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1 + 1e-9:  # entgegengesetzt: 180° um eine beliebige senkrechte Achse
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.asarray(np.eye(3) + vx + vx @ vx / (1 + c), dtype=np.float64)


def orientation(cam_from_world: list[Array], points: Array) -> Orientation:
    """Drehung, die oben auf −Y legt, und Verschiebung, die die Szenenmitte (Median der Punkte) in den
    Ursprung legt."""
    up = estimate_up(cam_from_world)
    rot = rotation_between(up, WORLD_UP)
    center = np.median(points, axis=0) if len(points) else np.zeros(3)
    tilt = float(np.degrees(np.arccos(np.clip(up @ WORLD_UP, -1.0, 1.0))))
    return Orientation(rotation=rot, translation=-(rot @ center), tilt_deg=tilt)


def baseline_ratio(centers: Array, points: Array) -> float:
    """Wie weit sich die Kamera bewegt hat, im Verhältnis zum Abstand der Szene.

    Grösster Abstand zweier Kamerapositionen geteilt durch den mittleren Abstand der Punkte zur
    Kameramitte. Nahe 0 heisst: Die Kamera hat sich nur gedreht. Dann fehlt die Tiefe, und der Splat wird
    unbrauchbar, auch wenn COLMAP alle Bilder verortet.
    """
    if len(centers) < 2 or len(points) == 0:
        return 0.0
    mid = centers.mean(axis=0)
    depth = float(np.median(np.linalg.norm(points - mid, axis=1)))
    if depth <= 0:
        return 0.0
    diffs = centers[:, None, :] - centers[None, :, :]
    return float(np.sqrt((diffs**2).sum(-1)).max() / depth)
