"""Lesen und Schreiben von Gaussian-Splat-PLY-Dateien (Format der 3DGS-Referenzimplementierung).

Eigene Implementierung, weil das verbreitete Paket ``plyfile`` unter GPL steht.
Eigenschaften je Gaussian: x y z nx ny nz f_dc_0..2 f_rest_* opacity scale_0..2 rot_0..3.
Opazität als Logit, Skalierung logarithmisch, Farbe als Kugelflächenfunktions-Koeffizienten.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SH_C0 = 0.28209479177387814

Array = NDArray[np.float32]


@dataclass
class GaussianCloud:
    means: Array  # (N, 3)
    sh_dc: Array  # (N, 3)
    sh_rest: Array  # (N, K, 3), K = (Grad+1)^2 - 1
    opacity_logit: Array  # (N,)
    log_scales: Array  # (N, 3)
    quats: Array  # (N, 4), w x y z

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def subset(self, keep: NDArray[np.bool_]) -> GaussianCloud:
        return GaussianCloud(
            self.means[keep],
            self.sh_dc[keep],
            self.sh_rest[keep],
            self.opacity_logit[keep],
            self.log_scales[keep],
            self.quats[keep],
        )


def _property_names(rest: int) -> list[str]:
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(rest * 3)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    return names


def write_ply(path: Path, cloud: GaussianCloud) -> None:
    n = len(cloud)
    rest = cloud.sh_rest.shape[1] if cloud.sh_rest.ndim == 3 else 0
    names = _property_names(rest)
    data = np.zeros((n, len(names)), dtype=np.float32)
    data[:, 0:3] = cloud.means
    data[:, 6:9] = cloud.sh_dc
    if rest:
        # Referenzformat: erst alle Koeffizienten für Rot, dann Grün, dann Blau
        data[:, 9 : 9 + rest * 3] = cloud.sh_rest.transpose(0, 2, 1).reshape(n, rest * 3)
    base = 9 + rest * 3
    data[:, base] = cloud.opacity_logit
    data[:, base + 1 : base + 4] = cloud.log_scales
    data[:, base + 4 : base + 8] = cloud.quats
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {name}" for name in names]
    header.append("end_header")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(data.astype("<f4").tobytes())
    tmp.replace(path)


_TYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "int": "<i4",
    "uint": "<u4",
    "short": "<i2",
    "ushort": "<u2",
}


def read_ply(path: Path) -> GaussianCloud:
    with path.open("rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path} ist keine PLY-Datei")
        count = 0
        props: list[tuple[str, str]] = []
        fmt = ""
        in_vertex = False
        while True:
            line = fh.readline().decode("ascii", errors="replace").strip()
            if not line:
                raise ValueError(f"{path}: Header unvollständig")
            if line == "end_header":
                break
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                props.append((parts[2], _TYPES[parts[1]]))
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: nur binary_little_endian wird unterstützt")
        dtype = np.dtype(props)
        raw = np.frombuffer(fh.read(dtype.itemsize * count), dtype=dtype, count=count)

    def cols(names: list[str]) -> Array:
        return np.stack([raw[n].astype(np.float32) for n in names], axis=1)

    names = {p[0] for p in props}
    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n.split("_")[-1]))
    rest = len(rest_names) // 3
    sh_rest = cols(rest_names).reshape(count, 3, rest).transpose(0, 2, 1) if rest else np.zeros((count, 0, 3))
    return GaussianCloud(
        means=cols(["x", "y", "z"]),
        sh_dc=cols(["f_dc_0", "f_dc_1", "f_dc_2"]),
        sh_rest=sh_rest.astype(np.float32),
        opacity_logit=raw["opacity"].astype(np.float32),
        log_scales=cols(["scale_0", "scale_1", "scale_2"]),
        quats=cols(["rot_0", "rot_1", "rot_2", "rot_3"]),
    )
