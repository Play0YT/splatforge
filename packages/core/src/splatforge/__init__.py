"""SplatForge Core: wandelt Videos in 3D Gaussian Splats um."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("splatforge")
except PackageNotFoundError:  # pragma: no cover - nur bei Ausführung ohne Installation
    __version__ = "0.0.0"
