"""Download und Prüfung von Modellgewichten.

Modelle liegen nicht im Repository, sondern werden beim ersten Gebrauch heruntergeladen und per SHA-256
geprüft. Die Quellen sind auf einen festen Stand (Commit) gepinnt, damit sich ein Modell nicht unbemerkt
ändert. Lizenzen: siehe THIRD_PARTY_LICENSES.md.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .errors import SplatForgeError

HF = "https://huggingface.co"
CHUNK = 1024 * 1024
DOWNLOAD_TIMEOUT_S = 60
ENV_MODELS_DIR = "SPLATFORGE_MODELS_DIR"


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class Model:
    key: str
    description: str
    license: str
    files: tuple[ModelFile, ...]

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


def _hf(repo: str, revision: str, path: str, sha256: str, size: int) -> ModelFile:
    return ModelFile(Path(path).name, f"{HF}/{repo}/resolve/{revision}/{path}", sha256, size)


_RTDETR = ("onnx-community/rtdetr_r18vd_coco_o365", "37122d8b9b89eb8384fba4d688d05051eaf7bb78")
_SAM_TINY = ("onnx-community/sam2.1-hiera-tiny-ONNX", "814a066640debee5a91e70aa401fb8e17e030503")
_SAM_SMALL = ("onnx-community/sam2.1-hiera-small-ONNX", "a7df49d8de14b9d2e4504d1687b0d568f905fd8d")

MODELS: dict[str, Model] = {
    "rtdetr-r18": Model(
        "rtdetr-r18",
        "RT-DETR R18 (COCO), Personen- und Objekterkennung",
        "Apache-2.0",
        (
            _hf(*_RTDETR, "onnx/model.onnx",
                "6c74a62251534f427294342508a162a7cd142418b4dd6222d0d0198851724bfa", 82572357),
        ),
    ),
    "sam2.1-tiny": Model(
        "sam2.1-tiny",
        "SAM 2.1 Hiera Tiny, Segmentierung",
        "Apache-2.0",
        (
            _hf(*_SAM_TINY, "onnx/vision_encoder.onnx",
                "4f30aacd3aaefbca81a0b7fe4c1fc96345570ea0a6f80ced599493d1b3be2e8c", 354238),
            _hf(*_SAM_TINY, "onnx/vision_encoder.onnx_data",
                "e83df9866a5afe68ea7f0f721f18f65137fc3acbf0da1c74e946d363e09c69cc", 134084864),
            _hf(*_SAM_TINY, "onnx/prompt_encoder_mask_decoder.onnx",
                "874414704c5d686db7d206a35f6e15d26563d50c8c4468fccc6739bd7e491dcf", 213114),
            _hf(*_SAM_TINY, "onnx/prompt_encoder_mask_decoder.onnx_data",
                "e9874d900dd4134ed60eab1e97910327c2419e0b2954485d8fd6e7f1a1470f47", 20958208),
        ),
    ),
    "sam2.1-small": Model(
        "sam2.1-small",
        "SAM 2.1 Hiera Small, Segmentierung (genauer, langsamer)",
        "Apache-2.0",
        (
            _hf(*_SAM_SMALL, "onnx/vision_encoder.onnx",
                "aacf1f7137bb6fffcf6bf166abcfabe28f57a76059254f3fb611c4a64a208119", 467440),
            _hf(*_SAM_SMALL, "onnx/vision_encoder.onnx_data",
                "260fd1f0a34e72a3dc79a739e563b4facc0ba75504818b433a1f808e66637456", 162476288),
            _hf(*_SAM_SMALL, "onnx/prompt_encoder_mask_decoder.onnx",
                "079c59b261f723ff5c6a125e69b0170a957b21c58738c28d2b0394ecd0587d7f", 213114),
            _hf(*_SAM_SMALL, "onnx/prompt_encoder_mask_decoder.onnx_data",
                "f9e59a584ab8ced21fa812c211bc01084204db1c9e92a5ef4fb3a49972b4e864", 20958208),
        ),
    ),
}  # fmt: skip


def default_models_dir() -> Path:
    """Cache-Ordner für Modelle, je nach Betriebssystem. Überschreibbar mit SPLATFORGE_MODELS_DIR."""
    if env := os.environ.get(ENV_MODELS_DIR):
        return Path(env)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "SplatForge" / "models"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "SplatForge" / "models"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "splatforge" / "models"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def is_installed(model: Model, models_dir: Path) -> bool:
    folder = models_dir / model.key
    paths = [(folder / f.name, f.size) for f in model.files]
    return all(path.is_file() and path.stat().st_size == size for path, size in paths)


def ensure_model(
    key: str, models_dir: Path | None = None, on_progress: Callable[[int, int], None] | None = None
) -> Path:
    """Stellt sicher, dass ein Modell vollständig und unverändert vorliegt. Gibt seinen Ordner zurück."""
    model = MODELS[key]
    folder = (models_dir or default_models_dir()) / model.key
    folder.mkdir(parents=True, exist_ok=True)
    done = 0
    for f in model.files:
        target = folder / f.name
        if target.is_file() and target.stat().st_size == f.size and _verified(target, f.sha256):
            done += f.size
            continue

        def report(received: int, base: int = done) -> None:
            if on_progress is not None:
                on_progress(base + received, model.size)

        _download(f, target, report)
        done += f.size
    return folder


def _verified(path: Path, sha256: str) -> bool:
    marker = path.with_name(path.name + ".sha256")
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == sha256:
        return True
    ok = sha256_of(path) == sha256
    if ok:
        marker.write_text(sha256, encoding="utf-8")
    return ok


def _download(f: ModelFile, target: Path, on_bytes: Callable[[int], None]) -> None:
    tmp = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(f.url, timeout=DOWNLOAD_TIMEOUT_S) as resp, tmp.open("wb") as out:  # noqa: S310 - feste https-URL
            received = 0
            digest = hashlib.sha256()
            while chunk := resp.read(CHUNK):
                out.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                on_bytes(received)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise SplatForgeError(
            f"Das Modell {f.name} konnte nicht heruntergeladen werden.",
            "Internetverbindung prüfen. Ohne Internet die Modelldateien von einem anderen Rechner in den "
            f"Ordner {target.parent} kopieren oder SPLATFORGE_MODELS_DIR setzen.",
            details=f"{f.url}: {exc}",
        ) from exc
    if digest.hexdigest() != f.sha256:
        tmp.unlink(missing_ok=True)
        raise SplatForgeError(
            f"Das heruntergeladene Modell {f.name} ist beschädigt (Prüfsumme stimmt nicht).",
            "Erneut versuchen. Bleibt der Fehler, bitte melden.",
            details=f"erwartet {f.sha256}, erhalten {digest.hexdigest()}",
        )
    shutil.move(str(tmp), target)
    target.with_name(target.name + ".sha256").write_text(f.sha256, encoding="utf-8")
