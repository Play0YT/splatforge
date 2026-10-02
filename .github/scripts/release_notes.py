"""Prüft, ob der Tag zur Paketversion passt, und gibt den Release-Text aus.

Der Text enthält alle Changelog-Abschnitte seit dem letzten Release desselben Pakets
(beim ersten Release also den ganzen Changelog).

Aufruf: python release_notes.py core-v0.1.3 packages/core
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

SECTION = re.compile(r"^## \[(?P<version>[^\]]+)\][^\n]*\n(?P<body>.*?)(?=^## \[|\Z)", re.M | re.S)


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", version))


def previous_release(prefix: str, current: str) -> str | None:
    result = subprocess.run(  # noqa: S603 - feste Argumentliste
        ["git", "tag", "--list", f"{prefix}*"],  # noqa: S607 - git aus dem Suchpfad des Runners
        capture_output=True,
        text=True,
        check=True,
    )
    versions = [t.removeprefix(prefix) for t in result.stdout.split() if t != f"{prefix}{current}"]
    older = [v for v in versions if _key(v) < _key(current)]
    return max(older, key=_key) if older else None


def main() -> int:
    tag, package = sys.argv[1], Path(sys.argv[2])
    prefix, tag_version = tag.rsplit("-v", 1)[0] + "-v", tag.rsplit("-v", 1)[1]
    version = tomllib.loads((package / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if tag_version != version:
        print(f"Tag {tag} passt nicht zur Paketversion {version}.", file=sys.stderr)
        return 1
    previous = previous_release(prefix, version)
    changelog = (package / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = [
        m for m in SECTION.finditer(changelog)
        if _key(m["version"]) <= _key(version) and (previous is None or _key(m["version"]) > _key(previous))
    ]  # fmt: skip
    if not sections or sections[0]["version"] != version:
        print(f"Kein Changelog-Eintrag für {version} gefunden.", file=sys.stderr)
        return 1
    if previous is None:
        print("Erstes Release. Enthält alle Änderungen seit Beginn des Projekts.\n")
    else:
        print(f"Änderungen seit {previous}.\n")
    for m in sections:
        print(f"## {m['version']}\n\n{m['body'].strip()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
