"""Gemeinsame Basis für Adapter auf externe Programme.

Regeln: Programme werden nie über eine Shell gestartet, Argumente immer als Liste übergeben,
Pfade immer als ``pathlib.Path``. Fehlt ein Programm oder ist es zu alt, gibt es eine
verständliche Meldung mit konkreter Maßnahme.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..errors import JobCancelledError, SplatForgeError, ToolMissingError

Version = tuple[int, ...]


def parse_version(text: str) -> Version | None:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        return None
    return tuple(int(g) for g in match.groups() if g is not None)


def format_version(version: Version) -> str:
    return ".".join(str(v) for v in version)


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str


class CancelToken:
    """Wird von der Pipeline gesetzt, wenn der Nutzer abbricht oder pausiert."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelledError()


class BinaryAdapter:
    """Basis für ein externes Programm mit Versionsprüfung."""

    name: str = ""
    executable: str = ""
    min_version: Version = (0,)
    version_args: Sequence[str] = ("-version",)
    install_hint: str = ""

    def __init__(self, configured_path: Path | None = None) -> None:
        self._configured_path = configured_path
        self._path: Path | None = None
        self._version: Version | None = None

    @property
    def path(self) -> Path:
        if self._path is None:
            self._path = self._locate()
        return self._path

    def available(self) -> bool:
        try:
            self.check()
        except ToolMissingError:
            return False
        return True

    def _locate(self) -> Path:
        if self._configured_path is not None:
            if self._configured_path.is_file():
                return self._configured_path
            raise ToolMissingError(
                f"{self.name} wurde unter {self._configured_path} nicht gefunden.",
                "Den Pfad in den Einstellungen korrigieren oder leer lassen.",
            )
        found = shutil.which(self.executable)
        if found is None:
            raise ToolMissingError(
                f"{self.name} ist nicht installiert oder nicht im Suchpfad.", self.install_hint
            )
        return Path(found)

    def version(self) -> Version | None:
        if self._version is None:
            result = self.run(list(self.version_args), timeout=30, check=False)
            self._version = parse_version(result.stdout + "\n" + result.stderr)
        return self._version

    def check(self) -> Version | None:
        """Prüft, ob das Programm vorhanden und neu genug ist."""
        found = self.version()
        if found is not None and found < self.min_version:
            raise ToolMissingError(
                f"{self.name} {format_version(found)} ist zu alt, "
                f"benötigt wird mindestens {format_version(self.min_version)}.",
                self.install_hint,
            )
        return found

    def run(self, args: Sequence[str | Path], timeout: float | None = None, check: bool = True) -> RunResult:
        cmd = [str(self.path), *(str(a) for a in args)]
        try:
            proc = subprocess.run(  # noqa: S603 - Argumentliste, keine Shell
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SplatForgeError(
                f"{self.name} hat nicht rechtzeitig geantwortet.",
                "Die Datei ist eventuell beschädigt. Eine andere Datei versuchen.",
                details=str(exc),
            ) from exc
        result = RunResult(proc.returncode, proc.stdout, proc.stderr)
        if check and proc.returncode != 0:
            raise self.error_from_output(result)
        return result

    def stream(
        self,
        args: Sequence[str | Path],
        on_line: Callable[[str], None],
        cancel: CancelToken | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        merge_stderr: bool = True,
    ) -> RunResult:
        """Startet das Programm und ruft ``on_line`` für jede Ausgabezeile auf."""
        cmd = [str(self.path), *(str(a) for a in args)]
        proc = subprocess.Popen(  # noqa: S603 - Argumentliste, keine Shell
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            env=env,
        )
        tail: list[str] = []
        stop_watch = threading.Event()

        def watch_cancel() -> None:
            while not stop_watch.wait(0.5):
                if cancel is not None and cancel.cancelled and proc.poll() is None:
                    proc.terminate()

        watcher = threading.Thread(target=watch_cancel, daemon=True)
        watcher.start()
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                tail.append(line)
                del tail[:-200]
                on_line(line)
            proc.wait()
        finally:
            stop_watch.set()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if cancel is not None and cancel.cancelled:
            raise JobCancelledError()
        result = RunResult(proc.returncode, "\n".join(tail), "")
        if proc.returncode != 0:
            raise self.error_from_output(result)
        return result

    def error_from_output(self, result: RunResult) -> SplatForgeError:
        """Übersetzt eine fehlgeschlagene Ausführung in eine verständliche Meldung."""
        output = (result.stderr or result.stdout).strip()
        last = output.splitlines()[-1] if output else ""
        return SplatForgeError(
            f"{self.name} ist mit einem Fehler abgebrochen.",
            "Details stehen im Log. Mit einer anderen Datei oder weniger Frames erneut versuchen.",
            details=f"Exit-Code {result.returncode}: {last}\n{output[-4000:]}",
        )
