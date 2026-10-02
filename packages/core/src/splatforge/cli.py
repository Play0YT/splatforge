"""Kommandozeile: ``splatforge run input.mp4 --preset standard --out ./result``.

Fortschritt wird als JSON-Lines ausgegeben (eine Zeile pro Event), wenn die Ausgabe nicht in ein
Terminal geht oder ``--json`` gesetzt ist. Im Terminal erscheinen lesbare Zeilen.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any

from . import __version__
from .adapters import CancelToken
from .config import JobConfig, Preset, TrainBackend
from .errors import SplatForgeError
from .events import EventSink
from .job import JobDir
from .pipeline import ExitCode, Pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="splatforge", description="Video zu 3D Gaussian Splat")
    parser.add_argument("--version", action="version", version=f"splatforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Neuen Job starten")
    run.add_argument("inputs", nargs="+", type=Path, help="Videodateien derselben Szene oder ein Bildordner")
    run.add_argument("--out", type=Path, required=True, help="Job-Ordner (leer oder neu)")
    run.add_argument("--preset", choices=[p.value for p in Preset], default=Preset.STANDARD.value)
    run.add_argument("--config", type=Path, help="JSON-Datei mit weiteren Job-Einstellungen")
    run.add_argument("--frames", type=int, help="Anzahl ausgewählter Frames (überschreibt das Preset)")
    run.add_argument("--max-edge", type=int, help="Maximale Bildkante in Pixeln")
    run.add_argument("--iterations", type=int, help="Trainings-Iterationen")
    run.add_argument("--backend", choices=[b.value for b in TrainBackend], help="Trainings-Backend")
    run.add_argument("--threads", type=int, help="Anzahl CPU-Threads (0 = alle)")
    _brush_arg(run)
    _output_args(run)

    resume = sub.add_parser("resume", help="Unterbrochenen Job fortsetzen")
    resume.add_argument("job", type=Path, help="Job-Ordner")
    _brush_arg(resume)
    _output_args(resume)

    analyze = sub.add_parser("analyze", help="Eingabe prüfen, ohne einen Job anzulegen")
    analyze.add_argument("input", type=Path)

    hardware = sub.add_parser("hardware", help="Erkannte Hardware und Backends anzeigen")
    _brush_arg(hardware)

    schema = sub.add_parser("schema", help="JSON-Schemas für Job-Konfiguration und Events schreiben")
    schema.add_argument("--out", type=Path, required=True)
    schema.add_argument("--force", action="store_true", help="Bestehende, abweichende Dateien ersetzen")
    return parser


def _brush_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--brush", type=Path, help="Pfad zur Brush-Programmdatei (sonst Suche im Suchpfad)")


def _output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="Events immer als JSON-Lines ausgeben")


def _load_config(args: argparse.Namespace) -> JobConfig:
    data: dict[str, Any] = {}
    if args.config:
        data = json.loads(args.config.read_text(encoding="utf-8"))
    data["inputs"] = [str(p) for p in args.inputs]
    data["preset"] = args.preset
    for key, value in (
        ("frames", args.frames),
        ("max_image_edge", args.max_edge),
        ("iterations", args.iterations),
    ):
        if value is not None:
            data[key] = value
    if args.backend:
        data.setdefault("train", {})["backend"] = args.backend
    if args.brush is not None:
        data.setdefault("tools", {})["brush"] = str(args.brush)
    if args.threads is not None:
        data.setdefault("resources", {})["num_threads"] = args.threads
    return JobConfig.model_validate(data)


def _run_pipeline(job: JobDir, config: JobConfig, as_json: bool) -> int:
    human = not as_json and sys.stdout.isatty()
    events = EventSink(log_file=job.events_file, stream=sys.stdout, human=human)
    cancel = CancelToken()

    def on_signal(signum: int, frame: FrameType | None) -> None:
        if cancel.cancelled:  # zweites Strg+C: sofort beenden
            raise KeyboardInterrupt
        cancel.cancel()

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)
    return int(Pipeline(job, config, events, cancel=cancel).run())


def _write_schemas(out: Path, force: bool) -> int:
    from .events import ProgressEvent

    out.mkdir(parents=True, exist_ok=True)
    files = {
        "job-config.schema.json": JobConfig.model_json_schema(),
        "progress-event.schema.json": ProgressEvent.model_json_schema(),
    }
    status = 0
    for name, schema in files.items():
        path = out / name
        text = json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != text and not force:
            print(f"{path} weicht ab und wurde nicht ersetzt (mit --force ersetzen).", file=sys.stderr)
            status = 1
            continue
        path.write_text(text, encoding="utf-8")
    return status


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            config = _load_config(args)
            job = JobDir(args.out)
            job.create(config)
            return _run_pipeline(job, job.load(), args.json)
        if args.command == "resume":
            job = JobDir(args.job)
            config = job.load()
            if args.brush is not None:
                # Gilt nur für diesen Lauf; job.json bleibt unverändert.
                tools = config.tools.model_copy(update={"brush": args.brush})
                config = config.model_copy(update={"tools": tools})
            return _run_pipeline(job, config, args.json)
        if args.command == "analyze":
            return _analyze(args.input)
        if args.command == "hardware":
            from .adapters import BrushAdapter
            from .hardware import detect

            info = detect(BrushAdapter(args.brush))
            print(json.dumps(info.to_dict(), indent=2, ensure_ascii=False))
            return 0
        if args.command == "schema":
            return _write_schemas(args.out, args.force)
    except SplatForgeError as exc:
        print(f"Fehler: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"Was tun: {exc.hint}", file=sys.stderr)
        return int(ExitCode.FAILED)
    return int(ExitCode.USAGE)


def _analyze(path: Path) -> int:
    import tempfile

    from .stages.analyze import ANALYSIS_FILE, AnalyzeStage
    from .stages.base import StageContext, Tools

    config = JobConfig(inputs=[path])
    with tempfile.TemporaryDirectory() as tmp:
        job = JobDir(Path(tmp))
        ctx = StageContext(job=job, config=config, events=EventSink(), tools=Tools.from_config(config))
        stage = AnalyzeStage()
        try:
            stage.run(ctx)
        except SplatForgeError as exc:
            print(json.dumps({"ok": False, "message": exc.message, "hint": exc.hint}, ensure_ascii=False))
            return int(ExitCode.FAILED)
        result = json.loads((stage.out_dir(ctx) / ANALYSIS_FILE).read_text(encoding="utf-8"))
    print(json.dumps({"ok": True, **result}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
