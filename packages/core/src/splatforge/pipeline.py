"""Pipeline-Runner: führt die Stufen der Reihe nach aus und überspringt fertige Stufen."""

from __future__ import annotations

import shutil
import time
import traceback
from enum import IntEnum

from .adapters import CancelToken
from .config import JobConfig
from .errors import InsufficientResourcesError, JobCancelledError, SplatForgeError
from .events import EventSink, EventType, ProgressEvent
from .job import JobDir
from .stages.analyze import AnalyzeStage
from .stages.base import Stage, StageContext, Tools
from .stages.export import ExportStage, write_report
from .stages.extract import ExtractStage
from .stages.mask import MaskStage
from .stages.select import SelectStage
from .stages.sfm import SfmStage
from .stages.train import TrainStage

MB = 1024 * 1024


class ExitCode(IntEnum):
    OK = 0
    FAILED = 1
    USAGE = 2
    CANCELLED = 130


def default_stages() -> list[Stage]:
    # Stufe 4 (360°-Aufbereitung) folgt mit Meilenstein 3.
    return [
        AnalyzeStage(),
        ExtractStage(),
        SelectStage(),
        MaskStage(),
        SfmStage(),
        TrainStage(),
        ExportStage(),
    ]


class Pipeline:
    def __init__(
        self,
        job: JobDir,
        config: JobConfig,
        events: EventSink,
        cancel: CancelToken | None = None,
        stages: list[Stage] | None = None,
        tools: Tools | None = None,
    ) -> None:
        self.ctx = StageContext(
            job=job,
            config=config,
            events=events,
            tools=tools or Tools.from_config(config),
            cancel=cancel or CancelToken(),
        )
        self.stages = stages if stages is not None else default_stages()

    def _check_disk(self, stage: Stage) -> None:
        ctx = self.ctx
        needed = stage.estimate_disk_bytes(ctx) + ctx.config.resources.min_free_disk_mb * MB
        free = shutil.disk_usage(ctx.job.root).free
        if free < needed:
            raise InsufficientResourcesError(
                f"Zu wenig freier Speicherplatz für die Stufe '{stage.name}': "
                f"{free // MB} MB frei, etwa {needed // MB} MB nötig.",
                "Speicherplatz freigeben, einen anderen Ausgabeordner wählen oder weniger Frames verwenden.",
            )

    def run(self) -> ExitCode:
        ctx = self.ctx
        events = ctx.events
        active = [s for s in self.stages if s.applies(ctx)]
        total_weight = sum(s.weight for s in active) or 1.0
        started = time.monotonic()
        events.emit(ProgressEvent(type=EventType.JOB_STARTED, message=f"Job in {ctx.job.root}", percent=0))
        done_weight = 0.0
        try:
            for stage in active:
                if not stage.is_done(ctx):
                    stage.preflight(ctx)
            for index, stage in enumerate(active):
                events.stage, events.stage_index, events.stage_count = stage.name, index, len(active)
                events.stage_weight_done = done_weight / total_weight
                events.stage_weight = stage.weight / total_weight
                if stage.is_done(ctx):
                    events.emit(ProgressEvent(type=EventType.STAGE_SKIPPED, message="bereits erledigt"))
                    done_weight += stage.weight
                    continue
                ctx.cancel.raise_if_cancelled()
                stage.cleanup(ctx)
                self._check_disk(stage)
                events.emit(
                    ProgressEvent(
                        type=EventType.STAGE_STARTED,
                        message=stage.name,
                        eta_seconds=stage.estimate_duration(ctx),
                    )
                )
                warnings_before = len(ctx.warnings)
                stage_start = time.monotonic()
                info = stage.run(ctx)
                info["seconds"] = round(time.monotonic() - stage_start, 1)
                info["warnings"] = ctx.warnings[warnings_before:]
                ctx.job.mark_done(stage.dirname, info)
                done_weight += stage.weight
                events.emit(
                    ProgressEvent(
                        type=EventType.STAGE_FINISHED,
                        message=f"fertig nach {info['seconds']} s",
                        percent=round(done_weight / total_weight * 100, 2),
                    )
                )
        except JobCancelledError:
            events.stage = None
            events.emit(
                ProgressEvent(
                    type=EventType.JOB_CANCELLED,
                    message="Abgebrochen. Der Job kann später mit 'splatforge resume' fortgesetzt werden.",
                )
            )
            return ExitCode.CANCELLED
        except SplatForgeError as exc:
            if exc.details:
                events.log(exc.details)
            events.emit(ProgressEvent(type=EventType.JOB_FAILED, message=exc.message, hint=exc.hint or None))
            return ExitCode.FAILED
        except Exception as exc:  # unerwartet: technische Details ins Log
            events.log(traceback.format_exc())
            events.emit(
                ProgressEvent(
                    type=EventType.JOB_FAILED,
                    message=f"Unerwarteter Fehler: {exc}",
                    hint="Den Job mit 'splatforge resume' erneut starten. Hilft das nicht, bitte das Log "
                    "(events.jsonl im Job-Ordner) melden.",
                )
            )
            return ExitCode.FAILED

        events.stage = None
        write_report(ctx, [(s.name, s.dirname) for s in active], time.monotonic() - started)
        events.emit(
            ProgressEvent(
                type=EventType.JOB_FINISHED,
                percent=100,
                message="Fertig",
                data={
                    "ply": str(ctx.job.stage_dir(ExportStage.dirname) / "splat.ply"),
                    "report": str(ctx.job.stage_dir(ExportStage.dirname) / "report.json"),
                },
            )
        )
        return ExitCode.OK
