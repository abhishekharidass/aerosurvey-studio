"""Background worker that runs pipeline stages off the UI thread."""
from __future__ import annotations

import os
from typing import List

from PySide6.QtCore import QThread, Signal

from ..model.project import Chunk
from .stages import Stage, StageContext, stage_by_key


class PipelineWorker(QThread):
    log = Signal(str, str)                 # message, level
    stage_started = Signal(str)            # stage name
    stage_progress = Signal(int)           # 0..100 within current stage
    overall_progress = Signal(int)         # 0..100 across the run
    stage_finished = Signal(str, bool)     # stage key, ok
    outputs_changed = Signal()             # chunk.outputs was updated
    run_finished = Signal(bool)            # overall ok / aborted

    def __init__(self, chunk: Chunk, stage_keys: List[str], workdir: str):
        super().__init__()
        self.chunk = chunk
        self.stage_keys = stage_keys
        self.workdir = workdir
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:
        os.makedirs(self.workdir, exist_ok=True)
        stages: List[Stage] = [stage_by_key(k) for k in self.stage_keys]
        n = len(stages)
        ok_all = True
        for i, stage in enumerate(stages):
            if self._cancel:
                self.log.emit("Processing cancelled by user.", "warn")
                ok_all = False
                break
            self.stage_started.emit(stage.name)
            self.log.emit(f"=== {stage.name} ({stage.engine.upper()}) ===", "stage")

            def progress(pct, _i=i):
                self.stage_progress.emit(pct)
                self.overall_progress.emit(int((_i + pct / 100.0) / n * 100))

            ctx = StageContext(self.chunk, self.workdir,
                               lambda m, lvl="info": self.log.emit(m, lvl),
                               progress, self._cancelled)
            try:
                ok = stage.run(ctx)
            except Exception as exc:  # keep the app alive on stage errors
                self.log.emit(f"{stage.name} failed: {exc}", "error")
                ok = False
            self.stage_finished.emit(stage.key, ok)
            if ok:
                self.outputs_changed.emit()
            else:
                ok_all = False
                if not self._cancel:
                    self.log.emit(f"Stopped: '{stage.name}' did not complete.", "error")
                break
        if ok_all:
            self.overall_progress.emit(100)
        self.run_finished.emit(ok_all)
