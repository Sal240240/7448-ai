"""Structured run log -- one append-only record of everything the pipeline does.

Every fetch, training run, and backtest appends a JSON line here. The point is
that "what is this model doing, on what data, and what came out" should be
answerable by reading one file, not by remembering which terminal a job was
started in. The webapp's model-transparency view reads the same file, so what
a visitor sees is the actual run history rather than numbers retyped into a
template.

Kept as JSONL rather than a database because it has to survive being read
mid-run by a separate process (the webapp) while a training job is still
appending to it, and because a plain text log is reviewable in a diff.

Usage:
    from runlog import log_run

    with log_run("train_elastic_net", params={"min_train_measured": 100}) as run:
        ...
        run.record(median_test_r=0.458, n_targets=1098)
"""
from __future__ import annotations

import json
import os
import platform
import socket
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "experiments" / "run_log.jsonl"


class RunRecord:
    """Accumulates metrics/notes for one pipeline step; written out on exit."""

    def __init__(self, step: str, params: dict | None = None):
        self.step = step
        self.params = params or {}
        self.metrics: dict = {}
        self.notes: list[str] = []
        self.artifacts: list[str] = []

    def record(self, **metrics) -> None:
        """Attach result metrics. Numpy scalars are coerced so json.dumps won't choke."""
        for key, value in metrics.items():
            self.metrics[key] = _jsonable(value)

    def note(self, message: str) -> None:
        """Attach a human-readable observation -- caveats, skipped data, judgment calls."""
        self.notes.append(message)

    def artifact(self, path) -> None:
        """Record a file this step produced, relative to the repo root where possible."""
        path = Path(path)
        root = LOG_PATH.parent.parent
        try:
            self.artifacts.append(str(path.resolve().relative_to(root)).replace("\\", "/"))
        except ValueError:
            self.artifacts.append(str(path))


def _jsonable(value):
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@contextmanager
def log_run(step: str, params: dict | None = None):
    """Time a pipeline step and append one JSONL record, success or failure.

    Failures are logged too -- a run log that only records successes is how you
    end up believing a stale artifact came from the last run.
    """
    record = RunRecord(step, params)
    started = time.time()
    status, error = "ok", None
    try:
        yield record
    except BaseException as e:  # noqa: BLE001 -- log then re-raise, including KeyboardInterrupt
        status = "interrupted" if isinstance(e, KeyboardInterrupt) else "failed"
        error = f"{type(e).__name__}: {e}"
        record.note(traceback.format_exc(limit=3).strip().splitlines()[-1])
        raise
    finally:
        entry = {
            "step": step,
            "status": status,
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - started, 2),
            "params": _jsonable(record.params),
            "metrics": record.metrics,
            "artifacts": record.artifacts,
            "notes": record.notes,
            "error": error,
            "host": {"machine": socket.gethostname(), "python": platform.python_version(), "pid": os.getpid()},
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def read_runs(step: str | None = None, limit: int | None = None) -> list[dict]:
    """Read back the log, newest first. Tolerates a partially-written trailing line."""
    if not LOG_PATH.exists():
        return []
    runs = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if step:
        runs = [r for r in runs if r.get("step") == step]
    runs.reverse()
    return runs[:limit] if limit else runs
