"""Events, explicit device resolution, atomic writes, and lifecycle helpers."""

from __future__ import annotations

import gc
import json
import logging
import os
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch

log = logging.getLogger("jdll_cross_goose")


class CancelledError(RuntimeError):
    """The caller requested cancellation; completed checkpoints remain on disk."""


class EnvironmentError(RuntimeError):
    """A required device, dependency, or runtime resource is unavailable."""


class TrainingError(RuntimeError):
    """Training failed (as distinct from validation or environment failure)."""


class ValidationError(TrainingError):
    """Validation failed; the run must not report successful completion."""


class Events:
    def __init__(self, callback=None, cancellation=None, task=None):
        self.callback = callback
        self.cancellation = cancellation
        self.task = task
        self.cancelled = False

    def check(self):
        if self.cancelled or (self.cancellation is not None and self.cancellation()):
            self.cancelled = True
            raise CancelledError("Operation cancelled by the caller")

    def emit(self, event, *, check=True):
        if check:
            self.check()
        log.info(event.get("message", event.get("type", "")))
        if self.callback is not None:
            try:
                if self.callback(dict(event)) is False:
                    self.cancelled = True
            except Exception:
                log.exception("Progress callback failed; continuing the operation")
        if self.task is not None:
            try:
                self.task.update(
                    message=json.dumps(event, allow_nan=False),
                    current=event.get("current", 0),
                    maximum=event.get("maximum", 1),
                )
            except Exception:
                log.exception("task.update failed; continuing the operation")
        if check:
            self.check()

    def message(self, message, *, kind="message", **fields):
        self.emit({"type": kind, "message": message, **fields})


def resolve_device(requested="cpu", *, allow_fallback=False, events=None):
    requested = str(requested).lower()
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'mps'")
    available = requested == "cpu"
    if requested == "cuda":
        available = torch.cuda.is_available()
    elif requested == "mps":
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if available:
        return torch.device(requested)
    if not allow_fallback:
        raise EnvironmentError(
            f"Requested {requested} device is unavailable in this PyTorch environment"
        )
    if events is not None:
        events.message(f"Requested {requested} is unavailable; using CPU", kind="warning")
    else:
        log.warning("Requested %s is unavailable; using CPU", requested)
    return torch.device("cpu")


def cleanup(device):
    gc.collect()
    if device is None:
        return
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        log.exception("Device cache cleanup failed")


def clear_error_frames(error):
    """Release tensors retained by completed traceback frames; keep traceback locations.

    A GUI may keep an exception for display indefinitely. Without clearing frame
    locals, that exception can retain a failed forward pass and all its device
    tensors even after the model has otherwise been released.
    """
    seen = set()
    pending = [error] if error is not None else []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)


def is_oom(error):
    return isinstance(error, (torch.OutOfMemoryError, MemoryError)) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def positive_int(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value
