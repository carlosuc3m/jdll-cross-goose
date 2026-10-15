"""Stable Python entry points. No Appose import or dependency is required."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

import numpy as np
import torch
from torch.nn import functional as F

from . import checkpoints
from .runtime import (
    CancelledError,
    EnvironmentError,
    Events,
    TrainingError,
    ValidationError,
    cleanup,
    clear_error_frames,
    is_oom,
    resolve_device,
)


def load_model(model_path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Load a model directory or explicit checkpoint; 'default' fetches upstream weights.

    The caller owns the returned model and may reuse it with infer(model=model).
    To select best weights explicitly, pass the path to checkpoints/best.ckpt.
    """
    return checkpoints.load(model_path, device)


def _report_failure(events, exc):
    if isinstance(exc, CancelledError):
        status = "cancelled"
    elif isinstance(exc, EnvironmentError):
        status = "environment_failure"
    elif isinstance(exc, ValidationError):
        status = "validation_failure"
    elif isinstance(exc, TrainingError):
        status = "training_failure"
    else:
        status = "failure"
    events.emit({"type": status, "status": status, "message": str(exc)}, check=False)


def infer(
    config: dict,
    inputs: dict,
    *,
    model: torch.nn.Module | None = None,
    callback: Callable[[dict], bool | None] | None = None,
    cancellation: Callable[[], bool] | None = None,
) -> dict:
    """Segment a contiguous CYX NumPy array. Returns YX integer labels and probability.

    config: model_path (or a supplied loaded model), device, checkpoint, tile_size,
    tile_overlap, n_steps, flow_batch_size, object_size, allow_device_fallback,
    cell_probability_threshold. See docs/options.md for defaults and semantics.
    """
    from .inference import _progress, prepare_input, resize_labels, segment_normalized

    events = Events(callback, cancellation)
    device = None
    owned_model = model is None
    original_training = model.training if model is not None else False
    image = result = None
    started = time.perf_counter()
    try:
        events.check()
        allowed = {
            "model_path",
            "device",
            "checkpoint",
            "tile_size",
            "tile_overlap",
            "n_steps",
            "flow_batch_size",
            "object_size",
            "allow_device_fallback",
            "cell_probability_threshold",
        }
        if config.keys() - allowed:
            raise ValueError(f"Unknown inference options: {sorted(config.keys() - allowed)}")
        if "image" not in inputs:
            raise ValueError("inputs requires an 'image' NumPy array in CYX order")
        requested = config.get("device", "cpu")
        device = resolve_device(
            requested, allow_fallback=config.get("allow_device_fallback", False), events=events
        )
        if owned_model:
            if not config.get("model_path"):
                raise ValueError("config.model_path is required when no model is supplied")
            model, metadata = checkpoints.load(
                config["model_path"],
                str(device),
                criterion=config.get("checkpoint", "last"),
                events=events,
            )
        else:
            metadata = getattr(model, "_jdll_metadata", None)
            if metadata is None:
                raise ValueError(
                    "Supplied model must carry saved config metadata; use load_model()"
                )
            if config.get("model_path") and str(config["model_path"]) not in {
                metadata["model_path"],
                metadata.get("checkpoint_path"),
            }:
                raise ValueError("config.model_path conflicts with the supplied model")
            model.to(device)
        model.eval()
        image, scale = prepare_input(inputs["image"], metadata["config"], config)
        result = segment_normalized(model, image, config, events)
        shape = inputs["image"].shape[-2:]
        if result["labels"].shape != shape:
            result["labels"] = resize_labels(result["labels"], shape)
            result["cell_probability"] = F.interpolate(
                torch.from_numpy(result["cell_probability"])[None, None],
                size=shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
        events.check()
        _progress(events, "inference_end", result["total_patches"], result["total_patches"])
        return {
            "outputs": {
                key: np.ascontiguousarray(result[key]) for key in ("labels", "cell_probability")
            },
            "metadata": {
                "model_path": metadata["model_path"],
                "checkpoint_path": metadata["checkpoint_path"],
                "device_requested": requested,
                "device_used": device.type,
                "input_axes": "cyx",
                "output_axes": "yx",
                "scale_factor": scale,
                "total_patches": result["total_patches"],
                "timings": {**result["timings"], "total": time.perf_counter() - started},
            },
        }
    except torch.OutOfMemoryError as exc:
        error = EnvironmentError(
            f"Inference ran out of {device.type if device else 'device'} memory; reduce tile_size or flow_batch_size"
        )
        _report_failure(events, error)
        raise error from exc
    except Exception as exc:
        if is_oom(exc):
            error = EnvironmentError(
                f"Inference ran out of {device.type if device else 'host'} memory"
            )
            _report_failure(events, error)
            raise error from exc
        _report_failure(events, exc)
        raise
    finally:
        if model is not None and not owned_model:
            model.train(original_training)
        if owned_model:
            model = None
        image = result = None
        clear_error_frames(sys.exception())
        cleanup(device)


def train(
    config: dict,
    *,
    callback: Callable[[dict], bool | None] | None = None,
    cancellation: Callable[[], bool] | None = None,
    task: object | None = None,
) -> dict:
    """Train on a canonical JDLL dataset. Cancellation raises CancelledError.

    If task is provided, events are JSON-encoded into task.update(message=...,
    current=..., maximum=...). Callback exceptions are logged and isolated.
    """
    events = Events(callback, cancellation, task)
    try:
        events.check()
        from .training import run

        return run(config, events)
    except Exception as exc:
        _report_failure(events, exc)
        raise
