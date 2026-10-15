"""Model configuration, deterministic checkpoint selection, and pretrained assets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from importlib.resources import files
from pathlib import Path

import torch
import yaml

from .runtime import (
    EnvironmentError,
    atomic_path,
    cleanup,
    clear_error_frames,
    is_oom,
    resolve_device,
)

UPSTREAM_REVISION = "4e7ffa3c71ad0c5d59b364dd3bf0e5787c8bd79f"
WEIGHTS_SHA256 = "03d7f4b3fa4f1e2e4e059d0c15c50ca2ebd4506334460033fd8161b446f8f2b9"


def default_config():
    return yaml.safe_load(
        files("jdll_cross_goose").joinpath("configs/upstream-default.yaml").read_text()
    )


def download_default(output_dir=None, *, events=None):
    """Fetch the pinned upstream model, verifying its SHA-256 before installation."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    directory = Path(
        output_dir or cache / "jdll-cross-goose" / UPSTREAM_REVISION / "default"
    ).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "weights.ckpt"
    config = directory / "config.yaml"
    if weights.exists():
        with weights.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != WEIGHTS_SHA256:
            raise ValueError(f"Existing weights fail checksum validation: {weights}")
    else:
        if events:
            events.message(f"Downloading upstream pretrained model to {directory}")
        url = (
            f"https://raw.githubusercontent.com/bia-pasteur/cross-goose/{UPSTREAM_REVISION}"
            "/crossgoose/saved_models/default/weights.ckpt"
        )
        digest = hashlib.sha256()
        with atomic_path(weights) as temporary:
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as dest:
                while chunk := source.read(1024 * 1024):
                    if events:
                        events.check()
                    digest.update(chunk)
                    dest.write(chunk)
            if digest.hexdigest() != WEIGHTS_SHA256:
                raise ValueError("Downloaded pretrained weights failed SHA-256 validation")
    expected = files("jdll_cross_goose").joinpath("configs/pretrained-default.yaml").read_text()
    if config.exists() and config.read_text() != expected:
        raise ValueError(f"Refusing to overwrite a different config: {config}")
    if not config.exists():
        with atomic_path(config) as temporary:
            temporary.write_text(expected, encoding="utf-8")
    return str(directory)


def read_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream) if path.suffix == ".json" else yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError(f"Config must contain a Cross-GOOSE model mapping: {path}")
    normalization = config.get("data", {}).get("image_normalization")
    if normalization not in {"M1P1", "N99"}:
        raise ValueError(f"Config requires data.image_normalization = M1P1 or N99: {path}")
    return config


def _epoch_step(path):
    def number(name):
        match = re.search(rf"{name}[=_-](\d+)", path.stem)
        return int(match.group(1)) if match else -1

    return number("epoch"), number("step"), path.name


def select_checkpoint(directory, criterion="last"):
    directory = Path(directory)
    if criterion not in {"last", "best", "best_ap_0.5", "best_ap_0.75", "best_ap_0.9"}:
        raise ValueError("checkpoint must be last, best, best_ap_0.5, best_ap_0.75, or best_ap_0.9")
    checkpoint_dir = directory / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("*.ckpt"))
    stable = checkpoint_dir / f"{criterion}.ckpt"
    if stable.is_file():
        return stable
    if not candidates and (directory / "weights.ckpt").is_file():
        return directory / "weights.ckpt"
    if criterion == "last":
        matches = [p for p in candidates if p.stem.startswith("last-")]
        if matches:
            return max(matches, key=_epoch_step)
    else:
        metric = "0.75" if criterion == "best" else criterion.removeprefix("best_ap_")
        scored = []
        for path in candidates:
            if not path.stem.startswith("best-"):
                continue
            match = re.search(
                rf"(?:val[_/-]ap[_=]){re.escape(metric)}[=_-]([-+\deE.]+)$", path.stem
            )
            if match:
                scored.append((float(match.group(1)), _epoch_step(path), path))
        if scored:
            return max(scored)[2]
        matches = [p for p in candidates if p.stem.startswith("best-")]
        if len(matches) == 1 and criterion == "best":
            return matches[0]
    raise FileNotFoundError(f"No unambiguous '{criterion}' checkpoint in {directory}")


def model_files(model_path, criterion="last", *, events=None):
    if str(model_path) == "default":
        model_path = download_default(events=events)
    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".ckpt":
            raise ValueError("Model path must be a directory or a .ckpt file")
        directory = path.parent.parent if path.parent.name == "checkpoints" else path.parent
        checkpoint = path
    else:
        directory = path
        checkpoint = select_checkpoint(directory, criterion)
    configs = [directory / name for name in ("config.yaml", "config.yml", "config.json")]
    config_path = next((p for p in configs if p.is_file()), None)
    if config_path is None:
        raise FileNotFoundError(
            f"Checkpoint requires compatible config.yaml, config.yml or config.json in {directory}"
        )
    return directory, checkpoint, config_path, read_config(config_path)


def load(model_path, device="cpu", *, criterion="last", events=None, allow_fallback=False):
    from .model import create_model

    resolved_device = resolve_device(device, allow_fallback=allow_fallback, events=events)
    model = state = None
    try:
        directory, checkpoint, config_path, config = model_files(
            model_path, criterion, events=events
        )
        model = create_model(config)
        # Full Lightning checkpoints from current upstream use tensor/primitive
        # state. Never load executable pickle globals from external checkpoints.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "state_dict" not in state:
            raise ValueError(f"Expected a complete Lightning checkpoint: {checkpoint}")
        try:
            model.load_state_dict(state["state_dict"], strict=True)
        except RuntimeError as exc:
            if is_oom(exc):
                raise
            raise ValueError(
                f"Checkpoint and model config are incompatible: {checkpoint}: {exc}"
            ) from exc
        state = None
        model.to(resolved_device).eval()
        metadata = {
            "model_path": str(directory),
            "checkpoint_path": str(checkpoint),
            "config_path": str(config_path),
            "config": config,
            "device_requested": str(device),
            "device_used": resolved_device.type,
        }
        model._jdll_metadata = metadata
        return model, metadata
    except torch.OutOfMemoryError as exc:
        model = None
        del state
        clear_error_frames(sys.exception())
        cleanup(resolved_device)
        raise EnvironmentError(
            f"Not enough {resolved_device.type} memory to load the model"
        ) from exc
    except BaseException as exc:
        model = None
        del state
        clear_error_frames(sys.exception())
        cleanup(resolved_device)
        if is_oom(exc):
            raise EnvironmentError(
                f"Not enough {resolved_device.type} memory to load the model"
            ) from exc
        raise


def copy_checkpoint(source, destination):
    with atomic_path(destination) as temporary:
        shutil.copyfile(source, temporary)
