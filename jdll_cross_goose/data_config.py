"""Serializable shared-data options, separate from Cross-GOOSE's scientific model."""

import math
from dataclasses import asdict

from jdll_unet.config import parse_training_config
from jdll_unet.scale import resolve_instance_scale_mode

DATA_DEFAULTS = {
    "data_cache_mb": "auto",
    "batch_prefetch": "auto",
    "foreground_oversampling": True,
    "foreground_probability": 0.4,
    "skip_empty_images": True,
    "skip_empty_patches": True,
    "empty_patch_fraction": 0.0,
    "empty_patch_max_retries": 8,
    "include_empty_patches_after_max_retries": False,
    "max_padding_ratio": 1.0,
    "steps_per_epoch": "auto",
    "minimum_steps_per_epoch": 1,
    "minimum_patches_per_epoch": 1000,
    "expected_patches_per_case": 10,
    "validation_steps": "auto",
    "normalization_sample_pixels": 1048576,
    "augmentation_profile": "fast",
    "augmentation": {},
}
DATA_OPTIONS = set(DATA_DEFAULTS) | {"annotation_preparation", "instance_scale_normalization"}


def resolve_data_options(config, request):
    for key, default in DATA_DEFAULTS.items():
        config[key] = request.get(key, default)
    config["seed"] = request.get("seed", 16)
    config["validation_fraction"] = request.get("validation_fraction", 0.15)
    options = request.get("instance_scale_normalization", {})
    options = {"enabled": options} if isinstance(options, bool) else options
    shared = parse_training_config(
        {
            "model_name": "crossgoose",
            "output_dir": ".",
            "dataset_path": ".",
            "annotation_preparation": request.get("annotation_preparation", {}),
            "instance_scale_normalization": options,
        }
    )
    annotation, scale = shared.annotation_preparation, shared.instance_scale_normalization
    scale.mode = resolve_instance_scale_mode(scale.mode, "2d")
    config["annotation_preparation"] = asdict(annotation)
    config["instance_scale_normalization"] = asdict(scale)
    for key in ("foreground_probability", "empty_patch_fraction", "validation_fraction"):
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{key} must be in [0, 1]")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    for key in (
        "skip_empty_images",
        "skip_empty_patches",
        "foreground_oversampling",
        "include_empty_patches_after_max_retries",
    ):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in (
        "minimum_steps_per_epoch",
        "minimum_patches_per_epoch",
        "expected_patches_per_case",
        "normalization_sample_pixels",
    ):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    retries = config["empty_patch_max_retries"]
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("empty_patch_max_retries must be a nonnegative integer")
    for key in ("steps_per_epoch", "validation_steps"):
        value = config[key]
        if value == "auto":
            config[key] = None
        elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be auto or a positive integer")
    for key in ("data_cache_mb", "max_padding_ratio"):
        value = config[key]
        if key == "data_cache_mb" and value == "auto":
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["batch_prefetch"] == "auto":
        config["batch_prefetch"] = request["device"] == "cuda"
    if not isinstance(config["batch_prefetch"], bool):
        raise ValueError("batch_prefetch must be boolean or auto")
    if config["augmentation_profile"] not in {"fast", "light-balanced", "balanced", "strong"}:
        raise ValueError("Unknown augmentation_profile")
    if not isinstance(config["augmentation"], dict):
        raise ValueError("augmentation must be a mapping")
