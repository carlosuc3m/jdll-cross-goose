"""Atomic previews compatible with JDLL's StardistValidationPreviewPanel."""

from pathlib import Path

import numpy as np

from .runtime import atomic_path, write_json


def save_array(path, array):
    with atomic_path(path) as temporary:
        with temporary.open("wb") as stream:
            np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    return str(Path(path).resolve())


def write_preview(model, pairs, directory, epoch, config, events):
    previews = getattr(model, "validation_previews", [])
    if not previews:
        events.message(
            "Validation previews unavailable within the configured memory budget.", kind="warning"
        )
        return
    return write_cached_preview(previews, directory, epoch, events)


def write_cached_preview(previews, directory, epoch, events):
    """Publish the exact regular-validation predictions; never forward the model again."""
    directory = Path(directory) / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, sample in enumerate(previews):
        events.check()
        prefix = directory / f"epoch-{epoch:04d}-sample-{index}"
        image = sample["image"]
        display = image[0] if np.array_equal(image[0], image[-1]) else image.transpose(1, 2, 0)
        samples.append(
            {
                **sample["metadata"],
                "image_path": save_array(f"{prefix}-image.npy", display),
                "label_path": save_array(f"{prefix}-labels.npy", sample["labels"]),
                "prediction_path": save_array(f"{prefix}-prediction.npy", sample["prediction"]),
                "prob_path": save_array(f"{prefix}-probability.npy", sample["probability"]),
                "initial_plane": 0,
            }
        )
    value = {"schema_version": 1, "epoch": epoch, "n_dim": 2, "samples": samples}
    path, latest = directory / f"epoch-{epoch:04d}.json", directory / "latest.json"
    write_json(path, value)
    write_json(latest, value)
    # Keep the latest two complete asset sets; publish before pruning.
    for old in directory.glob("epoch-*"):
        try:
            number = int(old.name.split("-")[1].split(".")[0])
        except ValueError:
            continue
        if number < epoch - 1:
            old.unlink(missing_ok=True)
    events.emit(
        {
            "type": "preview",
            "epoch": epoch,
            "preview_path": str(path.resolve()),
            "latest_preview_path": str(latest.resolve()),
            "samples": len(samples),
            "message": f"Saved validation examples for epoch {epoch} at: {path.resolve()}",
        }
    )
