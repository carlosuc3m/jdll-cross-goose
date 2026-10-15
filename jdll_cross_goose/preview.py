"""Atomic previews compatible with JDLL's StardistValidationPreviewPanel."""

from pathlib import Path

import numpy as np

from .images import read_image, read_mask
from .runtime import atomic_path, write_json


def save_array(path, array):
    with atomic_path(path) as temporary:
        with temporary.open("wb") as stream:
            np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    return str(Path(path).resolve())


def write_preview(model, pairs, directory, epoch, config, events):
    from .inference import prepare_input, segment_normalized

    directory = Path(directory) / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    samples = []
    was_training = model.training
    try:
        model.eval()
        for index, pair in enumerate(pairs[: config["adapter"]["preview_samples"]]):
            events.check()
            image = read_image(pair.image, config["data"].get("image_axes"))
            mask = read_mask(pair.mask, config["data"].get("mask_axes"))
            prepared, _ = prepare_input(image, config, {})
            result = segment_normalized(
                model, prepared, model.inference_config, events, progress=False
            )
            # Source arrays are cached once, avoiding duplicate image files each
            # epoch. Java's NPY decoder uses YX and YXC raster axes.
            image_path = directory / f"sample-{index}-image.npy"
            mask_path = directory / f"sample-{index}-labels.npy"
            if not image_path.exists():
                display = image[0] if image.shape[0] == 1 else image[:3].transpose(1, 2, 0)
                save_array(image_path, display)
                save_array(mask_path, mask)
            prefix = directory / f"epoch-{epoch:04d}-sample-{index}"
            samples.append(
                {
                    "image_path": str(image_path.resolve()),
                    "label_path": str(mask_path.resolve()),
                    "source_image_path": pair.image,
                    "source_label_path": pair.mask,
                    "prediction_path": save_array(f"{prefix}-prediction.npy", result["labels"]),
                    "prob_path": save_array(
                        f"{prefix}-probability.npy", result["cell_probability"]
                    ),
                    "initial_plane": 0,
                }
            )
    finally:
        model.train(was_training)
    value = {"schema_version": 1, "epoch": epoch, "n_dim": 2, "samples": samples}
    preview = directory / f"epoch-{epoch:04d}.json"
    latest = directory / "latest.json"
    write_json(preview, value)
    write_json(latest, value)
    events.emit(
        {
            "type": "preview",
            "epoch": epoch,
            "preview_path": str(preview.resolve()),
            "latest_preview_path": str(latest.resolve()),
        }
    )
