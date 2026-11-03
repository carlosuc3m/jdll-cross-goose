"""Optional case-wise evaluation, separate from regular patch validation."""

from pathlib import Path

import numpy as np
import torch
from crossgoose.cellpose.metrics import average_precision
from jdll_unet.crop_reading import CropArray
from torch.nn import functional as F

from .inference import resize_labels, segment_normalized
from .patches import ImageView
from .runtime import CancelledError, write_json


def compact(labels):
    ids, inverse = np.unique(labels, return_inverse=True)
    return (inverse + (0 if ids[0] == 0 else 1)).reshape(labels.shape).astype(np.int32)


def instance_ap(truths, predictions):
    with np.errstate(invalid="ignore", divide="ignore"):
        ap, _, _, _ = average_precision(
            [compact(v) for v in truths], [compact(v) for v in predictions], [0.5, 0.75, 0.9]
        )
    for i, (truth, prediction) in enumerate(zip(truths, predictions)):
        if not np.any(truth) and not np.any(prediction):
            ap[i] = 1.0
    return ap


@torch.inference_mode()
def evaluate(model, data, epoch, config, events):
    path = Path(data.directory) / "full_validation" / f"epoch-{epoch:04d}.json"
    results = []
    was_training = model.training
    try:
        model.eval()
        for index, pair in enumerate(data.val_pairs):
            events.check()
            record = data.records[(str(pair.image), pair.region)]
            height, width = pair.domain_shape
            scale = data.nominal_scale(record.diameter)
            shape = (max(1, round(height * scale)), max(1, round(width * scale)))
            # Embedding grids and global instance reconstruction scale with area.
            # Refuse excessive host allocations before decoding the source.
            estimate = max(height * width, np.prod(shape)) * (
                (2 * model.embeddings_dim + 1) * 8 + 64
            )
            limit = config["adapter"]["validation"]["full_memory_mb"] * 1024**2
            if estimate > limit:
                raise ValueError(
                    f"Full validation of {pair.image} needs approximately {estimate / 1024**2:.0f} MiB; "
                    "increase validation.full_memory_mb or use regular patch validation."
                )
            view = ImageView(data.raw_image(pair), record.normalization, data.config["grayscale"])
            image = view[:, :height, :width][:2]
            truth = np.asarray(CropArray(pair, data.reader, mask=True))
            if shape != (height, width):
                image = F.interpolate(
                    torch.from_numpy(image)[None],
                    size=shape,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )[0].numpy()
            result = segment_normalized(
                model, image, model.inference_config, events, progress=False
            )
            prediction = result["labels"]
            if prediction.shape != truth.shape:
                prediction = resize_labels(prediction, truth.shape)
            scores = instance_ap([truth], [prediction])[0]
            results.append(
                {
                    "source_image_path": str(pair.image),
                    "source_label_path": str(pair.mask),
                    "region": pair.region,
                    "scale_factor": scale,
                    "metrics": {
                        f"ap_{t}": float(score) for t, score in zip((0.5, 0.75, 0.9), scores)
                    },
                }
            )
            # Persist completed cases, even if a later case fails/cancels.
            write_json(path, {"epoch": epoch, "status": "running", "cases": results})
            events.emit(
                {
                    "type": "full_validation",
                    "status": "progress",
                    "epoch": epoch,
                    "current": index + 1,
                    "maximum": len(data.val_pairs),
                    "unit": "cases",
                    "message": f"Full validation: {index + 1}/{len(data.val_pairs)} cases.",
                }
            )
        metrics = {
            key: float(np.mean([case["metrics"][key] for case in results]))
            for key in results[0]["metrics"]
        }
        write_json(
            path, {"epoch": epoch, "status": "completed", "cases": results, "metrics": metrics}
        )
        return path, metrics
    except BaseException as exc:
        write_json(
            path,
            {
                "epoch": epoch,
                "status": "cancelled" if isinstance(exc, CancelledError) else "failed",
                "cases": results,
                "error": str(exc),
            },
        )
        raise
    finally:
        model.train(was_training)
