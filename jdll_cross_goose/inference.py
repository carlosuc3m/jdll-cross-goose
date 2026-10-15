"""Blend network embeddings, then follow Cross-GOOSE flows globally.

No tile-level instance labels are produced. The trained embedding channels share
one basis; weighted raw maps can be queried across tile boundaries by flow_fn.
The upstream backbone's image style varies with context, so tiled predictions
are an approximation, not bitwise equivalent to a full-image forward pass.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from crossgoose.cellpose.dynamics import get_masks_torch
from crossgoose.cellpose.transforms import get_pad_yx
from torch.nn import functional as F

from .images import channels, finite_image, normalize
from .runtime import positive_int


def starts(length, tile, overlap):
    if length <= tile:
        return [0]
    result = list(range(0, length - tile + 1, tile - overlap))
    if result[-1] != length - tile:
        result.append(length - tile)
    return result


def taper(height, width):
    y = np.maximum(np.sin(np.pi * (np.arange(height) + 0.5) / height) ** 2, 1e-3)
    x = np.maximum(np.sin(np.pi * (np.arange(width) + 0.5) / width) ** 2, 1e-3)
    return torch.from_numpy((y[:, None] * x[None, :]).astype(np.float32))


def _progress(events, phase, current, total, **fields):
    events.emit(
        {
            "type": "inference_progress",
            "phase": phase,
            "message": phase.replace("_", " "),
            "current": current,
            "maximum": total,
            "total_patches": total,
            **fields,
        }
    )


@torch.no_grad()
def merged_maps(model, image, config, events, *, progress=True):
    height, width = image.shape[-2:]
    tile = config.get("tile_size", 512)
    if tile is None:
        tile = max(height, width)
    positive_int(tile, "tile_size", minimum=32)
    if tile % 16:
        raise ValueError("tile_size must be a multiple of 16 (the backbone pooling stride)")
    overlap = config.get("tile_overlap", min(128, max(16, (tile // 64) * 16)))
    positive_int(overlap, "tile_overlap", minimum=0)
    if overlap >= tile or overlap % 16:
        raise ValueError("tile_overlap must be a multiple of 16 smaller than tile_size")
    if max(height, width) > tile and overlap < 16:
        raise ValueError("Tiled inference requires at least 16 pixels of overlap")
    patch_h, patch_w = min(tile, height), min(tile, width)
    positions = [
        (y, x) for y in starts(height, patch_h, overlap) for x in starts(width, patch_w, overlap)
    ]
    total = len(positions)
    if progress:
        _progress(events, "inference_start", 0, total)
    weight = taper(patch_h, patch_w)
    denominator = torch.zeros((height, width), dtype=torch.float32)
    maps = None
    device = next(model.parameters()).device
    for index, (y, x) in enumerate(positions, 1):
        events.check()
        if progress:
            _progress(events, "patch_start", index - 1, total, patch_index=index)
        patch = image[None, :, y : y + patch_h, x : x + patch_w].to(device)
        features = model.image_to_maps(patch, apply_sigmoids=False)
        raw = features["T1"][0].detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(raw).all():
            raise ValueError("Model produced non-finite embeddings or cell logits")
        if total == 1:
            # Preserve exact upstream raw outputs when no blending is needed.
            maps = raw
        else:
            if maps is None:
                maps = torch.zeros((raw.shape[0], height, width), dtype=torch.float32)
            maps[:, y : y + patch_h, x : x + patch_w] += raw * weight
            denominator[y : y + patch_h, x : x + patch_w] += weight
        del patch, features, raw
        if progress:
            _progress(events, "patch_end", index, total, patch_index=index)
    if progress:
        _progress(events, "merge_start", total, total)
    if total > 1:
        maps /= denominator
    return maps, total


@torch.no_grad()
def reconstruct(model, raw, config, events):
    height, width = raw.shape[-2:]
    cell_probability = torch.sigmoid(raw[-1])
    threshold = float(config.get("cell_probability_threshold", 0.5))
    if not 0 < threshold < 1:
        raise ValueError("cell_probability_threshold must be between zero and one")
    foreground = cell_probability > threshold
    origins = torch.nonzero(foreground)
    endpoints = torch.empty_like(origins)
    count = len(origins)
    n_steps = positive_int(config.get("n_steps", model.n_steps), "n_steps")
    chunk_size = positive_int(config.get("flow_batch_size", 65536), "flow_batch_size")
    device = next(model.parameters()).device
    embedding_0 = raw[: model.embeddings_dim]
    embedding_t = (
        embedding_0
        if model.shared_embedding
        else raw[model.embeddings_dim : 2 * model.embeddings_dim]
    )
    bounds = torch.tensor([height - 1, width - 1], device=device)
    # Raw grids stay in host memory to bound accelerator memory use. Only one
    # batch of sampled embeddings/trajectories is placed on the model device.
    for first in range(0, count, chunk_size):
        events.check()
        initial = origins[first : first + chunk_size]
        e0 = embedding_0[:, initial[:, 0], initial[:, 1]].T.to(device)
        position = initial.to(device=device, dtype=torch.float32)
        for _ in range(n_steps):
            events.check()
            index = position.long().cpu()
            et = embedding_t[:, index[:, 0], index[:, 1]].T.to(device)
            flow = model.flow_fn(e0, et) / model.flow_fac
            if not torch.isfinite(flow).all():
                raise ValueError("Model produced non-finite flows")
            position = torch.minimum(torch.clamp(position + flow, min=0), bounds)
        endpoints[first : first + len(initial)] = position.long().cpu()
    events.check()
    if count:
        labels = get_masks_torch(
            pt=endpoints.T.contiguous(), inds=tuple(origins.T.numpy()), shape0=(height, width)
        )
    else:
        labels = np.zeros((height, width), dtype=np.uint32)
    events.check()
    return {"labels": labels, "cell_probability": cell_probability.numpy()}


def resize_labels(labels, shape):
    # Integer indexing avoids float32 loss of precision for large instance IDs.
    y = np.minimum(
        (np.arange(shape[0]) * labels.shape[0] / shape[0]).astype(int), labels.shape[0] - 1
    )
    x = np.minimum(
        (np.arange(shape[1]) * labels.shape[1] / shape[1]).astype(int), labels.shape[1] - 1
    )
    return labels[y[:, None], x[None, :]]


@torch.no_grad()
def segment_normalized(model, image, config, events, *, progress=True):
    """Segment already-normalized CYX data, also used for validation patches."""
    events.check()
    started = time.perf_counter()
    image = np.ascontiguousarray(image, dtype=np.float32)
    if image.shape[0] == 1:
        image = np.repeat(image, 2, axis=0)
    image = torch.from_numpy(image[:2])
    height, width = image.shape[-2:]
    top, bottom, left, right = get_pad_yx(height, width, div=16, extra=1, min_size=None)
    image = F.pad(image, (left, right, top, bottom), value=-1)
    raw, total = merged_maps(model, image, config, events, progress=progress)
    mapped = time.perf_counter()
    outputs = reconstruct(model, raw, config, events)
    for key, array in outputs.items():
        outputs[key] = np.ascontiguousarray(array[top : top + height, left : left + width])
    ended = time.perf_counter()
    outputs["timings"] = {
        "network_and_merge": mapped - started,
        "flow_and_reconstruction": ended - mapped,
        "total": ended - started,
    }
    outputs["total_patches"] = total
    return outputs


def prepare_input(image, saved_config, config):
    finite_image(image)
    if not image.flags.c_contiguous:
        raise ValueError("inputs['image'] must be contiguous in CYX order")
    data_config = saved_config["data"]
    prepared = normalize(
        channels(image, data_config.get("grayscale", True)), data_config["image_normalization"]
    )
    scale = 1.0
    if config.get("object_size") is not None:
        object_size = float(config["object_size"])
        if not math.isfinite(object_size) or object_size <= 0:
            raise ValueError("object_size must be a finite positive diameter in input pixels")
        reference = saved_config.get("adapter", {}).get("object_size_statistics", {}).get("median")
        if reference is None:
            raise ValueError(
                "This model has no training object-size statistics; omit object_size or use an adapter-trained model"
            )
        scale = float(reference) / object_size
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Saved training object-size median must be finite and positive")
        size = tuple(max(1, round(d * scale)) for d in image.shape[-2:])
        prepared = F.interpolate(
            torch.from_numpy(prepared)[None],
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0].numpy()
    return prepared, scale
