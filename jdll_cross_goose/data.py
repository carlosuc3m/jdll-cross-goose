"""Canonical JDLL data, sequential flow caching, and upstream sampling/augmentation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from crossgoose.data.data_augment import AugmentationParams, random_transform
from crossgoose.data.dataset import flow_data_collate_fn
from crossgoose.data.points_sampling import RandomOnCellV2
from crossgoose.gridflow import GridFlow
from torch.utils.data import DataLoader, Dataset

from .images import channels, normalize, read_image, read_mask
from .runtime import atomic_path, write_json

IMAGE_EXTENSIONS = {".tiff", ".tif", ".png", ".jpeg", ".jpg", ".bmp"}
MASK_EXTENSIONS = {".tiff", ".tif", ".png", ".bmp"}


@dataclass
class Pair:
    image: str
    mask: str
    flow: str = ""


def stem(path, mask=False):
    name = path.stem.casefold()
    suffixes = ("_masks", "_mask", "_labels", "_label") if mask else ("_image", "_img")
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _indexed(directory, extensions, *, mask=False):
    result = {}
    for path in sorted(directory.iterdir()) if directory.is_dir() else []:
        if path.is_file() and path.suffix.lower() in extensions:
            key = stem(path, mask)
            if key in result:
                raise ValueError(f"Ambiguous normalized stem '{key}': {result[key]}, {path}")
            result[key] = path.resolve()
    return result


def pairs_in(directory):
    directory = Path(directory)
    images = _indexed(directory / "images", IMAGE_EXTENSIONS)
    masks = _indexed(directory / "masks", MASK_EXTENSIONS, mask=True)
    missing = images.keys() - masks.keys()
    extra = masks.keys() - images.keys()
    if missing or extra:
        raise ValueError(
            f"Unpaired files in {directory}: missing masks={sorted(missing)}, missing images={sorted(extra)}"
        )
    return [Pair(str(path), str(masks[key])) for key, path in sorted(images.items())]


def validate_dataset(config, events):
    root = Path(config["dataset_path"]).expanduser().resolve()
    train_pairs, val_pairs = pairs_in(root / "train"), pairs_in(root / "val")
    if not train_pairs:
        raise ValueError(f"No training image/mask pairs in {root / 'train'}")
    if not val_pairs:
        fraction = float(config.get("validation_fraction", 0.2))
        if not 0 < fraction < 1 or len(train_pairs) < 2:
            raise ValueError(
                "Provide val images/masks, or at least two train pairs and validation_fraction in (0, 1)"
            )
        order = np.random.default_rng(config.get("seed", 16)).permutation(len(train_pairs))
        count = min(len(train_pairs) - 1, max(1, round(len(train_pairs) * fraction)))
        validation_indices = set(order[:count].tolist())
        val_pairs = [p for i, p in enumerate(train_pairs) if i in validation_indices]
        train_pairs = [p for i, p in enumerate(train_pairs) if i not in validation_indices]
        events.message(
            f"Generated deterministic validation split: {len(train_pairs)} train, {len(val_pairs)} val"
        )
    diameters = []
    for split, pairs in (("train", train_pairs), ("val", val_pairs)):
        for pair in pairs:
            events.check()
            image = read_image(pair.image, config.get("image_axes"))
            labels = read_mask(pair.mask, config.get("mask_axes"), events)
            if image.shape[-2:] != labels.shape:
                raise ValueError(
                    f"Shape mismatch: {pair.image} {image.shape[-2:]} vs {pair.mask} {labels.shape}"
                )
            if split == "train":
                ids, counts = np.unique(labels, return_counts=True)
                diameters.extend((2 * np.sqrt(counts[ids > 0] / math.pi)).tolist())
    statistics = {
        "method": "equivalent_area_diameter",
        "count": len(diameters),
        "median": float(np.median(diameters)) if diameters else None,
        "mean": float(np.mean(diameters)) if diameters else None,
        "min": float(min(diameters)) if diameters else None,
        "max": float(max(diameters)) if diameters else None,
    }
    return train_pairs, val_pairs, statistics


def prepare_flows(pairs, directory, config, events):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index, pair in enumerate(pairs, 1):
        events.check()
        labels = read_mask(pair.mask, config.get("mask_axes"))
        # Include the validated mask contents and method in the key. Source
        # images stay in place and are never copied to an upstream directory.
        digest = hashlib.sha256(
            labels.tobytes() + str(labels.shape).encode() + config["center_method"].encode()
        ).hexdigest()[:24]
        flow_path = directory / f"{digest}.npz"
        pair.flow = str(flow_path.resolve())
        if not flow_path.exists():
            points, flows = {}, {}
            for label in np.unique(labels):
                if label == 0:
                    continue
                events.check()
                # Sequential per-instance calls preserve upstream flow science
                # and permit cancellation between objects without worker pools.
                gf = GridFlow.from_labels(
                    (labels == label).astype(np.int32),
                    n_interpol=1,
                    flow_center_method=config["center_method"],
                )
                points[f"points/{label}"] = gf.points[1].data
                flows[f"flows/{label}"] = gf.flows[1]
            with atomic_path(flow_path) as temporary:
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **points, **flows)
        events.message(
            f"Prepared flows {index}/{len(pairs)}: {flow_path}",
            kind="dataset",
            path=str(flow_path),
            current=index,
            maximum=len(pairs),
        )


def load_flow(path, n_interpol):
    with np.load(path, allow_pickle=False) as data:
        points = {
            int(k.split("/")[1]): data[k].copy() for k in data.files if k.startswith("points/")
        }
        flows = {int(k.split("/")[1]): data[k].copy() for k in data.files if k.startswith("flows/")}
    return GridFlow(points, flows, n_interpol)


class CanonicalDataset(Dataset):
    def __init__(self, pairs, config, *, validation=False):
        self.pairs = pairs
        self.config = config
        self.validation = validation
        self.bootstrap_factor = 1 if validation else config["bootstrap_factor"]
        self.sampler = RandomOnCellV2(**config["points_sampler"]["init_args"])
        augmentation = dict(config["augmentation_params"])
        if validation:
            augmentation.update(
                patch_size=config["val_patch_size"],
                flip_h=False,
                flip_v=False,
                rotate=False,
                rot90=False,
                scale_range=None,
                deterministic_patch=True,
            )
        self.augmentation = AugmentationParams(**augmentation)

    def __len__(self):
        return len(self.pairs) * self.bootstrap_factor

    def __getitem__(self, index):
        pair = self.pairs[index % len(self.pairs)]
        cfg = self.config
        image = read_image(pair.image, cfg.get("image_axes"))
        image = normalize(channels(image, cfg.get("grayscale", True)), cfg["image_normalization"])
        labels = read_mask(pair.mask, cfg.get("mask_axes"))
        flow = load_flow(pair.flow, cfg["gridflow_n_interpol"])
        image, labels, _, flow, _ = random_transform(
            torch.from_numpy(image),
            torch.from_numpy(labels),
            None,
            flow,
            v_fill_img=-1 if cfg["image_normalization"] == "M1P1" else 0,
            **asdict(self.augmentation),
        )
        coords, vectors, weights = self.sampler.sample(image, labels, flow)
        return {
            "image": image.float(),
            "labels": labels,
            "pts_coord": coords,
            "pts_flows": vectors,
            "pts_weights": weights,
            "pts_batch": torch.zeros((len(coords), 1), dtype=torch.long),
        }


def loaders(train_pairs, val_pairs, config):
    common = {
        "num_workers": config["num_workers"],
        "collate_fn": flow_data_collate_fn,
        "persistent_workers": False,
    }
    if config["num_workers"]:
        common["prefetch_factor"] = config["prefetch_factor"]
    train = DataLoader(
        CanonicalDataset(train_pairs, config),
        batch_size=config["batch_size"],
        shuffle=True,
        **common,
    )
    val = DataLoader(
        CanonicalDataset(val_pairs, config, validation=True),
        batch_size=config["val_batch_size"],
        shuffle=False,
        **common,
    )
    return train, val


def write_manifest(directory, train_pairs, val_pairs, statistics):
    path = Path(directory) / "dataset.json"
    write_json(
        path,
        {
            "train": [asdict(p) for p in train_pairs],
            "val": [asdict(p) for p in val_pairs],
            "object_size_statistics": statistics,
        },
    )
    return str(path.resolve())
