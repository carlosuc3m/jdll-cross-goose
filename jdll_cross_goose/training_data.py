"""Cross-GOOSE targets on JDLL-UNet's bounded 2D data infrastructure."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from crossgoose.data.dataset import flow_data_collate_fn
from jdll_unet.annotations import AnnotationPreparation
from jdll_unet.augment import EmptyPatchError, apply_augmentation, make_augmentation_config
from jdll_unet.config import AnnotationPreparationConfig, InstanceScaleNormalizationConfig
from jdll_unet.crop_reading import CropArray
from jdll_unet.crop_sampling import CropSampler
from jdll_unet.geometry import (
    DomainReader,
    assert_disjoint,
    case_record,
    inspect_pair,
    load_domain_image,
    load_domain_mask,
    padding_extents,
    resolve_data_cache_bytes,
    source_identity,
    spatial_holdout,
    split_sources,
    stable_seed,
)
from jdll_unet.image_reading import image_reading_session
from jdll_unet.io import ImageMaskPair
from jdll_unet.occlusion import plan_cutout
from jdll_unet.prefetch import BatchPrefetcher
from jdll_unet.scale import instance_scale_factor
from torch.utils.data import DataLoader, Dataset

from .data import pairs_in
from .flow_store import FlowStore
from .images import channels, finite_image, to_cyx
from .patches import ImageView, MaskView, WarpedFlow, sample_points
from .runtime import write_json


@dataclass
class Record:
    pair: ImageMaskPair
    analysis: object
    normalization: dict
    diameter: float | None
    flow: str
    label_ids: np.ndarray


class TrainingData:
    def __init__(self, config, directory, events):
        self.config, self.directory, self.events = config, Path(directory), events
        self.records = {}
        self.loaders = []
        self.cache_requests = {}
        self.context = image_reading_session(self.emit)
        self.session = self.context.__enter__()
        try:
            budget = resolve_data_cache_bytes(config.get("data_cache_mb", "auto"))
            self.reader = DomainReader(max_bytes=budget * 3 // 4, session=self.session)
            self.session.domain_reader = self.reader
            self.flows = FlowStore(self.directory / "dataset-cache", budget // 4, events)
            self.preparation = AnnotationPreparation(
                AnnotationPreparationConfig(**config.get("annotation_preparation", {})),
                emit=self.emit,
                cache_dir=self.directory / "annotation-cache",
                scale_config=InstanceScaleNormalizationConfig(
                    **config.get("instance_scale_normalization", {})
                ),
                seed=config["seed"],
            )
            self.session.annotations = self.preparation
            self.cache_budget = budget
        except BaseException:
            self.context.__exit__(None, None, None)
            raise

    def emit(self, event, **fields):
        self.events.emit({"type": event, **fields})

    def close(self):
        try:
            for loader in self.loaders:
                loader.close()
        finally:
            self.flows.close()
            self.reader.cache.clear()
            self.reader.bytes = 0
            self.context.__exit__(None, None, None)

    def _inspect(self, pair):
        self.events.check()
        domain = ImageMaskPair(Path(pair.image), Path(pair.mask), Path(pair.image).stem)
        if self.config.get("image_axes") or self.config.get("mask_axes"):
            # Explicit legacy hints override raster metadata, but never the
            # agreed NPY axis convention. The normal path remains header-driven.
            if domain.image.suffix.lower() == ".npy" or domain.mask.suffix.lower() == ".npy":
                domain, _ = inspect_pair(domain, reader=self.reader)
            else:
                image_axes = self.config.get("image_axes")
                mask_axes = self.config.get("mask_axes")
                raw_image = self.reader.read(domain.image)
                raw_mask = self.reader.read(domain.mask, role="mask")
                image = finite_image(to_cyx(raw_image, image_axes))
                mask = to_cyx(raw_mask, mask_axes)[0]
                if image.shape[-2:] != mask.shape:
                    raise ValueError(
                        f"Incompatible image/mask spatial geometry: {pair.image}, {pair.mask}"
                    )
                domain = replace(
                    domain,
                    image_axes=image_axes
                    or (
                        "YX"
                        if raw_image.ndim == 2
                        else "YXC"
                        if raw_image.shape[-1] <= 4
                        else "CYX"
                    ),
                    mask_axes=mask_axes
                    or (
                        "YX" if raw_mask.ndim == 2 else "YXC" if raw_mask.shape[-1] <= 4 else "CYX"
                    ),
                    spatial_shape=mask.shape,
                    image_channels=image.shape[0],
                    source_id=source_identity(domain.image),
                    source_kind="image_2d",
                )
        else:
            domain, _ = inspect_pair(domain, reader=self.reader)
        if len(domain.spatial_shape) != 2:
            raise ValueError(f"Cross-GOOSE training requires 2D images: {pair.image}")
        return domain

    def prepare(self, root):
        self.events.message(
            f"Data cache budget: {self.cache_budget / 1024**2:.0f} MiB; crop-first reading enabled.",
            kind="dataset",
        )
        root = Path(root)
        train, val = pairs_in(root / "train"), pairs_in(root / "val")
        if not train:
            train = pairs_in(root)
        train, val = [self._inspect(p) for p in train], [self._inspect(p) for p in val]
        if not train:
            raise ValueError(f"No training image/mask pairs in {root}")
        if not val:
            if len({p.source_id for p in train}) == 1:

                def feasible(pair, training):
                    try:
                        size = (
                            self.config["augmentation_params"]["patch_size"]
                            if training
                            else self.config["val_patch_size"]
                        )
                        padding_extents(
                            pair.domain_shape, (size, size), self.config["max_padding_ratio"]
                        )
                        return True
                    except ValueError:
                        return False

                train, val = spatial_holdout(
                    train[0],
                    self.config["validation_fraction"],
                    self.config["seed"],
                    feasible,
                    check_cancel=self.events.check,
                )
                # Connectivity repair is source-wide. Do not let a validation
                # region's object sizes determine training artifact filtering.
                self.preparation.config.minimum_diameter_ratio = 0.0
                self.config["annotation_preparation"]["minimum_diameter_ratio"] = 0.0
                self.events.message(
                    "Created disjoint spatial training/validation regions; "
                    "size-based annotation cleanup is disabled for this shared source.",
                    kind="dataset",
                )
            else:
                train, val = split_sources(
                    train, self.config["validation_fraction"], self.config["seed"]
                )
                self.events.message(
                    f"Generated deterministic validation split: {len(train)} train, {len(val)} val",
                    kind="dataset",
                )
        assert_disjoint(train, val)
        self.preparation.prepare(train + val, "2d", self.events.check, self.reader)
        self.preparation.report()
        statistics = []
        train_normalization = {}
        for split, pairs in (("train", train), ("val", val)):
            kept = []
            for pair in pairs:
                self.events.check()
                analysis = self.preparation.statistics(pair, "2d", self.reader)
                if split == "train" and not analysis.objects and self.config["skip_empty_images"]:
                    self.events.message(
                        f"Skipped unannotated training image: {pair.image}", kind="dataset"
                    )
                    continue
                key = (str(pair.image), pair.region)
                source_key = str(pair.image.resolve())
                normalization = (
                    train_normalization.get(source_key)
                    if pair.split_origin == "spatial_holdout"
                    else None
                )
                if normalization is None:
                    normalization = self.fit_normalization(pair)
                if split == "train":
                    train_normalization[source_key] = normalization
                scale_options = InstanceScaleNormalizationConfig(
                    **self.config["instance_scale_normalization"]
                )
                estimate = self.preparation.reference_size(
                    pair, "2d", (1, 1, 1), scale_options, self.config["seed"]
                )
                if estimate is None and analysis.objects:
                    estimate = self.preparation.reference_size(
                        pair,
                        "2d",
                        (1, 1, 1),
                        replace(scale_options, exclude_border_instances=False, min_instance_area=1),
                        self.config["seed"],
                    )
                    self.events.message(
                        f"Object-size estimate uses border/small-object fallback: {pair.image}",
                        kind="dataset",
                    )
                diameter = estimate.median_diameter_px if estimate is not None else None
                if split == "train" and diameter:
                    statistics.append(diameter)
                # Analysis already needs one mask at a time. Keep that same
                # domain view during flow preparation, avoiding repeated full
                # PNG decoding for every object when it exceeds the cache.
                mask = load_domain_mask(pair, "2d", self.reader, raw=True)
                manifest = self.flows.prepare(mask, analysis, self.config["center_method"])
                del mask
                self.records[key] = Record(
                    pair,
                    analysis,
                    normalization,
                    diameter,
                    manifest,
                    np.asarray((0, *analysis.labels), dtype=np.int64),
                )
                kept.append(pair)
                self.events.message(
                    f"Prepared source {len(kept)}/{len(pairs)}: {pair.image.name}",
                    kind="dataset",
                    current=len(kept),
                    maximum=len(pairs),
                )
            if split == "train":
                train = kept
            else:
                val = kept
        if not train:
            raise ValueError(
                "No annotated training images remain; disable skip_empty_images explicitly to train on negatives."
            )
        reference = float(np.median(statistics)) if statistics else None
        normalized = [diameter * self.nominal_scale(diameter) for diameter in statistics]
        self.statistics = {
            "method": "per_source_median_"
            + self.config["instance_scale_normalization"]["object_size_measure"],
            "count": len(statistics),
            "median": float(np.median(normalized)) if normalized else None,
            "source_median": reference,
            "mean": float(np.mean(normalized)) if normalized else None,
        }
        for pair in val:
            # A validation label must not supply a privileged scale hint.
            self.records[(str(pair.image), pair.region)].diameter = reference
        self.train_pairs, self.val_pairs = train, val
        write_json(self.directory / "annotation_statistics.json", self.preparation.diagnostics())
        write_json(
            self.directory / "dataset.json",
            {
                "train": [case_record(p) for p in train],
                "val": [case_record(p) for p in val],
                "object_size_statistics": self.statistics,
                "normalization": [
                    {"image": str(r.pair.image), "region": r.pair.region, **r.normalization}
                    for r in self.records.values()
                ],
            },
        )
        self.events.message(
            f"Dataset ready: {len(train)} training and {len(val)} validation sources.",
            kind="dataset",
        )
        return train, val, self.statistics

    def raw_image(self, pair):
        return CropArray(
            pair,
            self.reader,
            statistics={"type": "none", "channels": [(0, 1)] * pair.image_channels},
        )

    def fit_normalization(self, pair):
        # One full decode for non-region-readable images; a view for mapped
        # TIFF/NPY. Chunked arithmetic avoids a second whole float32 image.
        source = load_domain_image(pair, reader=self.reader, raw=True)
        height, width = pair.domain_shape
        limit = self.config["normalization_sample_pixels"]
        stride = max(1, math.ceil(2 * height * width / limit))
        samples, minimum, maximum, offset = [], math.inf, -math.inf, 0
        rows = max(1, 1024**2 // max(1, 2 * width))
        for y in range(0, height, rows):
            self.events.check()
            data = channels(
                source[:, slice(y, min(height, y + rows)), slice(0, width)],
                self.config["grayscale"],
            )
            minimum, maximum = min(minimum, float(data.min())), max(maximum, float(data.max()))
            if self.config["image_normalization"] == "N99":
                samples.append(data.reshape(-1)[(-offset) % stride :: stride].copy())
            offset += data.size
        if samples:
            minimum, maximum = (float(v) for v in np.percentile(np.concatenate(samples), [1, 99]))
            if stride > 1:
                self.events.message(
                    f"Estimated normalization from bounded intensity samples: {pair.image}",
                    kind="dataset",
                    sample_stride=stride,
                )
        return {
            "method": self.config["image_normalization"],
            "low": minimum,
            "high": maximum,
            "epsilon": 1e-3 if samples else 0,
            "sample_stride": stride if samples else 1,
        }

    def nominal_scale(self, diameter):
        options = self.config["instance_scale_normalization"]
        if not options["enabled"] or diameter is None:
            return 1.0
        target = (
            self.config["augmentation_params"]["patch_size"] * options["target_object_fraction"]
        )
        return instance_scale_factor(
            target,
            diameter,
            mode=options["mode"],
            min_scale=options["min_effective_scale"],
            max_scale=options["max_effective_scale"],
        )

    def make_loaders(self):
        cfg = self.config
        if cfg["num_workers"] != 0:
            raise ValueError(
                "The shared bounded data runtime requires num_workers=0; use batch_prefetch instead."
            )
        result = []
        for validation, pairs, batch in (
            (False, self.train_pairs, cfg["batch_size"]),
            (True, self.val_pairs, cfg["val_batch_size"]),
        ):
            dataset = PatchDataset(self, pairs, validation)
            loader = PrefetchLoader(
                dataset,
                batch_size=batch,
                shuffle=False,
                collate_fn=collate,
                num_workers=0,
                runtime=self,
                prefetch=cfg["batch_prefetch"] and not validation,
                generator=torch.Generator().manual_seed(cfg["seed"]),
            )
            self.loaders.append(loader)
            result.append(loader)
        self.refresh_cache_plan()
        return tuple(result)

    def refresh_cache_plan(self):
        requests = {}
        for loader in self.loaders:
            for key, count in loader.dataset.cache_requests.items():
                requests[key] = requests.get(key, 0) + count
        self.cache_requests = requests


class PatchDataset(Dataset):
    def __init__(self, runtime, pairs, validation):
        self.runtime, self.pairs, self.validation = runtime, pairs, validation
        cfg = runtime.config
        size = cfg["val_patch_size"] if validation else cfg["augmentation_params"]["patch_size"]
        self.augmentation = make_augmentation_config(
            cfg["augmentation_profile"],
            (size, size),
            cfg["foreground_oversampling"],
            cfg["foreground_probability"],
            cfg["augmentation"],
        )
        self.augmentation = replace(
            self.augmentation,
            **{
                key: cfg[key]
                for key in (
                    "skip_empty_patches",
                    "empty_patch_max_retries",
                    "include_empty_patches_after_max_retries",
                    "max_padding_ratio",
                )
            },
        )
        options = cfg["instance_scale_normalization"]
        self.augmentation = replace(
            self.augmentation,
            instance_scale_enabled=options["enabled"],
            target_object_diameter_px=cfg["augmentation_params"]["patch_size"]
            * options["target_object_fraction"],
            instance_scale_mode=options["mode"],
            training_scale_jitter=tuple(options["training_scale_jitter"]),
            min_effective_scale=options["min_effective_scale"],
            max_effective_scale=options["max_effective_scale"],
        )
        self.augmentation = replace(
            self.augmentation,
            jitter_distribution=options["jitter_distribution"],
            aggressive_scale_jitter=tuple(options["aggressive_scale_jitter"]),
            aggressive_jitter_probability=options["aggressive_jitter_probability"],
        )
        # Coordinate channels must receive geometry only, never brightness/noise/blur.
        self.geometry = replace(
            self.augmentation,
            **{
                f"{key}_probability": 0.0
                for key in (
                    "brightness",
                    "shift",
                    "contrast",
                    "gamma",
                    "noise",
                    "blur",
                    "lowres",
                    "channel_dropout",
                )
            },
        )
        self.photometric = replace(
            self.augmentation,
            instance_scale_enabled=False,
            flip_probability=0,
            rotate90_probability=0,
            affine_probability=0,
            elastic_probability=0,
            skip_empty_patches=False,
            foreground_oversampling=False,
        )
        self.samplers = {}
        weights = np.array([max(1, math.ceil(math.prod(p.domain_shape) / size**2)) for p in pairs])
        requested = cfg["validation_steps"] if validation else cfg["steps_per_epoch"]
        microbatches = (
            requested * (1 if validation else cfg["accumulation_steps"]) if requested else None
        )
        batch = cfg["val_batch_size"] if validation else cfg["batch_size"]
        self.sample_count = (
            microbatches * batch
            if microbatches
            else (
                int(weights.sum())
                if validation
                else max(
                    cfg["minimum_patches_per_epoch"],
                    cfg["minimum_steps_per_epoch"] * batch * cfg["accumulation_steps"],
                    int(weights.sum()) * cfg["bootstrap_factor"] * cfg["expected_patches_per_case"],
                )
            )
        )
        if not validation:
            effective = batch * cfg["accumulation_steps"]
            self.sample_count = math.ceil(self.sample_count / effective) * effective
        self.weights = weights / weights.sum()
        self.set_epoch(0)
        if validation:
            path = runtime.directory / "validation_plan.json"
            write_json(
                path,
                {
                    "seed": cfg["seed"],
                    "samples": len(self),
                    "sources": [str(p.image) for p in pairs],
                    "source_indices": self.indices.tolist(),
                    "patch_size": size,
                    "augmentation": asdict(self.augmentation),
                    "sampling": "deterministic_seed_per_patch",
                },
            )
            runtime.events.message(
                f"Validation plan: {len(self)} fixed patches from {len(pairs)} sources.",
                kind="validation_plan",
                status="ready",
                samples=len(self),
                batches=math.ceil(len(self) / batch),
                sources=len(pairs),
                path=str(path),
            )

    def set_epoch(self, epoch):
        self.epoch = 0 if self.validation else epoch
        rng = np.random.default_rng(
            stable_seed(self.runtime.config["seed"], self.epoch, "patch-plan")
        )
        # Cover every source before drawing additional size-weighted patches.
        base = rng.permutation(len(self.pairs))[: self.sample_count]
        extra = rng.choice(len(self.pairs), max(0, self.sample_count - len(base)), p=self.weights)
        self.indices = np.concatenate((base, extra))
        rng.shuffle(self.indices)
        self.empty_flags = np.zeros(len(self), dtype=bool)
        if not self.validation:
            self.empty_flags[
                rng.permutation(len(self))[
                    : math.ceil(len(self) * self.runtime.config["empty_patch_fraction"])
                ]
            ] = True
        self.cache_requests = {}
        counts = np.bincount(self.indices, minlength=len(self.pairs))
        for i, pair in enumerate(self.pairs):
            count = int(counts[i])
            for path, role in ((pair.image, "image"), (pair.mask, "mask")):
                key = (str(path.absolute()), role)
                self.cache_requests[key] = self.cache_requests.get(key, 0) + count
        self.runtime.refresh_cache_plan()

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        runtime, cfg = self.runtime, self.runtime.config
        runtime.events.check()
        seed = stable_seed(cfg["seed"], self.epoch, f"patch:{index}")
        rng = np.random.default_rng(seed)
        source = int(self.indices[index])
        order = (
            [source]
            if self.validation
            else [
                source,
                *rng.choice(
                    len(self.pairs),
                    size=min(len(self.pairs), cfg["empty_patch_max_retries"]),
                    replace=False,
                ).tolist(),
            ]
        )
        for source in order:
            pair = self.pairs[source]
            record = runtime.records[(str(pair.image), pair.region)]
            image = ImageView(runtime.raw_image(pair), record.normalization, cfg["grayscale"])
            mask = MaskView(CropArray(pair, runtime.reader, mask=True), record.label_ids)
            if source not in self.samplers:
                self.samplers[source] = CropSampler(
                    record.analysis, pair.domain_shape, seed=cfg["seed"], instances=True
                )
            sampler = self.samplers[source]
            geometry = self.geometry
            if record.diameter is None:
                geometry = replace(geometry, instance_scale_enabled=False)
            empty = True if self.empty_flags[index] else None
            if empty:
                geometry = replace(geometry, skip_empty_patches=False)
            try:
                with runtime.reader.cache_plan(runtime.cache_requests or self.cache_requests):
                    transformed, labels, valid = apply_augmentation(
                        image,
                        mask,
                        geometry,
                        rng=rng,
                        training=not self.validation,
                        object_diameter_px=record.diameter,
                        sampler=sampler,
                        empty=empty,
                        return_validity=True,
                    )
                labels = np.where(valid, labels, 0)
                if (
                    not self.validation
                    and geometry.skip_empty_patches
                    and not geometry.include_empty_patches_after_max_retries
                    and not np.any(labels)
                ):
                    raise EmptyPatchError(
                        "No foreground remains within the valid transformed patch."
                    )
                break
            except EmptyPatchError:
                if self.validation:
                    raise
        else:
            raise EmptyPatchError(
                "No eligible annotated patch found; check empty-patch and padding settings."
            )
        coordinates, image = transformed[-2:], transformed[:2]
        ids = np.unique(labels)
        ids = ids[ids > 0]
        mapping = {i + 1: int(mask.labels[label]) for i, label in enumerate(ids)}
        compact = np.zeros(labels.shape, np.int64)
        for i, label in enumerate(ids, 1):
            compact[labels == label] = i
        if not self.validation:
            image, _, _ = apply_augmentation(
                image, compact, self.photometric, rng=rng, return_validity=True
            )
            cutout = plan_cutout(
                valid,
                record.diameter or min(image.shape[-2:]) * 0.25,
                rng,
                probability=self.augmentation.cutout_probability,
                max_fraction=self.augmentation.cutout_max_fraction,
                diameter_fraction=self.augmentation.cutout_diameter_fraction,
                max_attempts=self.augmentation.cutout_max_attempts,
            )
            image[:, cutout] = 0
        image[:, ~valid] = -1 if cfg["image_normalization"] == "M1P1" else 0
        flow = runtime.flows.get(record.flow, mapping.values(), cfg["gridflow_n_interpol"])
        warped = WarpedFlow(flow, coordinates, mapping)
        image_t, labels_t = torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(compact)
        points, vectors, weights = sample_points(
            image_t,
            labels_t,
            warped,
            cfg["points_sampler"]["init_args"]["n_samples"],
            cfg["points_sampler"]["init_args"]["sigma"],
            torch.Generator().manual_seed(seed),
            valid,
        )
        support = coordinates[:, valid]
        return {
            "image": image_t,
            "labels": labels_t,
            "validity": torch.from_numpy(valid),
            "pts_coord": points,
            "pts_flows": vectors,
            "pts_weights": weights,
            "pts_batch": torch.zeros((len(points), 1), dtype=torch.long),
            "metadata": {
                "source_image_path": str(pair.image),
                "source_label_path": str(pair.mask),
                "region": pair.region,
                "patch_index": index,
                "source_bounds_yx": [
                    [float(a), float(b)] for a, b in zip(support.min(axis=1), support.max(axis=1))
                ],
                "scale_factor": runtime.nominal_scale(record.diameter),
            },
        }


def collate(samples):
    metadata = [sample["metadata"] for sample in samples]
    result = flow_data_collate_fn(
        [{k: v for k, v in sample.items() if k != "metadata"} for sample in samples]
    )
    result["metadata"] = metadata
    return result


class PrefetchLoader(DataLoader):
    def __init__(self, *args, runtime, prefetch, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime, self.prefetch, self.producer = runtime, prefetch, None

    def __iter__(self):
        if not self.prefetch:
            yield from super().__iter__()
            return
        self.producer = BatchPrefetcher(super().__iter__(), read_session=self.runtime.session)
        try:
            yield from self.producer
        finally:
            self.close()

    def close(self):
        if self.producer is not None:
            self.producer.close()
            self.producer = None
