"""Shared infrastructure must not change Cross-GOOSE's flow targets or sampling."""

import numpy as np
import pytest
import torch
from crossgoose.data.points_sampling import RandomOnCellV2
from crossgoose.gridflow import GridFlow
from jdll_unet.label_statistics import analyze_mask

from jdll_cross_goose.flow_store import FlowStore
from jdll_cross_goose.patches import MaskView, WarpedFlow, sample_points
from jdll_cross_goose.runtime import Events
from jdll_cross_goose.training import resolve_config
from jdll_cross_goose.training_data import TrainingData


def labels():
    result = np.zeros((40, 48), np.int64)
    result[7:20, 9:25] = 5
    result[22:37, 32:46] = 100_000_001
    return result


def test_bbox_flows_equal_upstream_and_cache_is_bounded(tmp_path):
    mask = labels()
    store = FlowStore(tmp_path, 100_000, Events())
    manifest = store.prepare(mask, analyze_mask(mask), "mass")
    actual = store.get(manifest, (5, 100_000_001), 4)
    for label in (5, 100_000_001):
        reference = GridFlow.from_labels(
            (mask == label).astype(np.int32), flow_center_method="mass", n_interpol=4
        )
        np.testing.assert_array_equal(actual.points[label].data, reference.points[1].data)
        np.testing.assert_array_equal(actual.flows[label], reference.flows[1])
    assert 0 < store.bytes <= store.max_bytes
    first = next(iter(store.cache.values()))
    store.get(manifest, (5,), 4)
    assert any(value is first for value in store.cache.values())
    store.close()
    assert not store.cache and store.bytes == 0


def test_explicit_rng_sampler_matches_upstream():
    mask = torch.from_numpy(labels())
    image = torch.zeros((2, *mask.shape))

    class Flow:
        def query_multiple_labels_threaded(self, pos, labels):
            return pos * np.array([0.2, -0.3])

    with torch.random.fork_rng():
        torch.manual_seed(123)
        reference = RandomOnCellV2(32, 1.4).sample(image, mask, Flow())
    actual = sample_points(
        image, mask, Flow(), 32, 1.4, torch.Generator().manual_seed(123), np.ones(mask.shape, bool)
    )
    for expected, value in zip(reference, actual):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "matrix", [np.eye(2), [[-1, 0], [0, 1]], [[0, 1], [-1, 0]], [[2, 0], [0, 2]]]
)
def test_warped_flow_transforms_direction_not_magnitude(matrix):
    matrix = np.asarray(matrix, float)
    yy, xx = np.mgrid[:32, :32]
    coordinates = np.einsum("ij,jhw->ihw", matrix, np.stack((yy, xx))) + 100

    class Flow:
        def query_multiple_labels(self, pos, labels):
            assert (labels == 77).all()
            return np.tile([0.6, 0.8], (len(pos), 1))

    result = WarpedFlow(Flow(), coordinates, {1: 77}).query_multiple_labels_threaded(
        pos=np.array([[10.5, 20.25], [12, 14]]), labels=np.array([1, 1])
    )
    expected = np.linalg.solve(matrix, [0.6, 0.8])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result, np.tile(expected, (2, 1)), atol=1e-6)


def test_mask_view_compacts_high_ids_before_float_interpolation():
    mask = labels()
    view = MaskView(mask, (5, 100_000_001))
    assert set(np.unique(view[:, :])) == {0, 1, 2}
    assert view.labels[2] == 100_000_001
    assert MaskView(mask, view.labels).labels is view.labels


def test_resolved_data_defaults_are_independent(training_config):
    first = resolve_config(training_config)
    first["data"]["augmentation"]["noise_probability"] = 0.7
    assert resolve_config(training_config)["data"]["augmentation"] == {}


@pytest.mark.parametrize("method", ["M1P1", "N99"])
def test_cached_normalization_preserves_original_calculation(training_config, tmp_path, method):
    from jdll_cross_goose.images import channels, normalize, read_image
    from jdll_cross_goose.patches import ImageView

    cfg = resolve_config({**training_config, "image_normalization": method})["data"]
    runtime = TrainingData(cfg, tmp_path / "runtime", Events())
    try:
        train, _, _ = runtime.prepare(training_config["dataset_path"])
        pair = train[0]
        record = runtime.records[(str(pair.image), pair.region)]
        h, w = pair.domain_shape
        actual = ImageView(runtime.raw_image(pair), record.normalization, cfg["grayscale"])[
            :, :h, :w
        ][:2]
        expected = normalize(channels(read_image(pair.image), cfg["grayscale"]), method)
        np.testing.assert_array_equal(actual, expected)
    finally:
        runtime.close()


@pytest.mark.parametrize("prefetch", [False, True])
def test_patch_validation_is_repeatable_and_loader_does_not_change_torch_rng(
    training_config, tmp_path, prefetch
):
    config = resolve_config({**training_config, "batch_prefetch": prefetch})["data"]
    runtime = TrainingData(config, tmp_path / "runtime", Events())
    try:
        runtime.prepare(training_config["dataset_path"])
        train, val = runtime.make_loaders()
        assert sum(runtime.cache_requests.values()) == 2 * (len(train.dataset) + len(val.dataset))
        torch.manual_seed(101)
        before = torch.random.get_rng_state().clone()
        batch = next(iter(train))
        assert torch.equal(before, torch.random.get_rng_state())
        assert batch["image"].shape == (1, 2, 32, 32)
        assert batch["pts_coord"].shape == (12, 2, 2)
        first, second = next(iter(val)), next(iter(val))
        for key in ("image", "labels", "pts_coord", "pts_flows", "pts_weights"):
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
        assert (tmp_path / "runtime" / "validation_plan.json").is_file()
    finally:
        runtime.close()
    assert not runtime.reader.cache and not runtime.flows.cache


def test_single_source_uses_disjoint_regions_and_training_normalization(training_config, tmp_path):
    import tifffile

    root = tmp_path / "single"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    mask = np.zeros((128, 160), np.uint16)
    for y in range(10, 120, 20):
        for x in range(10, 150, 20):
            mask[y : y + 8, x : x + 8] = 1 + y * 160 + x
    image = np.arange(mask.size, dtype=np.float32).reshape(mask.shape)
    tifffile.imwrite(root / "images" / "single.tif", image)
    tifffile.imwrite(root / "masks" / "single.tif", mask)
    config = resolve_config(training_config)["data"]
    runtime = TrainingData(config, tmp_path / "runtime", Events())
    try:
        train, val, _ = runtime.prepare(root)
        a, b = train[0], val[0]
        assert a.region is not None and b.region is not None
        assert any(a1 <= b0 or b1 <= a0 for (a0, a1), (b0, b1) in zip(a.region, b.region))
        train_record = runtime.records[(str(a.image), a.region)]
        val_record = runtime.records[(str(b.image), b.region)]
        assert train_record.normalization == val_record.normalization
        crop = image[tuple(slice(lo, hi) for lo, hi in a.region)]
        assert train_record.normalization["low"] == crop.min()
        assert train_record.normalization["high"] == crop.max()
    finally:
        runtime.close()


@pytest.mark.parametrize("format", ["npy", "misnamed-tiff"])
def test_shared_runtime_accepts_numpy_and_magic_detected_rasters(
    training_config, tiny_dataset, tmp_path, format
):
    import tifffile

    for path in tiny_dataset.rglob("*.tif"):
        if format == "npy":
            # The shared image convention requires spatial axes larger than 100.
            array = np.pad(tifffile.imread(path), ((48, 48), (60, 60)))
            np.save(path.with_suffix(".npy"), array, allow_pickle=False)
            path.unlink()
        else:
            path.rename(path.with_suffix(".png"))
    runtime = TrainingData(resolve_config(training_config)["data"], tmp_path / "runtime", Events())
    try:
        train, val, _ = runtime.prepare(tiny_dataset)
        assert len(train) == 2 and len(val) == 1
        first = next(iter(runtime.make_loaders()[0]))
        assert first["image"].shape == (1, 2, 32, 32)
        assert torch.isfinite(first["pts_flows"]).all()
    finally:
        runtime.close()


def test_normalization_does_not_decode_uncached_png_once_per_chunk(
    training_config, tmp_path, monkeypatch
):
    from jdll_unet.io import ImageMaskPair
    from PIL import Image

    path = tmp_path / "large.png"
    image = np.broadcast_to(np.arange(512, dtype=np.uint16).astype(np.uint8), (4096, 512))
    Image.fromarray(image).save(path)
    cfg = resolve_config({**training_config, "data_cache_mb": 0})["data"]
    runtime = TrainingData(cfg, tmp_path / "runtime", Events())
    calls = []
    original = runtime.session.pixels

    def read(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.session, "pixels", read)
    try:
        pair = ImageMaskPair(
            path, path, "large", image_axes="YX", spatial_shape=image.shape, image_channels=1
        )
        stats = runtime.fit_normalization(pair)
        assert stats["low"] == 0 and stats["high"] == 255
        assert calls == [path]
        assert runtime.reader.bytes == 0
    finally:
        runtime.close()
