import numpy as np
import pytest
import torch

from jdll_cross_goose.appose_api import infer, load_model
from jdll_cross_goose.inference import merged_maps, reconstruct, segment_normalized
from jdll_cross_goose.runtime import CancelledError, Events


@pytest.mark.integration
@pytest.mark.parametrize("channels", [1, 2, 4])
def test_inference_shape_finite_determinism_and_progress(saved_model, channels):
    image = np.random.default_rng(5).normal(size=(channels, 45, 63)).astype(np.float32)
    config = {"model_path": str(saved_model), "tile_size": 64, "tile_overlap": 16}
    events = []
    result = infer(config, {"image": image}, callback=events.append)
    repeat = infer(config, {"image": image})
    assert result["outputs"]["labels"].shape == (45, 63)
    assert result["outputs"]["labels"].dtype.kind in "ui"
    assert np.isfinite(result["outputs"]["cell_probability"]).all()
    np.testing.assert_array_equal(result["outputs"]["labels"], repeat["outputs"]["labels"])
    np.testing.assert_array_equal(
        result["outputs"]["cell_probability"], repeat["outputs"]["cell_probability"]
    )
    phases = [e["phase"] for e in events if e["type"] == "inference_progress"]
    patches = result["metadata"]["total_patches"]
    assert patches > 1
    assert phases == ["inference_start"] + ["patch_start", "patch_end"] * patches + [
        "merge_start",
        "inference_end",
    ]
    assert [e["patch_index"] for e in events if e.get("phase") == "patch_start"] == list(
        range(1, patches + 1)
    )
    assert all("current" in e and "maximum" in e and "message" in e for e in events)


@pytest.mark.integration
def test_scale_override_and_supplied_model(saved_model):
    model, _ = load_model(str(saved_model))
    image = np.zeros((1, 25, 35), dtype=np.float32)
    result = infer({"object_size": 40}, {"image": image}, model=model)
    assert result["metadata"]["scale_factor"] == 0.5
    assert result["outputs"]["labels"].shape == (25, 35)
    assert result["outputs"]["cell_probability"].shape == (25, 35)
    assert not model.training
    del model._jdll_metadata["config"]["adapter"]
    with pytest.raises(ValueError, match="no training object-size"):
        infer({"object_size": 40}, {"image": image}, model=model)


def test_callback_failure_isolation_and_cancellation(saved_model, caplog):
    image = np.zeros((1, 45, 63), dtype=np.float32)
    config = {"model_path": str(saved_model), "tile_size": 64, "tile_overlap": 16}

    def broken(event):
        raise RuntimeError("UI broke")

    assert infer(config, {"image": image}, callback=broken)["outputs"]["labels"].shape == (45, 63)
    assert "Progress callback failed" in caplog.text
    events = []

    def cancel(event):
        events.append(event)
        return False if event.get("phase") == "patch_end" else None

    with pytest.raises(CancelledError):
        infer(config, {"image": image}, callback=cancel)
    assert events[-1]["type"] == "cancelled"
    assert len([e for e in events if e.get("phase") == "patch_end"]) == 1
    with pytest.raises(CancelledError):
        infer(config, {"image": image}, cancellation=lambda: True)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((12, 12)),
        np.full((1, 12, 12), np.nan),
        np.zeros((1, 12, 12), dtype=complex),
        np.zeros((1, 12, 24))[:, :, ::2],
    ],
)
def test_invalid_inputs(saved_model, image):
    with pytest.raises(ValueError):
        infer({"model_path": str(saved_model)}, {"image": image})


class DirectedFlow(torch.nn.Module):
    def forward(self, e0, et):
        delta = (e0[:, :2] - et[:, 2:]).round()
        # Integer pixel steps preserve upstream's nearest embedding sampling.
        return torch.sign(delta) * 5


class OracleModel(torch.nn.Module):
    """Analytic, tile-independent maps with different centers for crossing objects."""

    embeddings_dim = 4
    shared_embedding = True
    flow_fac = 5
    n_steps = 100

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.flow_fn = DirectedFlow()

    def image_to_maps(self, image, apply_sigmoids=False):
        ids = image[:, 0]
        packed = image[:, 1]
        y = torch.floor(packed / 1000)
        x = packed % 1000
        cy = torch.where(ids == 1, 20, torch.where(ids == 2, 50, 10))
        cx = torch.where(ids == 1, 20, torch.where(ids == 2, 50, 1))
        cp = torch.where(ids > 0, 20.0, -20.0)
        return {"T1": torch.stack([cy, cx, y, x, cp], dim=1).float()}


def oracle_image():
    y, x = np.mgrid[:80, :80]
    labels = np.zeros((80, 80), np.float32)
    labels[(abs(y - x) < 4) & (x > 10) & (x < 70)] = 1
    labels[(abs(70 - y - x) < 4) & (x > 10) & (x < 70)] = 2
    labels[:20, :5] = 3  # border object
    return np.stack([labels, y * 1000 + x]).astype(np.float32)


def test_global_tiling_preserves_crossing_and_border_instances():
    model = OracleModel()
    image = oracle_image()
    full = segment_normalized(model, image, {"tile_size": None}, Events())
    tiled = segment_normalized(model, image, {"tile_size": 64, "tile_overlap": 32}, Events())
    np.testing.assert_array_equal(full["labels"], tiled["labels"])
    np.testing.assert_allclose(full["cell_probability"], tiled["cell_probability"], atol=1e-7)
    labels = tiled["labels"]
    for original_id in (1, 2, 3):
        predicted = np.unique(labels[image[0] == original_id])
        assert len(predicted) == 1 and predicted[0] > 0
    assert len(np.unique(labels)) == 4


def test_cancellation_checked_inside_flow_loop():
    model = OracleModel()
    image = torch.from_numpy(oracle_image())
    maps, _ = merged_maps(model, image, {"tile_size": None}, Events())
    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        return checks > 5

    with pytest.raises(CancelledError):
        reconstruct(model, maps, {"n_steps": 1000}, Events(cancellation=cancel))
    assert checks == 6


@pytest.mark.integration
def test_untiled_matches_upstream_reconstruction(saved_model):
    model, _ = load_model(str(saved_model))
    image = np.random.default_rng(5).uniform(-1, 1, (2, 45, 63)).astype(np.float32)
    # Compare raw maps/probabilities and labels using the actual upstream path.
    expected = model.segment_image(torch.from_numpy(image)[None])
    result = segment_normalized(model, image, {"tile_size": None}, Events())
    np.testing.assert_allclose(result["cell_probability"], expected["cellprob"], atol=2e-7)
    np.testing.assert_array_equal(result["labels"], expected["mask"])
