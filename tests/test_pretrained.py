"""Optional parity test against real upstream pretrained weights, without downloads."""

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from jdll_cross_goose.appose_api import load_model
from jdll_cross_goose.images import normalize
from jdll_cross_goose.inference import segment_normalized
from jdll_cross_goose.runtime import Events


@pytest.mark.integration
def test_pretrained_full_parity_and_tiled_consistency():
    directory = os.environ.get("CROSS_GOOSE_PRETRAINED")
    if directory is None or not Path(directory).is_dir():
        pytest.skip("Set CROSS_GOOSE_PRETRAINED to a downloaded upstream default model")
    model, _ = load_model(directory)
    image = tifffile.imread(Path(__file__).parent / "fixtures" / "bbbc010-w2.tif")
    image = normalize(np.repeat(image[None], 2, axis=0).astype(np.float32), "M1P1")
    upstream = model.segment_image(torch.from_numpy(image)[None])
    full = segment_normalized(model, image, {"tile_size": None}, Events())
    np.testing.assert_allclose(full["cell_probability"], upstream["cellprob"], atol=2e-7)
    np.testing.assert_array_equal(full["labels"], upstream["mask"])
    assert np.any(full["labels"]), (
        "Fixture must exercise actual foreground trajectories and instances"
    )
    tiled = segment_normalized(model, image, {"tile_size": 512, "tile_overlap": 128}, Events())
    assert tiled["total_patches"] > 1
    # Style/context varies per tile. Require useful numerical/foreground
    # agreement on this fixture, not impossible general bitwise neural parity.
    assert np.mean(np.abs(full["cell_probability"] - tiled["cell_probability"])) < 0.01
    a, b = full["labels"] > 0, tiled["labels"] > 0
    assert (a & b).sum() / (a | b).sum() > 0.95
