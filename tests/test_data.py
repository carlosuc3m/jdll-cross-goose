import numpy as np
import pytest
import tifffile
from PIL import Image

from jdll_cross_goose.data import CanonicalDataset, pairs_in, prepare_flows, validate_dataset
from jdll_cross_goose.images import channels, normalize, read_image, read_mask
from jdll_cross_goose.runtime import Events
from jdll_cross_goose.training import resolve_config


def test_split_and_statistics(tiny_dataset, training_config):
    train, val, stats = validate_dataset(training_config, Events())
    assert len(train) == 2 and len(val) == 1
    assert stats["count"] == 4 and stats["median"] > 0
    for path in (tiny_dataset / "val").rglob("*.tif"):
        path.unlink()
    first = validate_dataset(training_config, Events())
    second = validate_dataset(training_config, Events())
    assert first == second
    assert len(first[0]) == len(first[1]) == 1
    assert first[0][0].image != first[1][0].image


@pytest.mark.parametrize("image_extension", ["tif", "tiff", "png", "jpg", "jpeg", "bmp"])
def test_mixed_extensions_and_stems(tmp_path, image_extension):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    image = np.arange(120, dtype=np.uint8).reshape(10, 12)
    image_path = tmp_path / "images" / f"cell.{image_extension}"
    if image_extension.startswith("tif"):
        tifffile.imwrite(image_path, image)
    else:
        Image.fromarray(image).save(image_path)
    Image.fromarray((image > 50).astype(np.uint8)).save(tmp_path / "masks" / "cell_mask.png")
    (pair,) = pairs_in(tmp_path)
    assert read_image(pair.image).shape == (1, 10, 12)
    assert read_mask(pair.mask).shape == (10, 12)


def test_mask_validation_and_first_channel(tmp_path):
    path = tmp_path / "mask.tif"
    labels = np.zeros((3, 12, 16), np.uint16)
    labels[0, :, :8] = 7
    labels[1] = 99
    tifffile.imwrite(path, labels, metadata={"axes": "CYX"}, photometric="minisblack")
    events = []
    assert np.array_equal(read_mask(path, events=Events(events.append)), labels[0])
    assert events[0]["type"] == "warning"
    tifffile.imwrite(path, labels[0].astype(np.float32) + 0.1)
    with pytest.raises(ValueError, match="integer dtype"):
        read_mask(path)


def test_missing_mask_and_shape_mismatch(tiny_dataset, training_config):
    mask = tiny_dataset / "train" / "masks" / "sample0_mask.tif"
    tifffile.imwrite(mask, np.zeros((2, 2), np.uint16))
    with pytest.raises(ValueError, match="Shape mismatch"):
        validate_dataset(training_config, Events())
    mask.unlink()
    with pytest.raises(ValueError, match="missing masks"):
        pairs_in(tiny_dataset / "train")


def test_normalization_matches_upstream_and_handles_constant():
    from crossgoose.utils import normalize_image

    image = np.arange(120, dtype=np.float32).reshape(2, 6, 10)
    for method in ("M1P1", "N99"):
        np.testing.assert_array_equal(normalize(image, method), normalize_image(image, method))
        assert np.isfinite(normalize(np.ones_like(image), method)).all()
    assert channels(image[:1], False).shape[0] == 2
    np.testing.assert_array_equal(channels(np.concatenate([image, image]), False), image)


@pytest.mark.integration
def test_empty_masks_and_flow_cache_do_not_spawn_processes(
    tiny_dataset, training_config, tmp_path, monkeypatch
):
    import concurrent.futures

    def forbidden(*args, **kwargs):
        pytest.fail("Flow preparation must not create a process pool")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", forbidden)
    for path in tiny_dataset.rglob("*_mask.tif"):
        tifffile.imwrite(path, np.zeros((32, 40), np.uint16))
    pairs, _, stats = validate_dataset(training_config, Events())
    assert stats["count"] == 0
    cfg = resolve_config(training_config)["data"]
    prepare_flows(pairs, tmp_path / "cache", cfg, Events())
    batch = CanonicalDataset(pairs, cfg)[0]
    assert batch["pts_coord"].shape == (0, 2, 2)
    assert not batch["labels"].any()
    assert len(list((tmp_path / "cache").iterdir())) == 1
