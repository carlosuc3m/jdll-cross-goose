import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jdll_cross_goose.data import pairs_in
from jdll_cross_goose.images import channels, normalize, read_image, read_mask


@pytest.mark.parametrize("axes", ["YX", "CYX", "XYC", "TZCYX", "XYCZT"])
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_numpy_images_match_rasters_and_preserve_sources(tmp_path, axes, dtype):
    for directory in ("images", "masks"):
        (tmp_path / directory).mkdir()
    image = np.random.default_rng(8).integers(0, 60000, (1 if axes == "YX" else 2, 128, 160)).astype(dtype)
    target = np.zeros((128, 160), np.uint16)
    target[20:50, 40:70] = 300
    target[70:90, 100:120] = 65535
    array, order = image, "CYX"
    if "Z" in axes:
        array, order = array[:, None], "CZYX"
    if "T" in axes:
        array, order = array[None], "T" + order
    if "C" not in axes:
        array, order = array[0], order.replace("C", "")
    path = tmp_path / "images/case.npy"
    np.save(path, np.asfortranarray(array.transpose(tuple(order.index(axis) for axis in axes))))
    Image.fromarray(target).save(tmp_path / "masks/case_mask.png")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    pair, = pairs_in(tmp_path)
    loaded = read_image(pair.image)
    np.testing.assert_array_equal(loaded, image)
    np.testing.assert_array_equal(read_image(pair.image, axes="YXC"), image)
    np.testing.assert_array_equal(read_mask(pair.mask), target)
    if os.environ.get("JDLL_NUMPY_SMOKE") == "1" and axes == "CYX" and dtype == np.float32:
        from jdll_cross_goose.appose_api import train

        dataset = tmp_path / "smoke-dataset"
        for split, count in (("train", 2), ("val", 1)):
            (dataset / split / "images").mkdir(parents=True)
            (dataset / split / "masks").mkdir()
            for i in range(count):
                np.save(dataset / split / "images" / f"case-{i}.npy", image)
                np.save(dataset / split / "masks" / f"case-{i}_mask.npy", target)
        result = train({"model_name": "numpy", "output_dir": str(tmp_path / "trained"),
                        "dataset_path": str(dataset), "architecture": "default", "starting_point": "scratch",
                        "device": "cpu", "epochs": 1, "batch_size": 1, "val_batch_size": 1,
                        "patch_size": 32, "val_patch_size": 32, "n_samples": 12, "n_steps": 2, "previews": True})
        assert result["status"] == "completed"
        assert Path(result["best_checkpoint"]).is_file()
    for method in ("M1P1", "N99"):
        np.testing.assert_array_equal(normalize(channels(loaded, False), method), normalize(channels(image, False), method))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    # Reverse pairing: a TIFF input with a first-channel NumPy mask.
    import tifffile
    reverse = tmp_path / "reverse"
    (reverse / "images").mkdir(parents=True)
    (reverse / "masks").mkdir()
    tifffile.imwrite(reverse / "images/case.tif", image, metadata={"axes": "CYX"}, photometric="minisblack")
    np.save(reverse / "masks/case_mask.npy", np.stack([target, np.zeros_like(target)]))
    pair, = pairs_in(reverse)
    np.testing.assert_array_equal(read_mask(pair.mask), target)


def test_numpy_volumes_and_object_arrays_are_rejected(tmp_path):
    path = tmp_path / "unsupported.npy"
    for array in (np.zeros((4, 1, 128, 160), np.uint8), np.array([[object()]], object)):
        np.save(path, array)
        with pytest.raises(ValueError, match="unsupported.npy"):
            read_image(path)


def test_raster_images_take_precedence_over_numpy_caches_but_masks_remain_checked(tmp_path):
    for directory in ("images", "masks"):
        (tmp_path / directory).mkdir()
    image = np.ones((128, 160), np.uint16)
    Image.fromarray(image).save(tmp_path / "images/case.png")
    (tmp_path / "images/case.npy").write_bytes(b"stale cache")
    np.save(tmp_path / "images/other.npy", image)
    for name in ("case", "other"):
        Image.fromarray(image).save(tmp_path / f"masks/{name}_mask.png")
    assert [Path(pair.image).name for pair in pairs_in(tmp_path)] == ["case.png", "other.npy"]
    np.save(tmp_path / "masks/case_mask.npy", image)
    with pytest.raises(ValueError, match="Ambiguous normalized stem"):
        pairs_in(tmp_path)
