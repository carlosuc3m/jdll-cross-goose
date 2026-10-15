import numpy as np
import pytest
import tifffile
import torch


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_dataset(tmp_path):
    root = tmp_path / "dataset"
    rng = np.random.default_rng(16)
    for split, count in (("train", 2), ("val", 1)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "masks").mkdir()
        for index in range(count):
            yy, xx = np.mgrid[:32, :40]
            labels = np.zeros((32, 40), np.uint16)
            labels[(yy - 12) ** 2 + (xx - 12) ** 2 < 7**2] = 1
            labels[(yy - 23) ** 2 + (xx - 30) ** 2 < 5**2] = 9
            image = ((labels > 0) * 150 + rng.uniform(0, 20, labels.shape)).astype(np.uint8)
            tifffile.imwrite(root / split / "images" / f"sample{index}.tif", image)
            tifffile.imwrite(root / split / "masks" / f"sample{index}_mask.tif", labels)
    return root


@pytest.fixture
def training_config(tiny_dataset, tmp_path):
    return {
        "model_name": "smoke",
        "output_dir": str(tmp_path / "runs"),
        "dataset_path": str(tiny_dataset),
        "architecture": "default",
        "starting_point": "scratch",
        "device": "cpu",
        "epochs": 1,
        "batch_size": 1,
        "val_batch_size": 1,
        "patch_size": 32,
        "val_patch_size": 32,
        "n_samples": 12,
        "n_steps": 2,
        "previews": True,
        "seed": 16,
    }


@pytest.fixture
def saved_model(tmp_path):
    import yaml

    from jdll_cross_goose.checkpoints import default_config
    from jdll_cross_goose.model import create_model

    directory = tmp_path / "model"
    directory.mkdir()
    config = default_config()
    config["data"]["grayscale"] = False
    config["model"]["n_steps"] = 3
    config["adapter"] = {"object_size_statistics": {"median": 20.0}}
    (directory / "config.yaml").write_text(yaml.safe_dump(config))
    torch.manual_seed(16)
    model = create_model(config)
    torch.save({"state_dict": model.state_dict()}, directory / "weights.ckpt")
    return directory
