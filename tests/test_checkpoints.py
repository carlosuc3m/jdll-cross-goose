from pathlib import Path

import pytest
import torch
import yaml

from jdll_cross_goose.appose_api import load_model
from jdll_cross_goose.checkpoints import model_files, select_checkpoint


def test_selection_is_numeric_deterministic_and_stable_paths_win(tmp_path):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    for name in [
        "last-epoch=9-step=90.ckpt",
        "last-epoch=10-step=100.ckpt",
        "best-epoch=1-val_ap_0.75=0.9000.ckpt",
        "best-epoch=2-val_ap_0.75=0.8500.ckpt",
        "best-epoch=1-val_ap_0.9=0.9900.ckpt",
    ]:
        (directory / name).touch()
    assert select_checkpoint(tmp_path).name == "last-epoch=10-step=100.ckpt"
    assert select_checkpoint(tmp_path, "best").name == "best-epoch=1-val_ap_0.75=0.9000.ckpt"
    assert "0.9=0.9900" in select_checkpoint(tmp_path, "best_ap_0.9").name
    (directory / "last.ckpt").touch()
    (directory / "best.ckpt").touch()
    assert select_checkpoint(tmp_path).name == "last.ckpt"
    assert select_checkpoint(tmp_path, "best").name == "best.ckpt"


@pytest.mark.parametrize("extension", ["yaml", "yml", "json"])
def test_config_formats_and_explicit_checkpoint(saved_model, extension):
    import json

    source = saved_model / "config.yaml"
    config = yaml.safe_load(source.read_text())
    source.unlink()
    target = saved_model / f"config.{extension}"
    target.write_text(json.dumps(config) if extension == "json" else yaml.safe_dump(config))
    directory, checkpoint, found, _ = model_files(saved_model / "weights.ckpt")
    assert directory == saved_model
    assert checkpoint.name == "weights.ckpt"
    assert found == target


@pytest.mark.integration
def test_cpu_load_two_layouts_without_network(saved_model, monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        pytest.fail("Loading complete checkpoint must not download Cellpose weights")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    model, metadata = load_model(str(saved_model))
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert not model.training
    assert Path(metadata["checkpoint_path"]).name == "weights.ckpt"
    (saved_model / "checkpoints").mkdir()
    (saved_model / "weights.ckpt").rename(saved_model / "checkpoints" / "last.ckpt")
    reloaded, metadata = load_model(str(saved_model))
    assert torch.equal(next(model.parameters()), next(reloaded.parameters()))
    assert Path(metadata["checkpoint_path"]).name == "last.ckpt"


def test_missing_config_and_mismatched_weights(saved_model):
    source = saved_model / "config.yaml"
    original = source.read_text()
    source.unlink()
    with pytest.raises(FileNotFoundError, match="compatible config"):
        model_files(saved_model)
    config = yaml.safe_load(original)
    config["model"]["shared_embedding"] = False
    source.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="incompatible"):
        load_model(str(saved_model))


def test_ambiguous_best_is_rejected(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    for name in ("best-a.ckpt", "best-b.ckpt"):
        (tmp_path / "checkpoints" / name).touch()
    with pytest.raises(FileNotFoundError, match="unambiguous"):
        select_checkpoint(tmp_path, "best")
