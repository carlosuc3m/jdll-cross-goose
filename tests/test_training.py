import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from jdll_cross_goose.appose_api import infer, load_model, train
from jdll_cross_goose.runtime import CancelledError, TrainingError, ValidationError


@pytest.mark.integration
def test_scratch_and_fine_tune_checkpoints_progress_previews(training_config):
    events = []
    result = train(training_config, callback=events.append)
    assert result["status"] == "completed"
    assert Path(result["last_checkpoint"]).is_file()
    assert Path(result["best_checkpoint"]).is_file()
    assert (Path(result["model_dir"]) / "checkpoints" / "best.ckpt").is_file()
    state = torch.load(result["last_checkpoint"], map_location="cpu", weights_only=True)
    assert state["optimizer_states"] and state["lr_schedulers"] == []
    assert "loops" in state and "pytorch-lightning_version" in state
    assert state["global_step"] == 2
    config = yaml.safe_load(Path(result["config_path"]).read_text())
    assert config["data"]["num_workers"] == 0
    assert config["optimizer"]["class_path"] == "torch.optim.AdamW"
    assert config["adapter"]["object_size_statistics"]["median"] > 0
    (progress,) = [e for e in events if e["type"] == "progress"]
    assert progress["epoch"] == 1 and progress["step"] == progress["total_steps"] == 2
    assert set(progress["metrics"]) == {"val/ap_0.5", "val/ap_0.75", "val/ap_0.9"}
    assert np.isfinite(list(progress["losses"].values())).all()
    (preview,) = [e for e in events if e["type"] == "preview"]
    payload = json.loads(Path(preview["preview_path"]).read_text())
    assert payload == json.loads(Path(preview["latest_preview_path"]).read_text())
    sample = payload["samples"][0]
    for key in ("image_path", "label_path", "prediction_path", "prob_path"):
        assert Path(sample[key]).is_file()
        assert np.isfinite(np.load(sample[key], allow_pickle=False)).all()
    assert Path(sample["source_image_path"]).is_file()
    assert events[-1]["type"] == "completion"
    model, _ = load_model(result["model_dir"])
    inference = infer({}, {"image": np.zeros((1, 32, 40), np.float32)}, model=model)
    assert inference["outputs"]["labels"].shape == (32, 40)
    original_digest = hashlib.sha256(Path(result["last_checkpoint"]).read_bytes()).hexdigest()
    second = train(
        {
            **training_config,
            "starting_point": "fine_tune",
            "base_model": result["model_dir"],
            "model_name": "fine-tuned",
            "checkpoint": "last",
            "previews": False,
        }
    )
    assert second["status"] == "completed"
    assert (
        original_digest == hashlib.sha256(Path(result["last_checkpoint"]).read_bytes()).hexdigest()
    )
    assert load_model(second["model_dir"])[0].device.type == "cpu"


@pytest.mark.integration
def test_cancel_after_epoch_keeps_checkpoint_and_distinct_status(training_config):
    events = []

    def cancel(event):
        events.append(event)
        return False if event["type"] == "progress" else None

    with pytest.raises(CancelledError):
        train({**training_config, "epochs": 3}, callback=cancel)
    assert events[-1]["type"] == "cancelled"
    checkpoint = (
        Path(training_config["output_dir"])
        / training_config["model_name"]
        / "checkpoints"
        / "last.ckpt"
    )
    assert checkpoint.is_file()
    assert load_model(str(checkpoint))[0].device.type == "cpu"
    assert not any(e["type"] == "completion" for e in events)


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["training", "validation", "oom"])
def test_failure_types_and_cleanup(training_config, monkeypatch, failure):
    import jdll_cross_goose.training as training
    from jdll_cross_goose.model import AdapterModel

    calls = []
    original_cleanup = training.cleanup

    def cleanup(device):
        calls.append(device.type)
        original_cleanup(device)

    monkeypatch.setattr(training, "cleanup", cleanup)

    def explode(*args, **kwargs):
        if failure == "oom":
            raise torch.OutOfMemoryError("synthetic OOM")
        if failure == "validation":
            raise ValidationError("synthetic validation failure")
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(
        AdapterModel, "validation_step" if failure == "validation" else "training_step", explode
    )
    events = []
    from jdll_cross_goose.runtime import EnvironmentError

    expected = {"training": TrainingError, "validation": ValidationError, "oom": EnvironmentError}[
        failure
    ]
    with pytest.raises(expected):
        train(training_config, callback=events.append)
    assert calls == ["cpu"]
    assert (
        events[-1]["type"]
        == {
            "training": "training_failure",
            "validation": "validation_failure",
            "oom": "environment_failure",
        }[failure]
    )


def test_unsupported_options_fail_before_output_creation(training_config):
    with pytest.raises(ValueError, match="Unknown training"):
        train({**training_config, "learning_arte": 0.1})
    assert not Path(training_config["output_dir"]).exists()


@pytest.mark.integration
def test_empty_dataset_trains_and_validates_for_multiple_epochs(training_config, tiny_dataset):
    import tifffile

    for path in tiny_dataset.rglob("*_mask.tif"):
        tifffile.imwrite(path, np.zeros((32, 40), np.uint16))
    result = train({**training_config, "epochs": 2, "previews": False})
    assert result["history"]["epochs"] == [1, 2]
    assert np.isfinite(result["history"]["losses"]["train/total_loss"]).all()
    state = torch.load(result["last_checkpoint"], weights_only=True, map_location="cpu")
    assert state["epoch"] == 1 and state["global_step"] == 4
    assert Path(result["best_checkpoint"]).is_file()
