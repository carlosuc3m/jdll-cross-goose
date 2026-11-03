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
    (progress,) = (e for e in events if e.get("epoch_complete"))
    assert progress["epoch"] == 1 and progress["step"] == progress["total_steps"] == 2
    assert set(progress["metrics"]) == {"val/ap_0.5", "val/ap_0.75", "val/ap_0.9"}
    assert np.isfinite(list(progress["losses"].values())).all()
    steps = [e["step"] for e in events if e["type"] == "progress" and not e.get("epoch_complete")]
    assert steps == [0, 1, 2]
    (preview,) = (e for e in events if e["type"] == "preview")
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
        return False if event.get("epoch_complete") else None

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
def test_exact_model_directory_keeps_ui_log_and_protects_existing_weights(
    training_config, tmp_path
):
    directory = tmp_path / "selected-model"
    directory.mkdir()
    ui_log = directory / "training-ui.log"
    ui_log.write_text("Starting training\n")
    request = {**training_config, "model_dir": str(directory), "previews": False}
    result = train(request)
    assert Path(result["model_dir"]) == directory
    assert ui_log.read_text() == "Starting training\n"
    assert not (directory / training_config["model_name"]).exists()
    checkpoint = Path(result["last_checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="not empty"):
        train(request)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest


@pytest.mark.integration
def test_empty_dataset_trains_and_validates_for_multiple_epochs(training_config, tiny_dataset):
    import tifffile

    for path in tiny_dataset.rglob("*_mask.tif"):
        tifffile.imwrite(path, np.zeros((32, 40), np.uint16))
    result = train(
        {
            **training_config,
            "epochs": 2,
            "previews": False,
            "skip_empty_images": False,
            "skip_empty_patches": False,
        }
    )
    assert result["history"]["epochs"] == [1, 2]
    assert np.isfinite(result["history"]["losses"]["train/total_loss"]).all()
    state = torch.load(result["last_checkpoint"], weights_only=True, map_location="cpu")
    assert state["epoch"] == 1 and state["global_step"] == 4
    assert Path(result["best_checkpoint"]).is_file()


@pytest.mark.integration
def test_validation_forwards_once_and_accumulation_scheduler_count_updates(
    training_config, monkeypatch
):
    import jdll_cross_goose.training as training

    forwards = []
    original = training.create_model

    def create(config):
        model = original(config)
        model.backbone.register_forward_hook(lambda *args: forwards.append(torch.is_grad_enabled()))
        return model

    monkeypatch.setattr(training, "create_model", create)
    result = train(
        {**training_config, "effective_batch_size": 2, "lr_scheduler": {"type": "cosine"}}
    )
    state = torch.load(result["last_checkpoint"], weights_only=True, map_location="cpu")
    config = yaml.safe_load(Path(result["config_path"]).read_text())
    assert state["global_step"] == 2
    assert forwards.count(True) == 4
    assert forwards.count(False) == config["data"]["resolved_validation_patches"]
    assert state["optimizer_states"][0]["param_groups"][0]["lr"] == 0


@pytest.mark.integration
def test_full_validation_requests_are_one_shot_and_do_not_change_regular_selection(training_config):
    from jdll_unet.validation_control import FullValidationController

    control = FullValidationController()
    events = []

    def callback(event):
        events.append(event)
        if event["type"] == "epoch_start" and event["epoch"] == 1:
            control.request_full_validation("first")
        if (
            event["type"] == "full_validation"
            and event.get("status") == "started"
            and event["epoch"] == 1
        ):
            control.request_full_validation("second")

    result = train({**training_config, "epochs": 2}, callback=callback, control=control)
    passes = [
        e for e in events if e["type"] == "full_validation" and e.get("status") == "completed"
    ]
    assert [e["epoch"] for e in passes] == [1, 2]
    assert passes[0]["request_ids"] == ["first"] and passes[1]["request_ids"] == ["second"]
    assert all(e["next_epoch"] is None for e in passes)
    for event in passes:
        payload = json.loads(Path(event["path"]).read_text())
        assert payload["status"] == "completed" and len(payload["cases"]) == 1
    assert result["history"]["epochs"] == [1, 2]
    assert events[-1]["type"] == "completion"


@pytest.mark.integration
def test_full_validation_failure_preserves_regular_checkpoint(training_config, monkeypatch):
    import jdll_cross_goose.full_validation as full

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic full-validation failure")

    monkeypatch.setattr(full, "evaluate", fail)
    events = []
    with pytest.raises(ValidationError, match="synthetic full-validation"):
        train({**training_config, "validation": {"full_every": 1}}, callback=events.append)
    directory = Path(training_config["output_dir"]) / training_config["model_name"]
    assert (directory / "checkpoints" / "last.ckpt").is_file()
    assert (directory / "checkpoints" / "best.ckpt").is_file()
    assert events[-1]["type"] == "validation_failure"


def test_fine_tuning_uses_original_base_learning_rate(training_config):
    from jdll_cross_goose.training import resolve_config

    scratch = resolve_config({**training_config, "learning_rate": 0.002})
    request = {**training_config, "starting_point": "fine_tune", "base_model": "source"}
    first = resolve_config(request, scratch)
    second = resolve_config(request, first)
    assert first["optimizer"]["init_args"]["lr"] == second["optimizer"]["init_args"]["lr"] == 0.0002
