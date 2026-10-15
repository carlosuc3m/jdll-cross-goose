import gc
import weakref

import numpy as np
import pytest
import torch

from jdll_cross_goose import checkpoints
from jdll_cross_goose.appose_api import infer, train
from jdll_cross_goose.model import AdapterModel
from jdll_cross_goose.runtime import CancelledError, EnvironmentError, Events


@pytest.mark.parametrize("outcome", ["success", "cancel", "exception", "oom", "mps_oom"])
def test_inference_releases_owned_model_even_if_exception_is_retained(
    saved_model, monkeypatch, outcome
):
    references = []
    original = checkpoints.load

    def tracked_load(*args, **kwargs):
        model, metadata = original(*args, **kwargs)
        references.append(weakref.ref(model))
        return model, metadata

    monkeypatch.setattr(checkpoints, "load", tracked_load)
    if outcome in {"exception", "oom", "mps_oom"}:

        def fail(self, image, apply_sigmoids=False):
            # This frame contains the model and a temporary device tensor.
            temporary = torch.ones_like(image)
            references.append(weakref.ref(temporary))
            if outcome == "oom":
                raise torch.OutOfMemoryError("synthetic accelerator OOM")
            if outcome == "mps_oom":
                raise RuntimeError("MPS backend out of memory")
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(AdapterModel, "image_to_maps", fail)
    events = []

    def callback(event):
        events.append(event)
        if outcome == "cancel" and event.get("phase") == "patch_start":
            return False
        return None

    retained_exception = None
    try:
        infer(
            {"model_path": str(saved_model)},
            {"image": np.zeros((1, 32, 32), np.float32)},
            callback=callback,
        )
    except Exception as exc:
        retained_exception = exc
    if outcome == "success":
        assert retained_exception is None
    elif outcome == "cancel":
        assert isinstance(retained_exception, CancelledError)
    elif "oom" in outcome:
        assert isinstance(retained_exception, EnvironmentError)
    else:
        assert isinstance(retained_exception, RuntimeError)
    gc.collect()
    assert references and all(ref() is None for ref in references)


@pytest.mark.integration
@pytest.mark.parametrize("outcome", ["success", "exception", "cancel"])
def test_training_releases_model_and_optimizer(training_config, monkeypatch, outcome):
    from jdll_cross_goose import training

    references = []
    create = training.create_model
    configure = AdapterModel.configure_optimizers

    def tracked_create(*args, **kwargs):
        model = create(*args, **kwargs)
        references.append(weakref.ref(model))
        return model

    def tracked_optimizer(self):
        optimizer = configure(self)
        references.append(weakref.ref(optimizer))
        return optimizer

    monkeypatch.setattr(training, "create_model", tracked_create)
    monkeypatch.setattr(AdapterModel, "configure_optimizers", tracked_optimizer)
    if outcome == "exception":

        def fail(self, batch, batch_idx):
            raise RuntimeError("training failure")

        monkeypatch.setattr(AdapterModel, "training_step", fail)
    retained_exception = None
    try:
        train(
            {**training_config, "previews": False},
            callback=lambda event: (
                False if outcome == "cancel" and event["type"] == "validation" else None
            ),
        )
    except Exception as exc:
        retained_exception = exc
    assert (retained_exception is None) == (outcome == "success")
    gc.collect()
    assert len(references) == 2 and all(ref() is None for ref in references)


def test_task_bridge_uses_json_without_appose():
    import json

    class Task:
        def update(self, **kwargs):
            self.last = kwargs

    task = Task()
    payload = {"type": "progress", "message": "epoch 1", "current": 2, "maximum": 10}
    Events(task=task).emit(payload)
    assert json.loads(task.last["message"]) == payload
    assert task.last["current"] == 2
