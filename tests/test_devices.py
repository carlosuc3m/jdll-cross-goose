import gc

import numpy as np
import pytest
import torch

from jdll_cross_goose.appose_api import infer, load_model
from jdll_cross_goose.runtime import EnvironmentError, Events, cleanup, resolve_device


def test_explicit_device_resolution(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(EnvironmentError, match="unavailable"):
        resolve_device("cuda")
    events = []
    assert resolve_device("cuda", allow_fallback=True, events=Events(events.append)).type == "cpu"
    assert events[0]["type"] == "warning"
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_cpu_cleanup_does_not_call_accelerator_api(monkeypatch):
    def forbidden():
        pytest.fail("Unavailable device API invoked")

    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden)
    monkeypatch.setattr(torch.mps, "empty_cache", forbidden)
    cleanup(torch.device("cpu"))


@pytest.mark.accelerator
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_actual_model_and_tensor_placement(saved_model, device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    if device == "cuda" and torch.cuda.mem_get_info()[0] < 512 * 1024**2:
        pytest.skip("Less than 512 MiB GPU memory available")
    model, _ = load_model(str(saved_model), device)
    assert all(p.device.type == device for p in model.parameters())
    seen = []
    hook = model.backbone.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].device.type)
    )
    try:
        result = infer(
            {"device": device}, {"image": np.zeros((1, 32, 32), np.float32)}, model=model
        )
        assert result["metadata"]["device_used"] == device
        assert seen and set(seen) == {device}
    finally:
        hook.remove()
        del model
        gc.collect()
        cleanup(torch.device(device))


@pytest.mark.accelerator
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_training_placement(training_config, device, monkeypatch):
    from jdll_cross_goose.appose_api import train
    from jdll_cross_goose.model import AdapterModel

    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    if device == "cuda" and torch.cuda.mem_get_info()[0] < 512 * 1024**2:
        pytest.skip("Less than 512 MiB GPU memory available")
    seen = []
    original = AdapterModel.training_step

    def step(self, batch, batch_idx):
        seen.append((self.device.type, batch["image"].device.type, batch["pts_coord"].device.type))
        return original(self, batch, batch_idx)

    monkeypatch.setattr(AdapterModel, "training_step", step)
    result = train({**training_config, "device": device, "previews": False})
    assert result["device_used"] == device
    assert seen and all(placement == (device, device, device) for placement in seen)
