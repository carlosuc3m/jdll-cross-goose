"""A small constructor adapter; all network operations and losses are upstream.

The constructor is adapted from CrossGooseModel (MIT, Julie Mabon, 2026).
See THIRD_PARTY_LICENSES.txt. It avoids an unnecessary cyto3 download and makes
scratch training actually start with random weights, with identical topology.
"""

from __future__ import annotations

import copy
import inspect

import lightning
import torch
from crossgoose.cellpose.resnet_torch import batchconv
from crossgoose.model import flow_function
from crossgoose.model.models import CrossGooseModel
from crossgoose.model.modules import CPBackbone
from torch import nn
from torch.nn import functional as F


class ValidBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.validity = None

    def forward(self, logits, target):
        if self.validity is None or bool(self.validity.all()):
            return F.binary_cross_entropy_with_logits(logits, target)
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        axes = tuple(range(1, losses.ndim))
        return ((losses * self.validity).sum(axes) / self.validity.sum(axes).clamp_min(1)).mean()


class AdapterModel(CrossGooseModel):
    def __init__(
        self,
        flow_fn,
        embeddings_dim=8,
        crit_flow_weight=0.1,
        crit_cellprob_weight=2.0,
        n_steps=200,
        backbone_realease_delay=None,
        shared_embedding=False,
        train_on_trajectories=False,
        time_error_weighting=False,
    ):
        lightning.LightningModule.__init__(self)
        self.n_steps = n_steps
        self.embeddings_dim = embeddings_dim
        self.backbone_realease_delay = backbone_realease_delay
        self.shared_embedding = shared_embedding
        self.train_on_trajectories = train_on_trajectories
        self.time_error_weighting = time_error_weighting
        self.backbone = CPBackbone(device=self.device, load_pretrained=False)
        self.backbone_out = self.backbone.nbase[1]
        self.flow_fn = flow_fn
        out_c = (embeddings_dim if shared_embedding else 2 * embeddings_dim) + 1
        self.embedding_head = batchconv(self.backbone_out, out_c, sz=1)
        self.criterion_flow = nn.MSELoss(reduction="none")
        self.criterion_cellprob = ValidBCE()
        self.flow_fac = 5.0
        self.crit_flow_weight = crit_flow_weight
        self.crit_cellprob_weight = crit_cellprob_weight
        if backbone_realease_delay:
            self.backbone.requires_grad_(False)
        self.save_hyperparameters(ignore=["flow_fn"])
        self.optimizer_config = {"lr": 0.005, "weight_decay": 0.01}
        self.events = None
        self._cached_features = None
        self.validation_previews = []
        self.preview_limit = 0
        self.preview_max_bytes = 256 * 1024**2
        self.optimizer_name = "adamw"

    def configure_optimizers(self):
        optimizer = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}[
            self.optimizer_name
        ]
        return optimizer(self.parameters(), **self.optimizer_config)

    def image_to_maps(self, image, apply_sigmoids=False):
        if self._cached_features is not None:
            return self._cached_features
        return super().image_to_maps(image, apply_sigmoids)

    def compute_loss(self, batch, log_prefix=""):
        self.criterion_cellprob.validity = batch.get("validity")
        try:
            return super().compute_loss(batch, log_prefix)
        finally:
            self.criterion_cellprob.validity = None

    def training_step(self, batch, batch_idx):
        # Same loss/aggregation as upstream, without attempting to write to an
        # absent TensorBoard logger or render a progress bar in an Appose task.
        losses = self.compute_loss(batch)
        self.log_dict(
            losses,
            prog_bar=False,
            logger=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch["image"].shape[0],
        )
        return losses["loss"]

    def on_train_epoch_start(self):
        if self.backbone_realease_delay == self.current_epoch:
            self.backbone.requires_grad_(True)

    def validation_step(self, batch, batch_idx):
        # The loss and AP definition remain upstream. The inference wrapper adds
        # cancellation and guards the undefined empty/empty AP case.
        import numpy as np

        from .full_validation import compact, instance_ap
        from .inference import reconstruct
        from .runtime import CancelledError, EnvironmentError, ValidationError, is_oom

        try:
            self.events.check()
            self._cached_features = super().image_to_maps(
                batch["image"][:, :2], apply_sigmoids=False
            )
            try:
                losses = self.compute_loss(batch, log_prefix="val_")
                raw_maps = self._cached_features["T1"].detach().float().cpu()
            finally:
                self._cached_features = None
            predictions = []
            truths = []
            for index, image in enumerate(batch["image"]):
                self.events.check()
                result = reconstruct(self, raw_maps[index], self.inference_config, self.events)
                truth = batch["labels"][index].cpu().numpy()
                prediction = result["labels"].copy()
                if "validity" in batch:
                    valid = batch["validity"][index].cpu().numpy()
                    truth = np.where(valid, truth, 0)
                    prediction[~valid] = 0
                truth, prediction = compact(truth), compact(prediction)
                predictions.append(prediction)
                truths.append(truth)
                if len(self.validation_previews) < self.preview_limit:
                    sample = {
                        "image": image.detach().float().cpu().numpy().copy(),
                        "labels": truth,
                        "prediction": prediction,
                        "probability": result["cell_probability"],
                        "metadata": batch.get("metadata", [{}] * len(batch["image"]))[index],
                    }
                    size = sum(v.nbytes for v in sample.values() if isinstance(v, np.ndarray))
                    used = sum(
                        v.nbytes
                        for p in self.validation_previews
                        for v in p.values()
                        if isinstance(v, np.ndarray)
                    )
                    if used + size <= self.preview_max_bytes:
                        self.validation_previews.append(sample)
            ap = instance_ap(truths, predictions)
            for i, threshold in enumerate([0.5, 0.75, 0.9]):
                losses[f"val_ap_{threshold}"] = float(np.mean(ap[:, i]))
            if not all(np.isfinite(float(v)) for v in losses.values()):
                raise ValueError("Validation produced a non-finite loss or metric")
            self.log_dict(losses, batch_size=len(truths), on_epoch=True, logger=False)
        except CancelledError:
            raise
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:
            if is_oom(exc):
                raise EnvironmentError(f"Validation ran out of memory: {exc}") from exc
            raise ValidationError(f"Validation failed: {exc}") from exc


def create_model(config):
    params = copy.deepcopy(config.get("model", {}))
    if "init_args" in params:
        if params.get("class_path") != "crossgoose.model.models.CrossGooseModel":
            raise ValueError("Unsupported model class in config")
        params = params["init_args"]
    flow = params.pop("flow_fn", None)
    if not isinstance(flow, dict):
        raise ValueError("Compatible config must contain model.flow_fn")
    allowed = {
        f"crossgoose.model.flow_function.{name}": getattr(flow_function, name)
        for name in ("FlowLinear", "FlowAttention", "CPLike")
    }
    if flow.get("class_path") not in allowed:
        raise ValueError(f"Unsupported flow function: {flow.get('class_path')}")
    unknown = set(params) - (set(inspect.signature(AdapterModel).parameters) - {"flow_fn"})
    if unknown:
        raise ValueError(f"Unsupported model configuration keys: {sorted(unknown)}")
    return AdapterModel(flow_fn=allowed[flow["class_path"]](**flow.get("init_args", {})), **params)


def validate_default_architecture(config):
    params = config["model"].get("init_args", config["model"])
    flow = params.get("flow_fn", {})
    args = flow.get("init_args", {})
    if (
        flow.get("class_path") != "crossgoose.model.flow_function.FlowLinear"
        or params.get("embeddings_dim", 8) != 8
        or args.get("embedding_dim") != 8
        or args.get("hidden_dims") != [16, 8]
        or args.get("value_dim", 2) != 2
        or params.get("train_on_trajectories", False)
    ):
        raise ValueError("Fine-tuning requires the default FlowLinear architecture (8, [16, 8], 2)")
