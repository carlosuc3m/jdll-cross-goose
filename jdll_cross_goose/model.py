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
        self.criterion_cellprob = nn.BCEWithLogitsLoss(reduction="mean")
        self.flow_fac = 5.0
        self.crit_flow_weight = crit_flow_weight
        self.crit_cellprob_weight = crit_cellprob_weight
        if backbone_realease_delay:
            self.backbone.requires_grad_(False)
        self.save_hyperparameters(ignore=["flow_fn"])
        self.optimizer_config = {"lr": 0.005, "weight_decay": 0.01}
        self.events = None

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), **self.optimizer_config)

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
        from crossgoose.cellpose.metrics import average_precision

        from .inference import segment_normalized
        from .runtime import CancelledError, EnvironmentError, ValidationError, is_oom

        try:
            self.events.check()
            losses = self.compute_loss(batch, log_prefix="val_")
            predictions = []
            truths = []
            for index, image in enumerate(batch["image"]):
                result = segment_normalized(
                    self, image.cpu().numpy(), self.inference_config, self.events, progress=False
                )
                predictions.append(result["labels"])
                truths.append(batch["labels"][index].cpu().numpy())
            with np.errstate(invalid="ignore", divide="ignore"):
                ap, _, _, _ = average_precision(truths, predictions, [0.5, 0.75, 0.9])
            for i, (truth, prediction) in enumerate(zip(truths, predictions)):
                if not np.any(truth) and not np.any(prediction):
                    ap[i, :] = 1.0
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
