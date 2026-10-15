"""Lightning training with the upstream architecture, losses, sampler, and AdamW defaults."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import lightning as L
import torch
import yaml

from . import checkpoints
from .data import loaders, prepare_flows, validate_dataset, write_manifest
from .model import create_model, validate_default_architecture
from .preview import write_preview
from .runtime import (
    CancelledError,
    EnvironmentError,
    TrainingError,
    ValidationError,
    atomic_path,
    cleanup,
    clear_error_frames,
    is_oom,
    positive_int,
    resolve_device,
    write_json,
)

REQUIRED = {
    "model_name",
    "output_dir",
    "dataset_path",
    "architecture",
    "starting_point",
    "device",
    "epochs",
}
OPTIONS = {
    "base_model",
    "checkpoint",
    "batch_size",
    "val_batch_size",
    "num_workers",
    "epochs",
    "seed",
    "patch_size",
    "val_patch_size",
    "n_steps",
    "learning_rate",
    "weight_decay",
    "n_samples",
    "sigma",
    "grayscale",
    "image_normalization",
    "bootstrap_factor",
    "prefetch_factor",
    "gridflow_n_interpol",
    "center_method",
    "previews",
    "preview_samples",
    "validation_fraction",
    "image_axes",
    "mask_axes",
    "tile_size",
    "tile_overlap",
    "flow_batch_size",
    "allow_device_fallback",
    "deterministic",
}


def resolve_config(request, base_config=None):
    missing = REQUIRED - request.keys()
    if missing:
        raise ValueError(f"Missing training configuration: {sorted(missing)}")
    unknown = request.keys() - REQUIRED - OPTIONS
    if unknown:
        raise ValueError(f"Unknown training options: {sorted(unknown)}")
    if request["architecture"] != "default":
        raise ValueError("Only architecture='default' is supported")
    if request["starting_point"] not in {"scratch", "fine_tune"}:
        raise ValueError("starting_point must be scratch or fine_tune")
    if request["starting_point"] == "fine_tune" and not request.get("base_model"):
        raise ValueError("fine_tune requires base_model")
    name = request["model_name"]
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("model_name must be a single nonempty directory name")
    positive_int(request["epochs"], "epochs")
    cfg = checkpoints.default_config()
    # Preserve a fine-tuned model's topology and normalization, including the
    # original upstream default model's separate embedding heads.
    if base_config:
        cfg["model"] = copy.deepcopy(base_config["model"].get("init_args", base_config["model"]))
        for key in ("image_normalization", "grayscale"):
            cfg["data"][key] = base_config["data"].get(key, True if key == "grayscale" else "M1P1")
    cfg["seed_everything"] = request.get("seed", 16)
    positive_int(cfg["seed_everything"], "seed", minimum=0)
    cfg["data"].update(num_workers=0, keep_data_in_memory=False, lazy_flow_computing=False)
    for key in (
        "batch_size",
        "val_batch_size",
        "num_workers",
        "grayscale",
        "image_normalization",
        "bootstrap_factor",
        "prefetch_factor",
        "gridflow_n_interpol",
        "center_method",
        "image_axes",
        "mask_axes",
        "val_patch_size",
    ):
        if key in request:
            cfg["data"][key] = request[key]
    augmentation = cfg["data"]["augmentation_params"]
    augmentation.setdefault("deterministic_patch", False)
    if "patch_size" in request:
        augmentation["patch_size"] = request["patch_size"]
    for key, value in (
        ("patch_size", augmentation["patch_size"]),
        ("val_patch_size", cfg["data"]["val_patch_size"]),
    ):
        positive_int(value, key, minimum=32)
        if value % 16:
            raise ValueError(f"{key} must be a multiple of 16")
    for key in (
        "batch_size",
        "val_batch_size",
        "bootstrap_factor",
        "prefetch_factor",
        "gridflow_n_interpol",
    ):
        positive_int(cfg["data"][key], key)
    positive_int(cfg["data"]["num_workers"], "num_workers", minimum=0)
    if cfg["data"]["image_normalization"] not in {"M1P1", "N99"}:
        raise ValueError("image_normalization must be M1P1 or N99")
    from crossgoose.mask_utils import CENTER_METHODS

    if cfg["data"]["center_method"] not in CENTER_METHODS:
        raise ValueError(f"Unknown center_method: {cfg['data']['center_method']}")
    sampler = cfg["data"]["points_sampler"]["init_args"]
    for key in ("n_samples", "sigma"):
        if key in request:
            sampler[key] = request[key]
    positive_int(sampler["n_samples"], "n_samples")
    if not math.isfinite(float(sampler["sigma"])) or sampler["sigma"] < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if "n_steps" in request:
        cfg["model"]["n_steps"] = request["n_steps"]
    positive_int(cfg["model"]["n_steps"], "n_steps")
    optimizer = cfg["optimizer"]["init_args"]
    optimizer["lr"] = float(request.get("learning_rate", optimizer["lr"]))
    optimizer["weight_decay"] = float(request.get("weight_decay", optimizer["weight_decay"]))
    if not math.isfinite(optimizer["lr"]) or optimizer["lr"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(optimizer["weight_decay"]) or optimizer["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    cfg["trainer"] = {
        "max_epochs": request["epochs"],
        "accelerator": request["device"],
        "devices": 1,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": 1,
        "enable_checkpointing": False,
        "deterministic": request.get("deterministic", False),
    }
    cfg["adapter"] = {
        "version": 1,
        "upstream_revision": checkpoints.UPSTREAM_REVISION,
        "request": copy.deepcopy(request),
        "primary_metric": "val/ap_0.75",
        "previews": request.get("previews", True),
        "preview_samples": request.get("preview_samples", 1),
        "starting_point": request["starting_point"],
        "empty_patch_policy": "sample; cell probability loss only when no foreground",
    }
    positive_int(cfg["adapter"]["preview_samples"], "preview_samples")
    cfg["data"]["data_root"] = str(Path(request["dataset_path"]).expanduser().resolve())
    cfg["data"]["dataset"] = "."
    return cfg


class Progress(L.Callback):
    def __init__(self, events, directory, config, val_pairs):
        self.events, self.directory, self.config, self.val_pairs = (
            events,
            directory,
            config,
            val_pairs,
        )
        self.history = {"epochs": [], "losses": {}, "metrics": {}}
        self.best_score = -math.inf
        self.best_path = None
        self.last_path = directory / "checkpoints" / "last.ckpt"
        self.total_steps = 0

    def state_dict(self):
        return {
            "history": self.history,
            "best_score": self.best_score,
            "best_path": str(self.best_path) if self.best_path else None,
        }

    def on_train_start(self, trainer, pl_module):
        self.total_steps = int(trainer.estimated_stepping_batches)

    def on_train_epoch_start(self, trainer, pl_module):
        self.events.message(
            f"Starting epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
            kind="epoch_start",
            epoch=trainer.current_epoch + 1,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.events.check()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.events.check()
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if not torch.isfinite(loss).all():
            raise TrainingError("Training produced a non-finite loss")

    def on_validation_start(self, trainer, pl_module):
        self.events.message(f"Validating epoch {trainer.current_epoch + 1}", kind="validation")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.events.check()

    def on_train_epoch_end(self, trainer, pl_module):
        self.events.check()
        epoch = trainer.current_epoch + 1
        values = trainer.callback_metrics
        losses = {
            "train/total_loss": float(values["loss"]),
            "val/total_loss": float(values["val_loss"]),
        }
        metrics = {f"val/ap_{t}": float(values[f"val_ap_{t}"]) for t in (0.5, 0.75, 0.9)}
        if not all(math.isfinite(v) for v in [*losses.values(), *metrics.values()]):
            raise TrainingError("Epoch aggregation produced non-finite values")
        self.history["epochs"].append(epoch)
        for group, data in (("losses", losses), ("metrics", metrics)):
            for key, value in data.items():
                self.history[group].setdefault(key, []).append(value)
        score = metrics["val/ap_0.75"]
        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_path = (
                self.last_path.parent
                / f"best-epoch={epoch}-step={trainer.global_step}-val_ap_0.75={score:.4f}.ckpt"
            )
        # Save a full checkpoint atomically after every successfully validated
        # epoch, before callbacks/previews can cancel or fail.
        with atomic_path(self.last_path) as temporary:
            trainer.save_checkpoint(temporary, weights_only=False)
        if improved:
            checkpoints.copy_checkpoint(self.last_path, self.best_path)
            checkpoints.copy_checkpoint(self.last_path, self.last_path.parent / "best.ckpt")
        write_json(self.directory / "history.json", self.history)
        self.events.message(
            f"Saved last checkpoint: {self.last_path}",
            kind="checkpoint",
            checkpoint_path=str(self.last_path),
            epoch=epoch,
        )
        if improved:
            self.events.message(
                f"New best model: AP@0.75={score:.4f}: {self.best_path}",
                kind="checkpoint",
                checkpoint_path=str(self.best_path),
                best=True,
                epoch=epoch,
            )
        self.events.emit(
            {
                "type": "progress",
                "message": f"Epoch {epoch}: train loss {losses['train/total_loss']:.5f}, val AP@0.75 {score:.4f}",
                "epoch": epoch,
                "step": trainer.global_step,
                "total_epochs": trainer.max_epochs,
                "total_steps": self.total_steps,
                "current": trainer.global_step,
                "maximum": self.total_steps,
                "losses": losses,
                "metrics": metrics,
            }
        )
        if self.config["adapter"]["previews"]:
            try:
                write_preview(
                    pl_module, self.val_pairs, self.directory, epoch, self.config, self.events
                )
            except (CancelledError, torch.OutOfMemoryError):
                raise
            except Exception as exc:
                if is_oom(exc):
                    raise EnvironmentError(f"Validation preview ran out of memory: {exc}") from exc
                raise ValidationError(f"Validation preview failed: {exc}") from exc


def run(request, events):
    request = copy.deepcopy(request)
    cfg = resolve_config(request)
    directory = (Path(request["output_dir"]).expanduser() / request["model_name"]).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"Model output directory is not empty: {directory}; choose a new model_name"
        )
    device = None
    model = trainer = train_loader = val_loader = progress = None
    try:
        device = resolve_device(
            request["device"],
            allow_fallback=request.get("allow_device_fallback", False),
            events=events,
        )
        L.seed_everything(cfg["seed_everything"], workers=True, verbose=False)
        events.message("Validating dataset images and integer instance masks", kind="dataset")
        train_pairs, val_pairs, statistics = validate_dataset(request, events)
        if request["starting_point"] == "fine_tune":
            model, metadata = checkpoints.load(
                request["base_model"],
                str(device),
                criterion=request.get("checkpoint", "best"),
                events=events,
            )
            validate_default_architecture(metadata["config"])
            cfg = resolve_config(request, metadata["config"])
            cfg["adapter"]["source_checkpoint"] = metadata["checkpoint_path"]
            model.n_steps = cfg["model"]["n_steps"]
        else:
            events.message("Initializing upstream default architecture with random weights")
            model = create_model(cfg)
        cfg["adapter"].update(
            object_size_statistics=statistics,
            device_requested=request["device"],
            device_used=device.type,
        )
        cfg["trainer"]["accelerator"] = "gpu" if device.type == "cuda" else device.type
        directory.mkdir(parents=True, exist_ok=True)
        cfg_path = directory / "config.yaml"
        with atomic_path(cfg_path) as temporary:
            temporary.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        events.message(
            f"Resolved configuration: {cfg_path}",
            kind="configuration",
            config=cfg,
            config_path=str(cfg_path),
        )
        prepare_flows(train_pairs + val_pairs, directory / "dataset-cache", cfg["data"], events)
        manifest = write_manifest(directory, train_pairs, val_pairs, statistics)
        events.message(f"Dataset manifest: {manifest}", kind="dataset", path=manifest)
        train_loader, val_loader = loaders(train_pairs, val_pairs, cfg["data"])
        model.optimizer_config = cfg["optimizer"]["init_args"]
        model.events = events
        model.inference_config = {
            k: request[k] for k in ("tile_size", "tile_overlap", "flow_batch_size") if k in request
        }
        model._jdll_metadata = {
            "model_path": str(directory),
            "config_path": str(cfg_path),
            "config": cfg,
        }
        model.to(device).train()
        progress = Progress(events, directory, cfg, val_pairs)
        trainer = L.Trainer(**cfg["trainer"], callbacks=[progress], default_root_dir=str(directory))
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        events.check()
        result = {
            "model_dir": str(directory),
            "last_checkpoint": str(progress.last_path),
            "best_checkpoint": str(progress.best_path),
            "config_path": str(cfg_path),
            "history": progress.history,
            "dataset_manifest": manifest,
            "device_requested": request["device"],
            "device_used": device.type,
            "status": "completed",
        }
        write_json(directory / "result.json", result)
        events.message(f"Training completed: {directory}", kind="completion", **result)
        return result
    except (CancelledError, ValidationError, EnvironmentError):
        raise
    except torch.OutOfMemoryError as exc:
        raise EnvironmentError(
            f"Training ran out of {device.type if device else 'device'} memory"
        ) from exc
    except (ImportError, OSError) as exc:
        raise EnvironmentError(f"Training environment failure: {exc}") from exc
    except (ValueError, FileExistsError):
        raise
    except Exception as exc:
        if is_oom(exc):
            raise EnvironmentError(
                f"Training ran out of {device.type if device else 'host'} memory"
            ) from exc
        raise TrainingError(f"Training failed: {exc}") from exc
    finally:
        if trainer is not None:
            # Lightning normally tears down its strategy even on exceptions.
            # Detach the ownership cycle and optimizer state before collecting.
            trainer.optimizers.clear()
        if model is not None:
            model.events = None
            model._trainer = None
        model = trainer = train_loader = val_loader = progress = None
        clear_error_frames(sys.exception())
        cleanup(device)
