"""Lightning training with the upstream architecture, losses, sampler, and AdamW defaults."""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import asdict
from pathlib import Path

import lightning as L
import torch
import yaml

from . import checkpoints
from .data_config import DATA_OPTIONS, resolve_data_options
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
    "model_dir",
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
    "preview_memory_mb",
    "validation_fraction",
    "image_axes",
    "mask_axes",
    "tile_size",
    "tile_overlap",
    "flow_batch_size",
    "allow_device_fallback",
    "deterministic",
    "effective_batch_size",
    "mixed_precision",
    "lr_scheduler",
    "validation",
    "optimizer",
} | DATA_OPTIONS


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
    if request.get("batch_size", "auto") == "auto":
        cfg["data"]["batch_size"] = 2 if request["device"] == "cpu" else 8
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
        if key in request and request[key] != "auto":
            cfg["data"][key] = request[key]
    resolve_data_options(cfg["data"], request)
    effective = request.get("effective_batch_size", request.get("batch_size", 8))
    effective = 8 if effective == "auto" else effective
    positive_int(effective, "effective_batch_size")
    batch = cfg["data"]["batch_size"]
    if effective % batch:
        raise ValueError("effective_batch_size must be a multiple of batch_size")
    cfg["data"]["accumulation_steps"] = effective // batch
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
    base_lr = (
        float(
            base_config.get("adapter", {}).get(
                "base_learning_rate",
                base_config.get("optimizer", {}).get("init_args", {}).get("lr", optimizer["lr"]),
            )
        )
        if base_config
        else optimizer["lr"]
    )
    requested_lr = request.get("learning_rate", "auto")
    optimizer["lr"] = float(
        (base_lr * 0.1 if base_config else base_lr) if requested_lr == "auto" else requested_lr
    )
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
        "accumulate_grad_batches": cfg["data"]["accumulation_steps"],
        "precision": ("16-mixed" if request["device"] == "cuda" else "32-true")
        if request.get("mixed_precision", "auto") == "auto"
        else "16-mixed"
        if request["mixed_precision"] is True
        else "32-true",
    }
    from jdll_unet.config import LRSchedulerConfig, _validate_lr_scheduler

    scheduler = LRSchedulerConfig(**request.get("lr_scheduler", {"type": "none"}))
    scheduler.warmup_fraction = (
        0.0 if scheduler.warmup_fraction == "auto" else scheduler.warmup_fraction
    )
    _validate_lr_scheduler(scheduler)
    validation = {"full_every": 0, "full_memory_mb": 512, **request.get("validation", {})}
    if validation.keys() - {"full_every", "full_memory_mb"}:
        raise ValueError("Unsupported validation settings")
    positive_int(validation["full_every"], "validation.full_every", minimum=0)
    positive_int(validation["full_memory_mb"], "validation.full_memory_mb")
    optimizer_name = request.get("optimizer", "adamw")
    if optimizer_name not in {"adam", "adamw", "sgd"}:
        raise ValueError("optimizer must be adam, adamw or sgd")
    cfg["optimizer"]["class_path"] = {
        "adam": "torch.optim.Adam",
        "adamw": "torch.optim.AdamW",
        "sgd": "torch.optim.SGD",
    }[optimizer_name]
    if optimizer_name == "sgd":
        cfg["optimizer"]["init_args"] = {k: optimizer[k] for k in ("lr", "weight_decay")}
        cfg["optimizer"]["init_args"]["momentum"] = 0.9
    cfg["adapter"] = {
        "version": 1,
        "upstream_revision": checkpoints.UPSTREAM_REVISION,
        "request": copy.deepcopy(request),
        "primary_metric": "val/ap_0.75",
        "previews": request.get("previews", True),
        "preview_samples": request.get("preview_samples", 1),
        "preview_memory_mb": request.get("preview_memory_mb", 256),
        "starting_point": request["starting_point"],
        "empty_patch_policy": "configured shared image/patch filtering; negative patches have cell probability loss only",
        "optimizer": optimizer_name,
        "base_learning_rate": base_lr if base_config else optimizer["lr"],
        "lr_scheduler": asdict(scheduler),
        "validation": validation,
    }
    positive_int(cfg["adapter"]["preview_samples"], "preview_samples")
    positive_int(cfg["adapter"]["preview_memory_mb"], "preview_memory_mb")
    if request.get("mixed_precision", "auto") not in ("auto", True, False):
        raise ValueError("mixed_precision must be boolean or auto")
    for key in ("image_axes", "mask_axes"):
        if key in request and request[key] not in {"YX", "CYX", "YXC"}:
            raise ValueError(f"{key} must be YX, CYX or YXC")
    cfg["data"]["data_root"] = str(Path(request["dataset_path"]).expanduser().resolve())
    cfg["data"]["dataset"] = "."
    return cfg


class Progress(L.Callback):
    def __init__(self, events, directory, config, data, control=None):
        self.events, self.directory, self.config, self.val_pairs = (
            events,
            directory,
            config,
            data.val_pairs,
        )
        from jdll_unet.validation_control import FullValidationSchedule

        self.data = data
        self.schedule = FullValidationSchedule(
            config["adapter"]["validation"]["full_every"], control, self.control_event
        )
        self.scheduler = None
        self.previous_step = 0
        self.closed = False
        self.history = {"epochs": [], "losses": {}, "metrics": {}}
        self.best_score = -math.inf
        self.best_path = None
        self.best_by_threshold = {str(t): -math.inf for t in (0.5, 0.75, 0.9)}
        self.last_path = directory / "checkpoints" / "last.ckpt"
        self.total_steps = 0
        self.offset = 0
        self.run = None
        self.step_offset = 0

    def control_event(self, kind, **fields):
        self.events.emit({"type": kind, **fields}, check=False)

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.schedule.close(cancelled=self.events.cancelled)
            except Exception as exc:
                self.events.emit(
                    {"type": "warning", "message": f"Could not close validation control: {exc}"},
                    check=False,
                )

    def state_dict(self):
        return {
            "history": self.history,
            "best_score": self.best_score,
            "best_path": str(self.best_path) if self.best_path else None,
            "best_by_threshold": self.best_by_threshold,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "full_validation": self.schedule.state_dict(),
        }

    def on_train_start(self, trainer, pl_module):
        from jdll_unet.config import LRSchedulerConfig
        from jdll_unet.schedulers import LearningRateScheduler

        self.total_steps = int(trainer.estimated_stepping_batches)
        self.scheduler = LearningRateScheduler(
            trainer.optimizers[0],
            LRSchedulerConfig(**self.config["adapter"]["lr_scheduler"]),
            self.total_steps,
            trainer.max_epochs,
        )
        if self.run and self.run.selected:
            slot = self.run.selected
            steps = max(1, self.total_steps // trainer.max_epochs)
            self.step_offset = self.offset * steps
            self.total_steps += self.step_offset
            if slot["restore"]:
                bundle = torch.load(self.run.parent / slot["state"], map_location="cpu", weights_only=False)
                trainer.optimizers[0].load_state_dict(bundle["optimizer_states"][0])
                state = bundle["callbacks"][self.state_key]["scheduler"]
                exact = self.scheduler.continue_from(state, trainer.max_epochs, steps, self.run.validation_changed)
                if self.scheduler.type == "plateau" and self.run.validation_changed:
                    self.scheduler.best_score = self.best_score
                plugin = trainer.precision_plugin
                if plugin.__class__.__qualname__ in bundle:
                    plugin.load_state_dict(bundle[plugin.__class__.__qualname__])
                self.events.message("Restored optimizer and " + ("original LR schedule." if exact else "replanned remaining LR schedule."))
            else:
                self.events.message("Restoring weights with a fresh optimizer.")
            self.best_path = self.last_path.parent / "best.ckpt"
            with atomic_path(self.best_path) as temporary:
                trainer.save_checkpoint(temporary, weights_only=False)
            self.run.checkpoint("best", self.offset, self.best_path, self.best_path, self.best_score)
        self.schedule.event("ready", supported=True)
        self.emit_progress(trainer, epoch=self.offset)

    def emit_progress(self, trainer, *, epoch=None):
        self.events.emit(
            {
                "type": "progress",
                "message": "",
                "epoch": trainer.current_epoch + 1 + self.offset if epoch is None else epoch,
                "step": (trainer.global_step + self.step_offset),
                "total_epochs": (trainer.max_epochs + self.offset),
                "total_steps": self.total_steps,
                "current": (trainer.global_step + self.step_offset),
                "maximum": self.total_steps,
                "losses": {},
                "metrics": {},
            }
        )

    def on_train_epoch_start(self, trainer, pl_module):
        trainer.train_dataloader.dataset.set_epoch(trainer.current_epoch + self.offset)
        write_json(
            self.directory / "sampling" / f"epoch_{trainer.current_epoch + 1 + self.offset:04d}.json",
            {
                "epoch": trainer.current_epoch + 1 + self.offset,
                "source_requests": [
                    {"source": key[0], "role": key[1], "patches": value}
                    for key, value in trainer.train_dataloader.dataset.cache_requests.items()
                ],
            },
        )
        self.events.message(
            f"Starting epoch {trainer.current_epoch + 1 + self.offset}/{(trainer.max_epochs + self.offset)}",
            kind="epoch_start",
            epoch=trainer.current_epoch + 1 + self.offset,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.events.check()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.events.check()
        self.schedule.poll()
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if not torch.isfinite(loss).all():
            raise TrainingError("Training produced a non-finite loss")
        every = 1 if pl_module.device.type == "cpu" else 5
        if (trainer.global_step + self.step_offset) != self.previous_step:
            self.scheduler.step_batch()
            self.previous_step = (trainer.global_step + self.step_offset)
            if (trainer.global_step + self.step_offset) % every == 0:
                self.emit_progress(trainer)

    def on_validation_start(self, trainer, pl_module):
        self.full_pass = self.schedule.boundary(trainer.current_epoch + 1 + self.offset)
        pl_module.validation_previews = []
        self.events.message(f"Validating epoch {trainer.current_epoch + 1 + self.offset}", kind="validation")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.events.check()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.events.emit(
            {
                "type": "validation",
                "status": "progress",
                "epoch": trainer.current_epoch + 1 + self.offset,
                "current": batch_idx + 1,
                "maximum": trainer.num_val_batches[dataloader_idx],
                "unit": "batches",
                "message": "",
            }
        )

    def on_train_epoch_end(self, trainer, pl_module):
        self.events.check()
        epoch = trainer.current_epoch + 1 + self.offset
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
        improved_thresholds = [
            t for t in (0.5, 0.75, 0.9) if metrics[f"val/ap_{t}"] > self.best_by_threshold[str(t)]
        ]
        for threshold in improved_thresholds:
            self.best_by_threshold[str(threshold)] = metrics[f"val/ap_{threshold}"]
        self.scheduler.step_epoch(score)
        previous_best = self.best_path
        if improved:
            self.best_score = score
            self.best_path = (
                self.last_path.parent
                / f"best-epoch={epoch}-step={(trainer.global_step + self.step_offset)}-val_ap_0.75={score:.4f}.ckpt"
            )
        # Save a full checkpoint atomically after every successfully validated
        # epoch, before callbacks/previews can cancel or fail.
        with atomic_path(self.last_path) as temporary:
            trainer.save_checkpoint(temporary, weights_only=False)
        if improved:
            checkpoints.copy_checkpoint(self.last_path, self.best_path)
            checkpoints.copy_checkpoint(self.last_path, self.last_path.parent / "best.ckpt")
            if previous_best is not None and previous_best != self.best_path and previous_best.name != "best.ckpt":
                previous_best.unlink(missing_ok=True)
        if self.run:
            self.run.checkpoint("last", epoch, self.last_path, self.last_path, score)
            if improved:
                path = self.last_path.parent / "best.ckpt"
                self.run.checkpoint("best", epoch, path, path, score)
        for threshold in improved_thresholds:
            checkpoints.copy_checkpoint(
                self.last_path, self.last_path.parent / f"best_ap_{threshold}.ckpt"
            )
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
                "epoch_complete": True,
                "message": f"Epoch {epoch}: train loss {losses['train/total_loss']:.5f}, val AP@0.75 {score:.4f}",
                "epoch": epoch,
                "step": (trainer.global_step + self.step_offset),
                "total_epochs": (trainer.max_epochs + self.offset),
                "total_steps": self.total_steps,
                "current": (trainer.global_step + self.step_offset),
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
        if self.full_pass is not None:
            from .full_validation import evaluate

            self.schedule.started()
            try:
                path, metrics = evaluate(pl_module, self.data, epoch, self.config, self.events)
                self.schedule.finish(
                    "completed",
                    path=str(path),
                    metrics=metrics,
                    message=f"Full validation saved: {path}",
                )
            except CancelledError:
                self.schedule.finish(
                    "cancelled",
                    message="Full validation cancelled; regular checkpoints are preserved.",
                )
                raise
            except Exception as exc:
                self.schedule.finish(
                    "failed",
                    message=f"Full validation failed; regular checkpoints are preserved: {exc}",
                )
                if is_oom(exc):
                    raise EnvironmentError(f"Full validation ran out of memory: {exc}") from exc
                raise ValidationError(f"Full validation failed: {exc}") from exc
            finally:
                write_json(self.directory / "validation_state.json", self.schedule.state_dict())


def run(request, events, control=None):
    request = dict(request)
    run_state = request.pop("_jdll_run", None)
    request = copy.deepcopy(request)
    cfg = resolve_config(request)
    # JDLL already chooses a unique final directory and starts its UI log there.
    # Keep output_dir/model_name for existing standalone Python callers.
    directory = (
        Path(
            request.get("model_dir")
            or Path(request["output_dir"]).expanduser() / request["model_name"]
        )
        .expanduser()
        .resolve()
    )
    if directory.exists() and any(
        p.name not in {"training-ui.log", "split.json", "validation.sha256", "continuation.json", ".training.lock"}
        or not p.is_file() for p in directory.iterdir()
    ):
        raise FileExistsError(
            f"Model output directory is not empty: {directory}; choose a new model_name"
        )
    device = None
    model = trainer = train_loader = val_loader = progress = data = None
    try:
        device = resolve_device(
            request["device"],
            allow_fallback=request.get("allow_device_fallback", False),
            events=events,
        )
        runtime_request = {**request, "device": device.type}
        cfg = resolve_config(runtime_request)
        L.seed_everything(cfg["seed_everything"], workers=True, verbose=False)
        events.message("Validating dataset images and integer instance masks", kind="dataset")
        if request["starting_point"] == "fine_tune":
            model, metadata = checkpoints.load(
                run_state.parent / run_state.selected["state" if run_state.selected["restore"] else "weights"]
                if run_state and run_state.selected else request["base_model"],
                str(device),
                criterion=request.get("checkpoint", "best"),
                events=events,
            )
            validate_default_architecture(metadata["config"])
            cfg = resolve_config(runtime_request, metadata["config"])
            cfg["adapter"]["source_checkpoint"] = metadata["checkpoint_path"]
            if run_state and run_state.selected:
                previous = metadata["config"]
                cfg["optimizer"] = copy.deepcopy(previous["optimizer"])
                for name in ("optimizer", "lr_scheduler"):
                    cfg["adapter"][name] = copy.deepcopy(previous["adapter"][name])
            model.n_steps = cfg["model"]["n_steps"]
        else:
            events.message("Initializing upstream default architecture with random weights")
            model = create_model(cfg)
        from .training_data import TrainingData

        directory.mkdir(parents=True, exist_ok=True)
        data = TrainingData(cfg["data"], directory, events)
        train_pairs, val_pairs, statistics = data.prepare(request["dataset_path"], run_state)
        if run_state:
            run_state.record_domains(*[[(p.image, p.mask, p.region) for p in group] for group in (train_pairs, val_pairs)])
        train_loader, val_loader = data.make_loaders()
        cfg["data"]["resolved_patches_per_epoch"] = len(train_loader.dataset)
        cfg["data"]["resolved_validation_patches"] = len(val_loader.dataset)
        cfg["data"]["resolved_augmentation"] = asdict(train_loader.dataset.augmentation)
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
        manifest = str(directory / "dataset.json")
        events.message(f"Dataset manifest: {manifest}", kind="dataset", path=manifest)
        model.optimizer_config = cfg["optimizer"]["init_args"]
        model.optimizer_name = cfg["adapter"]["optimizer"]
        model.events = events
        model.preview_limit = cfg["adapter"]["preview_samples"] if cfg["adapter"]["previews"] else 0
        model.preview_max_bytes = cfg["adapter"]["preview_memory_mb"] * 1024**2
        model.inference_config = {
            k: request[k] for k in ("tile_size", "tile_overlap", "flow_batch_size") if k in request
        }
        model._jdll_metadata = {
            "model_path": str(directory),
            "config_path": str(cfg_path),
            "config": cfg,
        }
        model.to(device).train()
        progress = Progress(events, directory, cfg, data, control)
        progress.run = run_state
        progress.offset = run_state.start if run_state else 0
        trainer = L.Trainer(**cfg["trainer"], callbacks=[progress], default_root_dir=str(directory))
        if run_state and run_state.selected:
            score = run_state.selected.get("score")
            if run_state.validation_changed or score is None:
                trainer.validate(model, dataloaders=val_loader, verbose=False)
                score = float(trainer.callback_metrics["val_ap_0.75"])
                events.message(f"Scored the starting weights on the new validation split: AP@0.75 {score:.4f}.")
            progress.best_score = score
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        events.check()
        progress.close()
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
        if progress is not None:
            progress.close()
        if trainer is not None:
            # Lightning normally tears down its strategy even on exceptions.
            # Detach the ownership cycle and optimizer state before collecting.
            trainer.optimizers.clear()
        try:
            if data is not None:
                data.close()
        finally:
            if model is not None:
                model.events = None
                model._trainer = None
            model = trainer = train_loader = val_loader = progress = None
            clear_error_frames(sys.exception())
            cleanup(device)
