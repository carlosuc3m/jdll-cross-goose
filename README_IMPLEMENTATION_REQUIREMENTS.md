# jdll-cross-goose Implementation Requirements

## Purpose

`jdll-cross-goose` is a lightweight Python adapter around
[bia-pasteur/cross-goose](https://github.com/bia-pasteur/cross-goose). It must
provide stable callable APIs for JDLL/Appose without depending on Appose itself.
The upstream model and scientific behavior should remain authoritative; this
package supplies integration, dataset normalization, tiling, callbacks,
cancellation, checkpoint reporting, and tests.

Distribution name: `jdll-cross-goose`

Python module: `jdll_cross_goose`

Supported scope for the first release: 2D Cross-GOOSE instance segmentation,
using upstream's default architecture for training from scratch. CPU, CUDA, and
MPS must be selectable when supported by PyTorch and upstream operations.

## Public API

Create `jdll_cross_goose.appose_api` with these functions. The `task` argument is
an optional duck-typed bridge and must not import Appose.

```python
def load_model(model_path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]: ...

def infer(
    config: dict,
    inputs: dict,
    *,
    model: torch.nn.Module | None = None,
    callback: Callable[[dict], bool | None] | None = None,
    cancellation: Callable[[], bool] | None = None,
) -> dict: ...

def train(
    config: dict,
    *,
    callback: Callable[[dict], bool | None] | None = None,
    cancellation: Callable[[], bool] | None = None,
    task: object | None = None,
) -> dict: ...
```

If `task` is provided, `train` may adapt structured events to `task.update`, but
all core code and tests must work with plain callbacks and no Appose dependency.

## Model Layout

Accept both upstream layouts:

```text
model-name/
  config.yaml
  weights.ckpt
```

```text
model-name/
  config.yaml
  checkpoints/
    last.ckpt
    best-*.ckpt
```

Also accept `config.yml` and an equivalent `config.json`. Loading a checkpoint
without a compatible config must fail clearly. The adapter must select explicit
`last` and `best` checkpoints deterministically and report the chosen path.

Training output must use the second directory layout and preserve the complete
Lightning checkpoint plus the resolved config needed to reload it. Never modify
fine-tuning source weights. Maintain stable `checkpoints/last.ckpt` and
`checkpoints/best.ckpt` paths in addition to any metric-bearing checkpoint names,
so the Java UI can report predictable locations while the returned result still
contains the exact selected paths.

## Inference Contract

`inputs["image"]` is a contiguous NumPy array in `CYX` order. It may contain one
or two channels. More channels must be reduced deterministically to the first two;
a single channel must follow upstream's grayscale behavior. Validate finite
numeric input and normalize according to the saved model configuration.

`infer` returns:

```python
{
    "outputs": {
        "labels": np.ndarray,      # YX integer instance labels
        "cell_probability": np.ndarray,  # optional YX float map
    },
    "metadata": {
        "model_path": str,
        "checkpoint_path": str,
        "device_requested": str,
        "device_used": str,
        "input_axes": "cyx",
        "output_axes": "yx",
        "timings": dict,
    },
}
```

The final label image must match the original YX shape. Implement dense tiling
inside Python for images too large for one forward pass. Padding, overlap,
blending of raw maps/embeddings, flow following, and instance reconstruction must
be scientifically valid for Cross-GOOSE. Do not independently segment tiles and
naively concatenate instance IDs. If Cross-GOOSE cannot safely reconstruct
instances from merged raw outputs, document and test the chosen overlap-aware
instance reconciliation algorithm.

`config["object_size"]` is an optional approximate object diameter in input
pixels, derived from the user's reference rectangle. Persist the model's training
object-size statistics and, when this override is present, normalize inference
scale before tiling. Reconstruct raw outputs and instances at the normalized
scale, then return maps and labels at the original image resolution. Nearest
neighbor is acceptable only for final integer labels; interpolate continuous maps
before instance reconstruction whenever the algorithm permits it.

Emit callback events with `type="inference_progress"` and phases:

- `inference_start`, including `total_patches`
- `patch_start`, including one-based `patch_index` and `total_patches`
- `patch_end`, including one-based `patch_index` and `total_patches`
- `merge_start`
- `inference_end`

Each event contains `message`, `current`, and `maximum`. Check cancellation
between patches and during flow-following loops with bounded latency.

## Training Contract

Required configuration keys from JDLL:

```text
model_name
output_dir
dataset_path
architecture = "default"
starting_point = "scratch" | "fine_tune"
base_model                 # only for fine_tune
device = "cpu" | "cuda" | "mps"
epochs
```

Use all remaining upstream default architecture and optimizer parameters unless
explicitly overridden by a documented adapter option. Save the fully resolved
configuration to the model directory.

The canonical dataset provided by JDLL is:

```text
dataset/
  train/
    images/
      sample.ext
    masks/
      sample_mask.ext
  val/
    images/
    masks/
```

Support TIFF/TIF, PNG, JPEG/JPG, and BMP images, and TIFF/TIF, PNG, and BMP label
masks. Pair by normalized stem. Masks are integer instance-label images with zero
as background. The adapter may create an internal linked/cached upstream layout
(`sample_img.*`, `sample_masks.*`) but must not duplicate large images unless a
conversion is technically required. Report every generated location.

Training returns:

```python
{
    "model_dir": str,
    "last_checkpoint": str,
    "best_checkpoint": str,
    "config_path": str,
    "history": dict,
}
```

Emit structured progress events compatible with JDLL's shared training UI:

```python
{
    "type": "progress",
    "message": str,
    "epoch": int,
    "step": int,
    "total_epochs": int,
    "total_steps": int,
    "current": int,
    "maximum": int,
    "losses": {
        "train/total_loss": float,
        "val/total_loss": float,
    },
    "metrics": {
        "val/ap_0.5": float,
        "val/ap_0.75": float,
        "val/ap_0.9": float,
    },
}
```

Emit one validation-preview event per epoch when previews are enabled:

```python
{
    "type": "preview",
    "epoch": int,
    "preview_path": str,
    "latest_preview_path": str,
}
```

Use a JSON format that the existing JDLL `TrainingValidationPreview` parser can
render, or document the minimal required Java parser extension. Include source
image path, ground-truth label path, predicted labels, and cell-probability map
where available. Save previews atomically.

Forward human-readable dataset preparation, resolved configuration, epoch start,
epoch loss/metrics, validation, checkpoint, warning, cancellation, and completion
messages. Do not forward raw tqdm bars or duplicate every batch message.

Save `last.ckpt` every epoch and preserve the best validation checkpoint based on
upstream's primary AP metric. Report when a new best model is produced. A normal
finish, user cancellation, training failure, validation failure, and environment
failure must remain distinguishable.

## Device and Lifecycle Requirements

- Resolve the requested device explicitly. Never silently use CPU; if fallback is
  allowed, emit a warning and report requested/used devices.
- Set the model and all tensors on the resolved device.
- Support CPU training, even when slow. CUDA and MPS availability must be checked
  with PyTorch.
- Release model, optimizer, dataloader, and temporary tensors after success,
  cancellation, or failure. Run Python GC and the appropriate CUDA/MPS cache
  cleanup without invoking unavailable device APIs.
- Avoid multiprocessing workers that spawn outside Appose by default. The default
  data-loader worker count must be zero and configurable.
- Avoid hard-coded `model.to("cuda")`, which exists in upstream evaluation code.

## Dataset and Scientific Validation

- Reject shape-mismatched image/mask pairs before training.
- Reject non-integer or continuous masks rather than truncating them silently.
- Use the first channel of multichannel masks, matching JDLL's current mask
  policy, and emit a warning.
- Preserve touching, crossing, and border objects as distinct instance IDs.
- Handle empty masks without crashing and document whether empty training patches
  are sampled.
- Validate fine-tuning compatibility between checkpoint/config and the default
  architecture.

## Tests

Include unit and integration tests for:

- Both supported model-directory layouts and deterministic checkpoint selection.
- CPU model loading and inference on grayscale and two-channel images.
- CUDA and MPS resolution when available, with assertions on actual tensor/model
  placement.
- Exact output shape, integer labels, finite maps, deterministic seeded behavior,
  and tiled-versus-untiled consistency.
- Callback ordering, one-based patch progress, cancellation, and callback failure
  isolation.
- Canonical JDLL dataset ingestion, generated validation split input, mixed image
  extensions, missing masks, shape mismatch, multichannel masks, and empty masks.
- Scratch training and fine-tuning smoke tests with a tiny fixture dataset.
- Last/best checkpoint creation, config persistence, reloadability, progress
  metrics, and validation preview JSON.
- Cleanup after success, cancellation, Python exception, and accelerator OOM.

Keep dependencies as close as possible to upstream Cross-GOOSE. Do not add
Appose, GUI, Java, image-server, or unrelated scientific dependencies.
