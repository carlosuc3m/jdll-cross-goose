# API options

## Model loading

`load_model(model_path, device="cpu")` returns `(model, metadata)`. Accepts:

- `config.yaml`, `config.yml`, or `config.json` beside `weights.ckpt`;
- a config beside `checkpoints/last.ckpt` and `checkpoints/best-*.ckpt`;
- an explicit `.ckpt` path with its config in the containing model directory;
- `"default"`, which fetches the pinned pretrained model to the user cache.

The config must include `model.flow_fn` and `data.image_normalization`.
State dictionaries are loaded strictly and with `weights_only=True` and CPU map
location before device placement. Arbitrary config class imports are not used.
Supported upstream flow modules are FlowLinear, FlowAttention, and CPLike.
Training is restricted to the default FlowLinear architecture.

`infer` accepts a `checkpoint` selector: `last` (default), `best` (AP@0.75), or
`best_ap_0.5`, `best_ap_0.75`, `best_ap_0.9`. Stable exact names take precedence;
otherwise selection compares epoch/step numbers for last or parsed scores for
best, with filename tie-breaking. Ambiguous unscored sets fail. An explicit
checkpoint path selects that exact file. `weights.ckpt` is the sole choice in
the simple pretrained layout. Metadata reports the chosen absolute path.

## Inference

`inputs["image"]` must be a finite, contiguous, real numeric NumPy **CYX** array.
It must contain at least one channel. Output arrays have the original **YX** shape.

| Config key | Default | Meaning |
| --- | --- | --- |
| `model_path` | required unless supplying a loaded model | Model directory, checkpoint, or `default` |
| `device` | `cpu` | `cpu`, `cuda`, or `mps` |
| `checkpoint` | `last` | Selection described above |
| `allow_device_fallback` | `false` | Warn and report CPU if the requested device is unavailable |
| `tile_size` | `512` | Multiple of 16, at least 32; `None` disables tiling |
| `tile_overlap` | up to 128; approximately one quarter of tile size, rounded to a multiple of 16 | Multiple of 16 smaller than tile; at least 16 when tiled |
| `n_steps` | saved model value (normally 200) | Positive number of upstream flow steps |
| `flow_batch_size` | `65536` | Maximum foreground trajectories per device batch |
| `cell_probability_threshold` | `0.5` | Foreground probability threshold, strictly between 0 and 1 |
| `object_size` | absent | Approximate input object diameter in pixels; requires saved training statistics |

Normalize the whole image using the saved config. A one-channel image is
duplicated as upstream does. More than two channels are reduced to the first two.
With `grayscale=true`, retained channels are averaged before duplication;
otherwise the two channels are retained. This same policy is used for training.

Events with `type="inference_progress"` occur in this order:
`inference_start`, then `patch_start`/`patch_end` per patch, `merge_start`, and
`inference_end`. Every event includes `message`, `current`, `maximum`, and
`total_patches`. Patch events include one-based `patch_index`. `current` counts
completed patches. Failure/cancellation emits a distinct terminal event and raises.

Model/backbone/flow computation runs on the chosen device. Merged grids and final
instance reconstruction live in host memory; sampled embeddings are transferred
in bounded chunks for flow computation. This avoids storing the whole image's
embedding grids on the accelerator, but host RAM still grows with image area.

## Training

Required: `model_name`, `output_dir`, `dataset_path`, `architecture="default"`,
`starting_point="scratch"|"fine_tune"`, `device`, and positive integer `epochs`.
Fine-tuning additionally requires `base_model`. Paths in configs are resolved
relative to the process working directory, including CLI YAML configs.

Architecture, loss weights, AdamW options, augmentation, and sampler defaults come
from the pinned upstream `configs/model.yaml`. Scratch training uses random
weights with that topology. Optional documented overrides:

| Key | Default | Meaning |
| --- | --- | --- |
| `checkpoint` | `best` for fine-tuning | Base checkpoint selector |
| `batch_size`, `val_batch_size` | 8, 2 | Data-loader batch sizes |
| `num_workers` | 0 | Worker processes; zero runs data loading in the calling process |
| `prefetch_factor` | 4 | Used only when workers are enabled |
| `patch_size`, `val_patch_size` | 256, 512 | Square training/random and validation/center crops; multiples of 16, at least 32 |
| `seed` | 16 | Python, NumPy, PyTorch, split, and data-loader seed |
| `deterministic` | false | Lightning deterministic mode; may reject unsupported accelerator operations |
| `n_steps` | 200, or base model value | Validation and saved inference flow steps |
| `learning_rate`, `weight_decay` | 0.005, 0.01 | AdamW parameters |
| `n_samples`, `sigma` | 200, 4.0 | Upstream RandomOnCellV2 samples per patch and perturbation |
| `bootstrap_factor` | 1 | Number of sampled patches per source image per epoch |
| `gridflow_n_interpol` | 11 | Upstream sparse-flow interpolation neighbors |
| `center_method` | `graph_center_weighted` | Upstream flow center method |
| `grayscale` | true, or base model setting | Channel averaging policy |
| `image_normalization` | `M1P1`, or base model setting | `M1P1` or `N99` |
| `image_axes`, `mask_axes` | file metadata/detection | Optional `CYX` or `YXC` axis hints for multichannel files |
| `validation_fraction` | 0.2 | Seeded holdout if a validation subset was not supplied |
| `previews`, `preview_samples` | true, 1 | One event each epoch containing up to this many full validation images |
| `tile_size`, `tile_overlap`, `flow_batch_size` | inference defaults | Tiling and flow batching for validation and previews |
| `allow_device_fallback` | false | Explicit warning plus device metadata if CPU fallback is needed |

Unknown options fail to catch misspellings. Fully resolved configuration is saved.
Fine-tuning does not resume optimizer state; it creates a new run from selected
weights. `last.ckpt` is saved after every successfully validated epoch, and best
weights follow AP@0.75. Each checkpoint includes model, optimizer, Lightning loop,
and callback state. Cancellation during an epoch preserves earlier completed
epochs; an unvalidated partial epoch is not reported as a completed checkpoint.

Progress events contain epoch, step, total_epochs, total_steps, current, maximum,
`losses` (`train/total_loss`, `val/total_loss`) and `metrics` (`val/ap_0.5`,
`val/ap_0.75`, `val/ap_0.9`). Human-readable preparation, configuration, epoch,
validation, checkpoint, warning and terminal messages use separate events.
There are no tqdm bars or per-batch callback messages.

## Dataset and cancellation

Images: TIFF/TIF, PNG, JPEG/JPG, BMP. Masks: TIFF/TIF, PNG, BMP. Pairing is
case-insensitive by stem, stripping `_img`/`_image` from image names and
`_mask`/`_masks`/`_label`/`_labels` from mask names. Duplicate normalized stems,
unpaired files, invalid pixels, negative/non-integer masks, and mismatched shapes
fail before training. Floating masks are rejected even if their values happen to
be integral. Multichannel masks use channel zero and emit a warning.

No source-image copies or links are needed for training. Sequential flow caches
and the dataset manifest list generated paths. Preview NPY copies are necessary
for JDLL's existing NPY decoder and are generated only when previews are enabled.
Empty masks/patches are sampled, with cell probability loss and zero flow loss.
Empty/empty validation predictions receive AP=1 instead of an undefined NaN.

Callbacks returning exactly `False` cancel. Callback exceptions are logged and
isolated. The `cancellation` callable is checked before loading, between image
patches, on each flow step and trajectory chunk, between flow-cache objects, and
at training/validation batch boundaries. An individual PyTorch forward/backward,
file read, upstream ground-truth object flow computation, or upstream mask
clustering call cannot be interrupted midway. This bounds flow-loop cancellation
latency by one chunk computation, not the full number of flow steps.

`train(task=task)` sends JSON events as `task.update(message=..., current=...,
maximum=...)`; all core logic also works with plain callbacks. No Appose objects
are imported or required.

Errors live in `jdll_cross_goose.runtime`: `CancelledError`, `TrainingError`,
`ValidationError`, `EnvironmentError`. Configuration/input problems use
`ValueError`/`FileNotFoundError`. Cancellation, training failure, validation
failure, environment failure and successful completion are distinct events.

Internally owned models, optimizers and loaders are released in `finally`, then
Python GC and the selected available accelerator's cache cleanup run. A model
supplied to `infer` remains caller-owned; its previous training/eval mode is
restored. Loading intentionally returns a live model, so the caller must release
it when finished. Selecting a device for a supplied model moves it to that device.
