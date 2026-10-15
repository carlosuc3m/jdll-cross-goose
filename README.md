# jdll-cross-goose

Run [Cross-GOOSE](https://github.com/bia-pasteur/cross-goose) instance segmentation
and training from Python, the command line, or a JDLL/Appose task. Appose itself is
not a dependency. This adapter imports a pinned upstream package; no separate
fork or modified upstream installation is required.

## Install

Use Python **3.11 or newer** (3.12 recommended) and Git. Upstream currently uses
`typing.Self` and NumPy 2 APIs despite declaring older minimum versions.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For a CPU-only installation, install matching CPU builds of `torch` and
`torchvision` first, then install this package:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

For CUDA or MPS, use a compatible PyTorch installation for your platform. Device
selection is explicit; unavailable devices fail unless you request a reported
CPU fallback. Model loading and scratch training do not download Cellpose weights.

## Run the pretrained model

```bash
jdll-cross-goose download-model --output models/default
jdll-cross-goose infer --model models/default --input /path/to/image.tif \
  --output /path/to/labels.tif --device cpu
```

The download is approximately 80 MB and is verified against the pinned upstream
SHA-256. Inference writes integer labels, a `labels-probability.tif` map, and
`labels.json` metadata. Existing output files are protected against overwriting.
Use `--device cuda` or `--device mps` where supported. All commands also work as
`python -m jdll_cross_goose ...`.

To try the included real microscopy fixture immediately:

```bash
jdll-cross-goose infer --model models/default --input tests/fixtures/bbbc010-w2.tif \
  --output runs/example-labels.tif --device cpu
```

The CLI accepts grayscale and multichannel TIFF, PNG, JPEG, and BMP images.
For ambiguous image layouts, supply `--axes CYX` or `--axes YXC`.
The pretrained model is upstream's original default model; its usefulness on
your images depends on their similarity to its training data.

```python
import numpy as np
from jdll_cross_goose.appose_api import load_model, infer

model, metadata = load_model("models/default", device="cpu")
image = np.ascontiguousarray(my_image[None, ...])  # grayscale YX -> CYX
result = infer(
    {"device": "cpu", "tile_size": 512, "tile_overlap": 128},
    {"image": image},
    model=model,
    callback=lambda event: print(event),
)
labels = result["outputs"]["labels"]  # YX integer instance IDs, zero background
probability = result["outputs"]["cell_probability"]
```

Alternatively pass `{"model_path": "models/default"}` without a preloaded model.
`model_path="default"` downloads the pinned model to your user cache on first use.
An internally loaded model is released after inference. A model returned by
`load_model` belongs to the caller and can be reused.

## Train

Prepare a dataset with integer instance IDs (zero background):

```text
dataset/
  train/images/sample.tif
  train/masks/sample_mask.tif
  val/images/validation.tif
  val/masks/validation_mask.tif
```

Each subset may contain many image/mask pairs. Mixed image extensions are
supported. Without a `val` subset, a seeded 20% validation split is generated
from at least two training pairs. Source images and masks remain in place;
flow caches and a manifest are written under the new model directory.

Edit [examples/train.yaml](examples/train.yaml), then:

```bash
jdll-cross-goose train --config examples/train.yaml
```

```python
from jdll_cross_goose.appose_api import train

result = train({
    "model_name": "my-model",
    "output_dir": "runs",
    "dataset_path": "/path/to/dataset",
    "architecture": "default",
    "starting_point": "scratch",
    "device": "cpu",
    "epochs": 10,
}, callback=lambda event: print(event))
```

For fine-tuning, set `starting_point="fine_tune"`,
`base_model="models/default"` (or an adapter-trained model), and a new `model_name`.
Fine-tuning starts a new optimizer/run and leaves the source model untouched.
Both original separate embeddings and the current default shared embeddings are
supported for fine-tuning the default FlowLinear architecture.

Outputs include:

```text
runs/my-model/
  config.yaml
  checkpoints/last.ckpt
  checkpoints/best.ckpt
  checkpoints/best-epoch=...-step=...-val_ap_0.75=....ckpt
  dataset-cache/*.npz
  dataset.json
  history.json
  previews/latest.json
  previews/epoch-0001.json
  previews/*.npy
  result.json
```

Checkpoints contain full Lightning state and reload through `load_model`.
`best.ckpt` follows validation AP at IoU 0.75, the first monitor in upstream's
default config. The return value also reports the exact metric-bearing best
checkpoint. Existing nonempty model directories are never reused.

## Integration and scientific behavior

- [API options and lifecycle](docs/options.md)
- [Tiling, normalization, scale, and upstream adaptations](docs/scientific-notes.md)
- [JDLL validation preview format](docs/previews.md)

Only 2D segmentation is supported. Tiling blends raw embedding/logit maps before
global flow following and upstream instance reconstruction. Whole-image and
tiled neural predictions can differ because of context and the backbone's style
vector; use a larger tile/overlap when boundary artifacts matter.

The adapter records equivalent-area object diameters when training. The optional
`object_size` inference override scales an image relative to that training median.
Original upstream checkpoints lack these statistics, so that override is rejected
clearly for them instead of guessing a training diameter.

Callbacks return `False` to cancel, or supply `cancellation=lambda: ...`.
Cancellation raises `CancelledError`; completed epoch checkpoints remain available.
The CLI handles Ctrl+C through the same cancellation mechanism. See the options
document for error types and cancellation boundaries.

## Test

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=2 python -m pytest -q
python -m ruff check jdll_cross_goose tests
python -m build
```

Tests exercise the real upstream network, scratch training and fine-tuning on a
tiny generated dataset, plus analytic flow fixtures that test objects crossing
tile boundaries. Accelerator tests skip unavailable hardware. Optional pretrained
parity tests use `CROSS_GOOSE_PRETRAINED=/path/to/models/default`.

Upstream is pinned to commit
`4e7ffa3c71ad0c5d59b364dd3bf0e5787c8bd79f` (package version 1.3.5).
The small constructor adaptation and copied default configs retain upstream's
[MIT notice](THIRD_PARTY_LICENSES.txt). Cite the Cross-GOOSE authors' paper when
using the method in scientific work; citation details are in the upstream README.
