# Scientific behavior and upstream adaptations

Upstream commit: `4e7ffa3c71ad0c5d59b364dd3bf0e5787c8bd79f`.

## Reused methods

The adapter imports Cross-GOOSE's CPBackbone, batchconv embedding head,
FlowLinear/FlowAttention/CPLike, `compute_loss`, RandomOnCellV2,
`random_transform`, GridFlow ground-truth generation/interpolation,
`average_precision`, and `get_masks_torch`. These supply the model and scientific
algorithm. The training optimizer matches upstream's CLI config: AdamW at 0.005,
not the model class's fallback Adam at 0.001.

The small AdapterModel constructor reproduces the upstream module hierarchy and
state-dictionary names while calling `CPBackbone(load_pretrained=False)`.
Complete Cross-GOOSE checkpoints already contain the backbone. The upstream
constructor otherwise downloads cyto3 unnecessarily. For scratch training this
also provides genuinely random initial weights, using the current default shared
embedding topology. Legacy pretrained separate-head models retain their topology.

The shared JDLL-UNet data layer replaces eager process-pool loading. It reads
patches first, keeps bounded raw/flow caches, prepares sparse flows sequentially
from complete object bounding boxes, and does not spawn DataLoader workers.
RandomOnCellV2's calculation is retained with an explicit per-patch Torch RNG
to avoid interference between prefetch and model randomness. Tests verify exact
same-seed sampler outputs and exact sparse-flow equality against upstream.
No monkey-patches or changes to installed upstream files are made.
Normalization uses upstream M1P1/N99 with a finite constant-image guard:
constant M1P1 images become -1, and constant N99 images become 0.
For large images N99 percentiles use a logged, deterministic bounded sample.

Shared nnU-Net-inspired augmentation, scale normalization, annotation cleanup,
sampling and runtime optimizations are adapter policies, not claims about the
paper's training schedule. Geometric transforms carry a source-coordinate field;
its inverse Jacobian transforms sparse flow directions while preserving the
original interpolated magnitudes. Photometric operations never alter coordinates
or flow targets. A cropped object keeps its complete source-domain flow center.
Synthetic padding is excluded from cell-probability loss and AP evaluation.

## Tiling

1. Reduce channels and normalize the whole image using its saved config. If
   requested, bilinearly resample image intensities to the training object scale.
2. Apply upstream's symmetric padding (`get_pad_yx`, divisor 16, extra 1, value
   -1). Patch sizes/origins align to the backbone's pooling stride.
3. Compute raw embedding-head outputs per patch, before sigmoid. Accumulate each
   channel in a global host-memory raster using separable sine-squared weights
   with a positive floor, then divide by accumulated weights.
4. Apply sigmoid to the merged cell logits and select foreground pixels.
5. For each pixel retain its initial embedding e0. At each step gather et at the
   current integer coordinate and apply upstream `flow_fn(e0, et) / flow_fac`.
   Clamp coordinates to the padded image bounds. Follow flows across the full
   merged raster, never within isolated tile coordinates.
6. Reconstruct instances once with upstream `get_masks_torch` using all foreground
   endpoints. Remove padding. Resample continuous probabilities bilinearly and
   final integer labels by nearest-neighbor indexing if object scale was changed.

All patches use the same learned embedding channel basis; there are no tile-local
embedding permutations or positional-coordinate frames to reconcile. Blending
raw maps before the nonlinear flow function provides a continuous transition of
the input features used by the flow model. It does not concatenate instance IDs
or average already-integrated trajectories.

This is an approximation to full-image inference: receptive-field context and
the CPBackbone style vector differ per patch. Increasing tile size and overlap
can reduce discrepancies; exact equality for arbitrary trained models is not
claimed. The tests verify untiled parity with upstream, exact tile/full parity
for analytic maps, and instance continuity for objects crossing seams/borders.
These checks establish implementation consistency, not a benchmark of segmentation
quality on a user's microscopy data.

On the included 520 x 696 BBBC010 image with upstream default weights, CPU
untiled labels match upstream exactly (18 instances). Four 512-pixel tiles with
128-pixel overlap produced 18 instances, foreground IoU 0.987 against untiled
inference, and mean absolute probability difference 0.00045 in the development
environment (PyTorch 2.9.1). The optional pretrained regression test requires
foreground IoU >0.95 and probability difference <0.01 for this fixture.
These compare inference paths, not predictions against ground-truth annotations.

The full raw raster, mask histogram, and labels still require host RAM. Tiling
bounds network activation memory on the selected accelerator, and trajectory
batching bounds flow-function device memory. Very large images can still exhaust
host memory. Upstream reconstruction removes predicted objects occupying more
than 40% of the padded image and requires endpoint histogram peaks >10; these
upstream settings are retained.

## Scale and instance labels

Training estimates per-source equivalent-area diameters from prepared training
masks, with border-aware sampling and an explicit fallback. It records original
and nominally scale-normalized per-source medians. Given an
input diameter D, inference uses `training_median / D` as its isotropic scale.
It reconstructs continuous maps and instances at that normalized resolution,
then restores the original shape. The original pretrained config supplies no
training-diameter statistics: the adapter rejects an object-size override for
such models. It does not infer those statistics from Cellpose's backbone diameter.

Shared annotation preparation can repair disconnected components sharing an ID
and filter tiny artifacts; both policies are configurable and reported.
Touching different labels remain separate. Patch IDs are compacted before
interpolation to avoid float32 corruption of large original IDs. Original source
files remain intact. Upstream flow generation itself is unchanged. A single
YX label raster cannot encode two simultaneous labels at a crossing pixel. The
adapter follows JDLL's raster contract and does not claim multilabel overlap output.

## Devices

Inference bypasses upstream's CUDA-only evaluation script. Backbone and learned
flow computations run on the explicitly resolved CPU/CUDA/MPS device; caching,
sampling, global map storage, and instance reconstruction use CPU arrays/tensors.
This avoids MPS sparse-tensor and float64 operations in upstream postprocessing.
Hardware tests verify actual model/input placement when the backend is available.
