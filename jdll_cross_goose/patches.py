"""Lazy normalized patches and coordinate-correct augmentation of sparse flows."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import map_coordinates

from .images import channels


class MaskView:
    def __init__(self, source, labels):
        self.source = source
        self.labels = np.asarray((0, *labels), dtype=np.int64)
        self.shape = source.shape
        self.ndim = 2

    def __getitem__(self, selection):
        raw = self.source[selection]
        return np.searchsorted(self.labels, raw).astype(np.int64)


class ImageView:
    """Extra coordinate channels undergo exactly the same geometric transforms."""

    def __init__(self, source, statistics, grayscale):
        self.source, self.statistics, self.grayscale = source, statistics, grayscale
        self.shape = (4, *source.shape[-2:])
        self.ndim = 3

    def __getitem__(self, selection):
        channel, ys, xs = selection
        ys = slice(*ys.indices(self.shape[1]))
        xs = slice(*xs.indices(self.shape[2]))
        image = channels(self.source[(slice(None), ys, xs)], self.grayscale)
        low, high = self.statistics["low"], self.statistics["high"]
        if high - low <= self.statistics["epsilon"]:
            image.fill(-1 if self.statistics["method"] == "M1P1" else 0)
        elif self.statistics["method"] == "M1P1":
            image = ((image - np.float32(low)) / np.float32(high - low)) * 2 - 1
        else:
            image -= np.float32(low)
            image /= np.float32(high - low)
        y, x = np.meshgrid(
            np.arange(ys.start, ys.stop, dtype=np.float32),
            np.arange(xs.start, xs.stop, dtype=np.float32),
            indexing="ij",
        )
        return np.concatenate((image, y[None], x[None]))[channel]


class WarpedFlow:
    def __init__(self, flow, coordinates, label_mapping):
        self.flow, self.coordinates, self.mapping = flow, coordinates, label_mapping

    def query_multiple_labels_threaded(self, pos, labels):
        position = np.asarray(pos, dtype=np.float64).T
        original = np.stack(
            [map_coordinates(axis, position, order=1, mode="nearest") for axis in self.coordinates],
            axis=1,
        )
        source_labels = np.array([self.mapping[int(label)] for label in labels])
        vectors = self.flow.query_multiple_labels(original, source_labels)
        # Coordinates map augmented -> source. Its inverse Jacobian maps source
        # flow vectors back to augmented space, including reflections/elasticity.
        columns = []
        for axis in range(2):
            before, after = position.copy(), position.copy()
            before[axis] = np.maximum(0, before[axis] - 0.5)
            after[axis] = np.minimum(self.coordinates.shape[axis + 1] - 1, after[axis] + 0.5)
            denominator = np.maximum(after[axis] - before[axis], 1e-6)
            columns.append(
                np.stack(
                    [
                        (
                            map_coordinates(values, after, order=1, mode="nearest")
                            - map_coordinates(values, before, order=1, mode="nearest")
                        )
                        / denominator
                        for values in self.coordinates
                    ],
                    axis=1,
                )
            )
        jacobian = np.stack(columns, axis=-1)
        determinant = np.linalg.det(jacobian)
        result = np.zeros_like(vectors)
        valid = np.abs(determinant) > 1e-6
        self.valid_queries = valid
        result[valid] = np.linalg.solve(jacobian[valid], vectors[valid, :, None])[..., 0]
        norm = np.linalg.norm(result, axis=1, keepdims=True)
        # Preserve the upstream interpolation magnitude; only change direction.
        return result * np.linalg.norm(vectors, axis=1, keepdims=True) / np.maximum(norm, 1e-12)


def sample_points(image, labels, flow, n_samples, sigma, generator, valid):
    """Upstream RandomOnCellV2 with an explicit RNG for safe threaded prefetch.

    The modulo sampling and weighting deliberately match the MIT upstream
    implementation (Julie Mabon); no new flow objective or sampler is introduced.
    """
    height, width = labels.shape
    initial = torch.nonzero(labels)
    if len(initial) == 0:
        return torch.zeros((0, 2, 2)), torch.zeros((0, 2, 2)), torch.zeros((0, 2))
    initial = initial[torch.randint(len(initial), (n_samples,), generator=generator)]
    ids = labels[initial[:, 0], initial[:, 1]]
    points = {int(k): torch.nonzero(labels == k) for k in torch.unique(ids)}
    maximum = max(len(value) for value in points.values())
    choices = torch.randint(maximum, (n_samples,), generator=generator)
    target = torch.stack(
        [points[int(k)][choices[i] % len(points[int(k)])] for i, k in enumerate(ids)]
    ).float()
    target = torch.clamp(
        target + sigma * torch.randn(target.shape, generator=generator),
        min=torch.tensor([0, 0]),
        max=torch.tensor([height, width]) - 1,
    )
    vectors = torch.from_numpy(
        flow.query_multiple_labels_threaded(target.numpy(), ids.numpy())
    ).float()
    coordinates = torch.stack([initial, target], dim=1)
    vectors = torch.stack([torch.zeros_like(vectors), vectors], dim=1)
    weights = torch.ones((n_samples, 2))
    weights[:, 0] = 0
    index = target.long()
    weights[:, 1] *= torch.from_numpy(valid[index[:, 0], index[:, 1]].copy())
    if hasattr(flow, "valid_queries"):
        weights[:, 1] *= torch.from_numpy(flow.valid_queries.copy())
    weights /= weights.sum().clamp_min(1)
    return coordinates, vectors, weights
