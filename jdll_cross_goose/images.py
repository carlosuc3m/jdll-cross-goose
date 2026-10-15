"""Image axes, mask validation, and upstream-compatible normalization."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile


def finite_image(image):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or min(image.shape) < 1:
        raise ValueError("image must be a nonempty NumPy array in CYX order")
    if image.dtype.kind not in "uif" or not np.isfinite(image).all():
        raise ValueError("image must contain finite real numeric values")
    return image


def channels(image, grayscale):
    image = finite_image(image).astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Image values cannot be represented as finite float32")
    image = image[:2]
    if grayscale and image.shape[0] > 1:
        image = image.mean(axis=0, keepdims=True)
    if image.shape[0] == 1:
        image = np.repeat(image, 2, axis=0)
    return np.ascontiguousarray(image)


def normalize(image, method):
    from crossgoose.utils import normalize_image

    if method not in {"M1P1", "N99"}:
        raise ValueError("image_normalization must be M1P1 or N99")
    if float(image.max()) == float(image.min()):
        return np.full_like(image, -1.0 if method == "M1P1" else 0.0, dtype=np.float32)
    result = normalize_image(image, method).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Image normalization produced non-finite values")
    return np.ascontiguousarray(result)


def read_array(path):
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            return tif.asarray(), tif.series[0].axes
    return iio.imread(path), None


def to_cyx(array, axes=None):
    if array.ndim == 2:
        return array[None]
    if array.ndim != 3:
        raise ValueError(f"Only 2D images with optional channels are supported; got {array.shape}")
    if axes is not None:
        axes = axes.upper().replace("S", "C")
        if axes == "CYX":
            return array
        if axes == "YXC":
            return array.transpose(2, 0, 1)
        if any(a in axes for a in "ZT"):
            raise ValueError(f"Volume/time axes {axes} are outside the 2D scope")
    if array.shape[-1] <= 4:
        return array.transpose(2, 0, 1)
    if array.shape[0] <= 4:
        return array
    raise ValueError(f"Ambiguous channel axis for {array.shape}; specify image_axes or mask_axes")


def read_image(path, axes=None):
    image, stored_axes = read_array(path)
    return np.ascontiguousarray(finite_image(to_cyx(image, axes or stored_axes)))


def read_mask(path, axes=None, events=None):
    mask, stored_axes = read_array(path)
    if mask.ndim == 3:
        mask = to_cyx(mask, axes or stored_axes)
        if mask.shape[0] > 1 and events:
            events.message(f"Using first channel of multichannel mask: {path}", kind="warning")
        mask = mask[0]
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError(f"Mask must be a nonempty YX instance-label array: {path}")
    if mask.dtype.kind not in "uib" or (mask < 0).any():
        raise ValueError(f"Mask must use a nonnegative integer dtype, zero background: {path}")
    if mask.max() > np.iinfo(np.int64).max:
        raise ValueError(f"Mask labels exceed signed 64-bit integer range: {path}")
    return np.ascontiguousarray(mask, dtype=np.int64)
