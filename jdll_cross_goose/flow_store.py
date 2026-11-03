"""Per-object upstream flows: compute from complete objects, read only requested IDs."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from threading import RLock

import numpy as np
from crossgoose.gridflow import GridFlow
from scipy.spatial import KDTree

from .runtime import atomic_path, write_json


class FlowStore:
    def __init__(self, directory, max_bytes, events):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.events = events
        self.cache = OrderedDict()
        self.manifests = {}
        self.bytes = 0
        self.lock = RLock()

    def prepare(self, labels, analysis, center_method):
        digest = hashlib.sha256((str(labels.shape) + center_method + ":v2").encode())
        rows = max(1, 1024**2 // (8 * labels.shape[-1]))
        for start in range(0, labels.shape[0], rows):
            self.events.check()
            chunk = np.asarray(labels[start : start + rows, :], dtype=np.int64)
            digest.update(np.ascontiguousarray(chunk).tobytes())
        directory = self.directory / digest.hexdigest()
        directory.mkdir(exist_ok=True)
        manifest = directory / "manifest.json"
        if manifest.is_file():
            return str(manifest)
        objects = {}
        for index, (label, region) in enumerate(analysis.objects.items(), 1):
            self.events.check()
            filename = directory / f"{label}.npz"
            bounds = tuple(slice(a, b) for a, b in region.bounds)
            # Upstream immediately crops each object to this same bounding box.
            # Passing the complete object crop avoids scanning a full image per ID.
            if not filename.is_file():
                flow = GridFlow.from_labels(
                    (labels[bounds] == label).astype(np.int32),
                    n_interpol=1,
                    flow_center_method=center_method,
                )
                points = flow.points[1].data + np.array([a for a, _ in region.bounds])
                with atomic_path(filename) as temporary:
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, points=points, flows=flow.flows[1])
            objects[str(label)] = filename.name
            if index % 100 == 0:
                self.events.message(
                    f"Prepared object flows {index}/{len(analysis.objects)}",
                    kind="dataset",
                    current=index,
                    maximum=len(analysis.objects),
                )
        write_json(
            manifest, {"objects": objects, "shape": labels.shape, "center_method": center_method}
        )
        return str(manifest)

    def get(self, manifest, labels, n_interpol):
        import json

        manifest = Path(manifest)
        with self.lock:
            key = str(manifest)
            if key not in self.manifests:
                self.manifests[key] = json.loads(manifest.read_text())
            content = self.manifests[key]
            points, flows = {}, {}
            for label in labels:
                self.events.check()
                path = manifest.parent / content["objects"][str(int(label))]
                entry = self.cache.pop(path, None)
                if entry is None:
                    with np.load(path, allow_pickle=False) as data:
                        entry = KDTree(data["points"]), data["flows"]
                    size = self.entry_bytes(entry)
                    if size <= self.max_bytes:
                        while self.cache and self.bytes + size > self.max_bytes:
                            _, evicted = self.cache.popitem(last=False)
                            self.bytes -= self.entry_bytes(evicted)
                        self.cache[path] = entry
                        self.bytes += size
                else:
                    self.cache[path] = entry
                points[int(label)], flows[int(label)] = entry
        # Reuse the exact upstream KD-trees rather than rebuilding them per crop.
        result = GridFlow({}, {}, n_interpol)
        result.points, result.flows = points, flows
        return result

    @staticmethod
    def entry_bytes(entry):
        tree, vectors = entry
        return tree.data.nbytes + tree.indices.nbytes + tree.size * 128 + vectors.nbytes

    def close(self):
        with self.lock:
            self.cache.clear()
            self.manifests.clear()
            self.bytes = 0
