"""Command-line access to model download, inference, and training."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jdll-cross-goose", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download-model", help="Download the pinned pretrained default model"
    )
    download.add_argument("--output", default="models/default")
    inference = commands.add_parser(
        "infer", help="Segment a 2D image and save TIFF labels and probability"
    )
    inference.add_argument("--model", default="default")
    inference.add_argument("--input", required=True)
    inference.add_argument("--output", required=True, help="Output labels TIFF")
    inference.add_argument("--axes", choices=["YX", "CYX", "YXC"])
    inference.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    inference.add_argument("--checkpoint", default="last")
    inference.add_argument("--tile-size", type=int, default=512)
    inference.add_argument(
        "--tile-overlap", type=int, help="Overlap in pixels; defaults to up to 128"
    )
    inference.add_argument("--object-size", type=float)
    inference.add_argument("--n-steps", type=int)
    training = commands.add_parser("train", help="Train from a JSON or YAML config")
    training.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    from .runtime import CancelledError, Events, atomic_path, write_json

    cancelled = False

    def cancel(signum, frame):
        nonlocal cancelled
        cancelled = True

    previous_handler = signal.signal(signal.SIGINT, cancel)
    events = Events(cancellation=lambda: cancelled)
    try:
        if args.command == "download-model":
            from .checkpoints import download_default

            print(download_default(args.output, events=events))
        elif args.command == "infer":
            import numpy as np
            import tifffile

            from .appose_api import infer
            from .images import read_image

            image = read_image(args.input, args.axes)
            config = {
                "model_path": args.model,
                "device": args.device,
                "checkpoint": args.checkpoint,
                "tile_size": args.tile_size,
            }
            for key in ("object_size", "n_steps", "tile_overlap"):
                if getattr(args, key) is not None:
                    config[key] = getattr(args, key)
            output = Path(args.output).expanduser().resolve()
            if output.suffix.lower() not in {".tif", ".tiff"}:
                parser.error("--output must have a .tif or .tiff extension")
            probability = output.with_name(output.stem + "-probability.tif")
            metadata = output.with_suffix(".json")
            for path in (output, probability, metadata):
                if path.exists():
                    raise FileExistsError(f"Output already exists: {path}")
            result = infer(config, {"image": image}, cancellation=lambda: cancelled)
            for path, key, dtype in (
                (output, "labels", np.uint32),
                (probability, "cell_probability", np.float32),
            ):
                with atomic_path(path) as temporary:
                    tifffile.imwrite(
                        temporary,
                        result["outputs"][key].astype(dtype),
                        photometric="minisblack",
                        metadata={"axes": "YX"},
                    )
            write_json(metadata, result["metadata"])
            print(
                json.dumps(
                    {
                        "labels": str(output),
                        "cell_probability": str(probability),
                        "metadata": str(metadata),
                    }
                )
            )
        else:
            import yaml

            from .appose_api import train

            with open(args.config, encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
            result = train(config, cancellation=lambda: cancelled)
            print(json.dumps(result, allow_nan=False))
        return 0
    except CancelledError:
        logging.info("Cancelled")
        return 130
    except Exception as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous_handler)
