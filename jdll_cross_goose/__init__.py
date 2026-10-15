"""Cross-GOOSE integration without an Appose dependency."""

__version__ = "0.1.0"


def __getattr__(name):
    if name in {"load_model", "infer", "train"}:
        from . import appose_api

        return getattr(appose_api, name)
    raise AttributeError(name)
