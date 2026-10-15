# JDLL validation previews

The generated JSON matches the existing
`io.bioimage.modelrunner.gui.custom.stardist.StardistValidationPreviewPanel`
implementation of `TrainingValidationPreview`. Its `parseSamples` method reads
`samples[].image_path`, `label_path`, `prediction_path`, `prob_path`, plus
`epoch`, `n_dim`, and `initial_plane`. The panel loads arrays with `DecodeNumpy`.

```json
{
  "schema_version": 1,
  "epoch": 1,
  "n_dim": 2,
  "samples": [{
    "image_path": "/absolute/model/previews/sample-0-image.npy",
    "label_path": "/absolute/model/previews/sample-0-labels.npy",
    "source_image_path": "/absolute/dataset/val/images/sample.tif",
    "source_label_path": "/absolute/dataset/val/masks/sample_mask.tif",
    "prediction_path": "/absolute/model/previews/epoch-0001-sample-0-prediction.npy",
    "prob_path": "/absolute/model/previews/epoch-0001-sample-0-probability.npy",
    "initial_plane": 0
  }]
}
```

Paths are absolute. Images are YX or YXC, labels/probabilities are YX. Source
images/labels are saved once in NPY format; per-epoch predictions and probabilities
have immutable filenames. Every array and JSON file is installed by atomic rename
in the destination directory. `latest.json` is updated only after all referenced
files exist, so readers cannot observe partially written preview data.

One event per epoch contains `type="preview"`, `epoch`, `preview_path`, and
`latest_preview_path`. Java can route `preview_path` to the existing panel's
`loadPreview` method. No parser extension is needed when reusing that panel.
Other `TrainingValidationPreview` implementations may have different schemas;
the interface itself defines no JSON schema. Adding a Cross-GOOSE Java UI/service
is outside this Python adapter; this repo does not modify JDLL Java sources.
