# BBBC010 inference fixture

`bbbc010-w2.tif` is the original, unscaled GFP (w2) image for well A01 from
[BBBC010 version 2 images](https://bbbc.broadinstitute.org/BBBC010), provided by
Fred Ausubel's lab at MGH through the Broad Bioimage Benchmark Collection.
The source filename is recorded in `source.txt`.

Downloaded from
`https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip`.
The image is losslessly recompressed as a TIFF; all 520 x 696 uint16 pixels are
unchanged. BBBC010's dataset page states that the image set is **CC0**.
Reference: Ljosa et al., *Nature Methods* (2012), as requested by the dataset page.

This fixture exercises nonempty pretrained instance reconstruction and comparison
with upstream. It is not an independent held-out accuracy benchmark: upstream
trained on images from this collection, and its exact training split is not
recorded in the pretrained config.
