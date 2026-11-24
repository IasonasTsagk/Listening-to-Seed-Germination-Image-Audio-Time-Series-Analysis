# Seed-body V1 annotation protocol

Dataset version: `seed-body-v1`. Annotate five full, independent images in CVAT; do not configure tracking or interpolation. This pilot is one fixed scene, so timestamps are not independent samples.

Use one `seed_body` polygon per visible biological seed body. Trace its visible outer seed-body/seed-coat boundary. Exclude visible radicle, root, seedling, detached pale tissue, and background artifacts. For cracked seeds retain the identifiable seed-body portion, not long protrusions. Separate touching seeds when their boundaries are visible; otherwise do not invent a boundary and set `annotation_quality=ambiguous`. Annotate border seeds if useful and set `truncated=yes`.

Attributes are required metadata: `germinating` (`yes`, `no`, `uncertain`), `annotation_quality` (`clear`, `ambiguous`), `truncated` (`yes`, `no`), and `proposal_source` (`manual`, `sam21`, `other`). Germination metadata never changes the polygon. Do not assign germination stages.

Review order: early, early-middle, middle, middle-late, late. Periodically revisit contacts, cracks, pale emergence, borders, and ambiguous bodies. SAM is a review aid only: predicted IoU/stability do not validate a mask, and every retained proposal requires human review.

Annotate full frames once. Later crops assign an instance by centroid or majority area, use a configurable split-boundary buffer, and never duplicate an object across train and test. The selected 3×2 spatial split is fixed across timestamps. It is a within-experiment spatial holdout, not external generalization to a new batch, experiment, plate, camera, or lighting condition.

After review, export archival CVAT/COCO instance annotations. Because an Ultralytics export may not preserve CVAT attributes, retain the CVAT export and a sidecar CSV/JSON containing all four attributes. Do not export YOLO labels or train until `validate_cvat_export` succeeds and annotations are human reviewed.
