"""Validation utilities for future human-reviewed CVAT/COCO exports only."""
from __future__ import annotations

import json
from pathlib import Path
import math
import pandas as pd

from .annotation_preparation import ATTRIBUTE_VALUES, LABEL


def _valid_polygon(segmentation, width: int, height: int) -> bool:
    if not isinstance(segmentation, list) or not segmentation:
        return False
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            return False
        for x, y in zip(polygon[::2], polygon[1::2]):
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= width and 0 <= y <= height):
                return False
    return True


def validate_cvat_export(export_path: Path, expected_filenames: list[str] | None = None) -> dict:
    """Validate a future COCO-style CVAT export without writing training labels."""
    payload = json.loads(Path(export_path).read_text())
    images, annotations, categories = payload.get("images", []), payload.get("annotations", []), payload.get("categories", [])
    category_map = {item.get("id"): item.get("name") for item in categories}
    if set(category_map.values()) - {LABEL}:
        raise ValueError("Export contains an unsupported label.")
    files = [item.get("file_name") for item in images]
    if expected_filenames and set(files) != set(expected_filenames):
        raise ValueError("Export image filenames do not match the annotation manifest.")
    image_map = {item["id"]: item for item in images}
    ids = [item.get("id") for item in annotations]
    if len(ids) != len(set(ids)):
        raise ValueError("Annotation IDs must be unique.")
    for item in annotations:
        image = image_map.get(item.get("image_id"))
        if image is None or category_map.get(item.get("category_id")) != LABEL:
            raise ValueError("Annotation has an invalid image or label.")
        if not _valid_polygon(item.get("segmentation"), int(image["width"]), int(image["height"])):
            raise ValueError("Annotation contains an empty, non-finite, or out-of-bounds polygon.")
        for name, values in ATTRIBUTE_VALUES.items():
            if name in item.get("attributes", {}) and item["attributes"][name] not in values:
                raise ValueError(f"Invalid attribute value for {name}.")
    return {"image_count": len(images), "annotation_count": len(annotations), "valid": True}


def convert_cvat_or_coco_to_internal_annotations(export_path: Path) -> pd.DataFrame:
    """Return reviewed COCO polygons as an internal table; never creates YOLO labels."""
    validate_cvat_export(export_path)
    payload = json.loads(Path(export_path).read_text())
    image_map = {item["id"]: item for item in payload["images"]}
    return pd.DataFrame([{"annotation_id": item["id"], "filename": image_map[item["image_id"]]["file_name"],
                          "category": LABEL, "segmentation": json.dumps(item["segmentation"]),
                          "attributes": json.dumps(item.get("attributes", {}))} for item in payload["annotations"]])
