"""Phase 2.8A helpers for CVAT seed-body annotation preparation.

These utilities prepare working copies and split *plans*.  They never create
ground-truth instances, change raw images, or generate training labels.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from .preprocessing import SegmentationConfig, create_foreground_mask

ANNOTATION_VERSION = "seed-body-v1"
LABEL = "seed_body"
STAGES = ["early", "early-middle", "middle", "middle-late", "late"]
TIMESTAMPS = [
    "2026-07-14 12:01:32", "2026-07-14 18:11:11", "2026-07-15 00:04:10",
    "2026-07-15 06:13:59", "2026-07-15 12:06:59",
]
ATTRIBUTE_VALUES = {
    "germinating": ["yes", "no", "uncertain"],
    "annotation_quality": ["clear", "ambiguous"],
    "truncated": ["yes", "no"],
    "proposal_source": ["manual", "sam21", "other"],
}


def load_inventory(path: Path) -> pd.DataFrame:
    inventory = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"image_id", "timestamp", "filename", "path", "width", "height", "is_readable"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Inventory lacks required columns: {sorted(missing)}")
    return inventory


def select_annotation_frames(inventory: pd.DataFrame) -> pd.DataFrame:
    """Resolve the five approved times through the inventory, never filenames."""
    frames = inventory.copy()
    frames["timestamp"] = pd.to_datetime(frames["timestamp"], errors="coerce")
    requested = pd.DataFrame({"stage_label": STAGES, "timestamp": pd.to_datetime(TIMESTAMPS)})
    selected = requested.merge(frames, on="timestamp", how="left", validate="one_to_one")
    if selected["filename"].isna().any():
        missing = selected.loc[selected["filename"].isna(), "timestamp"].astype(str).tolist()
        raise ValueError(f"Required annotation timestamp(s) absent from inventory: {missing}")
    selected["chronological_index"] = selected["timestamp"].map(
        {stamp: index for index, stamp in enumerate(sorted(frames["timestamp"].dropna().unique()))}
    )
    return selected.sort_values("chronological_index").reset_index(drop=True)


def verify_annotation_frames(frames: pd.DataFrame) -> pd.DataFrame:
    if len(frames) != 5 or frames["timestamp"].duplicated().any():
        raise ValueError("Exactly five distinct annotation frames are required.")
    for record in frames.itertuples(index=False):
        if not bool(record.is_readable):
            raise ValueError(f"Required annotation image is unreadable: {record.filename}")
        source = Path(record.path)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Required annotation image cannot be opened: {source}")
        height, width = image.shape[:2]
        if int(record.width) != width or int(record.height) != height:
            raise ValueError(f"Inventory dimensions do not match image: {record.filename}")
    return frames


def prepare_cvat_upload_folder(frames: pd.DataFrame, upload_dir: Path) -> pd.DataFrame:
    """Copy only working copies; source raw images are never moved or changed."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    output = frames.copy()
    destinations = []
    for record in output.itertuples(index=False):
        source, destination = Path(record.path), upload_dir / record.filename
        shutil.copy2(source, destination)
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(f"Copy verification failed for {record.filename}")
        destinations.append(str(destination))
    output["cvat_upload_path"] = destinations
    return output


def build_cvat_task_spec() -> dict:
    return {
        "task_name": "seed-germination-seed-body-v1",
        "dataset_version": ANNOTATION_VERSION,
        "task_type": "image annotation",
        "tracking_or_interpolation": False,
        "labels": [{"name": LABEL, "shape": "polygon", "attributes": [
            {"name": name, "input_type": "select", "values": values}
            for name, values in ATTRIBUTE_VALUES.items()
        ]}],
        "ground_truth_policy": "Human-reviewed annotations only. SAM masks are optional proposals, never ground truth.",
    }


def calculate_tile_geometry(width: int, height: int, columns: int, rows: int) -> pd.DataFrame:
    if width <= 0 or height <= 0 or columns <= 0 or rows <= 0:
        raise ValueError("Frame dimensions and grid dimensions must be positive.")
    x_edges = np.rint(np.linspace(0, width, columns + 1)).astype(int)
    y_edges = np.rint(np.linspace(0, height, rows + 1)).astype(int)
    rows_out = []
    for row in range(rows):
        for column in range(columns):
            x0, x1, y0, y1 = x_edges[column], x_edges[column + 1], y_edges[row], y_edges[row + 1]
            rows_out.append({"tile_id": f"r{row:02d}_c{column:02d}", "row": row, "column": column,
                             "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1,
                             "width": x1 - x0, "height": y1 - y0})
    return pd.DataFrame(rows_out)


def generate_grid_layout(width: int, height: int, columns: int, rows: int) -> pd.DataFrame:
    return calculate_tile_geometry(width, height, columns, rows)


def calculate_otsu_tile_coverage(image_paths: Iterable[str], tiles: pd.DataFrame) -> pd.Series:
    """Average Otsu foreground coverage over images; components are not seed counts."""
    totals = np.zeros(len(tiles), dtype=float)
    count = 0
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image for coverage diagnostic: {path}")
        _, mask = create_foreground_mask(image, SegmentationConfig(method="otsu", dark_foreground=True))
        for index, tile in enumerate(tiles.itertuples(index=False)):
            totals[index] += float((mask[tile.y_min:tile.y_max, tile.x_min:tile.x_max] > 0).mean())
        count += 1
    return pd.Series(totals / count if count else totals, name="approximate_otsu_coverage")


def propose_spatial_split(tiles: pd.DataFrame, columns: int, rows: int) -> pd.DataFrame:
    """Deterministic spatial allocation: 3x2 gives 4/1/1 tiles (67/17/17%)."""
    out = tiles.copy()
    if (columns, rows) == (3, 2):
        mapping = {"r00_c00": "train", "r00_c01": "train", "r01_c00": "train", "r01_c01": "train",
                   "r00_c02": "validation", "r01_c02": "test"}
    elif (columns, rows) == (4, 2):
        mapping = {"r00_c00": "train", "r00_c01": "train", "r00_c02": "train", "r01_c00": "train",
                   "r01_c01": "train", "r00_c03": "validation", "r01_c02": "test", "r01_c03": "test"}
    else:
        raise ValueError("Only the evaluated 3x2 and 4x2 layouts are supported.")
    out["proposed_split"] = out["tile_id"].map(mapping)
    return out


def validate_spatial_split(tiles: pd.DataFrame, width: int, height: int) -> None:
    required = {"tile_id", "x_min", "y_min", "x_max", "y_max", "proposed_split"}
    if required - set(tiles.columns) or tiles["tile_id"].duplicated().any() or tiles["proposed_split"].isna().any():
        raise ValueError("Spatial split manifest is incomplete or non-deterministic.")
    canvas = np.zeros((height, width), dtype=np.uint8)
    for tile in tiles.itertuples(index=False):
        canvas[tile.y_min:tile.y_max, tile.x_min:tile.x_max] += 1
    if canvas.min() != 1 or canvas.max() != 1:
        raise ValueError("Tiles must cover each frame pixel exactly once without overlap.")
    if set(tiles["proposed_split"]) != {"train", "validation", "test"}:
        raise ValueError("Train, validation, and test regions are all required.")


def export_annotation_manifest(frames: pd.DataFrame, destination: Path) -> pd.DataFrame:
    manifest = frames[["image_id", "timestamp", "filename", "stage_label"]].copy()
    manifest["annotation_status"] = "not_started"
    manifest["number_of_reviewed_instances"] = 0
    manifest["notes"] = "Full-frame seed_body annotation required; SAM proposals are not ground truth."
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination, index=False)
    return manifest


def dataset_version_record(frames: pd.DataFrame) -> dict:
    return {"dataset_version": ANNOTATION_VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_version": "annotation_protocol_v1", "split_strategy_version": "spatial-fixed-grid-v1",
            "label_schema": {"label": LABEL, "attributes": ATTRIBUTE_VALUES},
            "source_images": frames[["timestamp", "filename", "source_path"]].to_dict("records")}

