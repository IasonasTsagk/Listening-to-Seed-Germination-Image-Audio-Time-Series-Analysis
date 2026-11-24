"""Image discovery, timestamp parsing, and metadata inventory utilities."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_TIMESTAMP = re.compile(
    r"(?P<year>20\d{2})[_-](?P<month>\d{1,2})[_-](?P<day>\d{1,2})"
    r"(?:[T_ -]+)(?P<hour>\d{1,2})[_:-](?P<minute>\d{1,2})[_:-](?P<second>\d{1,2})"
)


def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Parse common ``YYYY_MM_DD_HH_MM_SS`` filename timestamps, or return None."""
    match = _TIMESTAMP.search(Path(filename).stem)
    if not match:
        return None
    try:
        return datetime(**{key: int(value) for key, value in match.groupdict().items()})
    except ValueError:
        return None


def discover_images(image_dir: Path) -> list[Path]:
    """Recursively find experimental images, ordered by parsed time then filename.

    Notebook checkpoint directories and checkpoint-named files are editor artefacts,
    not experimental observations, so they are excluded without modifying raw data.
    """
    paths = [
        path for path in image_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and ".ipynb_checkpoints" not in path.relative_to(image_dir).parts
        and "-checkpoint" not in path.name.lower()
    ]
    return sorted(paths, key=lambda path: (parse_timestamp_from_filename(path.name) is None,
                                           parse_timestamp_from_filename(path.name) or datetime.max,
                                           path.name.lower()))


def inspect_image(path: Path) -> dict:
    """Read header-level image properties using OpenCV, recording unreadable files."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    row = {"filename": path.name, "path": str(path), "timestamp": parse_timestamp_from_filename(path.name),
           "file_size_bytes": path.stat().st_size, "is_readable": image is not None,
           "width": np.nan, "height": np.nan, "channels": np.nan}
    if image is not None:
        row["height"], row["width"] = image.shape[:2]
        row["channels"] = 1 if image.ndim == 2 else image.shape[2]
    return row


def build_inventory(image_dir: Path) -> pd.DataFrame:
    """Build a metadata table without retaining images in memory."""
    rows = [inspect_image(path) for path in discover_images(image_dir)]
    columns = ["image_id", "filename", "path", "timestamp", "width", "height", "channels",
               "file_size_bytes", "is_readable"]
    if not rows:
        return pd.DataFrame(columns=columns)
    inventory = pd.DataFrame(rows)
    inventory.insert(0, "image_id", range(len(inventory)))
    inventory["timestamp"] = pd.to_datetime(inventory["timestamp"])
    return inventory[columns]


def inventory_summary(inventory: pd.DataFrame) -> dict:
    """Return inspection metrics, including time gaps inferred from available timestamps."""
    timestamps = inventory["timestamp"].dropna().sort_values()
    intervals = timestamps.diff().dropna().dt.total_seconds()
    median_interval = float(intervals.median()) if not intervals.empty else None
    gaps = intervals[intervals > 1.5 * median_interval] if median_interval else intervals.iloc[0:0]
    return {
        "number_of_images": len(inventory), "readable_images": int(inventory["is_readable"].sum()) if len(inventory) else 0,
        "unreadable_images": int((~inventory["is_readable"]).sum()) if len(inventory) else 0,
        "missing_timestamps": int(inventory["timestamp"].isna().sum()) if len(inventory) else 0,
        "first_timestamp": timestamps.iloc[0] if not timestamps.empty else None,
        "last_timestamp": timestamps.iloc[-1] if not timestamps.empty else None,
        "median_interval_seconds": median_interval, "suspected_gap_count": len(gaps),
        "image_dimensions": inventory[["width", "height", "channels"]].drop_duplicates().to_dict("records"),
        "total_file_size_bytes": int(inventory["file_size_bytes"].sum()) if len(inventory) else 0,
    }


def save_csv(frame: pd.DataFrame, destination: Path) -> None:
    """Save numeric/metadata output; creates only derived-output directories."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
