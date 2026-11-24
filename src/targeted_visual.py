"""Human-led target-seed review of the existing 25-frame image pilot.

No seed is selected, tracked, or assigned a biological state by this module.
"""
from __future__ import annotations

from pathlib import Path
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .image_utils import load_image

ROI_COLUMNS = ["seed_id", "x_min", "y_min", "x_max", "y_max", "priority", "selection_notes"]
REVIEW_COLUMNS = ["seed_id", "pilot_frame_index", "frame_index", "image_filename", "image_timestamp", "x_min", "y_min", "x_max", "y_max", "roi_source", "identity_review", "manual_state", "manual_confidence", "notes"]
EVENT_REVIEW_COLUMNS = ["seed_id", "event_type", "last_negative_frame_index", "first_positive_frame_index", "priority", "visual_confidence", "notes"]
EVENT_COLUMNS = ["seed_id", "event_type", "last_negative_timestamp", "first_positive_timestamp", "coarse_interval_seconds", "visual_confidence", "roi_coordinates", "priority", "notes"]
STATES = {"intact", "possible_crack", "clear_crack", "radicle_visible", "advanced_germination", "occluded", "uncertain"}
PRIORITIES = {"high", "medium", "exclude"}
EVENT_TYPES = {"crack", "radicle_emergence"}


def load_pilot_inventory(path: Path) -> pd.DataFrame:
    """Use the existing inventory and retain exact, irregular timestamps."""
    frames = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"filename", "path", "timestamp", "width", "height", "is_readable"}
    if required - set(frames):
        raise ValueError(f"Inventory is missing: {sorted(required - set(frames))}")
    if frames.empty or frames["timestamp"].isna().any() or not frames["is_readable"].all():
        raise ValueError("Pilot inventory contains missing timestamps or unreadable images.")
    if frames["filename"].duplicated().any() or frames["timestamp"].duplicated().any():
        raise ValueError("Pilot inventory contains duplicate filenames or timestamps.")
    frames = frames.sort_values(["timestamp", "filename"]).reset_index(drop=True)
    frames.insert(0, "pilot_frame_index", np.arange(len(frames), dtype=int))
    return frames


def clip_roi(image: np.ndarray, roi: tuple[int, int, int, int], padding: int = 0) -> tuple[int, int, int, int]:
    """Return exclusive upper bounds, clipped to image dimensions."""
    if padding < 0 or any(not isinstance(v, (int, np.integer)) for v in roi):
        raise ValueError("ROI coordinates must be integers and padding nonnegative.")
    x0, y0, x1, y1 = map(int, roi)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("ROI must have positive width and height.")
    height, width = image.shape[:2]
    bounds = max(0, x0 - padding), max(0, y0 - padding), min(width, x1 + padding), min(height, y1 + padding)
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("ROI does not intersect the image.")
    return bounds


def crop_fixed_roi(image: np.ndarray, roi: tuple[int, int, int, int], padding: int = 0) -> np.ndarray:
    x0, y0, x1, y1 = clip_roi(image, roi, padding)
    return image[y0:y1, x0:x1].copy()


def validate_rois(rois: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
    if list(rois.columns) != ROI_COLUMNS:
        raise ValueError(f"ROI table must have columns: {ROI_COLUMNS}")
    if len(rois) > 10 or rois["seed_id"].duplicated().any():
        raise ValueError("Use at most ten distinct target seed IDs.")
    result = rois.copy()
    for row in result.itertuples(index=False):
        if not isinstance(row.seed_id, str) or not re.fullmatch(r"seed_(0[1-9]|10)", row.seed_id):
            raise ValueError(f"Invalid seed_id: {row.seed_id}")
        values = (row.x_min, row.y_min, row.x_max, row.y_max)
        if any(pd.isna(v) or int(v) != v for v in values):
            raise ValueError(f"ROI for {row.seed_id} needs integer coordinates.")
        x0, y0, x1, y1 = map(int, values)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"ROI for {row.seed_id} is outside the {width}x{height} frame.")
        if pd.notna(row.priority) and row.priority not in PRIORITIES:
            raise ValueError(f"Invalid priority for {row.seed_id}: {row.priority}")
    return result


def make_visual_review(rois: pd.DataFrame, frames: pd.DataFrame, frame_rois: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    by_key = {} if frame_rois is None else {(r.seed_id, int(r.frame_index)): r for r in frame_rois.itertuples(index=False)}
    for roi in rois.itertuples(index=False):
        for frame in frames.itertuples(index=False):
            per_frame = by_key.get((roi.seed_id, int(frame.pilot_frame_index)), roi)
            rows.append({"seed_id": roi.seed_id, "pilot_frame_index": frame.pilot_frame_index,
                         "frame_index": frame.pilot_frame_index,
                         "image_filename": frame.filename, "image_timestamp": frame.timestamp,
                         "x_min": per_frame.x_min, "y_min": per_frame.y_min, "x_max": per_frame.x_max, "y_max": per_frame.y_max,
                         "roi_source": getattr(per_frame, "roi_source", "reference"),
                         "identity_review": getattr(per_frame, "identity_review", ""),
                         "manual_state": "", "manual_confidence": "", "notes": ""})
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def save_overview(frames: pd.DataFrame, destination: Path) -> None:
    selected = frames.iloc[np.linspace(0, len(frames) - 1, 5).round().astype(int)]
    fig, axes = plt.subplots(5, 1, figsize=(16, 26), constrained_layout=True)
    for ax, row in zip(axes, selected.itertuples(index=False)):
        image = load_image(Path(row.path))
        if image is None:
            raise ValueError(f"Cannot read image: {row.path}")
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Pilot index {row.pilot_frame_index:02d}  |  {row.timestamp}")
        ax.set_xlabel("x pixels")
        ax.set_ylabel("y pixels")
        ax.grid(alpha=.18)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def save_seed_timeline(seed_id: str, roi: tuple[int, int, int, int], frames: pd.DataFrame,
                       destination: Path, padding: int = 30, frame_rois: pd.DataFrame | None = None) -> None:
    if len(frames) != 25:
        raise ValueError("A target timeline requires all 25 pilot frames.")
    fig, axes = plt.subplots(5, 5, figsize=(19, 19), constrained_layout=True)
    per_frame = {} if frame_rois is None else {int(r.frame_index): (int(r.x_min),int(r.y_min),int(r.x_max),int(r.y_max))
        for r in frame_rois.loc[frame_rois.seed_id.eq(seed_id)].itertuples(index=False)}
    for ax, row in zip(axes.flat, frames.itertuples(index=False)):
        image = load_image(Path(row.path))
        if image is None:
            raise ValueError(f"Cannot read image: {row.path}")
        ax.imshow(cv2.cvtColor(crop_fixed_roi(image, per_frame.get(int(row.pilot_frame_index), roi), padding), cv2.COLOR_BGR2RGB), interpolation="nearest")
        ax.set_title(f"{seed_id} | {row.pilot_frame_index:02d}\n{row.timestamp:%Y-%m-%d %H:%M:%S}", fontsize=9)
        ax.axis("off")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def build_reviewed_events(decisions: pd.DataFrame, review: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Emit events only from explicit human decisions and reviewed frame states."""
    if list(decisions.columns) != EVENT_REVIEW_COLUMNS:
        raise ValueError(f"Event decisions must have columns: {EVENT_REVIEW_COLUMNS}")
    if decisions.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    if decisions[["seed_id", "event_type"]].duplicated().any():
        raise ValueError("Duplicate seed/event decisions are not allowed.")
    frame_map = frames.set_index("pilot_frame_index")
    states = review.set_index(["seed_id", "pilot_frame_index"])
    rows = []
    for d in decisions.itertuples(index=False):
        if d.event_type not in EVENT_TYPES or d.priority not in PRIORITIES:
            raise ValueError(f"Invalid event type or priority for {d.seed_id}")
        if (pd.isna(d.last_negative_frame_index) or pd.isna(d.first_positive_frame_index)
                or str(d.last_negative_frame_index).strip() == ""
                or str(d.first_positive_frame_index).strip() == ""):
            continue  # unfinished manual decision, never an inferred event
        before, after = int(d.last_negative_frame_index), int(d.first_positive_frame_index)
        if before not in frame_map.index or after not in frame_map.index or after != before + 1:
            raise ValueError(f"Event for {d.seed_id} needs adjacent negative/positive pilot indices.")
        if (d.seed_id, before) not in states.index or (d.seed_id, after) not in states.index:
            raise ValueError(f"Missing manual review rows for {d.seed_id}")
        if "identity_review" in review and (states.loc[(d.seed_id, before), "identity_review"] != "confirmed"
            or states.loc[(d.seed_id, after), "identity_review"] != "confirmed"):
            raise ValueError(f"Human seed identity must be confirmed in both event-boundary frames for {d.seed_id}.")
        negative, positive = states.loc[(d.seed_id, before), "manual_state"], states.loc[(d.seed_id, after), "manual_state"]
        allowed_positive = {"crack": {"clear_crack"},
                            "radicle_emergence": {"radicle_visible", "advanced_germination"}}[d.event_type]
        allowed_negative = {"crack": {"intact"}, "radicle_emergence": {"intact", "possible_crack", "clear_crack"}}[d.event_type]
        if negative not in allowed_negative or positive not in allowed_positive:
            raise ValueError(f"Review states for {d.seed_id} do not support a clear {d.event_type} interval.")
        t0, t1 = frame_map.loc[before, "timestamp"], frame_map.loc[after, "timestamp"]
        roi = states.loc[(d.seed_id, before), ["x_min", "y_min", "x_max", "y_max"]].astype(int).tolist()
        rows.append({"seed_id": d.seed_id, "event_type": d.event_type,
                     "last_negative_timestamp": t0, "first_positive_timestamp": t1,
                     "coarse_interval_seconds": (t1 - t0).total_seconds(),
                     "visual_confidence": d.visual_confidence, "roi_coordinates": str(roi),
                     "priority": d.priority, "notes": d.notes})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
