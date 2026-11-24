"""Human-only visual milestone review. Audio is never consulted here."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .dense_visual import load_events, load_rois, window_for_seed, load_crop, _backup_review_csv

MILESTONE_COLUMNS = ["seed_id", "event_type", "last_negative_timestamp",
                     "first_positive_timestamp", "interval_seconds",
                     "visual_confidence", "notes", "source"]
EVENT_TYPES = ("crack_onset", "major_opening", "radicle_emergence")


def initialize_milestones(phase_b_path: Path, milestones_path: Path) -> pd.DataFrame:
    """Import Phase B cracks once; reject conflicts and keep later human rows."""
    phase_b = load_events(phase_b_path)
    cracks = phase_b.loc[phase_b.event_type.eq("crack")]
    if cracks.seed_id.duplicated().any():
        raise ValueError("Duplicate Phase B crack intervals")
    imported = pd.DataFrame([{"seed_id": row.seed_id, "event_type": "crack_onset",
                              "last_negative_timestamp": row.last_negative_timestamp,
                              "first_positive_timestamp": row.first_positive_timestamp,
                              "interval_seconds": (pd.Timestamp(row.first_positive_timestamp)-pd.Timestamp(row.last_negative_timestamp)).total_seconds(),
                              "visual_confidence": row.visual_confidence, "notes": row.notes,
                              "source": "phase_b"} for row in cracks.itertuples(index=False)], columns=MILESTONE_COLUMNS)
    if not milestones_path.exists():
        milestones_path.parent.mkdir(parents=True, exist_ok=True)
        imported.to_csv(milestones_path, index=False)
        return imported
    existing = pd.read_csv(milestones_path, keep_default_na=False)
    if set(MILESTONE_COLUMNS)-set(existing) or existing.duplicated(["seed_id", "event_type"]).any():
        raise ValueError("Existing visual milestone table has invalid schema or duplicate keys")
    for row in imported.itertuples(index=False):
        old = existing.loc[existing.seed_id.eq(row.seed_id) & existing.event_type.eq("crack_onset")]
        if len(old) == 1:
            if (str(old.iloc[0].last_negative_timestamp) != str(row.last_negative_timestamp) or
                str(old.iloc[0].first_positive_timestamp) != str(row.first_positive_timestamp) or
                old.iloc[0].source != "phase_b"):
                raise ValueError(f"Phase B crack boundary differs in milestone table: {row.seed_id}")
        elif old.empty:
            existing = pd.concat([existing, pd.DataFrame([row._asdict()])], ignore_index=True)
        else:
            raise ValueError("Duplicate crack milestone")
    if len(existing) > len(pd.read_csv(milestones_path)):
        _backup_review_csv(milestones_path)
        existing.to_csv(milestones_path, index=False)
    return existing


def seed_frames(inventory_path: Path, seed_id: str) -> pd.DataFrame:
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    window = window_for_seed(seed_id).window_id
    selected = inventory.loc[inventory.window_id.str.split(";").map(lambda ids: window in ids)].copy()
    selected["timestamp"] = pd.to_datetime(selected.timestamp)
    if selected.empty or selected.timestamp.duplicated().any():
        raise ValueError(f"Missing or duplicate dense images for {seed_id}")
    return selected.sort_values("timestamp").reset_index(drop=True)


def roi_for_seed(rois_path: Path, frame_rois_path: Path, seed_id: str, timestamp=None) -> tuple[int, int, int, int]:
    rois = load_rois(rois_path, frame_rois_path)
    roi = rois.loc[rois.seed_id.eq(seed_id)].iloc[0]
    bounds = tuple(int(roi[f"padded_{key}"]) for key in ("x_min", "y_min", "x_max", "y_max"))
    if timestamp is not None and frame_rois_path.exists():
        corrections = pd.read_csv(frame_rois_path, keep_default_na=False)
        match = corrections.loc[corrections.seed_id.eq(seed_id) &
                                corrections.image_timestamp.eq(str(pd.Timestamp(timestamp))) &
                                corrections.roi_source.eq("manual_correction")]
        if len(match) == 1:
            bounds = tuple(int(match.iloc[0][key]) for key in ("x_min", "y_min", "x_max", "y_max"))
    return bounds


def save_visual_milestone(milestones_path: Path, inventory_path: Path, seed_id: str,
                          event_type: str, last_negative: str, first_positive: str,
                          confidence: str, notes: str) -> None:
    if event_type not in ("major_opening", "radicle_emergence"):
        raise ValueError("Only Phase D later milestones can be saved here")
    if confidence not in ("", "high", "medium", "low"):
        raise ValueError("Invalid visual confidence")
    frames = seed_frames(inventory_path, seed_id)
    available = set(frames.timestamp)
    start, end = pd.Timestamp(last_negative), pd.Timestamp(first_positive)
    if start not in available or end not in available or start >= end:
        raise ValueError("Choose ordered last-negative and first-positive frames from the seed's dense inventory")
    table = pd.read_csv(milestones_path, keep_default_na=False)
    if table.duplicated(["seed_id", "event_type"]).any():
        raise ValueError("Duplicate milestone keys")
    new = {"seed_id": seed_id, "event_type": event_type,
           "last_negative_timestamp": str(start), "first_positive_timestamp": str(end),
           "interval_seconds": (end-start).total_seconds(), "visual_confidence": confidence,
           "notes": notes, "source": "phase_d"}
    match = table.seed_id.eq(seed_id) & table.event_type.eq(event_type)
    if match.sum() == 1:
        if table.loc[match].iloc[0].source != "phase_d":
            raise ValueError("Cannot overwrite a Phase B milestone")
        for key, value in new.items():
            table.loc[match, key] = value
    else:
        table = pd.concat([table, pd.DataFrame([new])], ignore_index=True)
    _backup_review_csv(milestones_path)
    table.to_csv(milestones_path, index=False)
