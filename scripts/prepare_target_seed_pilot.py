"""Prepare and refresh only the human-led target-seed review artifacts."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.targeted_visual import (EVENT_REVIEW_COLUMNS, REVIEW_COLUMNS, ROI_COLUMNS, STATES,
    build_reviewed_events, load_pilot_inventory, make_visual_review,
    save_overview, save_seed_timeline, validate_rois)
from src.target_seed_interactive import (effective_rois, camera_stability, stability_message,
    backup_file, save_gif, save_progression, make_frame_rois, FRAME_ROI_COLUMNS)


def main(root=ROOT, refresh_camera=False) -> None:
    root = Path(root)
    frames = load_pilot_inventory(root / "results/tables/dataset_inventory.csv")
    if len(frames) != 25:
        raise ValueError(f"Expected the existing 25-frame pilot inventory, found {len(frames)} rows.")
    tables = root / "results/tables/targeted_multimodal"
    figures = root / "results/figures/targeted_multimodal"
    timeline_dir = figures / "seed_timelines"
    tables.mkdir(parents=True, exist_ok=True)
    timeline_dir.mkdir(parents=True, exist_ok=True)
    roi_path = tables / "target_seed_rois.csv"
    frame_roi_path = tables / "target_seed_frame_rois.csv"
    review_path = tables / "target_seed_visual_review.csv"
    decisions_path = tables / "target_seed_event_review.csv"
    events_path = tables / "target_seed_events.csv"
    if not roi_path.exists():
        pd.DataFrame(columns=ROI_COLUMNS).to_csv(roi_path, index=False)
    if not frame_roi_path.exists():
        pd.DataFrame(columns=FRAME_ROI_COLUMNS).to_csv(frame_roi_path, index=False)
    if not decisions_path.exists():
        pd.DataFrame(columns=EVENT_REVIEW_COLUMNS).to_csv(decisions_path, index=False)
    if not review_path.exists():
        pd.DataFrame(columns=REVIEW_COLUMNS).to_csv(review_path, index=False)
    if not (figures / "pilot_frames_overview.png").exists():
        save_overview(frames, figures / "pilot_frames_overview.png")
    rois = effective_rois(pd.read_csv(roi_path, keep_default_na=False), int(frames.width.iloc[0]), int(frames.height.iloc[0]))
    camera_path = tables / "camera_stability_summary.csv"
    if refresh_camera and camera_path.exists():
        backup_file(camera_path)
    if refresh_camera or not camera_path.exists():
        camera_stability(frames).to_csv(camera_path, index=False)
    print(stability_message(pd.read_csv(camera_path), rois))
    if rois.empty:
        print(f"Loaded {len(frames)} pilot images: {frames.timestamp.iloc[0]} to {frames.timestamp.iloc[-1]}")
        print(f"Overview ready: {figures / 'pilot_frames_overview.png'}")
        print("Use the mouse editor in notebook 11 to select and save 5–10 seeds, then generate timelines.")
        return
    existing_frame_rois = pd.read_csv(frame_roi_path, keep_default_na=False)
    frame_rois = make_frame_rois(rois, frames, existing_frame_rois)
    if not frame_rois.astype(str).equals(existing_frame_rois.astype(str)):
        if not existing_frame_rois.empty:
            backup_file(frame_roi_path)
        frame_rois.to_csv(frame_roi_path, index=False)
    expected = make_visual_review(rois, frames, frame_rois)
    review = pd.read_csv(review_path, keep_default_na=False)
    keys = ["seed_id", "pilot_frame_index", "frame_index", "image_filename", "image_timestamp", "x_min", "y_min", "x_max", "y_max", "roi_source", "identity_review"]
    if review.empty:
        review = expected
        review.to_csv(review_path, index=False)
    else:
        if "frame_index" not in review:
            review.insert(2, "frame_index", review["pilot_frame_index"])
        if "roi_source" not in review:
            review.insert(9, "roi_source", "reference")
        if "identity_review" not in review:
            review.insert(10, "identity_review", "")
        actual_keys = review[keys].astype(str).reset_index(drop=True)
        expected_keys = expected[keys].astype(str).reset_index(drop=True)
        if not actual_keys.equals(expected_keys):
            # Preserve every reviewed row. Changed/reused IDs must be reconciled by a human.
            new_by_key = expected.set_index(["seed_id", "pilot_frame_index"])
            for row in review.itertuples(index=False):
                if any(str(getattr(row, c)).strip() for c in ["manual_state", "manual_confidence", "notes"]):
                    key = (row.seed_id, row.pilot_frame_index)
                    if key not in new_by_key.index or any(str(getattr(row,c)) != str(new_by_key.loc[key,c]) for c in keys[2:]):
                        raise ValueError("ROI edits affect human-reviewed rows. Original review retained; reconcile it manually before regenerating.")
                    for c in ["manual_state", "manual_confidence", "notes"]:
                        expected.loc[(expected.seed_id == row.seed_id) & (expected.pilot_frame_index == row.pilot_frame_index), c] = getattr(row,c)
            backup_file(review_path)
            review = expected
            review.to_csv(review_path, index=False)
        invalid = set(review.manual_state) - ({""} | STATES)
        if invalid:
            raise ValueError(f"Unknown manual states: {sorted(invalid)}")
    for roi in rois.itertuples(index=False):
        save_seed_timeline(roi.seed_id, (int(roi.x_min), int(roi.y_min), int(roi.x_max), int(roi.y_max)),
                           frames, timeline_dir / f"{roi.seed_id}_timeline.png", padding=0, frame_rois=frame_rois)
        save_gif(roi.seed_id, (int(roi.x_min), int(roi.y_min), int(roi.x_max), int(roi.y_max)),
                 frames, timeline_dir / f"{roi.seed_id}_timeline.gif", frame_rois=frame_rois)
    save_progression(rois, frames, figures / "selected_seed_progression_overview.png", frame_rois=frame_rois)
    decisions = pd.read_csv(decisions_path, keep_default_na=False)
    if not decisions.empty:
        events = build_reviewed_events(decisions, review, frames)
        if not events.empty:
            events.to_csv(events_path, index=False)
    print(f"Loaded {len(frames)} frames and {len(rois)} human-selected ROIs.")
    print(f"Timeline sheets: {timeline_dir}")
    print(f"Manual review table: {review_path}")
    print("GIF playback is constant 700 ms/frame, NOT proportional to irregular real elapsed time.")


if __name__ == "__main__":
    main()
