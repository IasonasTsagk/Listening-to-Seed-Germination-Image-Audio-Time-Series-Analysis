from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.targeted_visual import (EVENT_REVIEW_COLUMNS, ROI_COLUMNS, REVIEW_COLUMNS,
    build_reviewed_events, clip_roi, crop_fixed_roi, load_pilot_inventory,
    make_visual_review, validate_rois, save_seed_timeline)
from src.target_seed_interactive import (INTERACTIVE_COLUMNS, normalize_rectangle,
    padded_rectangle, selection_row, effective_rois, save_selections, load_selections,
    chronological_crops, save_gif, estimate_shift, estimate_orb_shift,
    FRAME_ROI_COLUMNS, recenter_rectangle, make_frame_rois)


def test_fixed_crop_and_boundary_clipping():
    image = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
    assert clip_roi(image, (1, 1, 3, 3), padding=2) == (0, 0, 5, 5)
    crop = crop_fixed_roi(image, (1, 1, 3, 3), padding=2)
    assert crop.shape == (5, 5)
    assert np.array_equal(crop, image[:5, :5])
    assert np.array_equal(image, np.arange(6 * 8, dtype=np.uint8).reshape(6, 8))
    with pytest.raises(ValueError, match="does not intersect"):
        crop_fixed_roi(image, (9, 1, 12, 3))


def test_chronology_timestamps_review_and_event_interval(tmp_path):
    inventory = pd.DataFrame({
        "filename": ["later.png", "earlier.png"],
        "path": ["later.png", "earlier.png"],
        "timestamp": ["2026-07-14 13:08:36", "2026-07-14 12:01:32"],
        "width": [8, 8], "height": [6, 6], "is_readable": [True, True],
    })
    path = tmp_path / "inventory.csv"
    inventory.to_csv(path, index=False)
    frames = load_pilot_inventory(path)
    assert frames.filename.tolist() == ["earlier.png", "later.png"]
    rois = pd.DataFrame([["seed_01", 1, 1, 3, 3, "high", "human selection"]], columns=ROI_COLUMNS)
    rois = validate_rois(rois, 8, 6)
    review = make_visual_review(rois, frames)
    assert review.image_timestamp.tolist() == list(frames.timestamp)
    assert review.manual_state.tolist() == ["", ""]
    assert list(review.columns) == REVIEW_COLUMNS
    review.loc[:, "manual_state"] = ["intact", "clear_crack"]
    review.loc[:, "identity_review"] = ["confirmed", "confirmed"]
    decisions = pd.DataFrame([["seed_01", "crack", 0, 1, "high", "clear", "manual review"]], columns=EVENT_REVIEW_COLUMNS)
    events = build_reviewed_events(decisions, review, frames)
    assert len(events) == 1
    assert events.coarse_interval_seconds.iloc[0] == 4024
    assert events.last_negative_timestamp.iloc[0] == pd.Timestamp("2026-07-14 12:01:32")
    assert events.first_positive_timestamp.iloc[0] == pd.Timestamp("2026-07-14 13:08:36")
    review.loc[1, "manual_state"] = "uncertain"
    with pytest.raises(ValueError, match="do not support"):
        build_reviewed_events(decisions, review, frames)


def test_no_event_without_manual_decision():
    frames = pd.DataFrame({"pilot_frame_index": [0], "timestamp": [pd.Timestamp("2026-07-14 12:01:32")]})
    empty = build_reviewed_events(pd.DataFrame(columns=EVENT_REVIEW_COLUMNS), pd.DataFrame(columns=REVIEW_COLUMNS), frames)
    assert empty.empty


def test_mouse_rectangle_padding_clipping_and_bad_inputs():
    assert normalize_rectangle((7.8, 5.6, 1.2, 0.1), 8, 6) == (1, 0, 8, 6)
    assert padded_rectangle((2, 2, 4, 4), 8, 6, 3) == (0, 0, 7, 6)
    assert padded_rectangle((0, 0, 2, 2), 8, 6, 4) == (0, 0, 6, 6)
    with pytest.raises(ValueError):
        normalize_rectangle((2, 2, 2, 2), 8, 6)
    with pytest.raises(ValueError):
        normalize_rectangle((9, 1, 10, 3), 8, 6)
    with pytest.raises(ValueError):
        padded_rectangle((1, 1, 3, 3), 8, 6, -1)


def test_roi_csv_explicit_replace_and_duplicate_guard(tmp_path):
    frame = pd.Series(dict(filename="frame.png", timestamp=pd.Timestamp("2026-07-14 12:01:32"), width=8, height=6))
    rows = [selection_row(f"seed_{i:02d}", (0, 0, 2, 2), frame, padding=2) for i in range(1, 6)]
    table = pd.DataFrame(rows, columns=INTERACTIVE_COLUMNS)
    path = tmp_path / "rois.csv"
    save_selections(table, path, 8, 6)
    assert load_selections(path, frame).seed_id.tolist() == [f"seed_{i:02d}" for i in range(1, 6)]
    assert effective_rois(table, 8, 6).iloc[0]["x_max"] == 4
    with pytest.raises(ValueError, match="confirmation"):
        save_selections(table, path, 8, 6)
    backup = save_selections(table, path, 8, 6, confirm_replace=True)
    assert backup.exists()
    duplicate = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="distinct"):
        effective_rois(duplicate, 8, 6)
    bad = table.copy()
    bad.loc[0, "padded_x_max"] = 5
    with pytest.raises(ValueError, match="Padded ROI"):
        effective_rois(bad, 8, 6)


def test_chronological_fixed_crops_and_gif(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        cv2.imwrite(str(path), np.full((6, 8, 3), index*50, np.uint8))
        paths.append(path)
    frames = pd.DataFrame(dict(pilot_frame_index=[2, 0, 1],
        filename=["2.png", "0.png", "1.png"], path=[str(paths[i]) for i in [2,0,1]],
        timestamp=pd.to_datetime(["2026-07-14 14:00:03", "2026-07-14 12:01:32", "2026-07-14 13:08:36"]),
        width=8, height=6))
    crops = chronological_crops(frames, (0, 0, 3, 4))
    assert [f.pilot_frame_index for f, _ in crops] == [0, 1, 2]
    assert {crop.shape for _, crop in crops} == {(4, 3, 3)}
    assert [int(crop[0,0,0]) for _, crop in crops] == [0, 50, 100]
    gif = tmp_path / "timeline.gif"
    save_gif("seed_01", (0, 0, 3, 4), frames, gif)
    with Image.open(gif) as animation:
        assert animation.n_frames == 3
        assert animation.size == (300, 76)


def test_camera_shift_estimate_direction_and_weak_feature_fallback():
    rng = np.random.default_rng(5)
    image = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    shifted = cv2.warpAffine(image, np.float32([[1,0,3],[0,1,-2]]), (128,128))
    dx, dy, response = estimate_shift(image, shifted)
    assert abs(dx-3) < .5 and abs(dy+2) < .5 and response > .15
    orb = estimate_orb_shift(image, shifted)
    if orb is not None:
        assert abs(orb[0]-3) < 1 and abs(orb[1]+2) < 1


def test_manual_recenter_preserves_scale_and_reference_fallback(tmp_path):
    assert recenter_rectangle((0, 0, 4, 4), (6, 4, 8, 6), 8, 6) == (4, 2, 8, 6)
    rois = pd.DataFrame([["seed_01", 0, 0, 4, 4, "", ""]], columns=ROI_COLUMNS)
    paths = []
    for i in range(3):
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        if i == 1:
            image[3:5, 5:7] = 255  # a seed moves right/down in the middle frame
        else:
            image[1:3, 1:3] = 255
        path = tmp_path / f"{i}.png"
        cv2.imwrite(str(path), image)
        paths.append(str(path))
    frames = pd.DataFrame(dict(pilot_frame_index=[0,1,2], filename=[f"{i}.png" for i in range(3)],
        path=paths, timestamp=pd.to_datetime(["2026-07-14 12:01:32", "2026-07-14 13:08:36", "2026-07-14 14:00:03"]),
        width=8, height=6))
    baseline = make_frame_rois(rois, frames)
    assert list(baseline.columns) == FRAME_ROI_COLUMNS
    assert baseline.roi_source.tolist() == ["reference"]*3
    corrected = baseline.copy()
    corrected.loc[1, list(("x_min","y_min","x_max","y_max"))] = [4,2,8,6]
    corrected.loc[1, "roi_source"] = "manual_correction"
    corrected.loc[1, "identity_review"] = "confirmed"
    stored = tmp_path / "frame_rois.csv"
    corrected.to_csv(stored, index=False)
    loaded = make_frame_rois(rois, frames, pd.read_csv(stored, keep_default_na=False))
    assert loaded.roi_source.tolist() == ["reference", "manual_correction", "reference"]
    assert loaded.identity_review.tolist() == ["", "confirmed", ""]
    crops = chronological_crops(frames, (0,0,4,4), loaded, "seed_01")
    assert {crop.shape for _, crop in crops} == {(4,4,3)}
    assert [int(crop.max()) for _, crop in crops] == [255,255,255]
    review = make_visual_review(rois, frames, loaded)
    assert review.roi_source.tolist() == loaded.roi_source.tolist()
    assert review.identity_review.tolist() == loaded.identity_review.tolist()
    gif = tmp_path / "moved.gif"
    save_gif("seed_01", (0,0,4,4), frames, gif, frame_rois=loaded)
    with Image.open(gif) as animation:
        assert animation.n_frames == 3
        assert animation.size == (300, 76)
    bad = loaded.copy()
    bad.loc[1, "x_max"] = 7
    with pytest.raises(ValueError, match="scale"):
        make_frame_rois(rois, frames, bad)
    with pytest.raises(ValueError, match="duplicate"):
        make_frame_rois(rois, frames, pd.concat([loaded, loaded.iloc[[0]]]))


def test_25_frame_timeline_uses_only_explicit_correction(tmp_path, monkeypatch):
    import src.targeted_visual as visual
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), np.zeros((6, 8, 3), dtype=np.uint8))
    times = pd.date_range("2026-07-14 12:01:32", periods=25, freq="67min")
    frames = pd.DataFrame(dict(pilot_frame_index=range(25), path=[str(image_path)]*25,
                               timestamp=times, width=8, height=6,
                               filename=[f"{i}.png" for i in range(25)]))
    rois = pd.DataFrame([["seed_01", 0, 0, 4, 4, "", ""]], columns=ROI_COLUMNS)
    per_frame = make_frame_rois(rois, frames)
    per_frame.loc[12, ["x_min", "y_min", "x_max", "y_max"]] = [4, 2, 8, 6]
    per_frame.loc[12, "roi_source"] = "manual_correction"
    used = []
    original = visual.crop_fixed_roi
    def recording_crop(image, roi, padding):
        used.append(tuple(roi))
        return original(image, roi, padding)
    monkeypatch.setattr(visual, "crop_fixed_roi", recording_crop)
    output = tmp_path / "timeline.png"
    save_seed_timeline("seed_01", (0,0,4,4), frames, output, padding=0, frame_rois=per_frame)
    assert output.exists()
    assert len(used) == 25
    assert used[12] == (4,2,8,6)
    assert all(roi == (0,0,4,4) for i, roi in enumerate(used) if i != 12)
