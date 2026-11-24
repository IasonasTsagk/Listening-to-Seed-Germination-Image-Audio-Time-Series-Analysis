from pathlib import Path
import sys

import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dense_visual import (WINDOWS, clip_bounds, discover_dense_images, initial_events,
                              intermediate_frame_count, load_crop, load_rois, make_manifest, narrowed_frames,
                              prepare_stage2, sample_nearest, save_event_figure,
                              row_has_human_review, validate_final_event, validate_interval,
                              window_for_seed, window_images, load_events, load_review_manifest,
                              save_events, save_review_manifest, save_stage2_interval,
                              MANIFEST_COLUMNS, REVIEW_KEY)


def test_seed_mapping_and_missing_seed():
    assert window_for_seed("seed_01").window_id == "WINDOW_A"
    assert window_for_seed("seed_05").window_id == "WINDOW_B"
    assert window_for_seed("seed_02").window_id == "WINDOW_C"
    with pytest.raises(ValueError, match="No unique"):
        window_for_seed("seed_99")


def test_phase_a_frame_roi_correction_is_reused(tmp_path):
    from src.dense_visual import WINDOWS
    roi_rows = []
    for window in WINDOWS:
        for seed in window.seed_ids:
            roi_rows.append({"seed_id": seed, "padded_x_min": 1, "padded_y_min": 2,
                             "padded_x_max": 5, "padded_y_max": 6})
    roi_path = tmp_path / "rois.csv"
    pd.DataFrame(roi_rows).to_csv(roi_path, index=False)
    frame_path = tmp_path / "frame_rois.csv"
    pd.DataFrame([{"seed_id": "seed_05", "image_timestamp": "2026-07-14 19:01:36",
                   "x_min": 3, "y_min": 4, "x_max": 7, "y_max": 8,
                   "roi_source": "manual_correction"}]).to_csv(frame_path, index=False)
    rois = load_rois(roi_path, frame_path)
    seed = rois.loc[rois.seed_id == "seed_05"].iloc[0]
    assert seed.padded_x_min == 3
    assert seed.dense_roi_source == "phase_a_frame_correction"


def test_interval_validation_and_bounds():
    assert validate_interval("2026-07-14 20:01:57", "2026-07-14 20:02:07") == 10
    with pytest.raises(ValueError, match="must precede"):
        validate_interval("2026-07-14 20:02:07", "2026-07-14 20:01:57")
    with pytest.raises(ValueError, match="exceeds"):
        validate_interval("2026-07-14 20:01:57", "2026-07-14 20:02:07", allowed_end="2026-07-14 20:02:06")
    assert clip_bounds((0, 0, 5, 5), 8, 8, 3) == (0, 0, 8, 8)
    with pytest.raises(ValueError, match="does not intersect"):
        clip_bounds((20, 20, 30, 30), 8, 8)


def test_nearest_sampling_preserves_actual_time_and_order():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-07-14 12:03:04", "2026-07-14 12:00:00",
                                                  "2026-07-14 12:01:26", "2026-07-14 12:04:52"])})
    result = sample_nearest(frame, 90)
    assert list(result.timestamp) == sorted(result.timestamp)
    assert result.timestamp.iloc[0] == pd.Timestamp("2026-07-14 12:00:00")
    assert result.timestamp.iloc[-1] == pd.Timestamp("2026-07-14 12:04:52")
    assert pd.Timestamp("2026-07-14 12:01:26") in list(result.timestamp)


def test_shared_boundary_stored_once_and_membership(tmp_path):
    group = tmp_path / "0"
    group.mkdir()
    for name in ("recording__2026_07_14_19_01_36.png", "recording__2026_07_14_20_01_57.png",
                 "recording__2026_07_14_21_00_07.png", "recording__2026_07_14_20_01_57-checkpoint.png"):
        Image.new("RGB", (8, 6)).save(group / name)
    inv = discover_dense_images(tmp_path, windows=WINDOWS[1:], groups=[group])
    assert len(inv) == 3
    assert len(window_images(inv, "WINDOW_B")) == 2
    assert len(window_images(inv, "WINDOW_C")) == 2
    boundary = inv.loc[inv.filename.str.contains("20_01_57")].iloc[0]
    assert boundary.window_id == "WINDOW_B;WINDOW_C"
    assert list(inv.timestamp) == sorted(inv.timestamp)


def test_manifest_blank_human_fields_and_lazy_crop(tmp_path, monkeypatch):
    group = tmp_path / "0"
    group.mkdir()
    image = group / "recording__2026_07_14_20_01_57.png"
    Image.new("RGB", (8, 6)).save(image)
    assert load_crop(str(image), (1, 1, 5, 5)).size == (8, 6)
    monkeypatch.setattr(Image, "open", lambda *_: (_ for _ in ()).throw(AssertionError("decoded")))
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-07-14 20:01:57", "2026-07-14 20:02:07"])})
    sample_nearest(frame)  # Metadata-only sampling never decodes a frame.


def test_final_event_accepts_consecutive_and_wider_available_frame_intervals():
    frames = pd.DataFrame({"window_id": ["WINDOW_B", "WINDOW_B", "WINDOW_B"],
                           "filename": ["a.png", "b.png", "c.png"],
                           "timestamp": pd.to_datetime(["2026-07-14 19:01:36", "2026-07-14 19:01:46", "2026-07-14 19:01:56"]),
                           "source_path": ["a", "b", "c"]})
    row = initial_events().loc[lambda table: table.seed_id == "seed_05"].iloc[0].copy()
    row.event_type = "crack"
    row.stage1_last_negative_timestamp = "2026-07-14 19:01:36"
    row.stage1_first_positive_timestamp = "2026-07-14 19:01:56"
    row.last_negative_timestamp = "2026-07-14 19:01:36"
    row.first_positive_timestamp = "2026-07-14 19:01:46"
    assert validate_final_event(row, frames) == 10
    row.first_positive_timestamp = "2026-07-14 19:01:56"
    assert validate_final_event(row, frames) == 20
    assert intermediate_frame_count(frames, "seed_05", row.last_negative_timestamp,
                                    row.first_positive_timestamp) == 1


def test_stage2_reuses_saved_human_review_and_roi(tmp_path):
    inventory = pd.DataFrame({"window_id": ["WINDOW_B"] * 3,
                              "filename": ["a.png", "b.png", "c.png"],
                              "timestamp": pd.to_datetime(["2026-07-14 19:01:36", "2026-07-14 19:01:46", "2026-07-14 19:01:56"]),
                              "source_path": ["a", "b", "c"]})
    # Keep the synthetic fixture limited to WINDOW_B, without changing production mapping.
    from src.dense_visual import MANIFEST_COLUMNS
    manifest = pd.DataFrame([["seed_05", "WINDOW_B", "a.png", "2026-07-14 19:01:36", 0,
                              1, 2, 5, 6, "phase_a_padded", "stage1", "", "", ""]], columns=MANIFEST_COLUMNS)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    events = initial_events()
    events.loc[events.seed_id == "seed_05", "stage1_last_negative_timestamp"] = "2026-07-14 19:01:36"
    events.loc[events.seed_id == "seed_05", "stage1_first_positive_timestamp"] = "2026-07-14 19:01:56"
    events_path = tmp_path / "events.csv"
    events.to_csv(events_path, index=False)
    first = prepare_stage2("seed_05", inventory, manifest_path, events_path)
    assert len(first) == 3
    saved = pd.read_csv(manifest_path, keep_default_na=False)
    mask = (saved.seed_id == "seed_05") & (saved.review_stage == "stage2") & (saved.image_filename == "b.png")
    saved.loc[mask, ["manual_state", "x_min", "roi_source"]] = ["clear_crack", 2, "manual_correction"]
    saved.to_csv(manifest_path, index=False)
    second = prepare_stage2("seed_05", inventory, manifest_path, events_path)
    middle = second.loc[second.image_filename == "b.png"].iloc[0]
    assert middle.manual_state == "clear_crack"
    assert middle.x_min == 2
    assert middle.roi_source == "manual_correction"


def _seed01_stage2_fixture(tmp_path):
    stamps = pd.date_range("2026-07-14 17:03:56", periods=5, freq="10s")
    inventory = pd.DataFrame({"window_id": ["WINDOW_A"] * 5,
                              "filename": [f"frame_{i}.png" for i in range(5)],
                              "timestamp": stamps, "source_path": [f"path_{i}" for i in range(5)]})
    stage1 = pd.DataFrame([
        ["seed_01", "WINDOW_A", "frame_0.png", str(stamps[0]), 0, 1, 2, 5, 6,
         "phase_a_padded", "stage1", "intact", "high", "reviewed stage 1"],
        ["seed_02", "WINDOW_C", "other.png", "2026-07-14 20:01:57", 0, 7, 8, 11, 12,
         "phase_a_padded", "stage1", "", "", ""],
    ], columns=MANIFEST_COLUMNS)
    path = tmp_path / "dense_visual_review_manifest.csv"
    stage1.to_csv(path, index=False)
    events = initial_events()
    events.loc[events.seed_id == "seed_01", "stage1_last_negative_timestamp"] = str(stamps[0])
    events.loc[events.seed_id == "seed_01", "stage1_first_positive_timestamp"] = str(stamps[-1])
    events_path = tmp_path / "refined_visual_events.csv"
    events.to_csv(events_path, index=False)
    return inventory, path, events_path, stage1, stamps


def test_stage2_regeneration_is_idempotent_and_backs_up(tmp_path):
    inventory, path, events_path, stage1, stamps = _seed01_stage2_fixture(tmp_path)
    original_events = events_path.read_bytes()
    first = prepare_stage2("seed_01", inventory, path, events_path)
    assert len(first) == 5
    after_first = path.read_bytes()
    backups = list(tmp_path.glob("dense_visual_review_manifest.backup_*.csv"))
    assert len(backups) == 1
    assert pd.read_csv(backups[0], keep_default_na=False).shape[0] == 2
    second = prepare_stage2("seed_01", inventory, path, events_path)
    assert len(second) == 5
    assert path.read_bytes() == after_first
    assert len(list(tmp_path.glob("dense_visual_review_manifest.backup_*.csv"))) == 1
    saved = pd.read_csv(path, keep_default_na=False)
    assert not saved.duplicated(REVIEW_KEY).any()
    pd.testing.assert_frame_equal(saved.loc[saved.review_stage == "stage1", stage1.columns].reset_index(drop=True),
                                  stage1.reset_index(drop=True), check_dtype=False)
    assert events_path.read_bytes() == original_events


def test_narrowed_stage2_removes_only_unreviewed_rows(tmp_path):
    inventory, path, events_path, stage1, stamps = _seed01_stage2_fixture(tmp_path)
    prepare_stage2("seed_01", inventory, path, events_path)
    events = pd.read_csv(events_path, keep_default_na=False)
    events.loc[events.seed_id == "seed_01", "stage1_first_positive_timestamp"] = str(stamps[2])
    events.to_csv(events_path, index=False)
    current = prepare_stage2("seed_01", inventory, path, events_path)
    assert current.image_timestamp.tolist() == [str(x) for x in stamps[:3]]
    saved = pd.read_csv(path, keep_default_na=False)
    assert saved.loc[(saved.seed_id == "seed_01") & (saved.review_stage == "stage2")].shape[0] == 3
    assert not saved.duplicated(REVIEW_KEY).any()
    pd.testing.assert_frame_equal(saved.loc[saved.review_stage == "stage1", stage1.columns].reset_index(drop=True),
                                  stage1.reset_index(drop=True), check_dtype=False)


def test_reviewed_outside_window_and_matching_human_fields_survive(tmp_path):
    inventory, path, events_path, stage1, stamps = _seed01_stage2_fixture(tmp_path)
    prepare_stage2("seed_01", inventory, path, events_path)
    saved = pd.read_csv(path, keep_default_na=False)
    outside = (saved.seed_id == "seed_01") & (saved.review_stage == "stage2") & (saved.image_timestamp == str(stamps[4]))
    matching = (saved.seed_id == "seed_01") & (saved.review_stage == "stage2") & (saved.image_timestamp == str(stamps[1]))
    saved.loc[outside, ["manual_state", "manual_confidence", "notes"]] = ["uncertain", "low", "old review"]
    saved.loc[matching, ["manual_state", "manual_confidence", "notes", "x_min", "roi_source"]] = ["possible_crack", "medium", "keep me", 3, "manual_correction"]
    saved.to_csv(path, index=False)
    events = pd.read_csv(events_path, keep_default_na=False)
    events.loc[events.seed_id == "seed_01", "stage1_first_positive_timestamp"] = str(stamps[2])
    events.to_csv(events_path, index=False)
    expected_events = events_path.read_bytes()
    current = prepare_stage2("seed_01", inventory, path, events_path)
    assert len(current) == 3  # Obsolete reviewed row is kept in CSV, not shown in current viewer.
    persisted = pd.read_csv(path, keep_default_na=False)
    old_review = persisted.loc[(persisted.seed_id == "seed_01") & (persisted.image_timestamp == str(stamps[4])) &
                               (persisted.review_stage == "stage2")].iloc[0]
    assert (old_review.manual_state, old_review.manual_confidence, old_review.notes) == ("uncertain", "low", "old review")
    current_review = persisted.loc[(persisted.seed_id == "seed_01") & (persisted.image_timestamp == str(stamps[1])) &
                                   (persisted.review_stage == "stage2")].iloc[0]
    assert (current_review.manual_state, current_review.manual_confidence, current_review.notes) == ("possible_crack", "medium", "keep me")
    assert current_review.x_min == 3 and current_review.roi_source == "manual_correction"
    assert not persisted.duplicated(REVIEW_KEY).any()
    pd.testing.assert_frame_equal(persisted.loc[persisted.review_stage == "stage1", stage1.columns].reset_index(drop=True),
                                  stage1.reset_index(drop=True), check_dtype=False)
    assert events_path.read_bytes() == expected_events
    prepare_stage2("seed_01", inventory, path, events_path)  # Reopening twice remains safe.


def test_row_review_detection():
    assert not row_has_human_review({"manual_state": " ", "manual_confidence": "", "notes": "", "roi_source": "phase_a_padded"})
    assert row_has_human_review({"manual_state": "", "manual_confidence": "0", "notes": ""})
    assert row_has_human_review({"roi_source": "manual_correction"})


def test_stage2_interval_saves_nullable_numeric_and_preserves_reviews(tmp_path):
    inventory, manifest_path, events_path, _, stamps = _seed01_stage2_fixture(tmp_path)
    events = load_events(events_path)
    events.loc[events.seed_id == "seed_01", "event_type"] = "crack"
    events.loc[events.seed_id == "seed_01", "notes"] = "existing human note"
    save_events(events, events_path)
    before = load_events(events_path)
    assert str(before.refined_interval_seconds.dtype) == "Float64"
    assert before.refined_interval_seconds.isna().all()
    manifest_before = manifest_path.read_bytes()

    seconds = save_stage2_interval(events_path, inventory, "seed_01", "crack",
                                   str(stamps[1]), str(stamps[2]), "high", "confirmed")
    assert seconds == 10.0
    saved = load_events(events_path)
    row = saved.loc[saved.seed_id == "seed_01"].iloc[0]
    assert row.refined_interval_seconds == 10.0
    assert str(saved.refined_interval_seconds.dtype) == "Float64"
    assert saved.loc[saved.seed_id != "seed_01", "refined_interval_seconds"].isna().all()
    assert row.notes == "existing human note"
    assert row.stage1_last_negative_timestamp == before.loc[before.seed_id == "seed_01", "stage1_last_negative_timestamp"].iloc[0]
    assert row.stage1_first_positive_timestamp == before.loc[before.seed_id == "seed_01", "stage1_first_positive_timestamp"].iloc[0]
    pd.testing.assert_frame_equal(saved.loc[saved.seed_id != "seed_01"].reset_index(drop=True),
                                  before.loc[before.seed_id != "seed_01"].reset_index(drop=True))
    assert manifest_path.read_bytes() == manifest_before
    backups = list(tmp_path.glob("refined_visual_events.backup_*.csv"))
    assert len(backups) == 2  # One for test setup, one immediately before the human save.
    assert any(load_events(backup).equals(before) for backup in backups)


def test_stage2_saves_wide_uncertainty_interval_with_ambiguous_frames(tmp_path):
    inventory, manifest_path, events_path, _, stamps = _seed01_stage2_fixture(tmp_path)
    prepare_stage2("seed_01", inventory, manifest_path, events_path)
    manifest = load_review_manifest(manifest_path)
    for index, state in ((1, "possible_crack"), (2, "uncertain"), (3, "occluded")):
        mask = (manifest.seed_id == "seed_01") & (manifest.review_stage == "stage2") & \
               (manifest.image_timestamp == str(stamps[index]))
        manifest.loc[mask, "manual_state"] = state
    save_review_manifest(manifest, manifest_path)
    manifest_before = manifest_path.read_bytes()
    events_before = load_events(events_path)

    seconds = save_stage2_interval(events_path, inventory, "seed_01", "crack",
                                   str(stamps[0]), str(stamps[4]), "medium", "confirmed")
    assert seconds == 40.0
    assert intermediate_frame_count(inventory, "seed_01", str(stamps[0]), str(stamps[4])) == 3
    saved = load_events(events_path)
    row = saved.loc[saved.seed_id == "seed_01"].iloc[0]
    assert (row.last_negative_timestamp, row.first_positive_timestamp) == (str(stamps[0]), str(stamps[4]))
    assert row.refined_interval_seconds == 40.0
    assert row.stage1_last_negative_timestamp == events_before.loc[0, "stage1_last_negative_timestamp"]
    assert row.stage1_first_positive_timestamp == events_before.loc[0, "stage1_first_positive_timestamp"]
    assert manifest_path.read_bytes() == manifest_before
    assert load_review_manifest(manifest_path).loc[
        lambda table: (table.seed_id == "seed_01") & (table.review_stage == "stage2") &
                      table.manual_state.ne(""), "manual_state"].tolist() == ["possible_crack", "uncertain", "occluded"]


@pytest.mark.parametrize("negative,positive", [("", "2026-07-14 17:04:06"),
                                                 ("2026-07-14 17:03:56", ""),
                                                 ("2026-07-14 17:04:06", "2026-07-14 17:04:06"),
                                                 ("2026-07-14 17:04:16", "2026-07-14 17:04:06")])
def test_stage2_rejects_missing_or_unordered_bounds_without_changing_events(tmp_path, negative, positive):
    inventory, _, events_path, _, _ = _seed01_stage2_fixture(tmp_path)
    before = events_path.read_bytes()
    with pytest.raises(ValueError, match="required|must precede"):
        save_stage2_interval(events_path, inventory, "seed_01", "crack", negative, positive)
    assert events_path.read_bytes() == before


def test_event_figure_uses_actual_nonconsecutive_boundary_frames(tmp_path, monkeypatch):
    inventory, _, events_path, _, stamps = _seed01_stage2_fixture(tmp_path)
    events = load_events(events_path)
    row = events.loc[events.seed_id == "seed_01"].iloc[0].copy()
    row.event_type = "crack"
    row.last_negative_timestamp = str(stamps[1])
    row.first_positive_timestamp = str(stamps[3])
    seen = []

    def fake_crop(source_path, *_args, **_kwargs):
        seen.append(source_path)
        return Image.new("RGB", (8, 8))

    monkeypatch.setattr("src.dense_visual.load_crop", fake_crop)
    destination = tmp_path / "event.png"
    save_event_figure(row, inventory, (0, 0, 8, 8), destination)
    assert destination.exists()
    assert seen == ["path_0", "path_1", "path_3", "path_4"]


def test_manifest_schema_and_backed_up_save_preserve_human_text(tmp_path):
    _, manifest_path, _, _, _ = _seed01_stage2_fixture(tmp_path)
    original = manifest_path.read_bytes()
    manifest = load_review_manifest(manifest_path)
    assert all(str(manifest[column].dtype) == "Int64" for column in ("frame_index", "x_min", "y_min", "x_max", "y_max"))
    assert manifest.loc[0, "manual_state"] == "intact"
    save_review_manifest(manifest, manifest_path)
    backups = list(tmp_path.glob("dense_visual_review_manifest.backup_*.csv"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    reloaded = load_review_manifest(manifest_path)
    assert reloaded.loc[0, "manual_state"] == "intact"
    assert reloaded.loc[0, "manual_confidence"] == "high"
    assert reloaded.loc[0, "notes"] == "reviewed stage 1"


def test_reject_nonnumeric_refined_interval_without_overwriting(tmp_path):
    path = tmp_path / "refined_visual_events.csv"
    events = initial_events()
    events.loc[events.seed_id == "seed_01", "refined_interval_seconds"] = 10.0
    save_events(events, path)
    assert load_events(path).loc[0, "refined_interval_seconds"] == 10.0
    original = path.read_bytes()
    malformed = pd.read_csv(path, dtype="string", keep_default_na=False)
    malformed.loc[0, "refined_interval_seconds"] = "not a duration"
    with pytest.raises(ValueError, match="Non-numeric"):
        save_events(malformed, path)
    assert path.read_bytes() == original
