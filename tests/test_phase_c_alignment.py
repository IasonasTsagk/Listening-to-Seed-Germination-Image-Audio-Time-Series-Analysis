from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.phase_c_alignment import (build_alignment, choose_control_windows,
    closest_peak, coverage_summary, group_visual_intervals, load_visual_intervals,
    overlap_seconds, signed_lag_seconds, window_metrics)
from src.dense_visual import initial_events


def _audio_fixture():
    base = pd.Timestamp("2026-07-14 17:00:00")
    starts = [base + pd.Timedelta(seconds=10*i) for i in range(1200)]
    features = pd.DataFrame({"relative_path": [f"{i}/a.wav" for i in range(len(starts))],
                             "start": starts, "end": [value+pd.Timedelta(seconds=2.75) for value in starts],
                             "rms_energy": [.01]*len(starts), "spectral_centroid_hz": [100.]*len(starts),
                             "spectral_flatness": [.1]*len(starts)})
    peaks = pd.DataFrame({"timestamp": [base+pd.Timedelta(seconds=x) for x in (9, 31, 32, 61)],
                          "source_relative_path": ["0/a.wav", "3/a.wav", "3/a.wav", "6/a.wav"],
                          "prominence": [.1,.2,.3,.4], "envelope_peak_value": [.1]*4,
                          "width_seconds": [.02]*4, "high": [False, True, True, False]})
    return features, peaks


def test_signed_lag_inside_before_after_and_endpoint():
    start, end = "2026-07-14 17:00:20", "2026-07-14 17:00:40"
    assert signed_lag_seconds("2026-07-14 17:00:10", start, end) == -10
    assert signed_lag_seconds("2026-07-14 17:00:20", start, end) == 0
    assert signed_lag_seconds("2026-07-14 17:00:30", start, end) == 0
    assert signed_lag_seconds("2026-07-14 17:00:50", start, end) == 10


def test_interval_grouping_and_shared_peak_counted_once():
    base = pd.Timestamp("2026-07-14 17:00:00")
    events = pd.DataFrame({"seed_id": ["seed_01", "seed_03", "seed_04", "seed_05"],
                           "event_type": ["crack"]*4,
                           "start": [base, base+pd.Timedelta(seconds=15), base+pd.Timedelta(seconds=2), base+pd.Timedelta(hours=1)],
                           "end": [base+pd.Timedelta(seconds=20), base+pd.Timedelta(seconds=35),
                                   base+pd.Timedelta(seconds=10), base+pd.Timedelta(hours=1, seconds=20)]})
    groups = group_visual_intervals(events)
    assert len(groups) == 2
    assert set(groups.iloc[0].seed_ids.split(";")) == {"seed_01", "seed_03", "seed_04"}
    features, peaks = _audio_fixture()
    first = window_metrics(groups.iloc[0].group_start, groups.iloc[0].group_end, features, peaks)
    assert first["peak_count"] == 3  # Peaks at 9, 31 and 32; not multiplied by three seeds.


def test_coverage_union_and_no_recording_is_not_silence():
    start = pd.Timestamp("2026-07-14 17:00:00")
    features = pd.DataFrame({"start": [start, start+pd.Timedelta(seconds=2)],
                             "end": [start+pd.Timedelta(seconds=3), start+pd.Timedelta(seconds=5)],
                             "relative_path": ["a.wav", "b.wav"], "rms_energy": [.1,.1]})
    assert overlap_seconds(start, start+pd.Timedelta(seconds=10), features) == 5
    summary = coverage_summary(start, start+pd.Timedelta(seconds=10), features)
    assert summary["recorded_audio_seconds"] == 5
    assert summary["coverage_fraction"] == .5
    empty = coverage_summary(start+pd.Timedelta(minutes=2), start+pd.Timedelta(minutes=3), features)
    assert empty["audio_coverage_class"] == "none"
    assert empty["recorded_audio_seconds"] == 0
    peaks = pd.DataFrame(columns=["timestamp", "high"])
    metrics = window_metrics(start+pd.Timedelta(minutes=2), start+pd.Timedelta(minutes=3), features, peaks)
    assert np.isnan(metrics["peaks_per_recorded_minute"])


def test_coverage_normalized_peak_rate():
    features, peaks = _audio_fixture()
    base = pd.Timestamp("2026-07-14 17:00:00")
    metric = window_metrics(base, base+pd.Timedelta(seconds=40), features, peaks)
    assert metric["recorded_audio_seconds"] == 11
    assert metric["peak_count"] == 3
    assert metric["high_prominence_peak_count"] == 2
    assert metric["peaks_per_recorded_minute"] == pytest.approx(3*60/11)


def test_controls_deterministic_and_exclude_all_visual_event_windows():
    features, peaks = _audio_fixture()
    base = pd.Timestamp("2026-07-14 17:20:00")
    groups = pd.DataFrame({"visual_group_id": ["visual_group_01", "visual_group_02"],
                           "seed_ids": ["seed_01", "seed_02"],
                           "number_of_seed_events": [1,1], "event_types": ["crack","crack"],
                           "group_start": [base, base+pd.Timedelta(hours=1)],
                           "group_end": [base+pd.Timedelta(minutes=2), base+pd.Timedelta(hours=1, minutes=2)]})
    first = choose_control_windows(groups, features)
    second = choose_control_windows(groups, features)
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("matched_visual_group_id").size().to_dict() == {"visual_group_01": 2, "visual_group_02": 2}
    for control in first.itertuples(index=False):
        assert control.control_start.date() == base.date()
        assert control.control_end - control.control_start == pd.Timedelta(minutes=12)
        for group in groups.itertuples(index=False):
            assert not (control.control_start < group.group_end+pd.Timedelta(minutes=5) and
                        control.control_end > group.group_start-pd.Timedelta(minutes=5))


def test_seed01_left_censoring_from_human_table(tmp_path):
    table = initial_events()
    for index, row in table.iterrows():
        table.at[index, "event_type"] = "crack"
        table.at[index, "last_negative_timestamp"] = row.coarse_start_timestamp
        table.at[index, "first_positive_timestamp"] = str(pd.Timestamp(row.coarse_start_timestamp)+pd.Timedelta(seconds=20))
    path = tmp_path / "events.csv"
    table.to_csv(path, index=False)
    events = load_visual_intervals(path)
    seed01 = events.loc[events.seed_id == "seed_01"].iloc[0]
    assert bool(seed01.left_censored)
    assert seed01.visual_timing_quality == "limited_possible_left_censoring"


def test_later_visual_event_type_uses_same_interval_loader(tmp_path):
    table = initial_events().iloc[[0]].copy()
    table.loc[:, "event_type"] = "major_opening"
    table.loc[:, "last_negative_timestamp"] = "2026-07-14 17:10:00"
    table.loc[:, "first_positive_timestamp"] = "2026-07-14 17:10:30"
    path = tmp_path / "later_events.csv"
    table.to_csv(path, index=False)
    loaded = load_visual_intervals(path)
    assert loaded.event_type.iloc[0] == "major_opening"
    assert loaded.duration_seconds.iloc[0] == 30


def test_closest_peak_signed_lag():
    _, peaks = _audio_fixture()
    start = pd.Timestamp("2026-07-14 17:00:40")
    end = start+pd.Timedelta(seconds=10)
    closest = closest_peak(start, end, peaks)
    assert closest["closest_peak_relationship"] == "before"
    assert closest["closest_peak_lag_seconds"] == -8


def test_post_event_windows_are_separate_from_symmetric_windows():
    features, peaks = _audio_fixture()
    base = pd.Timestamp("2026-07-14 17:00:00")
    events = pd.DataFrame({"seed_id": ["seed_01"], "event_type": ["crack"],
                           "start": [base + pd.Timedelta(seconds=20)],
                           "end": [base + pd.Timedelta(seconds=40)],
                           "duration_seconds": [20.0],
                           "visual_timing_quality": ["reviewed_interval"],
                           "left_censored": [False]})
    groups = group_visual_intervals(events)
    result = build_alignment(events, groups, features, peaks)
    seed_windows = result["windows"].loc[result["windows"].entity_type == "seed"].set_index("window_scale")
    assert {"exact", "30s", "1min", "5min", "post_30s", "post_1min", "post_5min"} <= set(seed_windows.index)
    assert seed_windows.loc["post_30s", "window_start"] == base + pd.Timedelta(seconds=40)
    assert seed_windows.loc["post_30s", "window_end"] == base + pd.Timedelta(seconds=70)
    assert seed_windows.loc["post_30s", "peak_count"] == 1  # Peak at 61 s only.
    assert result["group"].iloc[0].candidate_peaks_post_30s == 1
