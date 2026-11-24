from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dense_visual import initial_events
from src.phase_d_visual import initialize_milestones, save_visual_milestone
from src.phase_d_audio import (SIMILARITY_FEATURES, build_candidate_rows,
    blind_identifiers, prepare_blind_review, preserve_manual_candidate_fields,
    rank_candidates, reference_similarity,
    save_blind_review, top_distinct_wavs)
from src.phase_c_alignment import window_metrics


def test_milestone_import_preserves_phase_b_and_human_later_rows(tmp_path):
    phase_b = initial_events()
    for i in phase_b.index:
        phase_b.loc[i, "event_type"] = "crack"
        phase_b.loc[i, "last_negative_timestamp"] = "2026-07-14 17:00:00"
        phase_b.loc[i, "first_positive_timestamp"] = "2026-07-14 17:00:20"
    source, target = tmp_path / "phase_b.csv", tmp_path / "milestones.csv"
    phase_b.to_csv(source, index=False)
    before = source.read_bytes()
    first = initialize_milestones(source, target)
    assert len(first) == 5 and set(first.event_type) == {"crack_onset"}
    later = {"seed_id": "seed_01", "event_type": "major_opening",
             "last_negative_timestamp": "2026-07-14 17:01:00",
             "first_positive_timestamp": "2026-07-14 17:01:30", "interval_seconds": 30,
             "visual_confidence": "medium", "notes": "human", "source": "phase_d"}
    pd.concat([first, pd.DataFrame([later])], ignore_index=True).to_csv(target, index=False)
    again = initialize_milestones(source, target)
    assert len(again) == 6 and again.iloc[-1].notes == "human"
    assert source.read_bytes() == before
    assert again.loc[again.seed_id.eq("seed_01") & again.event_type.eq("crack_onset"),
                     "first_positive_timestamp"].iloc[0] == "2026-07-14 17:00:20"


def test_later_milestone_requires_ordered_inventory_frames(tmp_path):
    source = tmp_path / "phase_b.csv"
    base = initial_events()
    for i in base.index:
        base.loc[i, "event_type"] = "crack"
        base.loc[i, "last_negative_timestamp"] = "2026-07-14 17:00:00"
        base.loc[i, "first_positive_timestamp"] = "2026-07-14 17:00:20"
    base.to_csv(source, index=False)
    target = tmp_path / "milestones.csv"
    initialize_milestones(source, target)
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame({"window_id": ["WINDOW_A"]*4,
                  "timestamp": pd.date_range("2026-07-14 17:00:00", periods=4, freq="10s")}).to_csv(inventory, index=False)
    with pytest.raises(ValueError):
        save_visual_milestone(target, inventory, "seed_01", "crack_onset",
                              "2026-07-14 17:00:20", "2026-07-14 17:00:30", "high", "")
    with pytest.raises(ValueError):
        save_visual_milestone(target, inventory, "seed_01", "major_opening",
                              "2026-07-14 17:00:30", "2026-07-14 17:00:20", "high", "")
    save_visual_milestone(target, inventory, "seed_01", "major_opening",
                          "2026-07-14 17:00:20", "2026-07-14 17:00:30", "high", "human")
    saved = pd.read_csv(target, keep_default_na=False)
    assert saved.loc[saved.event_type.eq("major_opening"), "interval_seconds"].iloc[0] == 10
    assert saved.loc[saved.event_type.eq("crack_onset") & saved.seed_id.eq("seed_01"),
                     "first_positive_timestamp"].iloc[0] == "2026-07-14 17:00:20"
    assert len(list(tmp_path.glob("milestones.backup_*.csv"))) == 1


def test_reference_similarity_is_deterministic_and_equal_weight():
    population = pd.DataFrame({key: np.linspace(1, 10, 30) for key in SIMILARITY_FEATURES})
    reference = population.iloc[[12]]
    candidates = population.iloc[[12, 0, 29]]
    scores, settings = reference_similarity(population, candidates, reference)
    assert scores[0] == pytest.approx(1)
    assert scores[0] > scores[1] and scores[0] > scores[2]
    assert scores.tolist() == reference_similarity(population, candidates, reference)[0].tolist()
    assert len(settings["features"]) == len(SIMILARITY_FEATURES)


def test_deterministic_rank_and_manual_candidate_field_preservation(tmp_path):
    population = pd.DataFrame({key: np.linspace(1, 10, 30) for key in SIMILARITY_FEATURES})
    candidates = population.iloc[[12, 0]].rename(columns={"prominence": "peak_prominence",
        "width_seconds": "peak_width_seconds", "rms_energy": "rms"}).copy()
    candidates["visual_group_id"] = "group_01"
    candidates["wav_relative_path"] = ["a.wav", "b.wav"]
    candidates["peak_timestamp"] = pd.to_datetime(["2026-07-14 17:00:01", "2026-07-14 17:00:02"])
    candidates["is_control"] = [False, True]
    candidates["candidate_id"] = ["candidate_a", "candidate_b"]
    ranked, _ = rank_candidates(candidates, population, population.iloc[[12]])
    assert ranked.reference_similarity_rank.tolist() == [1, 2]
    path = tmp_path / "candidates.csv"
    ranked.loc[0, "manual_review_label"] = "uncertain"
    ranked.loc[0, "manual_review_confidence"] = "low"
    ranked.loc[0, "notes"] = "human"
    ranked.to_csv(path, index=False)
    again, _ = rank_candidates(candidates, population, population.iloc[[12]])
    kept = preserve_manual_candidate_fields(again, path)
    assert kept.loc[0, ["manual_review_label", "manual_review_confidence", "notes"]].tolist() == ["uncertain", "low", "human"]


def test_no_recording_is_unknown_not_silence():
    start = pd.Timestamp("2026-07-14 17:00:00")
    features = pd.DataFrame({"start": pd.to_datetime([]), "end": pd.to_datetime([])})
    peaks = pd.DataFrame({"timestamp": pd.to_datetime([]), "high": pd.Series(dtype=bool),
                          "prominence": pd.Series(dtype=float)})
    result = window_metrics(start, start+pd.Timedelta(minutes=1), features, peaks)
    assert result["audio_coverage_class"] == "none"
    assert np.isnan(result["peaks_per_recorded_minute"])


def test_peak_event_control_separation_and_shared_peak_deduplication():
    base = pd.Timestamp("2026-07-14 17:00:00")
    groups = pd.DataFrame({"visual_group_id": ["group_01", "group_02"],
                           "seed_ids": ["seed_01", "seed_02"], "event_types": ["crack_onset"]*2,
                           "group_start": [base, base+pd.Timedelta(minutes=8)],
                           "group_end": [base+pd.Timedelta(seconds=30), base+pd.Timedelta(minutes=8, seconds=30)]})
    controls = pd.DataFrame({"control_id": ["control_01"], "matched_visual_group_id": ["group_01"],
                             "control_start": [base+pd.Timedelta(minutes=20)],
                             "control_end": [base+pd.Timedelta(minutes=21)]})
    times = [base+pd.Timedelta(minutes=4), base+pd.Timedelta(minutes=20, seconds=10)]
    peaks = pd.DataFrame({"timestamp": times, "filename": ["a.wav", "b.wav"],
                          "source_relative_path": ["a.wav", "b.wav"], "start": times,
                          "prominence": [.01, .02], "width_seconds": [.05, .05]})
    for key in SIMILARITY_FEATURES:
        if key not in peaks: peaks[key] = 1.0
    rows = build_candidate_rows(groups, controls, peaks)
    assert len(rows) == 2  # The 17:04 peak lies in two expanded event windows, assigned once.
    assert rows.is_control.sum() == 1
    assert rows.loc[~rows.is_control, "visual_group_id"].iloc[0] == "group_01"


def test_blind_ids_mapping_and_manual_label_preservation(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("a.wav", "b.wav"):
        (raw / name).write_bytes(b"wav" + name.encode())
    selected = pd.DataFrame({"candidate_id": ["candidate_a", "candidate_b"],
                             "is_control": [False, True], "visual_group_id": ["group_01"]*2,
                             "seed_ids": ["seed_01"]*2, "visual_event_type": ["crack_onset"]*2,
                             "wav_filename": ["a.wav", "b.wav"], "wav_relative_path": ["a.wav", "b.wav"],
                             "wav_timestamp": ["2026-07-14 17:00:00"]*2,
                             "lag_seconds": [0, np.nan], "reference_similarity_score": [.9, .8],
                             "peak_timestamp": ["2026-07-14 17:00:01"]*2})
    manifest, mapping, blind = prepare_blind_review(selected, raw, tmp_path / "copies")
    assert len(set(blind.blind_id)) == 2
    assert blind.blind_id.tolist() == blind_identifiers(2)
    assert set(mapping.event_or_control) == {"event", "control"}
    assert "event_or_control" not in blind.columns
    assert all(Path(path).is_file() for path in blind.listening_path)
    assert (raw / "a.wav").read_bytes() == b"wava.wav"
    manifest_path, blind_path = tmp_path / "manifest.csv", tmp_path / "blind.csv"
    manifest.to_csv(manifest_path, index=False); blind.to_csv(blind_path, index=False)
    save_blind_review(blind_path, manifest_path, blind.blind_id.iloc[0], "uncertain", "low", "human")
    assert pd.read_csv(manifest_path, keep_default_na=False).manual_audio_label.iloc[0] == "uncertain"
    assert pd.read_csv(blind_path, keep_default_na=False).notes.iloc[0] == "human"
    with pytest.raises(ValueError):
        save_blind_review(blind_path, manifest_path, blind.blind_id.iloc[0], "germination_sound", "high", "")
    mapping_path = tmp_path / "mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    regenerated, remap, _ = prepare_blind_review(selected, raw, tmp_path / "copies",
                                                  manifest_path, mapping_path)
    assert remap.set_index("wav_relative_path").blind_id.to_dict() == mapping.set_index("wav_relative_path").blind_id.to_dict()
    assert regenerated.loc[regenerated.wav_relative_path.eq("a.wav"), "manual_audio_label"].iloc[0] == "uncertain"
    assert len(top_distinct_wavs(pd.DataFrame({"visual_group_id": ["group_01"]*2,
        "is_control": [False]*2, "reference_similarity_score": [.9, .8],
        "peak_timestamp": ["a", "b"], "candidate_id": ["x", "y"],
        "wav_relative_path": ["same.wav"]*2}), "group_01", False)) == 1
