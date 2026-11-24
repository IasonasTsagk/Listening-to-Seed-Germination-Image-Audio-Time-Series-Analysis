from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.phase_c2_audio_review import (analyze_original_wav, merge_review_fields,
    normalize_for_listening, review_summary, save_manual_review,
    select_adjacent_recordings, write_listening_copy)


def test_normalized_copy_does_not_overwrite_source_and_avoids_clipping(tmp_path):
    rate = 48000
    source = tmp_path / "original.wav"
    samples = np.tile(np.array([[100, -200], [50, -100]], dtype=np.int16), (2400, 1))
    wavfile.write(source, rate, samples)
    before = source.read_bytes()
    output = tmp_path / "original_normalized_for_listening.wav"
    write_listening_copy(source, output)
    assert source.read_bytes() == before
    out_rate, normalized = wavfile.read(output)
    assert out_rate == rate
    assert normalized.shape == samples.shape
    assert np.max(np.abs(normalized)) <= .900001
    assert np.max(np.abs(normalized)) == pytest.approx(.9)
    with pytest.raises(ValueError):
        write_listening_copy(source, source)
    assert np.max(np.abs(normalize_for_listening(np.zeros((10, 2))))) == 0


def test_original_measurements_do_not_use_listening_copy(tmp_path):
    rate = 48000
    source, normalized = tmp_path / "original.wav", tmp_path / "listening.wav"
    samples = (np.sin(np.arange(4800)*2*np.pi*500/rate)*400).astype(np.int16)
    wavfile.write(source, rate, samples)
    peaks = pd.DataFrame(columns=["relative_peak_time_seconds", "prominence",
                                  "absolute_peak_timestamp", "width_seconds"])
    original_before, _ = analyze_original_wav(source, peaks)
    write_listening_copy(source, normalized)
    original_after, _ = analyze_original_wav(source, peaks)
    listening_metrics, _ = analyze_original_wav(normalized, peaks)
    assert original_before["rms"] == original_after["rms"]
    assert original_before["max_abs_amplitude"] == original_after["max_abs_amplitude"]
    assert listening_metrics["rms"] > original_after["rms"]


def test_adjacent_selection_and_timestamp_gaps():
    start = pd.Timestamp("2026-07-14 17:00:00")
    features = pd.DataFrame({"relative_path": [f"{i}.wav" for i in range(6)],
                             "start": [start+pd.Timedelta(seconds=i*10) for i in range(6)],
                             "end": [start+pd.Timedelta(seconds=i*10+2.75) for i in range(6)]})
    context = select_adjacent_recordings(features, "3.wav")
    assert context.role.tolist() == ["previous_2", "previous_1", "target", "next_1", "next_2"]
    assert context.start_lag_from_target_seconds.tolist() == [-20, -10, 0, 10, 20]
    assert pd.isna(context.iloc[0].unrecorded_gap_from_previous_seconds)
    assert context.unrecorded_gap_from_previous_seconds.iloc[1:].tolist() == [7.25]*4
    assert len(select_adjacent_recordings(features, "0.wav")) == 3


def test_manual_review_preserved_and_summary_counts(tmp_path):
    manifest_path, metrics_path = tmp_path / "manifest.csv", tmp_path / "metrics.csv"
    manifest = pd.DataFrame({"visual_group_id": ["group_01", "group_02"],
                             "wav_relative_path": ["a.wav", "b.wav"],
                             "manual_audio_label": ["", ""],
                             "manual_audio_confidence": ["", ""], "notes": ["", ""]})
    metrics = manifest.copy()
    metrics.insert(0, "review_id", ["review_01", "review_02"])
    metrics["relationship_to_visual_event"] = ["closest_candidate_peak", "representative_low_rms_control"]
    metrics["lag_seconds"] = [0.0, ""]
    manifest.to_csv(manifest_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    save_manual_review(manifest_path, metrics_path, "review_01", "uncertain", "low", "faint event")
    updated = pd.read_csv(metrics_path, keep_default_na=False)
    assert updated.loc[0, ["manual_audio_label", "manual_audio_confidence", "notes"]].tolist() == ["uncertain", "low", "faint event"]
    assert pd.read_csv(manifest_path, keep_default_na=False).loc[0, "notes"] == "faint event"
    assert len(list(tmp_path.glob("*.backup_*.csv"))) == 2
    regenerated = metrics.copy()
    preserved = merge_review_fields(regenerated, updated)
    assert preserved.loc[0, "notes"] == "faint event"
    summary = review_summary(updated)
    assert summary["total_review_wavs"] == 2
    assert summary["labeled_review_wavs"] == 1
    assert summary["label_counts"]["uncertain"] == 1
    assert summary["window_label_counts"]["inside_visual_interval"]["uncertain"] == 1
    assert summary["window_label_counts"]["control"]["uncertain"] == 0
    with pytest.raises(ValueError):
        save_manual_review(manifest_path, metrics_path, "review_02", "germination_sound", "high", "")
