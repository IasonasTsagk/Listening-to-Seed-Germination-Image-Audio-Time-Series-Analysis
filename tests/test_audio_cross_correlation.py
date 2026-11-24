from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio_cross_correlation import FFTZNCC, make_template, read_original_float, score_wav
from src.audio_io import parse_audio_timestamp


def _synthetic():
    rng = np.random.default_rng(17)
    rate = 8000
    signal = rng.normal(0, .02, 4000).astype(np.float32)
    signal[1700:2000] += .45*np.hanning(300).astype(np.float32)
    stereo = np.column_stack([signal, signal*.94]).astype(np.float32)
    template = make_template(stereo, rate, .08, .22)
    return rate, stereo, template


def test_self_scaled_offset_and_lag_recovery():
    rate, stereo, template = _synthetic()
    engine = FFTZNCC(template)
    own = engine.compare(stereo)
    assert own["max_zncc"] == pytest.approx(1, abs=2e-5)
    assert own["lag_seconds"] == pytest.approx(template.start_seconds, abs=1/rate)
    scaled_offset = stereo*.3 + .15
    result = engine.compare(scaled_offset)
    assert result["max_zncc"] == pytest.approx(1, abs=2e-5)
    assert result["lag_seconds"] == pytest.approx(template.start_seconds, abs=1/rate)
    shifted = np.zeros_like(stereo)
    shifted[500:] = stereo[:-500]
    recovered = engine.compare(shifted)
    assert recovered["max_zncc"] == pytest.approx(1, abs=2e-5)
    assert recovered["lag_seconds"] == pytest.approx(template.start_seconds+500/rate, abs=1/rate)


def test_unrelated_noise_lower_and_silence_safe():
    rate, stereo, template = _synthetic()
    engine = FFTZNCC(template)
    noise = np.random.default_rng(99).normal(0, .1, stereo.shape).astype(np.float32)
    assert engine.compare(noise)["max_zncc"] < .5
    silent = engine.compare(np.zeros_like(stereo))
    assert silent["valid_comparison"] is False
    assert "all_windows_near_zero_energy" in silent["warning"]
    assert not any(np.isinf(v) for v in silent.values() if isinstance(v, float))


def test_timestamp_reference_flag_and_raw_unchanged(tmp_path):
    rate, stereo, template = _synthetic()
    root = tmp_path / "raw"
    root.mkdir()
    path = root / "er_file_2026_07_14_19_23_28.wav"
    wavfile.write(path, rate, stereo)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded_rate, loaded = read_original_float(path)
    assert loaded_rate == rate and loaded.shape == stereo.shape
    result = score_wav(path, root, FFTZNCC(template), path)
    assert result["is_reference_file"] is True
    assert result["valid_comparison"] is True
    assert result["max_zncc"] == pytest.approx(1, abs=2e-5)
    assert result["wav_start_timestamp"] == "2026-07-14 19:23:28"
    assert pd.Timestamp(result["absolute_match_timestamp"]) == pd.Timestamp("2026-07-14 19:23:28.220000")
    assert parse_audio_timestamp(path.name).isoformat() == "2026-07-14T19:23:28"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    # The external ranking explicitly excludes self-matches.
    frame = pd.DataFrame([result, {**result, "filename": "other.wav", "is_reference_file": False}])
    assert frame.loc[frame.valid_comparison & ~frame.is_reference_file].filename.tolist() == ["other.wav"]
