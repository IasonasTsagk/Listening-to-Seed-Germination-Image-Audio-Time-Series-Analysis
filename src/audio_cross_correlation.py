"""Streaming, minimally processed waveform cross-correlation.

Primary processing: original stereo PCM -> per-channel DC removal in each
comparison window -> FFT numerator / exact running-energy denominator.
No filtering, denoising, envelope matching or source attribution is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import os
import wave

import numpy as np
from scipy import fft
from scipy.io import wavfile

from .audio_io import parse_audio_timestamp

REFERENCE_NAME = "er_file_2026_07_14_19_23_28.wav"
REFERENCE_PEAK_SECONDS = 0.2225
PRIMARY_SECONDS = 0.30
EXPECTED_RATE = 48_000
EXPECTED_CHANNELS = 2
EXPECTED_DURATION = 2.75


def discover_wavs(root: Path) -> list[tuple[datetime | None, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for directory, _, names in os.walk(root):
        for name in names:
            if Path(name).suffix.lower() == ".wav" and not name.startswith("._"):
                rows.append((parse_audio_timestamp(name), Path(directory) / name))
    return sorted(rows, key=lambda item: (item[0] is None, item[0] or datetime.max, str(item[1])))


def read_original_float(path: Path) -> tuple[int, np.ndarray]:
    rate, data = wavfile.read(path)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("Expected non-empty multi-channel WAV")
    if np.issubdtype(data.dtype, np.integer):
        scale = np.iinfo(data.dtype).max
        samples = data.astype(np.float32) / scale
    elif np.issubdtype(data.dtype, np.floating):
        samples = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV dtype: {data.dtype}")
    if not np.isfinite(samples).all():
        raise ValueError("WAV contains non-finite samples")
    return rate, samples


def channel_diagnostics(samples: np.ndarray, rate: int, reference_crop: tuple[int, int] | None = None) -> dict:
    if samples.shape[1] != 2:
        raise ValueError("Stereo diagnostics require two channels")
    left, right = samples.T.astype(np.float64)
    rms = [float(np.sqrt(np.mean(channel**2))) for channel in (left, right)]
    peak = [float(np.max(np.abs(channel))) for channel in (left, right)]
    result = {"channel_pearson_correlation": float(np.corrcoef(left, right)[0, 1]) if min(np.std(left), np.std(right)) > 0 else np.nan,
              "left_rms": rms[0], "right_rms": rms[1],
              "left_peak_amplitude": peak[0], "right_peak_amplitude": peak[1],
              "left_peak_time_seconds": float(np.argmax(np.abs(left))/rate),
              "right_peak_time_seconds": float(np.argmax(np.abs(right))/rate),
              "peak_timing_difference_seconds": float(abs(np.argmax(np.abs(left))-np.argmax(np.abs(right)))/rate)}
    if reference_crop is not None:
        start, end = reference_crop
        section = samples[start:end].astype(np.float64)
        result["crop_channel_pearson_correlation"] = float(np.corrcoef(section.T)[0, 1])
        result["crop_left_rms"] = float(np.sqrt(np.mean(section[:, 0]**2)))
        result["crop_right_rms"] = float(np.sqrt(np.mean(section[:, 1]**2)))
    return result


@dataclass
class Template:
    samples: np.ndarray  # shape (channels, template_samples), zero mean per channel
    rate: int
    start_seconds: float
    end_seconds: float
    peak_seconds: float

    @property
    def length(self) -> int:
        return self.samples.shape[1]

    @property
    def rms(self) -> np.ndarray:
        return np.sqrt(np.mean(self.samples.astype(np.float64)**2, axis=1))


def make_template(samples: np.ndarray, rate: int, length_seconds: float = PRIMARY_SECONDS,
                  peak_seconds: float = REFERENCE_PEAK_SECONDS) -> Template:
    length = round(length_seconds*rate)
    start = round(peak_seconds*rate)-length//2
    end = start+length
    if start < 0 or end > len(samples):
        raise ValueError("Reference template extends outside WAV")
    crop = samples[start:end].T.astype(np.float32, copy=True)
    crop -= crop.mean(axis=1, keepdims=True)
    if np.any(np.sum(crop.astype(np.float64)**2, axis=1) <= 0):
        raise ValueError("Reference template has zero energy")
    return Template(crop, rate, start/rate, end/rate, peak_seconds)


class FFTZNCC:
    """Reuse template FFT for same-sized WAVs; score both channels separately."""

    def __init__(self, template: Template):
        self.template = template
        self._cached_n = None
        self._template_fft = None
        self._nfft = None
        self.template_energy = np.sum(template.samples.astype(np.float64)**2, axis=1)

    def _prepare(self, signal_length: int) -> None:
        if self._cached_n != signal_length:
            self._nfft = fft.next_fast_len(signal_length+self.template.length-1)
            self._template_fft = np.conj(fft.rfft(self.template.samples, self._nfft, axis=1))
            self._cached_n = signal_length

    def compare(self, samples: np.ndarray) -> dict:
        if samples.ndim != 2 or samples.shape[1] != self.template.samples.shape[0]:
            raise ValueError("Candidate channel count differs from template")
        n, channels = samples.shape
        m = self.template.length
        if n < m:
            return {"valid_comparison": False, "warning": "shorter_than_template"}
        self._prepare(n)
        signals = samples.T.astype(np.float32, copy=False)
        numerator = fft.irfft(fft.rfft(signals, self._nfft, axis=1)*self._template_fft,
                             self._nfft, axis=1)[:, :n-m+1].astype(np.float64)
        results = []
        low_energy_count = 0
        for channel in range(channels):
            x = signals[channel].astype(np.float64)
            sums = np.empty(n+1, dtype=np.float64)
            squares = np.empty(n+1, dtype=np.float64)
            sums[0] = squares[0] = 0
            np.cumsum(x, out=sums[1:])
            np.cumsum(x*x, out=squares[1:])
            window_sum = sums[m:]-sums[:-m]
            window_square = squares[m:]-squares[:-m]
            centered_energy = np.maximum(window_square-window_sum**2/m, 0)
            # Numerical precision guard only; not a biological amplitude cutoff.
            energy_floor = 100*np.finfo(np.float64).eps*max(1.0, float(window_square.max()))
            valid = centered_energy > energy_floor
            low_energy_count += int((~valid).sum())
            if not valid.any():
                results.append(None)
                continue
            zncc = np.full(len(window_square), np.nan, dtype=np.float64)
            zncc[valid] = numerator[channel, valid] / np.sqrt(
                self.template_energy[channel]*centered_energy[valid])
            zncc[valid] = np.clip(zncc[valid], -1, 1)
            positive_index = int(np.nanargmax(zncc))
            negative_index = int(np.nanargmin(zncc))
            local_rms = float(np.sqrt(window_square[positive_index]/m))
            results.append({"max": float(zncc[positive_index]), "lag": positive_index,
                            "min": float(zncc[negative_index]), "negative_lag": negative_index,
                            "local_rms": local_rms})
        if not any(result is not None for result in results):
            return {"valid_comparison": False, "warning": "all_windows_near_zero_energy",
                    "near_zero_energy_window_count": low_energy_count}
        best_channel = max((i for i, result in enumerate(results) if result is not None),
                           key=lambda i: results[i]["max"])
        best = results[best_channel]
        return {"valid_comparison": True, "warning": "near_zero_energy_windows_skipped" if low_energy_count else "",
                "near_zero_energy_window_count": low_energy_count,
                "max_zncc": best["max"], "lag_seconds": best["lag"]/self.template.rate,
                "local_candidate_RMS": best["local_rms"],
                "reference_RMS": float(self.template.rms[best_channel]),
                "channel": "left" if best_channel == 0 else "right",
                "channel_method": "separate_channels_max_positive",
                "left_max_zncc": results[0]["max"] if results[0] is not None else np.nan,
                "right_max_zncc": results[1]["max"] if results[1] is not None else np.nan,
                "left_lag_seconds": results[0]["lag"]/self.template.rate if results[0] is not None else np.nan,
                "right_lag_seconds": results[1]["lag"]/self.template.rate if results[1] is not None else np.nan,
                "strongest_negative_zncc": min(result["min"] for result in results if result is not None)}


def score_wav(path: Path, root: Path, engine: FFTZNCC, reference_path: Path) -> dict:
    relative = str(path.relative_to(root))
    stamp = parse_audio_timestamp(path.name)
    result = {"filename": path.name, "relative_path": relative,
              "wav_start_timestamp": stamp.isoformat(sep=" ") if stamp else "",
              "is_reference_file": path == reference_path,
              "valid_comparison": False, "warning": ""}
    try:
        rate, samples = read_original_float(path)
        result["sample_rate_hz"] = rate
        result["channels"] = samples.shape[1]
        result["duration_seconds"] = samples.shape[0]/rate
        if rate != engine.template.rate or samples.shape[1] != engine.template.samples.shape[0]:
            result["warning"] = "incompatible_sample_rate_or_channels"
            return result
        score = engine.compare(samples)
        result.update(score)
        if abs(result["duration_seconds"]-EXPECTED_DURATION) > 1/rate:
            result["warning"] = (result.get("warning", "") + ";unexpected_duration").strip(";")
        if stamp and result.get("valid_comparison"):
            result["match_window_start_timestamp"] = (stamp+timedelta(seconds=result["lag_seconds"])).isoformat(sep=" ", timespec="microseconds")
            result["absolute_match_timestamp"] = (stamp+timedelta(seconds=result["lag_seconds"]+
                engine.template.peak_seconds-engine.template.start_seconds)).isoformat(sep=" ", timespec="microseconds")
    except Exception as error:
        result["warning"] = f"{type(error).__name__}: {error}"
    return result
