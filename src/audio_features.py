"""Lightweight per-snippet acoustic features; raw audio remains untouched."""
from __future__ import annotations
import numpy as np
from scipy import signal
from scipy.io import wavfile

def read_wav_float(path):
    rate, data = wavfile.read(path)
    if data.size == 0: raise ValueError("Empty WAV")
    if np.issubdtype(data.dtype, np.integer): data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else: data = data.astype(np.float64)
    return rate, data if data.ndim == 2 else data[:, None]

def channel_diagnostics(samples: np.ndarray) -> dict:
    left, right = samples[:,0], samples[:, min(1, samples.shape[1]-1)]
    metrics = {}
    for name, x in (("left",left),("right",right)):
        metrics.update({f"{name}_rms": float(np.sqrt(np.mean(x*x))), f"{name}_peak_abs": float(np.max(np.abs(x))), f"{name}_mean_abs":float(np.mean(np.abs(x))), f"{name}_std":float(np.std(x))})
    metrics["channel_correlation"] = float(np.corrcoef(left,right)[0,1]) if len(left)>1 and np.std(left)>0 and np.std(right)>0 else np.nan
    metrics["rms_ratio_right_to_left"] = metrics["right_rms"] / metrics["left_rms"] if metrics["left_rms"] else np.nan
    return metrics

def zero_crossing_rate(x, rate): return float(np.count_nonzero(np.diff(np.signbit(x))) / len(x) * rate) if len(x) else np.nan
def rms_envelope(x, rate, window_seconds, hop_seconds, smoothing_frames=1):
    width, hop = max(1,round(rate*window_seconds)), max(1,round(rate*hop_seconds)); starts=np.arange(0,max(1,len(x)-width+1),hop)
    values=np.array([np.sqrt(np.mean(x[i:i+width]**2)) for i in starts]);
    if smoothing_frames>1: values=signal.medfilt(values, kernel_size=smoothing_frames if smoothing_frames%2 else smoothing_frames+1)
    return starts/rate + width/(2*rate), values
def basic_features(x, rate):
    freqs, power = signal.periodogram(x, rate); total=power.sum(); centroid=float((freqs*power).sum()/total) if total else np.nan
    bandwidth=float(np.sqrt((((freqs-centroid)**2)*power).sum()/total)) if total else np.nan
    csum=np.cumsum(power); rolloff=float(freqs[np.searchsorted(csum, .85*total)]) if total else np.nan
    return {"rms_energy":float(np.sqrt(np.mean(x*x))), "peak_absolute_amplitude":float(np.max(np.abs(x))), "mean_absolute_amplitude":float(np.mean(np.abs(x))), "standard_deviation":float(np.std(x)), "zero_crossing_rate_hz":zero_crossing_rate(x,rate), "spectral_centroid_hz":centroid, "spectral_bandwidth_hz":bandwidth, "spectral_rolloff_hz":rolloff, "dominant_frequency_hz":float(freqs[np.argmax(power)]) if total else np.nan, "spectral_flatness":float(np.exp(np.mean(np.log(power+1e-15)))/np.mean(power+1e-15)) if len(power) else np.nan}
