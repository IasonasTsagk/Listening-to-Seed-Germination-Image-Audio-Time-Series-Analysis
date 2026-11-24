"""Small, read-only acoustic inspection set for Phase C2 human review.

Measurements use original WAVs. Amplitude-normalized copies are listening aids
only, never inputs to the metrics or interpretation.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
from scipy.io import wavfile

from .audio_features import basic_features, read_wav_float, rms_envelope

ALLOWED_LABELS = (
    "low_activity", "obvious_speech_or_household", "impact_or_handling",
    "mechanical_or_continuous", "short_transient_unexplained", "uncertain",
)
ALLOWED_CONFIDENCE = ("high", "medium", "low")
REVIEW_COLUMNS = ("manual_audio_label", "manual_audio_confidence", "notes")


def review_identifier(index: int) -> str:
    return f"review_{index + 1:02d}"


def select_adjacent_recordings(features: pd.DataFrame, relative_path: str, count: int = 2) -> pd.DataFrame:
    """Select recorded neighbors by timestamp; expose the actual unrecorded gaps."""
    ordered = features.sort_values(["start", "relative_path"]).reset_index(drop=True)
    positions = ordered.index[ordered.relative_path.eq(relative_path)].tolist()
    if len(positions) != 1:
        raise ValueError(f"Expected one audio inventory row for {relative_path}")
    target_index = positions[0]
    indices = range(max(0, target_index-count), min(len(ordered), target_index+count+1))
    target_start = ordered.loc[target_index, "start"]
    rows = []
    previous_end = None
    for position in indices:
        row = ordered.loc[position]
        offset = position-target_index
        rows.append({"role": "target" if offset == 0 else f"previous_{-offset}" if offset < 0 else f"next_{offset}",
                     "wav_relative_path": row.relative_path, "wav_filename": Path(row.relative_path).name,
                     "start_timestamp": row.start, "end_timestamp": row.end,
                     "start_lag_from_target_seconds": float((row.start-target_start).total_seconds()),
                     "unrecorded_gap_from_previous_seconds": float((row.start-previous_end).total_seconds()) if previous_end is not None else np.nan})
        previous_end = row.end
    return pd.DataFrame(rows)


def normalize_for_listening(samples: np.ndarray, target_peak: float = 0.90) -> np.ndarray:
    """Apply one scalar gain to all channels, preserving channel balance."""
    if not 0 < target_peak < 1:
        raise ValueError("target_peak must be strictly between zero and one")
    peak = float(np.max(np.abs(samples)))
    if not np.isfinite(peak):
        raise ValueError("Audio contains non-finite samples")
    if peak == 0:
        return np.zeros_like(samples, dtype=np.float32)
    return (samples * (target_peak / peak)).astype(np.float32)


def write_listening_copy(source: Path, destination: Path) -> Path:
    if source.resolve() == destination.resolve():
        raise ValueError("Listening copy cannot overwrite its source")
    rate, samples = read_wav_float(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination, rate, normalize_for_listening(samples))
    return destination


def analyze_original_wav(source: Path, peak_rows: pd.DataFrame) -> tuple[dict, dict]:
    """Return original-signal metrics and plotting arrays, never normalized data."""
    rate, samples = read_wav_float(source)
    mono = samples.mean(axis=1)
    base = basic_features(mono, rate)
    envelope_time, envelope = rms_envelope(mono, rate, .025, .010, 3)
    ordered = peak_rows.sort_values(["prominence", "relative_peak_time_seconds"], ascending=[False, True])
    strongest = ordered.iloc[0] if len(ordered) else None
    metrics = {"sample_rate_hz": rate, "channel_count": samples.shape[1],
               "duration_seconds": len(samples)/rate,
               "rms": base["rms_energy"],
               "max_abs_amplitude": base["peak_absolute_amplitude"],
               "mean_abs_amplitude": base["mean_absolute_amplitude"],
               "standard_deviation": base["standard_deviation"],
               "zero_crossing_rate_hz": base["zero_crossing_rate_hz"],
               "spectral_centroid_hz": base["spectral_centroid_hz"],
               "spectral_bandwidth_hz": base["spectral_bandwidth_hz"],
               "spectral_rolloff_hz": base["spectral_rolloff_hz"],
               "dominant_frequency_hz": base["dominant_frequency_hz"],
               "candidate_peak_count": len(peak_rows),
               "strongest_peak_prominence": float(strongest.prominence) if strongest is not None else np.nan,
               "strongest_peak_timestamp": strongest.absolute_peak_timestamp if strongest is not None else "",
               "strongest_peak_relative_seconds": float(strongest.relative_peak_time_seconds) if strongest is not None else np.nan,
               "strongest_peak_width_seconds": float(strongest.width_seconds) if strongest is not None else np.nan}
    plotting = {"rate": rate, "mono": mono, "samples": samples,
                "envelope_time": envelope_time, "envelope": envelope,
                "peaks": peak_rows.copy()}
    return metrics, plotting


def merge_review_fields(new: pd.DataFrame, existing: pd.DataFrame, key: str = "review_id") -> pd.DataFrame:
    """Keep old human entries when regenerating measurements; never discard reviewed rows."""
    if existing.empty:
        return new
    if existing[key].duplicated().any() or new[key].duplicated().any():
        raise ValueError("Duplicate review identifiers")
    old = existing.set_index(key)
    out = new.copy()
    for index, row in out.iterrows():
        if row[key] in old.index:
            for field in REVIEW_COLUMNS:
                prior = old.loc[row[key], field]
                if pd.notna(prior) and str(prior) != "" and str(row[field]) == "":
                    out.at[index, field] = prior
    missing = old.loc[~old.index.isin(out[key])]
    reviewed = missing.loc[(missing[list(REVIEW_COLUMNS)].fillna("").astype(str) != "").any(axis=1)]
    if len(reviewed):
        out = pd.concat([out, reviewed.reset_index()], ignore_index=True)
    return out


def backup_csv(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def save_manual_review(manifest_path: Path, metrics_path: Path, review_id: str,
                       label: str, confidence: str, notes: str) -> None:
    """Save one human decision to both tables after backing up existing CSVs."""
    if label not in ("", *ALLOWED_LABELS) or confidence not in ("", *ALLOWED_CONFIDENCE):
        raise ValueError("Use only the documented neutral labels and confidence levels")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    metrics = pd.read_csv(metrics_path, keep_default_na=False)
    rows = metrics.loc[metrics.review_id.eq(review_id)]
    if len(rows) != 1:
        raise ValueError(f"Unknown or duplicate review ID: {review_id}")
    key = (rows.iloc[0].visual_group_id, rows.iloc[0].wav_relative_path)
    match = manifest.visual_group_id.eq(key[0]) & manifest.wav_relative_path.eq(key[1])
    if match.sum() != 1:
        raise ValueError("Review ID does not identify exactly one manifest row")
    for frame, mask in ((manifest, match), (metrics, metrics.review_id.eq(review_id))):
        frame.loc[mask, "manual_audio_label"] = label
        frame.loc[mask, "manual_audio_confidence"] = confidence
        frame.loc[mask, "notes"] = notes
    for path, frame in ((manifest_path, manifest), (metrics_path, metrics)):
        backup_csv(path)
        temporary = path.with_name(path.name + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    write_review_summary(metrics_path, metrics_path.with_name("phase_c2_review_summary.json"))


def review_summary(metrics: pd.DataFrame) -> dict:
    labels = metrics.manual_audio_label.fillna("").astype(str)
    counts = {label: int(labels.eq(label).sum()) for label in ALLOWED_LABELS}
    lag = pd.to_numeric(metrics.lag_seconds, errors="coerce")
    control = metrics.relationship_to_visual_event.fillna("").str.contains("control", case=False)
    window_masks = {"inside_visual_interval": lag.eq(0) & ~control,
                    "within_30_seconds": lag.abs().le(30) & ~control,
                    "within_1_minute": lag.abs().le(60) & ~control,
                    "within_5_minutes": lag.abs().le(300) & ~control,
                    "control": control}
    category_by_window = {
        window: {label: int(labels.loc[mask].eq(label).sum()) for label in ALLOWED_LABELS}
        for window, mask in window_masks.items()
    }
    return {"total_review_wavs": len(metrics), "labeled_review_wavs": int(labels.ne("").sum()),
            "unlabeled_review_wavs": int(labels.eq("").sum()), "label_counts": counts,
            "window_label_counts": category_by_window,
            "window_rule": "Cumulative signed-lag windows based on the selected candidate peak; controls are separate. A WAV can appear in multiple windows.",
            "caution": "Neutral human acoustic descriptions only; no source or biological causation established."}


def write_review_summary(metrics_path: Path, output_path: Path) -> dict:
    summary = review_summary(pd.read_csv(metrics_path, keep_default_na=False))
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
