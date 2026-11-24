"""Transparent, exploratory reference similarity for precomputed RMS-envelope peaks."""
from __future__ import annotations

from pathlib import Path
import hashlib
import shutil

import numpy as np
import pandas as pd
from scipy import signal

from .phase_c_alignment import (group_visual_intervals, choose_control_windows,
    window_metrics, signed_lag_seconds, expanded_bounds)
from .phase_c2_audio_review import (ALLOWED_LABELS, ALLOWED_CONFIDENCE,
    backup_csv, analyze_original_wav)

REFERENCE_WAV = "er_file_2026_07_14_19_23_28.wav"
SIMILARITY_FEATURES = ("prominence", "width_seconds", "rms_energy",
                       "peak_absolute_amplitude", "zero_crossing_rate_hz",
                       "spectral_centroid_hz", "spectral_bandwidth_hz",
                       "spectral_rolloff_hz", "dominant_frequency_hz", "spectral_flatness")
LOG10_FEATURES = {"prominence", "width_seconds", "rms_energy", "peak_absolute_amplitude", "spectral_flatness"}


def milestone_events(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.duplicated(["seed_id", "event_type"]).any():
        raise ValueError("Duplicate visual milestone")
    rows = frame.loc[frame.last_negative_timestamp.ne("") & frame.first_positive_timestamp.ne("")].copy()
    rows["start"] = pd.to_datetime(rows.last_negative_timestamp)
    rows["end"] = pd.to_datetime(rows.first_positive_timestamp)
    if (rows.start >= rows.end).any():
        raise ValueError("Visual milestone interval must be positive")
    rows["duration_seconds"] = (rows.end-rows.start).dt.total_seconds()
    return rows.sort_values(["start", "end", "seed_id"]).reset_index(drop=True)


def load_day_audio(features_path: Path, peaks_path: Path, day: str):
    features = pd.read_csv(features_path)
    features = features.loc[features.start_timestamp.str.startswith(day)].copy()
    features["start"] = pd.to_datetime(features.start_timestamp)
    features["end"] = pd.to_datetime(features.end_timestamp)
    if features.relative_path.duplicated().any():
        raise ValueError("Duplicate WAV relative path")
    columns = ["source_audio_file", "source_relative_path", "file_start_timestamp",
               "relative_peak_time_seconds", "absolute_peak_timestamp", "envelope_peak_value",
               "prominence", "width_seconds"]
    chunks = []
    for chunk in pd.read_csv(peaks_path, usecols=columns, chunksize=100_000):
        selected = chunk.loc[chunk.file_start_timestamp.str.startswith(day)]
        if len(selected): chunks.append(selected)
    peaks = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)
    peaks["timestamp"] = pd.to_datetime(peaks.absolute_peak_timestamp)
    peaks["high"] = False  # Not used for similarity; retained for shared window_metrics API.
    peaks = peaks.drop_duplicates(["source_relative_path", "relative_peak_time_seconds"])
    if not set(peaks.source_relative_path).issubset(set(features.relative_path)):
        raise ValueError("Peak table references absent audio features")
    return features.sort_values("start").reset_index(drop=True), peaks.sort_values("timestamp").reset_index(drop=True)


def reference_profile(raw_root: Path, features: pd.DataFrame, peaks: pd.DataFrame,
                      human_review_path: Path | None = None) -> dict:
    matched = features.loc[features.filename.eq(REFERENCE_WAV)]
    if len(matched) != 1:
        raise ValueError("Reference WAV missing or duplicate")
    file_peaks = peaks.loc[peaks.source_relative_path.eq(matched.iloc[0].relative_path)]
    if file_peaks.empty:
        raise ValueError("Reference has no precomputed candidate peak")
    strongest = file_peaks.sort_values(["prominence", "timestamp"], ascending=[False, True]).iloc[0]
    raw = (raw_root / matched.iloc[0].relative_path).resolve()
    if not raw.is_relative_to(raw_root.resolve()) or not raw.is_file():
        raise FileNotFoundError(raw)
    original, plotting = analyze_original_wav(raw, file_peaks)
    center = int(float(strongest.relative_peak_time_seconds)*plotting["rate"])
    radius = int(.1*plotting["rate"])
    segment = plotting["mono"][max(0, center-radius):min(len(plotting["mono"]), center+radius)]
    frequencies, power = signal.periodogram(segment, plotting["rate"])
    broadband_fraction = float(power[(frequencies >= 1000) & (frequencies <= 20000)].sum()/power.sum()) if power.sum() else np.nan
    human_label = ""
    if human_review_path is not None and human_review_path.exists():
        review = pd.read_csv(human_review_path, keep_default_na=False)
        matched_review = review.loc[review.wav_filename.eq(REFERENCE_WAV)]
        if len(matched_review) != 1:
            raise ValueError("Reference WAV must have one existing human review row")
        human_label = matched_review.iloc[0].manual_audio_label
    result = {"reference_wav": REFERENCE_WAV,
              "reference_relative_path": matched.iloc[0].relative_path,
              "reference_peak_timestamp": str(strongest.timestamp),
              "reference_peak_prominence": float(strongest.prominence),
              "reference_peak_width_seconds": float(strongest.width_seconds),
              "reference_peak_envelope_value": float(strongest.envelope_peak_value),
              "human_label": human_label,
              "source_attribution": "unknown"}
    result.update({key: original[key] for key in ("rms", "max_abs_amplitude", "zero_crossing_rate_hz",
                   "spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_rolloff_hz", "dominant_frequency_hz")})
    result["spectral_flatness"] = float(matched.iloc[0].spectral_flatness)
    result["peak_local_1k_20k_power_fraction"] = broadband_fraction
    result["peak_local_window_seconds"] = .2
    return result


def peak_feature_frame(peaks: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    columns = ["relative_path", "filename", "start", *[key for key in SIMILARITY_FEATURES if key not in {"prominence", "width_seconds"}]]
    return peaks.merge(features[columns], left_on="source_relative_path", right_on="relative_path",
                       how="left", validate="many_to_one")


def _transformed_features(frame: pd.DataFrame) -> np.ndarray:
    values = []
    for key in SIMILARITY_FEATURES:
        numbers = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float)
        numbers = np.maximum(numbers, 0)
        values.append(np.log10(numbers+1e-8) if key in LOG10_FEATURES else np.log1p(numbers))
    return np.column_stack(values)


def reference_similarity(population: pd.DataFrame, candidates: pd.DataFrame,
                         reference: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Equal-weight robust-scaled distance, transformed per feature; no fitted classifier."""
    pop = _transformed_features(population)
    center = np.nanmedian(pop, axis=0)
    lower, upper = np.nanpercentile(pop, [25, 75], axis=0)
    spread = np.maximum(upper-lower, 1e-6)
    query = _transformed_features(candidates)
    ref = _transformed_features(reference)
    query = np.where(np.isfinite(query), query, center)
    ref = np.where(np.isfinite(ref), ref, center)
    distance = np.sqrt(np.mean(((query-ref[0])/spread)**2, axis=1))
    score = 1/(1+distance)
    settings = {"features": list(SIMILARITY_FEATURES),
                "transformation": "log10(x+1e-8) for amplitude, width and flatness; log1p(x) for frequency/rate features",
                "standardization": "July 14 peak population IQR per feature; equal feature weights; missing values imputed by population median",
                "score": "1/(1+sqrt(mean(((candidate-reference)/IQR)^2)))",
                "caution": "File-level spectral and RMS features, peak-level prominence and width; similarity does not establish source."}
    return score, settings


def _relationship(lag: float) -> str:
    if lag == 0: return "inside_visual_interval"
    if abs(lag) <= 30: return "within_30_seconds"
    if abs(lag) <= 60: return "within_1_minute"
    return "within_5_minutes"


def build_candidate_rows(groups: pd.DataFrame, controls: pd.DataFrame,
                         peak_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in groups.itertuples(index=False):
        left, right = expanded_bounds(group.group_start, group.group_end, 300)
        nearby = peak_features.loc[(peak_features.timestamp > left) & (peak_features.timestamp <= right)]
        for peak in nearby.itertuples(index=False):
            lag = signed_lag_seconds(peak.timestamp, group.group_start, group.group_end)
            rows.append({"visual_group_id": group.visual_group_id, "seed_ids": group.seed_ids,
                         "visual_event_type": group.event_types, "visual_interval_start": group.group_start,
                         "visual_interval_end": group.group_end, "wav_filename": peak.filename,
                         "wav_relative_path": peak.source_relative_path, "wav_timestamp": peak.start,
                         "peak_timestamp": peak.timestamp, "relationship_to_visual_event": _relationship(lag),
                         "lag_seconds": lag, "peak_prominence": peak.prominence,
                         "peak_width_seconds": peak.width_seconds, "rms": peak.rms_energy,
                         "spectral_centroid_hz": peak.spectral_centroid_hz,
                         "spectral_bandwidth_hz": peak.spectral_bandwidth_hz,
                         "spectral_rolloff_hz": peak.spectral_rolloff_hz,
                         "dominant_frequency_hz": peak.dominant_frequency_hz,
                         "spectral_flatness": peak.spectral_flatness,
                         "zero_crossing_rate_hz": peak.zero_crossing_rate_hz,
                         "peak_absolute_amplitude": peak.peak_absolute_amplitude,
                         "control_id": "", "is_control": False})
    for control in controls.itertuples(index=False):
        nearby = peak_features.loc[(peak_features.timestamp > control.control_start) &
                                   (peak_features.timestamp <= control.control_end)]
        group = groups.loc[groups.visual_group_id.eq(control.matched_visual_group_id)].iloc[0]
        for peak in nearby.itertuples(index=False):
            rows.append({"visual_group_id": group.visual_group_id, "seed_ids": group.seed_ids,
                         "visual_event_type": group.event_types, "visual_interval_start": group.group_start,
                         "visual_interval_end": group.group_end, "wav_filename": peak.filename,
                         "wav_relative_path": peak.source_relative_path, "wav_timestamp": peak.start,
                         "peak_timestamp": peak.timestamp, "relationship_to_visual_event": "matched_control",
                         "lag_seconds": np.nan, "peak_prominence": peak.prominence,
                         "peak_width_seconds": peak.width_seconds, "rms": peak.rms_energy,
                         "spectral_centroid_hz": peak.spectral_centroid_hz,
                         "spectral_bandwidth_hz": peak.spectral_bandwidth_hz,
                         "spectral_rolloff_hz": peak.spectral_rolloff_hz,
                         "dominant_frequency_hz": peak.dominant_frequency_hz,
                         "spectral_flatness": peak.spectral_flatness,
                         "zero_crossing_rate_hz": peak.zero_crossing_rate_hz,
                         "peak_absolute_amplitude": peak.peak_absolute_amplitude,
                         "control_id": control.control_id, "is_control": True})
    table = pd.DataFrame(rows)
    if table.empty: return table
    # A physical peak may intersect multiple expanded windows. Assign it once,
    # preferring the group with the smallest absolute lag.
    table["_abs_lag"] = pd.to_numeric(table.lag_seconds, errors="coerce").abs().fillna(np.inf)
    table = table.sort_values(["is_control", "_abs_lag", "visual_group_id", "peak_timestamp"])
    table = table.drop_duplicates(["wav_relative_path", "peak_timestamp", "is_control"], keep="first")
    table = table.sort_values(["is_control", "peak_timestamp", "visual_group_id"]).reset_index(drop=True)
    table["candidate_id"] = ["candidate_" + hashlib.sha256(
        f"{row.visual_group_id}|{row.wav_relative_path}|{row.peak_timestamp}|{row.is_control}".encode()
    ).hexdigest()[:12] for row in table.itertuples(index=False)]
    return table.drop(columns="_abs_lag")


def rank_candidates(candidates: pd.DataFrame, population: pd.DataFrame,
                    reference_row: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if candidates.empty: return candidates, {}
    source = candidates.rename(columns={"peak_prominence": "prominence", "peak_width_seconds": "width_seconds",
                                         "rms": "rms_energy"})
    score, settings = reference_similarity(population, source, reference_row)
    ranked = candidates.copy()
    ranked["reference_similarity_score"] = score
    # Assign deterministic ranks in sorted order, then restore chronological table order.
    ordered = ranked.sort_values(["reference_similarity_score", "peak_timestamp", "candidate_id"],
                                 ascending=[False, True, True]).index
    ranked.loc[ordered, "reference_similarity_rank"] = np.arange(1, len(ranked)+1)
    ranked["manual_review_label"] = ""
    ranked["manual_review_confidence"] = ""
    ranked["notes"] = ""
    return ranked, settings


def preserve_manual_candidate_fields(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists(): return new
    old = pd.read_csv(path, keep_default_na=False)
    old["peak_timestamp"] = pd.to_datetime(old.peak_timestamp)
    new = new.copy()
    new["peak_timestamp"] = pd.to_datetime(new.peak_timestamp)
    key = ["visual_group_id", "wav_relative_path", "peak_timestamp", "is_control"]
    if old.duplicated(key).any() or new.duplicated(key).any():
        raise ValueError("Duplicate candidate keys")
    prior = old.set_index(key)
    out = new.copy()
    for index, row in out.iterrows():
        identifier = tuple(row[column] for column in key)
        if identifier in prior.index:
            for field in ("manual_review_label", "manual_review_confidence", "notes"):
                out.at[index, field] = prior.loc[identifier, field]
    removed = old.loc[~old.set_index(key).index.isin(out.set_index(key).index)]
    if (removed[["manual_review_label", "manual_review_confidence", "notes"]].fillna("").astype(str) != "").any(axis=1).any():
        raise ValueError("Regeneration would remove human-reviewed candidate rows")
    return out


def top_distinct_wavs(candidates: pd.DataFrame, group_id: str, control: bool, limit=3) -> pd.DataFrame:
    selected = candidates.loc[candidates.visual_group_id.eq(group_id) & candidates.is_control.eq(control)]
    return (selected.sort_values(["reference_similarity_score", "peak_timestamp", "candidate_id"],
                                 ascending=[False, True, True])
            .drop_duplicates("wav_relative_path").head(limit))


def blind_identifiers(count: int, seed: int = 20261025) -> list[str]:
    order = np.random.default_rng(seed).permutation(count)
    return [f"audio_A{int(value)+1:02d}" for value in order]


def prepare_blind_review(selected: pd.DataFrame, raw_root: Path, destination: Path,
                         existing_review_path: Path | None = None,
                         existing_mapping_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if selected.wav_relative_path.duplicated().any():
        raise ValueError("Blind review must not repeat a WAV")
    destination.mkdir(parents=True, exist_ok=True)
    mapping, manifest, blind_review = [], [], []
    prior = pd.read_csv(existing_review_path, keep_default_na=False) if existing_review_path and existing_review_path.exists() else pd.DataFrame()
    old_mapping = pd.read_csv(existing_mapping_path, keep_default_na=False) if existing_mapping_path and existing_mapping_path.exists() else pd.DataFrame()
    old_ids = dict(zip(old_mapping.wav_relative_path, old_mapping.blind_id)) if not old_mapping.empty else {}
    if not prior.empty:
        reviewed = prior.loc[(prior[["manual_audio_label", "manual_audio_confidence", "notes"]].fillna("").astype(str) != "").any(axis=1)]
        if not set(reviewed.wav_relative_path).issubset(set(selected.wav_relative_path)):
            raise ValueError("Regeneration would drop a human-reviewed blind WAV")
    unused_ids = (identifier for identifier in blind_identifiers(99) if identifier not in set(old_ids.values()))
    fresh_ids = iter(blind_identifiers(len(selected))) if not old_ids else unused_ids
    for _, row in selected.iterrows():
        blind_id = old_ids.get(row.wav_relative_path) or next(fresh_ids)
        source = (raw_root / row.wav_relative_path).resolve()
        if not source.is_relative_to(raw_root.resolve()) or not source.is_file():
            raise FileNotFoundError(source)
        copy = destination / f"{blind_id}.wav"
        if copy.exists() and hashlib.sha256(copy.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
            raise ValueError("Existing blind copy differs from selected raw WAV")
        if not copy.exists(): shutil.copy2(source, copy)
        record = {"candidate_id": row.candidate_id, "event_or_control": "control" if row.is_control else "event",
                  "visual_group_id": row.visual_group_id, "seed_ids": row.seed_ids,
                  "event_type": row.visual_event_type, "wav_filename": row.wav_filename,
                  "wav_relative_path": row.wav_relative_path, "wav_timestamp": row.wav_timestamp,
                  "lag_seconds": row.lag_seconds, "similarity_score": row.reference_similarity_score,
                  "blind_id": blind_id, "manual_audio_label": "",
                  "manual_audio_confidence": "", "notes": ""}
        if not prior.empty:
            matched = prior.loc[prior.wav_relative_path.eq(row.wav_relative_path)]
            if len(matched) == 1:
                for field in ("manual_audio_label", "manual_audio_confidence", "notes"):
                    record[field] = matched.iloc[0][field]
        manifest.append(record)
        mapping.append({"blind_id": blind_id, "candidate_id": row.candidate_id,
                        "event_or_control": record["event_or_control"], "visual_group_id": row.visual_group_id,
                        "wav_relative_path": row.wav_relative_path, "peak_timestamp": row.peak_timestamp})
        blind_review.append({"blind_id": blind_id, "listening_path": str(copy),
                             "manual_audio_label": record["manual_audio_label"],
                             "manual_audio_confidence": record["manual_audio_confidence"],
                             "notes": record["notes"]})
    return pd.DataFrame(manifest), pd.DataFrame(mapping), pd.DataFrame(blind_review)


def save_blind_review(blind_path: Path, manifest_path: Path, blind_id: str,
                      label: str, confidence: str, notes: str) -> None:
    if label not in ("", *ALLOWED_LABELS) or confidence not in ("", *ALLOWED_CONFIDENCE):
        raise ValueError("Use documented neutral labels and confidence levels")
    blind = pd.read_csv(blind_path, keep_default_na=False)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    blind_match, manifest_match = blind.blind_id.eq(blind_id), manifest.blind_id.eq(blind_id)
    if blind_match.sum() != 1 or manifest_match.sum() != 1:
        raise ValueError("Blind ID must identify exactly one review WAV")
    for frame, mask in ((blind, blind_match), (manifest, manifest_match)):
        frame.loc[mask, "manual_audio_label"] = label
        frame.loc[mask, "manual_audio_confidence"] = confidence
        frame.loc[mask, "notes"] = notes
    for path, frame in ((blind_path, blind), (manifest_path, manifest)):
        backup_csv(path)
        temporary = path.with_name(path.name+".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
