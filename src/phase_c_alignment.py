"""Exploratory visual/audio alignment for human-reviewed event intervals.

One microphone observes the whole box. Peaks are acoustic candidates, never
seed-attributed or biological labels. Raw WAVs are read-only inputs.
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .dense_visual import load_events

SCALES_SECONDS = (0, 30, 60, 300)
SCALE_NAMES = {0: "exact", 30: "30s", 60: "1min", 300: "5min"}
# Fraction of an interval occupied by the union of actually recorded WAV spans.
COVERAGE_THRESHOLDS = {"good": 0.50, "partial": 0.20, "poor": 0.0}
CONTROL_OFFSETS_MINUTES = (-20, 20, -30, 30, -45, 45, -60, 60, -90, 90, -120, 120)


def load_visual_intervals(path: Path) -> pd.DataFrame:
    frame = load_events(path)
    if frame.empty or frame.seed_id.eq("").any() or frame.event_type.eq("").any():
        raise ValueError("Visual event table needs seed IDs and human-reviewed event types")
    for column in ("last_negative_timestamp", "first_positive_timestamp"):
        if frame[column].eq("").any():
            raise ValueError(f"Human visual boundary missing: {column}")
    frame = frame.copy()
    frame["start"] = pd.to_datetime(frame.last_negative_timestamp)
    frame["end"] = pd.to_datetime(frame.first_positive_timestamp)
    if (frame.end <= frame.start).any():
        raise ValueError("Visual interval must have positive duration")
    frame["duration_seconds"] = (frame.end - frame.start).dt.total_seconds()
    if frame.duplicated(["seed_id", "event_type", "start", "end"]).any():
        raise ValueError("Duplicate human visual event interval")
    frame["left_censored"] = (frame.seed_id.eq("seed_01") & frame.event_type.eq("crack") &
                              frame.start.eq(pd.to_datetime(frame.coarse_start_timestamp)))
    frame["visual_timing_quality"] = np.where(frame.left_censored, "limited_possible_left_censoring", "reviewed_interval")
    return frame.sort_values(["start", "end", "seed_id"]).reset_index(drop=True)


def group_visual_intervals(events: pd.DataFrame) -> pd.DataFrame:
    """Connected components of intervals with positive temporal overlap."""
    ordered = events.sort_values(["start", "end", "seed_id"])
    groups = []
    current = []
    start = end = None
    for row in ordered.itertuples(index=False):
        if current and row.start >= end:  # touching endpoints alone is not overlap
            groups.append((start, end, current))
            current = []
        if not current:
            start, end = row.start, row.end
        else:
            end = max(end, row.end)
        current.append(row)
    if current:
        groups.append((start, end, current))
    rows = []
    for number, (group_start, group_end, members) in enumerate(groups, 1):
        rows.append({"visual_group_id": f"visual_group_{number:02d}",
                     "seed_ids": ";".join(sorted(member.seed_id for member in members)),
                     "number_of_seed_events": len(members), "group_start": group_start,
                     "group_end": group_end,
                     "event_types": ";".join(sorted({member.event_type for member in members}))})
    return pd.DataFrame(rows)


def load_audio_tables(features_path: Path, peaks_path: Path, day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read existing features and peak CSVs; never decode or reprocess WAVs."""
    features = pd.read_csv(features_path)
    required_features = {"relative_path", "start_timestamp", "end_timestamp", "duration_seconds", "rms_energy"}
    if required_features - set(features):
        raise ValueError("Audio feature table lacks required columns")
    features = features.loc[features.start_timestamp.astype(str).str.startswith(day)].copy()
    features["start"] = pd.to_datetime(features.start_timestamp)
    features["end"] = pd.to_datetime(features.end_timestamp)
    if (features.end <= features.start).any() or features.relative_path.duplicated().any():
        raise ValueError("Audio file intervals or identifiers are invalid")
    features = features.sort_values("start").reset_index(drop=True)
    chunks = []
    wanted = ["source_audio_file", "source_relative_path", "file_start_timestamp",
              "relative_peak_time_seconds", "absolute_peak_timestamp", "envelope_peak_value",
              "prominence", "width_seconds", "high"]
    for chunk in pd.read_csv(peaks_path, usecols=wanted, chunksize=100_000):
        selected = chunk.loc[chunk.file_start_timestamp.astype(str).str.startswith(day)]
        if not selected.empty:
            chunks.append(selected)
    peaks = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=wanted)
    peaks["timestamp"] = pd.to_datetime(peaks.absolute_peak_timestamp)
    peaks["high"] = peaks.high.astype(str).str.lower().eq("true")
    peaks = peaks.drop_duplicates(["source_relative_path", "relative_peak_time_seconds"]).sort_values("timestamp").reset_index(drop=True)
    return features, peaks


def overlap_seconds(start, end, recordings: pd.DataFrame) -> float:
    """Duration of the union of recorded intervals clipped to [start, end]."""
    if pd.Timestamp(end) <= pd.Timestamp(start):
        raise ValueError("Interval must have positive duration")
    spans = []
    for row in recordings.loc[(recordings.end > start) & (recordings.start < end)].itertuples(index=False):
        spans.append((max(pd.Timestamp(start), row.start), min(pd.Timestamp(end), row.end)))
    if not spans:
        return 0.0
    spans.sort()
    total = 0.0
    left, right = spans[0]
    for next_left, next_right in spans[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += (right - left).total_seconds()
            left, right = next_left, next_right
    return total + (right - left).total_seconds()


def coverage_summary(start, end, features: pd.DataFrame) -> dict:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    nearby = features.loc[(features.end > start) & (features.start < end)]
    recorded = overlap_seconds(start, end, nearby)
    fraction = recorded / (end - start).total_seconds()
    category = "good" if fraction >= 0.50 else "partial" if fraction >= 0.20 else "poor" if recorded > 0 else "none"
    before = features.loc[features.end <= start].sort_values("end")
    after = features.loc[features.start >= end].sort_values("start")
    nearest_before = before.iloc[-1] if len(before) else None
    nearest_after = after.iloc[0] if len(after) else None
    distances = [float((start - nearest_before.end).total_seconds())] if nearest_before is not None else []
    if nearest_after is not None:
        distances.append(float((nearest_after.start - end).total_seconds()))
    return {"recording_count": len(nearby), "recorded_audio_seconds": recorded,
            "coverage_fraction": fraction, "audio_coverage_class": category,
            "nearest_wav_before": nearest_before.relative_path if nearest_before is not None else "",
            "nearest_wav_after": nearest_after.relative_path if nearest_after is not None else "",
            "distance_to_nearest_recording_seconds": 0.0 if recorded else min(distances) if distances else np.nan}


def signed_lag_seconds(peak_time, start, end) -> float:
    """Negative before, zero within (start,end], positive after."""
    peak, start, end = map(pd.Timestamp, (peak_time, start, end))
    if peak <= start:
        return float((peak - start).total_seconds())
    if peak > end:
        return float((peak - end).total_seconds())
    return 0.0


def closest_peak(start, end, peaks: pd.DataFrame, limit_seconds=300) -> dict:
    expanded_start = pd.Timestamp(start) - pd.Timedelta(seconds=limit_seconds)
    expanded_end = pd.Timestamp(end) + pd.Timedelta(seconds=limit_seconds)
    nearby = peaks.loc[(peaks.timestamp > expanded_start) & (peaks.timestamp <= expanded_end)].copy()
    if nearby.empty:
        return {"closest_peak_timestamp": "", "closest_peak_relationship": "none_within_5min",
                "closest_peak_lag_seconds": np.nan, "closest_peak_prominence": np.nan,
                "closest_peak_envelope_value": np.nan, "closest_peak_width_seconds": np.nan}
    nearby["lag"] = nearby.timestamp.map(lambda value: signed_lag_seconds(value, start, end))
    selected = nearby.sort_values(["lag", "timestamp"], key=lambda col: col.abs() if col.name == "lag" else col).iloc[0]
    lag = float(selected.lag)
    return {"closest_peak_timestamp": selected.timestamp, "closest_peak_relationship": "inside" if lag == 0 else "before" if lag < 0 else "after",
            "closest_peak_lag_seconds": lag, "closest_peak_prominence": selected.prominence,
            "closest_peak_envelope_value": selected.envelope_peak_value,
            "closest_peak_width_seconds": selected.width_seconds}


def window_metrics(start, end, features: pd.DataFrame, peaks: pd.DataFrame) -> dict:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    coverage = coverage_summary(start, end, features)
    selected = peaks.loc[(peaks.timestamp > start) & (peaks.timestamp <= end)]
    audio = features.loc[(features.end > start) & (features.start < end)]
    recorded = coverage["recorded_audio_seconds"]
    if len(audio):
        weights = np.minimum(audio.end, end).sub(np.maximum(audio.start, start)).dt.total_seconds()
        mean_rms = float(np.average(audio.rms_energy, weights=weights))
        median_rms = float(audio.rms_energy.median())
        max_rms = float(audio.rms_energy.max())
        spectral_centroid = float(audio.spectral_centroid_hz.median()) if "spectral_centroid_hz" in audio else np.nan
        spectral_flatness = float(audio.spectral_flatness.median()) if "spectral_flatness" in audio else np.nan
    else:
        mean_rms = median_rms = max_rms = spectral_centroid = spectral_flatness = np.nan
    return {**coverage, "peak_count": len(selected), "high_prominence_peak_count": int(selected.high.sum()),
            "peaks_per_recorded_minute": len(selected) * 60 / recorded if recorded else np.nan,
            "high_prominence_peaks_per_recorded_minute": int(selected.high.sum()) * 60 / recorded if recorded else np.nan,
            "mean_peak_prominence": float(selected.prominence.mean()) if len(selected) else np.nan,
            "median_peak_prominence": float(selected.prominence.median()) if len(selected) else np.nan,
            "maximum_peak_prominence": float(selected.prominence.max()) if len(selected) else np.nan,
            "mean_rms": mean_rms, "median_rms": median_rms, "maximum_rms": max_rms,
            "median_spectral_centroid_hz": spectral_centroid,
            "median_spectral_flatness": spectral_flatness}


def expanded_bounds(start, end, seconds: int):
    offset = pd.Timedelta(seconds=seconds)
    return pd.Timestamp(start)-offset, pd.Timestamp(end)+offset


def choose_control_windows(groups: pd.DataFrame, features: pd.DataFrame, count=2) -> pd.DataFrame:
    """Predeclared clock offsets; choose by coverage proximity, never peaks."""
    event_bounds = [expanded_bounds(row.group_start, row.group_end, 300)
                    for row in groups.itertuples(index=False)]
    chosen_global = []
    rows = []
    for group in groups.itertuples(index=False):
        event_start, event_end = expanded_bounds(group.group_start, group.group_end, 300)
        duration = event_end - event_start
        event_coverage = coverage_summary(event_start, event_end, features)["coverage_fraction"]
        candidates = []
        for rank, minutes in enumerate(CONTROL_OFFSETS_MINUTES):
            control_start = event_start + pd.Timedelta(minutes=minutes)
            control_end = control_start + duration
            if control_start.date() != event_start.date() or control_end.date() != event_end.date():
                continue
            if any(control_start < blocked_end and control_end > blocked_start
                   for blocked_start, blocked_end in event_bounds):
                continue
            coverage = coverage_summary(control_start, control_end, features)
            if coverage["recorded_audio_seconds"] == 0:
                continue
            candidates.append((abs(coverage["coverage_fraction"] - event_coverage),
                               abs(minutes), rank, minutes, control_start, control_end))
        for _, _, _, minutes, control_start, control_end in sorted(candidates):
            if any(control_start < used_end and control_end > used_start
                   for used_start, used_end in chosen_global):
                continue
            rows.append({"control_id": f"{group.visual_group_id}_control_{sum(r['matched_visual_group_id']==group.visual_group_id for r in rows)+1:02d}",
                         "matched_visual_group_id": group.visual_group_id,
                         "control_start": control_start, "control_end": control_end,
                         "selection_rule": f"fixed_offset_{minutes:+d}min; selected_by_audio_coverage_not_peaks"})
            chosen_global.append((control_start, control_end))
            if sum(r["matched_visual_group_id"] == group.visual_group_id for r in rows) >= count:
                break
    return pd.DataFrame(rows)


def _window_rows(kind: str, identifier: str, start, end, features, peaks) -> list[dict]:
    rows = []
    for seconds in SCALES_SECONDS:
        left, right = expanded_bounds(start, end, seconds)
        rows.append({"entity_type": kind, "entity_id": identifier, "window_scale": SCALE_NAMES[seconds],
                     "window_start": left, "window_end": right, "window_seconds": (right-left).total_seconds(),
                     **window_metrics(left, right, features, peaks)})
    for seconds in (30, 60, 300):
        left, right = pd.Timestamp(end), pd.Timestamp(end)+pd.Timedelta(seconds=seconds)
        rows.append({"entity_type": kind, "entity_id": identifier,
                     "window_scale": f"post_{SCALE_NAMES[seconds]}",
                     "window_start": left, "window_end": right,
                     "window_seconds": float(seconds), **window_metrics(left, right, features, peaks)})
    return rows


def build_alignment(events: pd.DataFrame, groups: pd.DataFrame, features: pd.DataFrame,
                    peaks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create coverage-aware seed/group windows and deterministic controls."""
    group_for_seed = {seed: group.visual_group_id for group in groups.itertuples(index=False)
                      for seed in group.seed_ids.split(";")}
    metrics_rows = []
    seed_rows = []
    for event in events.itertuples(index=False):
        windows = _window_rows("seed", event.seed_id, event.start, event.end, features, peaks)
        metrics_rows.extend(windows)
        by_scale = {row["window_scale"]: row for row in windows}
        exact = by_scale["exact"]
        near = by_scale["5min"]
        closest = closest_peak(event.start, event.end, peaks)
        seed_rows.append({"seed_id": event.seed_id, "visual_group_id": group_for_seed[event.seed_id],
                          "event_type": event.event_type, "visual_interval_start": event.start,
                          "visual_interval_end": event.end, "visual_interval_seconds": event.duration_seconds,
                          "visual_timing_quality": event.visual_timing_quality,
                          "left_censored": bool(event.left_censored),
                          "audio_coverage_class": exact["audio_coverage_class"],
                          "audio_recorded_seconds_exact": exact["recorded_audio_seconds"],
                          "audio_coverage_fraction_exact": exact["coverage_fraction"],
                          "wav_count_exact": exact["recording_count"],
                          "nearest_wav_before": exact["nearest_wav_before"],
                          "nearest_wav_after": exact["nearest_wav_after"],
                          "distance_to_nearest_recording_seconds": exact["distance_to_nearest_recording_seconds"],
                          **{f"candidate_peaks_{scale}": by_scale[scale]["peak_count"] for scale in SCALE_NAMES.values()},
                          **{f"candidate_peaks_post_{scale}": by_scale[f"post_{scale}"]["peak_count"] for scale in ("30s", "1min", "5min")},
                          **{f"high_prominence_peaks_post_{scale}": by_scale[f"post_{scale}"]["high_prominence_peak_count"] for scale in ("30s", "1min", "5min")},
                          **closest, "mean_rms_near_event": near["mean_rms"],
                          "high_prominence_rate_near_event": near["high_prominence_peaks_per_recorded_minute"],
                          "manual_audio_label": "", "interpretation": "insufficient_audio_coverage" if exact["audio_coverage_class"] in {"none", "poor"} else "candidate_activity_only_no_seed_attribution",
                          "notes": "Possible left censoring: crack may already be present at dense-window start." if event.left_censored else ""})
    group_rows = []
    for group in groups.itertuples(index=False):
        windows = _window_rows("group", group.visual_group_id, group.group_start, group.group_end, features, peaks)
        metrics_rows.extend(windows)
        by_scale = {row["window_scale"]: row for row in windows}
        exact, near = by_scale["exact"], by_scale["5min"]
        closest = closest_peak(group.group_start, group.group_end, peaks)
        group_rows.append({"visual_group_id": group.visual_group_id, "seed_ids": group.seed_ids,
                           "number_of_seed_events": group.number_of_seed_events,
                           "group_start": group.group_start, "group_end": group.group_end,
                           "event_types": group.event_types,
                           "audio_coverage": exact["audio_coverage_class"],
                           "audio_recorded_seconds_exact": exact["recorded_audio_seconds"],
                           "audio_coverage_fraction_exact": exact["coverage_fraction"],
                           **{f"candidate_peaks_{scale}": by_scale[scale]["peak_count"] for scale in SCALE_NAMES.values()},
                           **{f"candidate_peaks_post_{scale}": by_scale[f"post_{scale}"]["peak_count"] for scale in ("30s", "1min", "5min")},
                           **{f"high_prominence_peaks_post_{scale}": by_scale[f"post_{scale}"]["high_prominence_peak_count"] for scale in ("30s", "1min", "5min")},
                           **closest, "peak_activity_summary": "candidate_acoustic_activity_only",
                           "control_activity_summary": "", "exploratory_event_control_ratio": np.nan,
                           "manual_interpretation": "", "notes": "Shared microphone; peaks cannot be attributed to individual seeds."})
    controls = choose_control_windows(groups, features)
    control_rows = []
    for control in controls.itertuples(index=False):
        metric = window_metrics(control.control_start, control.control_end, features, peaks)
        control_rows.append({"control_id": control.control_id,
                             "matched_visual_group_id": control.matched_visual_group_id,
                             "control_start": control.control_start, "control_end": control.control_end,
                             "selection_rule": control.selection_rule,
                             "recorded_audio_seconds": metric["recorded_audio_seconds"],
                             "coverage_fraction": metric["coverage_fraction"],
                             "peak_count": metric["peak_count"],
                             "peaks_per_recorded_minute": metric["peaks_per_recorded_minute"],
                             "high_prominence_peak_count": metric["high_prominence_peak_count"],
                             "high_prominence_peaks_per_recorded_minute": metric["high_prominence_peaks_per_recorded_minute"],
                             "mean_peak_prominence": metric["mean_peak_prominence"],
                             "median_peak_prominence": metric["median_peak_prominence"],
                             "maximum_peak_prominence": metric["maximum_peak_prominence"],
                             "mean_rms": metric["mean_rms"], "median_rms": metric["median_rms"],
                             "median_spectral_centroid_hz": metric["median_spectral_centroid_hz"],
                             "notes": "Control selected without candidate-peak information."})
    control_table = pd.DataFrame(control_rows)
    group_table = pd.DataFrame(group_rows)
    for idx, row in group_table.iterrows():
        matched = control_table.loc[control_table.matched_visual_group_id == row.visual_group_id] if not control_table.empty else pd.DataFrame()
        if matched.empty:
            group_table.at[idx, "control_activity_summary"] = "no_eligible_control"
            continue
        control_rate = matched.peaks_per_recorded_minute.mean()
        event_rate = next(metric["peaks_per_recorded_minute"] for metric in metrics_rows
                          if metric["entity_type"] == "group" and metric["entity_id"] == row.visual_group_id and metric["window_scale"] == "5min")
        group_table.at[idx, "control_activity_summary"] = f"{len(matched)} matched controls; mean {control_rate:.3g} candidate peaks/recorded min"
        if pd.notna(event_rate) and pd.notna(control_rate) and control_rate > 0:
            group_table.at[idx, "exploratory_event_control_ratio"] = event_rate / control_rate
    return {"seed": pd.DataFrame(seed_rows), "group": group_table,
            "controls": control_table, "windows": pd.DataFrame(metrics_rows)}


def targeted_audio_review(groups: pd.DataFrame, controls: pd.DataFrame,
                          features: pd.DataFrame, peaks: pd.DataFrame) -> pd.DataFrame:
    """Select a few distinct WAVs per group for later human listening."""
    rows = []
    feature_by_path = features.set_index("relative_path")
    for group in groups.itertuples(index=False):
        start, end = group.group_start, group.group_end
        choices: dict[str, list[tuple[str, object]]] = {}
        near5 = peaks.loc[(peaks.timestamp > start-pd.Timedelta(minutes=5)) &
                          (peaks.timestamp <= end+pd.Timedelta(minutes=5))].copy()
        if len(near5):
            near5["abs_lag"] = near5.timestamp.map(lambda value: abs(signed_lag_seconds(value, start, end)))
            nearest = near5.sort_values(["abs_lag", "timestamp"]).iloc[0]
            choices.setdefault(nearest.source_relative_path, []).append(("closest_candidate_peak", nearest))
            near1 = near5.loc[near5.abs_lag <= 60]
            if len(near1):
                strongest1 = near1.sort_values(["prominence", "timestamp"], ascending=[False, True]).iloc[0]
                choices.setdefault(strongest1.source_relative_path, []).append(("strongest_within_1min", strongest1))
            strongest5 = near5.sort_values(["prominence", "timestamp"], ascending=[False, True]).iloc[0]
            if strongest5.source_relative_path not in choices:
                choices.setdefault(strongest5.source_relative_path, []).append(("strongest_within_5min", strongest5))
        matched_controls = controls.loc[controls.matched_visual_group_id == group.visual_group_id]
        if len(matched_controls):
            control = matched_controls.iloc[0]
            low = features.loc[(features.end > control.control_start) &
                               (features.start < control.control_end)].sort_values(["rms_energy", "start"])
            if len(low):
                choices.setdefault(low.iloc[0].relative_path, []).append(("representative_low_rms_control", None))
        for relative_path, reasons in choices.items():
            if relative_path not in feature_by_path.index:
                continue
            feature = feature_by_path.loc[relative_path]
            peak_reasons = [(reason, peak) for reason, peak in reasons if peak is not None]
            selected_peak = max((peak for _, peak in peak_reasons),
                                key=lambda item: item.prominence, default=None)
            lag = signed_lag_seconds(selected_peak.timestamp, start, end) if selected_peak is not None else np.nan
            rows.append({"visual_group_id": group.visual_group_id, "seed_ids": group.seed_ids,
                         "visual_event_type": group.event_types,
                         "visual_interval_start": start, "visual_interval_end": end,
                         "wav_filename": Path(relative_path).name,
                         "wav_relative_path": relative_path,
                         "wav_timestamp": feature.start,
                         "relationship_to_visual_event": ";".join(reason for reason, _ in reasons),
                         "peak_timestamp": selected_peak.timestamp if selected_peak is not None else "",
                         "lag_seconds": lag,
                         "peak_prominence": selected_peak.prominence if selected_peak is not None else np.nan,
                         "rms": feature.rms_energy,
                         "manual_audio_label": "", "manual_audio_confidence": "", "notes": ""})
    return pd.DataFrame(rows)


def preserve_manual_audio_review(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return new
    old = pd.read_csv(path, keep_default_na=False)
    key = ["visual_group_id", "wav_relative_path"]
    if old.duplicated(key).any() or new.duplicated(key).any():
        raise ValueError("Audio review manifest contains duplicate group/WAV keys")
    if set(old.columns) != set(new.columns):
        raise ValueError("Existing audio review manifest schema differs; reconcile manually")
    merged = new.copy()
    indexed = old.set_index(key)
    for index, row in merged.iterrows():
        pair = (row.visual_group_id, row.wav_relative_path)
        if pair in indexed.index:
            for column in ("manual_audio_label", "manual_audio_confidence", "notes"):
                merged.at[index, column] = indexed.loc[pair, column]
    old_keys = set(zip(old.visual_group_id, old.wav_relative_path))
    new_keys = set(zip(merged.visual_group_id, merged.wav_relative_path))
    reviewed = old.loc[(old[["manual_audio_label", "manual_audio_confidence", "notes"]].astype(str) != "").any(axis=1)]
    retained = reviewed.loc[[pair not in new_keys for pair in zip(reviewed.visual_group_id, reviewed.wav_relative_path)]]
    if len(retained):
        merged = pd.concat([merged, retained], ignore_index=True)
    return merged


def copy_review_wavs(manifest: pd.DataFrame, raw_audio_root: Path, destination: Path) -> pd.DataFrame:
    """Copy only selected short WAVs; validate paths stay inside raw root."""
    destination.mkdir(parents=True, exist_ok=True)
    root = raw_audio_root.resolve()
    output = manifest.copy()
    local_paths = []
    for relative in manifest.wav_relative_path:
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise FileNotFoundError(f"Selected raw WAV unavailable: {relative}")
        target = destination / Path(relative).name
        if target.exists() and target.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing review copy differs: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        local_paths.append(str(target))
    output["review_copy_path"] = local_paths
    return output


def findings_summary(events: pd.DataFrame, groups: pd.DataFrame,
                     seed_alignment: pd.DataFrame, group_alignment: pd.DataFrame,
                     review: pd.DataFrame) -> dict:
    return {
        "status": "exploratory_visual_audio_alignment_only",
        "seed_level_visual_events": len(events),
        "unique_temporal_visual_groups": len(groups),
        "groups_with_good_or_partial_exact_audio_coverage": int(group_alignment.audio_coverage.isin(["good", "partial"]).sum()),
        "groups_with_candidate_inside_exact_interval": int(group_alignment.candidate_peaks_exact.gt(0).sum()),
        "groups_with_candidate_within_30s": int(group_alignment.candidate_peaks_30s.gt(0).sum()),
        "groups_with_candidate_within_1min": int(group_alignment.candidate_peaks_1min.gt(0).sum()),
        "groups_with_candidate_within_5min": int(group_alignment.candidate_peaks_5min.gt(0).sum()),
        "groups_with_candidate_in_post_5min": int(group_alignment.candidate_peaks_post_5min.gt(0).sum()),
        "groups_with_no_candidate_within_5min": int(group_alignment.candidate_peaks_5min.eq(0).sum()),
        "groups_with_insufficient_exact_audio_coverage": int(group_alignment.audio_coverage.isin(["none", "poor"]).sum()),
        "manual_audio_files_prepared": int(review.wav_relative_path.nunique()),
        "manual_obvious_environmental_noise_labels": int(review.manual_audio_label.eq("obvious_speech_or_household").sum()),
        "manual_unexplained_transient_labels": int(review.manual_audio_label.eq("short_transient_unexplained").sum()),
        "seed_01_possible_left_censoring": bool(events.loc[events.seed_id == "seed_01", "left_censored"].iloc[0]),
        "caution": "One shared microphone, intermittent recordings and household noise prevent seed attribution or causal claims.",
    }
