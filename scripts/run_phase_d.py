"""Phase D: reference-like precomputed peaks around independent visual milestones."""
from __future__ import annotations

from pathlib import Path
import os
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.phase_c_alignment import group_visual_intervals, choose_control_windows, window_metrics, expanded_bounds
from src.phase_c2_audio_review import analyze_original_wav, backup_csv
from src.phase_d_visual import initialize_milestones
from src.phase_d_audio import (REFERENCE_WAV, SIMILARITY_FEATURES, build_candidate_rows,
    load_day_audio, milestone_events, peak_feature_frame, prepare_blind_review,
    preserve_manual_candidate_fields, rank_candidates, reference_profile, top_distinct_wavs)
from scripts.run_phase_c2 import diagnostic_figure

TABLES = PROJECT / "results/tables/targeted_multimodal"
FIGURES = PROJECT / "results/figures/targeted_multimodal/phase_d"
RAW_AUDIO = Path(os.environ.get("SEED_GERMINATION_AUDIO_ROOT", PROJECT / "data/raw/audio"))


def visual_timeline(events: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    seeds = sorted(events.seed_id.unique())
    colors = {"crack_onset": "#d95f02", "major_opening": "#1b9e77", "radicle_emergence": "#7570b3"}
    for position, seed in enumerate(seeds):
        for row in events.loc[events.seed_id.eq(seed)].itertuples(index=False):
            ax.plot([row.start, row.end], [position, position], linewidth=8,
                    color=colors.get(row.event_type, "gray"), solid_capstyle="butt")
            ax.scatter([row.start, row.end], [position, position], s=35,
                       color=colors.get(row.event_type, "gray"))
    ax.set_yticks(range(len(seeds)), seeds)
    ax.set_ylim(-.6, len(seeds)-.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel(str(events.start.iloc[0].date()) + " — independent visual intervals")
    ax.set_title("Human-reviewed milestones; bar = visual timing uncertainty")
    ax.grid(axis="x", alpha=.2)
    fig.autofmt_xdate()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def reference_figure(profile: dict, output: Path) -> None:
    keys = ["reference_peak_prominence", "reference_peak_width_seconds", "rms",
            "max_abs_amplitude", "zero_crossing_rate_hz", "spectral_centroid_hz",
            "spectral_bandwidth_hz", "spectral_rolloff_hz", "dominant_frequency_hz",
            "peak_local_1k_20k_power_fraction"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axis("off")
    data = [[key, f"{profile[key]:.6g}"] for key in keys]
    table = ax.table(cellText=data, colLabels=["Original-WAV reference feature", "Value"],
                     cellLoc="left", colLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("review_06 reference unexplained transient\nSimilarity uses the documented subset of features", pad=14)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def ranked_figure(candidates: pd.DataFrame, output: Path) -> None:
    top = candidates.sort_values("reference_similarity_score", ascending=False).groupby("is_control", group_keys=False).head(30)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for control, color, label in ((False, "#d95f02", "Visual ±5 min"), (True, "#7570b3", "Matched controls")):
        subset = top.loc[top.is_control.eq(control)].sort_values("reference_similarity_score", ascending=False)
        ax.scatter(np.arange(1, len(subset)+1), subset.reference_similarity_score,
                   color=color, s=23, label=label)
    ax.set_xlabel("Rank within event or control candidates")
    ax.set_ylabel("Reference similarity (descriptive)")
    ax.set_title("Top precomputed peak candidates by robust scaled feature distance")
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output, dpi=170); plt.close(fig)


def distribution_figure(candidates: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    event = candidates.loc[~candidates.is_control, "reference_similarity_score"]
    control = candidates.loc[candidates.is_control, "reference_similarity_score"]
    ax.boxplot([event, control], tick_labels=[f"Visual ±5 min\n(n={len(event)})", f"Controls\n(n={len(control)})"],
               showfliers=False)
    ax.scatter([1, 2], [event.max(), control.max()], color="#d95f02", marker="D", s=40,
               zorder=3, label="Observed maximum (including omitted boxplot outliers)")
    ax.set_ylabel("Reference similarity")
    ax.set_title("Descriptive distributions; peaks within a WAV are not independent")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(output, dpi=170); plt.close(fig)


def group_timeline(group, features, candidates, output: Path) -> None:
    left, right = expanded_bounds(group.group_start, group.group_end, 300)
    audio = features.loc[(features.end > left) & (features.start < right)]
    peaks = candidates.loc[candidates.visual_group_id.eq(group.visual_group_id) & ~candidates.is_control]
    fig, ax = plt.subplots(figsize=(11, 3.7))
    ax.axvspan(group.group_start, group.group_end, color="#d95f02", alpha=.25,
               label="Visual uncertainty interval")
    for row in audio.itertuples(index=False):
        ax.plot([max(row.start, left), min(row.end, right)], [0, 0],
                color="#1b9e77", lw=5, solid_capstyle="butt")
    scatter = ax.scatter(peaks.peak_timestamp, peaks.reference_similarity_score,
                         c=peaks.reference_similarity_score, cmap="viridis", s=11, alpha=.55)
    ax.set_xlim(left, right); ax.set_ylim(-.05, 1.05)
    ax.set_ylabel("Similarity; green line = recorded WAVs")
    ax.set_xlabel("Actual timestamp; blank stretches lack recordings")
    ax.set_title(f"{group.visual_group_id}: {group.seed_ids} · {group.event_types}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.colorbar(scatter, ax=ax, label="Similarity score")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output, dpi=170); plt.close(fig)


def summary_figure(summary: pd.DataFrame, output: Path) -> None:
    x = np.arange(len(summary))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].bar(x, summary.audio_coverage_fraction_exact, color="#1b9e77")
    axes[0].set_ylabel("Exact-interval\naudio coverage")
    axes[0].set_ylim(0, 1)
    axes[1].bar(x-.18, summary.top_reference_similarity, .35, color="#d95f02", label="Best visual-window candidate")
    axes[1].bar(x+.18, summary.top_control_similarity, .35, color="#7570b3", label="Best matched control")
    axes[1].set_ylabel("Highest similarity")
    axes[1].legend()
    axes[1].set_xticks(x, summary.visual_group_id)
    for ax in axes: ax.grid(axis="y", alpha=.2)
    fig.suptitle("Shared microphone: independent visual groups are the comparison unit")
    fig.tight_layout(); fig.savefig(output, dpi=170); plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    milestone_path = TABLES / "visual_milestones.csv"
    initialize_milestones(TABLES / "refined_visual_events.csv", milestone_path)
    events = milestone_events(milestone_path)
    if events.empty: raise ValueError("No independently reviewed visual milestones")
    groups = group_visual_intervals(events)
    day = str(events.start.iloc[0].date())
    if any(events.start.dt.date.astype(str).ne(day)):
        raise ValueError("Phase D pilot currently expects one experimental day")
    audio_tables = PROJECT / "results/tables/audio"
    features, peaks = load_day_audio(audio_tables / "audio_file_features_july13_18.csv",
                                     audio_tables / "audio_candidate_peaks_july13_18.csv", day)
    controls = choose_control_windows(groups, features)
    enriched = peak_feature_frame(peaks, features)
    reference = reference_profile(RAW_AUDIO, features, peaks,
                                  TABLES / "targeted_audio_review_manifest.csv")
    ref_population = enriched.loc[(enriched.source_audio_file.eq(REFERENCE_WAV)) &
                                  (enriched.timestamp.eq(pd.Timestamp(reference["reference_peak_timestamp"])))]
    if len(ref_population) != 1: raise ValueError("Reference peak feature row not unique")
    candidates = build_candidate_rows(groups, controls, enriched)
    if candidates.empty: raise ValueError("No recorded acoustic candidates in requested windows")
    candidates, settings = rank_candidates(candidates, enriched, ref_population)
    candidate_path = TABLES / "phase_d_acoustic_candidates.csv"
    candidates = preserve_manual_candidate_fields(candidates, candidate_path)
    if candidate_path.exists(): backup_csv(candidate_path)
    candidates.to_csv(candidate_path, index=False)
    controls.to_csv(TABLES / "phase_d_control_windows.csv", index=False)
    group_rows = []
    for group in groups.itertuples(index=False):
        exact = window_metrics(group.group_start, group.group_end, features, peaks)
        expanded_left, expanded_right = expanded_bounds(group.group_start, group.group_end, 300)
        expanded = window_metrics(expanded_left, expanded_right, features, peaks)
        group_candidates = candidates.loc[candidates.visual_group_id.eq(group.visual_group_id) & ~candidates.is_control]
        matched = candidates.loc[candidates.visual_group_id.eq(group.visual_group_id) & candidates.is_control]
        strongest = group_candidates.sort_values(["peak_prominence", "peak_timestamp"], ascending=[False, True]).head(1)
        closest = group_candidates.assign(abs_lag=group_candidates.lag_seconds.abs()).sort_values(
            ["abs_lag", "peak_timestamp"]).head(1)
        matched_controls = controls.loc[controls.matched_visual_group_id.eq(group.visual_group_id)]
        control_recorded = sum(window_metrics(row.control_start, row.control_end, features, peaks)["recorded_audio_seconds"]
                               for row in matched_controls.itertuples(index=False))
        top = group_candidates.sort_values("reference_similarity_score", ascending=False).head(1)
        group_rows.append({"visual_group_id": group.visual_group_id, "seed_ids": group.seed_ids,
                           "event_type": group.event_types, "visual_start": group.group_start,
                           "visual_end": group.group_end, "audio_coverage": exact["audio_coverage_class"],
                           "audio_coverage_fraction_exact": exact["coverage_fraction"],
                           "audio_recorded_seconds_exact": exact["recorded_audio_seconds"],
                           "wav_count_exact": exact["recording_count"],
                           "audio_recorded_seconds_5min": expanded["recorded_audio_seconds"],
                           "candidate_rate_per_recorded_minute_5min": expanded["peaks_per_recorded_minute"],
                           "matched_control_recorded_seconds": control_recorded,
                           "matched_control_candidate_rate_per_recorded_minute": len(matched)*60/control_recorded if control_recorded else np.nan,
                           "candidate_count": len(group_candidates),
                           "candidate_count_exact": int(((group_candidates.peak_timestamp > group.group_start) &
                                                         (group_candidates.peak_timestamp <= group.group_end)).sum()),
                           "candidate_count_within_30s": int(group_candidates.lag_seconds.abs().le(30).sum()),
                           "candidate_count_within_1min": int(group_candidates.lag_seconds.abs().le(60).sum()),
                           "candidate_count_within_5min": len(group_candidates),
                           "strongest_peak_timestamp": strongest.iloc[0].peak_timestamp if len(strongest) else "",
                           "strongest_peak_prominence": float(strongest.iloc[0].peak_prominence) if len(strongest) else np.nan,
                           "closest_peak_timestamp": closest.iloc[0].peak_timestamp if len(closest) else "",
                           "closest_peak_lag_seconds": float(closest.iloc[0].lag_seconds) if len(closest) else np.nan,
                           "top_reference_similarity": float(top.iloc[0].reference_similarity_score) if len(top) else np.nan,
                           "top_candidate_wav": top.iloc[0].wav_filename if len(top) else "",
                           "top_candidate_lag": float(top.iloc[0].lag_seconds) if len(top) else np.nan,
                           "top_candidate_prominence": float(top.iloc[0].peak_prominence) if len(top) else np.nan,
                           "top_control_similarity": float(matched.reference_similarity_score.max()) if len(matched) else np.nan,
                           "median_event_similarity": float(group_candidates.reference_similarity_score.median()) if len(group_candidates) else np.nan,
                           "median_control_similarity": float(matched.reference_similarity_score.median()) if len(matched) else np.nan,
                           "control_similarity_distribution": json.dumps({"count": len(matched),
                               "median": float(matched.reference_similarity_score.median()) if len(matched) else None,
                               "maximum": float(matched.reference_similarity_score.max()) if len(matched) else None}),
                           "manual_review_result": "pending_blinded_review",
                           "event_vs_control_interpretation": "descriptive_only_no_source_attribution",
                           "notes": "One shared microphone; unrecorded intervals are unknown, not silence."})
    summary = pd.DataFrame(group_rows)
    summary.to_csv(TABLES / "phase_d_event_summary.csv", index=False)
    selected = pd.concat([top_distinct_wavs(candidates, group.visual_group_id, control)
                          for group in groups.itertuples(index=False) for control in (False, True)], ignore_index=True)
    selected = selected.drop_duplicates("wav_relative_path").reset_index(drop=True)
    review_path = TABLES / "phase_d_manual_review_manifest.csv"
    mapping_path = TABLES / "phase_d_blind_mapping.csv"
    blind_path = TABLES / "phase_d_blind_review.csv"
    manifest, mapping, blind = prepare_blind_review(selected, RAW_AUDIO,
        PROJECT / "data/processed/targeted_multimodal/phase_d_audio_review",
        review_path, mapping_path)
    for path, frame in ((review_path, manifest), (mapping_path, mapping), (blind_path, blind)):
        if path.exists(): backup_csv(path)
        frame.to_csv(path, index=False)
    visual_timeline(events, FIGURES / "visual_milestone_timelines.png")
    reference_figure(reference, FIGURES / "reference_event_feature_profile.png")
    ranked_figure(candidates, FIGURES / "ranked_similarity.png")
    distribution_figure(candidates, FIGURES / "event_control_similarity_distribution.png")
    summary_figure(summary, FIGURES / "visual_groups_summary.png")
    for group in groups.itertuples(index=False):
        group_timeline(group, features, candidates, FIGURES / f"{group.visual_group_id}_timeline.png")
    for index, row in selected.iterrows():
        file_peaks = peaks.loc[peaks.source_relative_path.eq(row.wav_relative_path)]
        raw = RAW_AUDIO / row.wav_relative_path
        _, plotting = analyze_original_wav(raw, file_peaks)
        image_row = pd.Series({"review_id": f"phase_d_top_{index+1:02d}", "wav_filename": row.wav_filename,
                               "visual_group_id": row.visual_group_id, "wav_timestamp": row.wav_timestamp,
                               "relationship_to_visual_event": "blinded review selection",
                               "lag_seconds": "withheld"})
        diagnostic_figure(image_row, plotting, FIGURES / f"top_candidate_{index+1:02d}.png")
    # Phase C predeclared good coverage as >=50%. Fewer than two independent
    # well-recorded groups cannot establish reproducibility of an absence/presence.
    classification = ("insufficient_audio_coverage" if summary.audio_coverage.eq("good").sum() < 2
                      else "weak_or_inconsistent_pattern")
    findings = {"status": "automatic_similarity_ranking_pending_blinded_human_review",
                "preliminary_reproducibility_classification": classification,
                "classification_rule": "At least two independent visual groups must meet the predeclared Phase C good exact-interval coverage criterion (>=50%) before a reproducibility pattern can be assessed.",
                "visual_milestones": len(events), "visual_groups": len(groups),
                "groups_with_good_or_partial_exact_coverage": int(summary.audio_coverage.isin(["good", "partial"]).sum()),
                "event_candidate_peaks": int((~candidates.is_control).sum()),
                "control_candidate_peaks": int(candidates.is_control.sum()),
                "manual_review_wavs": len(manifest), "blinded_review_wavs": len(blind),
                "reference_profile": reference, "similarity_method": settings,
                "acoustic_candidate_source": "Existing Audio Phase 1c candidate table: scipy.signal.find_peaks on the RMS/energy envelope; no new full-dataset detection.",
                "control_selection": "Same day, fixed clock offsets, no overlap with any visual ±5-minute window, coverage matched without using peak activity; two requested per group.",
                "human_review_status": "18 blinded clips prepared; neutral labels initially blank. Existing Phase C2 label for reference is reported only as prior human input.",
                "limitations": ["One shared microphone and one experimental run.",
                                "Only existing independently reviewed milestones were analyzed.",
                                "Many peaks within each WAV are correlated, not independent observations.",
                                "Audio is intermittent; gaps are unknown, not silence.",
                                "File-level spectral features are not peak-specific.",
                                "Household and mechanical sources remain possible; blind manual review is pending.",
                                "A visual change may be gradual, too faint, substrate-borne, masked, or in an audio gap."]}
    (TABLES / "phase_d_initial_findings.json").write_text(json.dumps(findings, indent=2, default=str)+"\n")
    narrative = ["# Phase D exploratory findings", "",
                 f"{len(events)} independently reviewed visual milestones form {len(groups)} temporal groups. Only Phase B crack-onset intervals are available so far.",
                 f"{int(summary.audio_coverage.eq('good').sum())} groups have good exact-interval audio coverage under the predeclared >=50% criterion.",
                 f"The search reused {int((~candidates.is_control).sum())} event-window and {int(candidates.is_control.sum())} control candidate peaks from the existing find_peaks table.",
                 f"Preliminary classification: **{classification}**. Blinded listening remains pending.", "",
                 "The reference is an unexplained transient of unknown source. Its self-similarity is 1 by construction; this is not independent replication.",
                 "High feature similarity does not identify a physical source, and gaps without WAVs are unknown rather than silent.",
                 "One microphone observes all seeds; overlapping visual intervals were grouped before acoustic comparison.",
                 "A dramatic visual change may be gradual, too faint, substrate-borne, masked, or occur during an audio gap.", "",
                 f"Similarity method: {settings['standardization']}; {settings['score']}",
                 f"Controls: {findings['control_selection']}", "",
                 "Next: independently review later visual milestones in notebook 14 and label neutral-ID WAVs in notebook 15 before opening the mapping.", ""]
    (TABLES / "phase_d_initial_findings.md").write_text("\n".join(narrative))
    print(json.dumps({key: findings[key] for key in ("visual_milestones", "visual_groups", "groups_with_good_or_partial_exact_coverage",
        "event_candidate_peaks", "control_candidate_peaks", "manual_review_wavs", "preliminary_reproducibility_classification")}, indent=2))


if __name__ == "__main__": main()
