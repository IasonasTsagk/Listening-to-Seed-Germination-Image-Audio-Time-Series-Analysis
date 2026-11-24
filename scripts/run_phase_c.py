"""Run Phase C from existing human visual and precomputed audio tables."""
from __future__ import annotations

from pathlib import Path
import os
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.dense_visual import _backup_review_csv
from src.phase_c_alignment import (build_alignment, copy_review_wavs, findings_summary,
    group_visual_intervals, load_audio_tables, load_visual_intervals, preserve_manual_audio_review,
    targeted_audio_review, expanded_bounds)


def timeline_figure(group, features, peaks, output):
    left, right = expanded_bounds(group.group_start, group.group_end, 300)
    audio = features.loc[(features.end > left) & (features.start < right)]
    candidates = peaks.loc[(peaks.timestamp > left) & (peaks.timestamp <= right)]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axvspan(group.group_start, group.group_end, color="#d95f02", alpha=.22,
               label="Visual uncertainty interval")
    for recording in audio.itertuples(index=False):
        ax.plot([max(recording.start, left), min(recording.end, right)], [0, 0],
                color="#1b9e77", linewidth=5, solid_capstyle="butt")
    ax.scatter(candidates.timestamp, np.ones(len(candidates)), s=7, alpha=.30,
               color="#7570b3", label="Candidate acoustic peak")
    high = candidates.loc[candidates.high]
    ax.scatter(high.timestamp, np.full(len(high), 1.5), s=13, alpha=.65,
               color="#e7298a", label="High-prominence candidate")
    ax.set_xlim(left, right)
    ax.set_yticks([0, 1, 1.5], ["Recorded WAV spans", "All candidates", "High prominence"])
    ax.set_ylim(-.4, 1.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_title(f"{group.visual_group_id}: {group.seed_ids} (shared microphone)")
    ax.set_xlabel(f"{group.group_start.date()} recording timestamp; blank gaps have no WAV")
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def comparison_figure(groups, windows, controls, output):
    group_ids = groups.visual_group_id.tolist()
    x = np.arange(len(group_ids))
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    measures = [
        ("peaks_per_recorded_minute", "Candidate peaks / recorded min"),
        ("high_prominence_peaks_per_recorded_minute", "High-prominence peaks / recorded min"),
        ("median_peak_prominence", "Median peak prominence"),
        ("mean_rms", "Mean WAV RMS"),
    ]
    for ax, (column, label) in zip(axes, measures):
        event = [windows.loc[(windows.entity_type == "group") & (windows.entity_id == gid) &
                             (windows.window_scale == "5min"), column].iloc[0]
                 for gid in group_ids]
        control = [controls.loc[controls.matched_visual_group_id == gid, column].mean()
                   for gid in group_ids]
        ax.bar(x-.18, event, .35, label="Visual group ±5 min", color="#1b9e77")
        ax.bar(x+.18, control, .35, label="Matched control mean", color="#7570b3")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend()
    axes[-1].set_xticks(x, group_ids)
    fig.suptitle("Descriptive event–control activity; controls chosen without peak data")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summary_figure(groups, group_alignment, output):
    table = group_alignment.set_index("visual_group_id").loc[groups.visual_group_id]
    x = np.arange(len(table))
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    axes[0].bar(x, table.audio_coverage_fraction_exact, color="#1b9e77")
    axes[0].set_ylabel("Recorded fraction\nof exact interval")
    axes[0].set_ylim(0, 1)
    axes[1].bar(x-.18, table.candidate_peaks_exact, .35, label="Exact interval", color="#d95f02")
    axes[1].bar(x+.18, table.candidate_peaks_5min, .35, label="Expanded ±5 min", color="#7570b3")
    axes[1].set_ylabel("Candidate peaks\n(count; not normalized)")
    axes[1].legend()
    axes[1].set_xticks(x, table.index)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Three unique visual groups; one shared microphone")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    tables = PROJECT / "results/tables"
    targeted = tables / "targeted_multimodal"
    audio = tables / "audio"
    figures = PROJECT / "results/figures/targeted_multimodal/audio_alignment"
    review_copies = PROJECT / "data/processed/targeted_multimodal/audio_review"
    raw_audio = Path(os.environ.get("SEED_GERMINATION_AUDIO_ROOT", PROJECT / "data/raw/audio"))
    if not raw_audio.is_dir():
        raise FileNotFoundError(f"Raw audio root is not mounted: {raw_audio}")
    events = load_visual_intervals(targeted / "refined_visual_events.csv")
    events = events.loc[events.event_type == "crack"].copy()
    expected_seeds = {f"seed_{i:02d}" for i in range(1, 6)}
    if set(events.seed_id) != expected_seeds or events.seed_id.duplicated().any():
        raise ValueError("Current crack pilot needs one reviewed interval for each of five seeds")
    if len({stamp.date() for stamp in events.start}) != 1:
        raise ValueError("Current Phase C pilot expects events on one date")
    day = str(events.start.iloc[0].date())
    groups = group_visual_intervals(events)
    features, peaks = load_audio_tables(audio / "audio_file_features_july13_18.csv",
                                        audio / "audio_candidate_peaks_july13_18.csv", day)
    outputs = build_alignment(events, groups, features, peaks)
    figures.mkdir(parents=True, exist_ok=True)
    for group in groups.itertuples(index=False):
        timeline_figure(group, features, peaks, figures / f"{group.visual_group_id}_timeline.png")
    comparison_figure(groups, outputs["windows"], outputs["controls"], figures / "event_control_activity.png")
    summary_figure(groups, outputs["group"], figures / "visual_groups_summary.png")
    review = targeted_audio_review(groups, outputs["controls"], features, peaks)
    review = copy_review_wavs(review, raw_audio, review_copies)
    review_path = targeted / "targeted_audio_review_manifest.csv"
    review = preserve_manual_audio_review(review, review_path)
    if review_path.exists():
        _backup_review_csv(review_path)
    review.to_csv(review_path, index=False)
    outputs["seed"].to_csv(targeted / "visual_audio_alignment.csv", index=False)
    outputs["group"].to_csv(targeted / "visual_audio_group_alignment.csv", index=False)
    outputs["controls"].to_csv(targeted / "audio_control_windows.csv", index=False)
    outputs["windows"].to_csv(targeted / "visual_audio_window_metrics.csv", index=False)
    findings = findings_summary(events, groups, outputs["seed"], outputs["group"], review)
    findings["coverage_criteria"] = "good >= 50%; partial >= 20%; poor > 0%; none = 0% recorded fraction"
    findings["control_rule"] = "Same date and duration as group ±5 min; fixed offsets; exclude all event ±5 min spans; choose coverage match, then clock proximity; never use peaks for selection."
    findings["signed_lag_definition"] = "Negative before visual interval; zero inside (start,end]; positive after. Closest candidate searched within ±5 min."
    (targeted / "phase_c_initial_findings.json").write_text(json.dumps(findings, indent=2) + "\n")
    narrative = ["# Phase C exploratory findings", "",
                 f"Five human-reviewed crack intervals form {len(groups)} unique visual temporal groups.",
                 f"{findings['groups_with_good_or_partial_exact_audio_coverage']} groups have good/partial exact-interval audio coverage.",
                 f"{findings['groups_with_candidate_inside_exact_interval']} have at least one acoustic candidate in the exact visual interval.",
                 f"{findings['groups_with_candidate_in_post_5min']} have at least one candidate in the separate five-minute post-event window.",
                 "",
                 "These are temporal co-occurrences, not evidence that a seed made a sound.",
                 "One shared microphone observes the box, recordings are intermittent, and household noise can confound peaks.",
                 "The five seeds came from one run; overlapping seeds do not provide independent acoustic observations.",
                 "Seed_01 may be left-censored because evidence may precede its first dense frame.",
                 "Visual selection was independent of audio. Manual audio review remains blank and is the next action.",
                 "Suggested neutral manual labels: obvious_speech_or_household, impact_or_handling, mechanical_or_continuous, short_transient_unexplained, low_activity, uncertain.",
                 "",
                 f"Coverage: {findings['coverage_criteria']}",
                 f"Controls: {findings['control_rule']}",
                 f"Lag: {findings['signed_lag_definition']}", ""]
    (targeted / "phase_c_initial_findings.md").write_text("\n".join(narrative))
    print(json.dumps(findings, indent=2))
    print(f"Prepared {len(review)} targeted group/WAV review references and copies")


if __name__ == "__main__":
    main()
