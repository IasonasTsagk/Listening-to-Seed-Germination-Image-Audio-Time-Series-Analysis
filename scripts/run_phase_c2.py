"""Prepare eleven original-signal diagnostics and separate listening copies."""
from __future__ import annotations

from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.phase_c2_audio_review import (analyze_original_wav, backup_csv,
    merge_review_fields, review_identifier, select_adjacent_recordings,
    write_listening_copy, write_review_summary)

RAW_ROOT = Path(os.environ.get("SEED_GERMINATION_AUDIO_ROOT", PROJECT / "data/raw/audio"))
TABLES = PROJECT / "results/tables/targeted_multimodal"
FIGURES = PROJECT / "results/figures/targeted_multimodal/audio_detailed_review"
LISTENING = PROJECT / "data/processed/targeted_multimodal/audio_review/normalized_for_listening"


def diagnostic_figure(row, plotting: dict, output: Path) -> None:
    rate, mono = plotting["rate"], plotting["mono"]
    peaks = plotting["peaks"]
    times = np.arange(len(mono))/rate
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    axes[0].plot(times, mono, color="#294e70", lw=.5)
    bound = max(float(np.max(np.abs(mono)))*1.08, 1e-8)
    axes[0].set_ylim(-bound, bound)
    axes[0].set_ylabel("Original mean-mono\namplitude")
    axes[0].set_title(f"{row.review_id} · {row.wav_filename} · {row.visual_group_id}")
    axes[1].plot(plotting["envelope_time"], plotting["envelope"], color="#208766", lw=.9)
    if len(peaks):
        axes[1].scatter(peaks.relative_peak_time_seconds, peaks.envelope_peak_value,
                        s=9, alpha=.5, color="#b34949", label="Existing candidate peaks")
        strongest = peaks.sort_values("prominence", ascending=False).iloc[0]
        axes[1].axvline(strongest.relative_peak_time_seconds, color="#c22b2b",
                        lw=1, linestyle="--", label="Strongest candidate")
        axes[1].legend(fontsize=8)
    axes[1].set_ylabel("Original RMS envelope")
    f, t, power = signal.spectrogram(mono, rate, nperseg=2048, noverlap=1536)
    magnitude_db = 10*np.log10(np.maximum(power, 1e-15))
    image = axes[2].pcolormesh(t, f/1000, magnitude_db, shading="auto", cmap="magma")
    axes[2].set_ylim(0, rate/2000)
    axes[2].set_ylabel("Frequency (kHz)")
    axes[2].set_xlabel("Seconds within recorded WAV; gaps are not shown")
    fig.colorbar(image, ax=axes[2], label="Power spectral density (dB/Hz)")
    for ax in axes[:2]:
        ax.set_xlim(0, len(mono)/rate)
        ax.set_xlabel("Seconds within recorded WAV")
        ax.grid(alpha=.15)
    relation = row.relationship_to_visual_event
    lag_text = str(row.lag_seconds) if str(row.lag_seconds) != "" else "not available"
    fig.suptitle(f"{row.wav_timestamp} · {relation} · lag {lag_text} s\n"
                 "Candidate peak ≠ germination sound · measurements use original WAV", fontsize=10)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    manifest_path = TABLES / "targeted_audio_review_manifest.csv"
    metrics_path = TABLES / "targeted_audio_detailed_metrics.csv"
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    if len(manifest) != 11 or manifest.duplicated(["visual_group_id", "wav_relative_path"]).any():
        raise ValueError("Expected eleven unique current review entries")
    features = pd.read_csv(PROJECT / "results/tables/audio/audio_file_features_july13_18.csv")
    features["start"] = pd.to_datetime(features.start_timestamp)
    features["end"] = pd.to_datetime(features.end_timestamp)
    peaks = pd.read_csv(PROJECT / "results/tables/audio/audio_candidate_peaks_july13_18.csv")
    peaks = peaks.loc[peaks.source_relative_path.isin(manifest.wav_relative_path)].copy()
    FIGURES.mkdir(parents=True, exist_ok=True)
    LISTENING.mkdir(parents=True, exist_ok=True)
    rows, context_rows = [], []
    root = RAW_ROOT.resolve()
    for index, item in manifest.reset_index(drop=True).iterrows():
        review_id = review_identifier(index)
        source = (root / item.wav_relative_path).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise FileNotFoundError(f"Raw review WAV missing: {item.wav_relative_path}")
        clip_peaks = peaks.loc[peaks.source_relative_path.eq(item.wav_relative_path)]
        metrics, plotting = analyze_original_wav(source, clip_peaks)
        listening_path = LISTENING / f"{source.stem}_normalized_for_listening.wav"
        write_listening_copy(source, listening_path)
        row = {"review_id": review_id, "visual_group_id": item.visual_group_id,
               "seed_ids": item.seed_ids, "wav_filename": item.wav_filename,
               "wav_relative_path": item.wav_relative_path, "wav_timestamp": item.wav_timestamp,
               "visual_interval_start": item.visual_interval_start,
               "visual_interval_end": item.visual_interval_end,
               "relationship_to_visual_event": item.relationship_to_visual_event,
               "lag_seconds": item.lag_seconds, **metrics,
               "original_raw_path": str(source), "listening_copy_path": str(listening_path),
               "diagnostic_figure_path": str(FIGURES / f"{review_id}_audio_diagnostics.png"),
               "manual_audio_label": item.manual_audio_label,
               "manual_audio_confidence": item.manual_audio_confidence, "notes": item.notes}
        rows.append(row)
        diagnostic_figure(pd.Series(row), plotting, FIGURES / f"{review_id}_audio_diagnostics.png")
        context = select_adjacent_recordings(features, item.wav_relative_path)
        context.insert(0, "review_id", review_id)
        context.insert(1, "visual_group_id", item.visual_group_id)
        context["original_raw_path"] = context.wav_relative_path.map(lambda relative: str(root / relative))
        context_rows.append(context)
    detailed = pd.DataFrame(rows)
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path, keep_default_na=False)
        detailed = merge_review_fields(detailed, existing)
        backup_csv(metrics_path)
    detailed.to_csv(metrics_path, index=False)
    pd.concat(context_rows, ignore_index=True).to_csv(TABLES / "targeted_audio_context.csv", index=False)
    summary = write_review_summary(metrics_path, TABLES / "phase_c2_review_summary.json")
    print(f"Inspected {len(manifest)} original WAVs; wrote {len(rows)} diagnostics and listening copies")
    print(f"Context rows: {sum(map(len, context_rows))}; labeled clips: {summary['labeled_review_wavs']}")


if __name__ == "__main__":
    main()
