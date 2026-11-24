"""Full, resumable raw-waveform ZNCC search; no acoustic source attribution."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.audio_cross_correlation import (EXPECTED_CHANNELS, EXPECTED_DURATION,
    EXPECTED_RATE, PRIMARY_SECONDS, REFERENCE_NAME, REFERENCE_PEAK_SECONDS,
    FFTZNCC, channel_diagnostics, discover_wavs, make_template,
    read_original_float, score_wav)

RAW = Path(os.environ.get("SEED_GERMINATION_AUDIO_ROOT", PROJECT / "data/raw/audio"))
TABLES = PROJECT / "results/tables/targeted_multimodal"
FIGURES = PROJECT / "results/figures/targeted_multimodal/cross_correlation"
REVIEW = PROJECT / "data/processed/targeted_multimodal/cross_correlation_review"
CHECKPOINTS = TABLES / "cross_correlation_checkpoints"
BATCH_SIZE = 500
RESULT_FIELDS = ["filename", "relative_path", "wav_start_timestamp", "absolute_match_timestamp",
    "match_window_start_timestamp", "max_zncc", "lag_seconds", "local_candidate_RMS",
    "reference_RMS", "channel", "channel_method", "left_max_zncc", "right_max_zncc",
    "left_lag_seconds", "right_lag_seconds", "strongest_negative_zncc",
    "near_zero_energy_window_count", "sample_rate_hz", "channels", "duration_seconds",
    "is_reference_file", "valid_comparison", "warning"]
MANUAL_FIELDS = ["manual_label", "similarity_to_reference", "confidence",
                 "obvious_household_or_environmental", "notes"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_diagnostics(reference_path: Path, sample_paths, template) -> pd.DataFrame:
    rows = []
    for role, path in [("reference", reference_path), *[("reproducible_corpus_sample", p) for p in sample_paths]]:
        rate, samples = read_original_float(path)
        metrics = channel_diagnostics(samples, rate,
            (round(template.start_seconds*rate), round(template.end_seconds*rate)) if role == "reference" else None)
        rows.append({"role": role, "filename": path.name,
                     "relative_path": str(path.relative_to(RAW)), "sample_rate_hz": rate,
                     "channels": samples.shape[1], "duration_seconds": len(samples)/rate, **metrics})
    return pd.DataFrame(rows)


def reference_figures(samples: np.ndarray, template, rate: int) -> None:
    mono = samples.mean(axis=1)
    limit = round(.5*rate)
    time_axis = np.arange(limit)/rate
    width = max(1, round(.005*rate))
    rms = np.sqrt(np.convolve(mono[:limit].astype(np.float64)**2,
                              np.ones(width)/width, mode="same"))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax in axes:
        ax.axvspan(template.start_seconds, template.end_seconds, color="#d95f02", alpha=.2,
                   label="Fixed 0.30 s reference crop")
        ax.axvline(REFERENCE_PEAK_SECONDS, color="#a31e1e", ls="--", lw=1,
                   label="Existing candidate-peak timestamp")
        ax.grid(alpha=.15)
    axes[0].plot(time_axis, samples[:limit, 0], lw=.5, label="Original left")
    axes[0].plot(time_axis, samples[:limit, 1], lw=.5, alpha=.6, label="Original right")
    axes[0].set_ylabel("Amplitude (original PCM scale)"); axes[0].legend(fontsize=8)
    axes[1].plot(time_axis, rms, color="#1b9e77", lw=.8)
    axes[1].set_ylabel("Mean-channel RMS (5 ms display window)")
    axes[1].set_xlabel("Seconds from reference WAV start")
    fig.suptitle("Reference transient: onset, peak and decay within fixed crop")
    fig.tight_layout(); fig.savefig(FIGURES / "reference_transient_waveform.png", dpi=170); plt.close(fig)
    section = mono[:limit]
    frequency, times, power = signal.spectrogram(section, rate, nperseg=2048, noverlap=1536)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot = ax.pcolormesh(times, frequency/1000, 10*np.log10(np.maximum(power, 1e-15)),
                         shading="auto", cmap="magma")
    ax.axvspan(template.start_seconds, template.end_seconds, facecolor="none",
               edgecolor="cyan", linewidth=2)
    ax.axvline(REFERENCE_PEAK_SECONDS, color="white", ls="--", lw=.8)
    ax.set_ylim(0, rate/2000)
    ax.set_xlabel("Seconds from reference WAV start")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("Reference spectrogram; original mean of stereo channels for display only")
    fig.colorbar(plot, ax=ax, label="Power spectral density (dB/Hz)")
    fig.tight_layout(); fig.savefig(FIGURES / "reference_transient_spectrogram.png", dpi=170); plt.close(fig)


def run_checkpointed(paths, reference_path: Path, template, metadata: dict) -> Path:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    metadata_path = CHECKPOINTS / "search_metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Existing checkpoints use different corpus/template/method; reconcile before resuming")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
    engine = FFTZNCC(template)
    total_batches = (len(paths)+BATCH_SIZE-1)//BATCH_SIZE
    started = time.perf_counter()
    completed = 0
    for batch_number in range(total_batches):
        selected = paths[batch_number*BATCH_SIZE:(batch_number+1)*BATCH_SIZE]
        output = CHECKPOINTS / f"batch_{batch_number:06d}.csv"
        expected = [str(path.relative_to(RAW)) for _, path in selected]
        if output.exists():
            previous = pd.read_csv(output, usecols=["relative_path"], keep_default_na=False)
            if previous.relative_path.tolist() != expected:
                raise ValueError(f"Checkpoint has wrong file list: {output}")
        else:
            records = [score_wav(path, RAW, engine, reference_path) for _, path in selected]
            temporary = output.with_name(output.name+".tmp")
            pd.DataFrame(records).reindex(columns=RESULT_FIELDS).to_csv(temporary, index=False)
            temporary.replace(output)
        completed += len(selected)
        if (batch_number+1) % 10 == 0 or batch_number+1 == total_batches:
            print(f"Processed/checkpointed {completed:,}/{len(paths):,} WAVs "
                  f"({100*completed/len(paths):.1f}%, elapsed {(time.perf_counter()-started)/60:.1f} min)", flush=True)
    all_results = TABLES / "cross_correlation_all_results.csv"
    temporary = all_results.with_name(all_results.name+".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for batch_number in range(total_batches):
            with (CHECKPOINTS / f"batch_{batch_number:06d}.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                for row in reader:
                    writer.writerow(row)
    temporary.replace(all_results)
    return all_results


def visual_intervals() -> pd.DataFrame:
    path = TABLES / "visual_milestones.csv"
    if path.exists():
        events = pd.read_csv(path, keep_default_na=False)
    else:
        events = pd.read_csv(TABLES / "refined_visual_events.csv", keep_default_na=False)
        events["event_type"] = events.event_type.replace({"crack": "crack_onset"})
    events = events.loc[events.last_negative_timestamp.ne("") & events.first_positive_timestamp.ne("")].copy()
    events["start"] = pd.to_datetime(events.last_negative_timestamp)
    events["end"] = pd.to_datetime(events.first_positive_timestamp)
    return events


def align_visual(top: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in top.itertuples(index=False):
        stamp = pd.Timestamp(record.absolute_match_timestamp)
        matches = []
        for event in events.itertuples(index=False):
            if event.start < stamp <= event.end:
                lag = 0.0
            elif stamp <= event.start:
                lag = (stamp-event.start).total_seconds()
            else:
                lag = (stamp-event.end).total_seconds()
            if abs(lag) <= 300:
                matches.append((abs(lag), event.seed_id, event.event_type, lag))
        if matches:
            matches.sort()
            best_abs, _, _, lag = matches[0]
            relation = ("inside_known_visual_interval" if best_abs == 0 else
                        "within_30_seconds" if best_abs <= 30 else
                        "within_1_minute" if best_abs <= 60 else "within_5_minutes")
        else:
            lag, relation = np.nan, "outside_all_known_target_seed_windows"
        rows.append({"rank": record.rank, "filename": record.filename,
                     "relative_path": record.relative_path,
                     "absolute_match_timestamp": record.absolute_match_timestamp,
                     "max_zncc": record.max_zncc,
                     "visual_relationship": relation,
                     "known_seed_ids_within_5min": ";".join(sorted({item[1] for item in matches})),
                     "known_event_types_within_5min": ";".join(sorted({item[2] for item in matches})),
                     "lag_to_nearest_visual_interval_seconds": lag})
    return pd.DataFrame(rows)


def score_plots(valid: pd.DataFrame, top: pd.DataFrame) -> None:
    scores = valid.max_zncc.dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(scores, bins=100, color="#294e70")
    ax.set_xlabel("Maximum positive ZNCC per WAV")
    ax.set_ylabel("WAV count")
    ax.set_title("Full non-reference score distribution")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "correlation_score_distribution.png", dpi=170); plt.close(fig)
    tail = float(np.percentile(scores, 99))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(scores[scores >= tail], bins=70, color="#7570b3")
    ax.set_xlabel(f"Maximum positive ZNCC (>= 99th percentile, {tail:.4f})")
    ax.set_ylabel("WAV count")
    ax.set_title("Upper score tail; reference excluded")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "correlation_score_histogram_zoomed_top_tail.png", dpi=170); plt.close(fig)
    timeline = top.copy()
    timeline["time"] = pd.to_datetime(timeline.absolute_match_timestamp)
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.scatter(timeline.time, timeline.max_zncc, c=timeline.max_zncc, cmap="viridis", s=35)
    ax.set_xlim(pd.to_datetime(valid.wav_start_timestamp).min(), pd.to_datetime(valid.wav_start_timestamp).max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xlabel("Full experiment timeline")
    ax.set_ylabel("ZNCC of top 50 non-reference WAVs")
    ax.set_title("Timing of waveform-similar acoustic candidates")
    ax.grid(alpha=.2)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIGURES / "top_matches_timeline.png", dpi=170); plt.close(fig)
    counts = timeline.time.dt.date.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(day) for day in counts.index], counts.values, color="#1b9e77")
    ax.set_ylabel("Top-50 WAVs")
    ax.set_xlabel("Recording date")
    ax.set_title("Top waveform-similar candidates by day; no biological attribution")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "top_matches_by_day.png", dpi=170); plt.close(fig)


def diagnostic_figure(rank: int, row, candidate: np.ndarray, template, rate: int) -> None:
    channel = 0 if row.channel == "left" else 1
    start = round(row.lag_seconds*rate)
    segment = candidate[start:start+template.length, channel]
    reference = template.samples[channel]
    seconds = np.arange(template.length)/rate
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(seconds, reference, lw=.5, color="#294e70")
    axes[0].set_ylabel("Reference amplitude")
    axes[1].plot(seconds, segment, lw=.5, color="#d95f02")
    axes[1].set_ylabel("Candidate amplitude")
    for axis, samples, title in ((axes[2], reference, "Reference"), (axes[3], segment, "Candidate")):
        f, t, power = signal.spectrogram(samples, rate, nperseg=2048, noverlap=1536)
        axis.pcolormesh(t, f/1000, 10*np.log10(np.maximum(power, 1e-15)), shading="auto", cmap="magma")
        axis.set_ylim(0, rate/2000)
        axis.set_ylabel(f"{title} kHz")
    for axis in axes[:2]:
        axis.set_xlim(0, template.length/rate)
        axis.grid(alpha=.1)
    axes[3].set_xlabel("Seconds within aligned 0.30 s segments")
    fig.suptitle(f"Top match {rank:02d} · {row.filename} · {row.absolute_match_timestamp}\n"
                 f"ZNCC={row.max_zncc:.4f}, {row.channel}, lag={row.lag_seconds:.4f}s, "
                 f"local RMS={row.local_candidate_RMS:.6g}; original amplitudes", fontsize=10)
    fig.savefig(FIGURES / f"top_match_{rank:02d}.png", dpi=160)
    plt.close(fig)


def review_outputs(top: pd.DataFrame, template) -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    manifest_path = TABLES / "cross_correlation_manual_review.csv"
    existing = pd.read_csv(manifest_path, keep_default_na=False) if manifest_path.exists() else pd.DataFrame()
    rows = []
    for index, row in enumerate(top.head(20).itertuples(index=False), 1):
        source = RAW / row.relative_path
        destination = REVIEW / f"{index:02d}_{row.filename}"
        if destination.exists():
            if sha256(destination) != sha256(source):
                raise ValueError(f"Existing review copy differs: {destination}")
        else:
            shutil.copy2(source, destination)
        if sha256(destination) != sha256(source):
            raise IOError(f"Review copy failed hash verification: {source}")
        rate, samples = read_original_float(source)
        diagnostic_figure(index, row, samples, template, rate)
        item = {"rank": index, "filename": row.filename, "relative_path": row.relative_path,
                "timestamp": row.absolute_match_timestamp, "max_zncc": row.max_zncc,
                "lag_seconds": row.lag_seconds, "local_RMS": row.local_candidate_RMS,
                "review_copy_path": str(destination),
                **{field: "" for field in MANUAL_FIELDS}}
        if not existing.empty:
            prior = existing.loc[existing.relative_path.eq(row.relative_path)]
            if len(prior) == 1:
                for field in MANUAL_FIELDS: item[field] = prior.iloc[0][field]
        rows.append(item)
    if not existing.empty:
        reviewed = existing.loc[existing[MANUAL_FIELDS].fillna("").astype(str).ne("").any(axis=1)]
        if not set(reviewed.relative_path).issubset({row["relative_path"] for row in rows}):
            raise ValueError("Regeneration would remove human-reviewed top-match WAVs")
        backup = manifest_path.with_name(f"{manifest_path.stem}.backup_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
        shutil.copy2(manifest_path, backup)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


def sensitivity(top_pool: pd.DataFrame, reference_samples: np.ndarray, rate: int) -> pd.DataFrame:
    pool = top_pool.head(200).copy()
    for duration, suffix in ((.20, "0p20"), (.40, "0p40")):
        template = make_template(reference_samples, rate, duration)
        engine = FFTZNCC(template)
        scores = []
        for row in pool.itertuples(index=False):
            candidate_rate, samples = read_original_float(RAW / row.relative_path)
            result = engine.compare(samples) if candidate_rate == rate else {"valid_comparison": False}
            scores.append(result.get("max_zncc", np.nan))
        pool[f"score_{suffix}"] = scores
        pool[f"rank_{suffix}_within_top200_pool"] = pool[f"score_{suffix}"].rank(ascending=False, method="first").astype("Int64")
    pool["score_0p30_primary"] = pool.max_zncc
    pool["rank_0p30_primary"] = pool["rank"]
    columns = ["rank_0p30_primary", "filename", "relative_path", "absolute_match_timestamp",
               "score_0p20", "score_0p30_primary", "score_0p40",
               "rank_0p20_within_top200_pool", "rank_0p40_within_top200_pool"]
    output = pool[columns]
    output.to_csv(TABLES / "cross_correlation_template_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    subset = pool.head(50)
    ax.scatter(subset["rank"], subset.rank_0p20_within_top200_pool, s=25, label="0.20 s")
    ax.scatter(subset["rank"], subset.rank_0p40_within_top200_pool, s=25, label="0.40 s")
    ax.plot([1, 50], [1, 50], color="gray", ls="--", lw=1)
    ax.set_xlabel("Primary 0.30 s rank (full corpus)")
    ax.set_ylabel("Alternative-template rank within primary top-200 pool")
    ax.set_title("Template-length sensitivity; alternative ranks are pool-limited")
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "template_rank_stability.png", dpi=170); plt.close(fig)
    return output


def main() -> None:
    if not RAW.is_dir(): raise FileNotFoundError(f"Mount raw audio volume: {RAW}")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = discover_wavs(RAW)
    references = [path for _, path in paths if path.name == REFERENCE_NAME]
    if len(references) != 1: raise ValueError("Expected one reference WAV in complete corpus")
    reference_path = references[0]
    rate, samples = read_original_float(reference_path)
    if rate != EXPECTED_RATE or samples.shape[1] != EXPECTED_CHANNELS or abs(len(samples)/rate-EXPECTED_DURATION) > 1/rate:
        raise ValueError("Reference WAV format differs from documented expectation")
    template = make_template(samples, rate)
    sample_paths = [paths[i][1] for i in np.linspace(0, len(paths)-1, 24, dtype=int)]
    diagnostics = reference_diagnostics(reference_path, sample_paths, template)
    diagnostics.to_csv(TABLES / "cross_correlation_channel_diagnostics.csv", index=False)
    reference_figures(samples, template, rate)
    corpus_hash = hashlib.sha256("\n".join(str(path.relative_to(RAW)) for _, path in paths).encode()).hexdigest()
    metadata = {"reference_relative_path": str(reference_path.relative_to(RAW)),
                "reference_sha256": sha256(reference_path), "reference_peak_seconds": REFERENCE_PEAK_SECONDS,
                "reference_start_seconds": template.start_seconds, "reference_end_seconds": template.end_seconds,
                "template_samples": template.length, "sample_rate_hz": rate,
                "channel_method": "separate_channels_max_positive",
                "primary_method": "Per-channel zero-mean normalized raw-waveform cross-correlation; FFT numerator, exact rolling denominator; positive maximum; no filtering",
                "score_timestamp": "Aligned reference peak: WAV start + matched template-start lag + reference peak offset within template",
                "corpus_count": len(paths), "corpus_path_hash": corpus_hash,
                "batch_size": BATCH_SIZE}
    all_path = run_checkpointed(paths, reference_path, template, metadata)
    all_rows = pd.read_csv(all_path, low_memory=False)
    valid = all_rows.loc[all_rows.valid_comparison.eq(True) & ~all_rows.is_reference_file.eq(True)].copy()
    valid = valid.sort_values(["max_zncc", "wav_start_timestamp", "relative_path"],
                              ascending=[False, True, True]).reset_index(drop=True)
    valid["rank"] = np.arange(1, len(valid)+1)
    top = valid.head(50).copy()
    top.to_csv(TABLES / "cross_correlation_top_matches.csv", index=False)
    percentiles = {f"p{value}": float(np.percentile(valid.max_zncc, value))
                   for value in (50, 90, 95, 99, 99.5, 99.9, 99.95, 99.99)}
    summary = {"corpus_wavs": len(all_rows), "valid_wavs": int(all_rows.valid_comparison.eq(True).sum()),
               "invalid_or_failed_wavs": int(all_rows.valid_comparison.ne(True).sum()),
               "reference_self_score": float(all_rows.loc[all_rows.is_reference_file.eq(True), "max_zncc"].iloc[0]),
               "reference_crop_start_seconds": template.start_seconds,
               "reference_crop_end_seconds": template.end_seconds,
               "channel_method": metadata["channel_method"],
               "reference_channel_correlation": diagnostics.iloc[0].channel_pearson_correlation,
               "sample_min_channel_correlation": diagnostics.iloc[1:].channel_pearson_correlation.min(),
               **percentiles}
    pd.DataFrame([summary]).to_csv(TABLES / "cross_correlation_summary.csv", index=False)
    score_plots(valid, top)
    align_visual(top, visual_intervals()).to_csv(TABLES / "cross_correlation_visual_alignment.csv", index=False)
    review_outputs(top, template)
    sensitivity(valid, samples, rate)
    (TABLES / "cross_correlation_method.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("Completed full raw-waveform ZNCC search:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
