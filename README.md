# Listening to Seed Germination

### Image, Audio & Time-Series Analysis

Proof-of-concept research project developed with Maastricht University and Bejo. The repository contains analysis code, review notebooks, tests, documentation, and selected compact results. Raw experimental images and audio remain outside Git.

## Project overview

The project asks whether visible seed changes and short acoustic transients can be examined together without assuming that a recorded sound came from a particular seed. The current analysis is exploratory. It does not establish a germination-specific acoustic signature.

## Research question

Can independently reviewed visual intervals, intermittent audio recordings, and waveform diagnostics reveal useful temporal associations or recurring acoustic patterns during seed germination?

## Experimental setup and dataset

A fixed camera recorded a shared seed tray. Two microphones recorded the tray and its surrounding environment, so the audio is not localised to individual seeds. The complete local snapshot collection contains 114,565 timestamped 1920×1080 images, spanning 2026-07-04 16:10:57 through 2026-07-18 16:35:06. A separate 25-image pilot subset was used for initial visual inspection. Timestamps come from filenames; intervals are not assumed to be exactly regular.

The audio inventory contains 120,311 stereo WAV recordings at 48 kHz, each 2.75 seconds long. Acquisition was intermittent. Gaps between WAVs are unrecorded time, not silence. See [data/README.md](data/README.md) for the expected local layout and data availability.

## Methodology

1. Select a small set of visually trackable seeds without consulting audio candidates. Five seeds formed the current proof-of-concept subset.
2. Review each seed in a fixed image region. Record the last definitely negative image and the first definitely positive image for a visible crack or radicle observation.
3. Express each event as an uncertainty interval `(last negative, first positive]`. Establish these intervals independently of audio.
4. Inspect the corresponding recorded WAV coverage. Use RMS envelopes and `scipy.signal.find_peaks` to propose acoustic candidates, then examine original waveforms, spectrograms, and manual listening notes.
5. Search the complete WAV corpus for recurrence of one unexplained transient using zero-mean normalized cross-correlation on the original waveforms. Compare stereo channels separately, and retain the highest **positive** score per WAV.

The visual analysis also includes classical preprocessing and segmentation diagnostics, pretrained-model smoke tests, and exploratory reverse visual inspection. These are documented as diagnostics, not validated automatic germination measurements.

## Key results

- The five initially reviewed crack intervals are approximately **41–141 seconds** wide. The first positive image gives an upper bound, not the exact biological event time.
- An unexplained acoustic candidate was recorded near **2026-07-14 19:23:28.22**. The shared microphone setup does not permit attribution to an individual seed.
- The 0.30-second reference transient was compared with all **120,311** WAVs. The highest non-reference zero-mean normalized cross-correlation score was approximately **0.855**.
- Waveform-similar events occur at other times. Similarity does not show that the events share a physical source or were caused by germination. Current evidence is insufficient to define a germination-specific acoustic signature.

Selected summary tables and plots are kept under `results/`. Full inventories, checkpoint batches, model weights, copied review WAVs, and raw-frame presentation animations remain local.

## Repository structure

| Directory | Contents |
| --- | --- |
| `src/` | Reusable image, audio, review, and comparison code |
| `scripts/` | Command-line and batch entry points |
| `notebooks/` | Dataset inspection and human-review workflows |
| `tests/` | Automated checks for core logic and preservation rules |
| `docs/` | Annotation protocol and project documentation |
| `results/tables/` | Selected compact quantitative results and review summaries |
| `results/figures/` | Selected plots without raw-scene presentation footage |
| `data/` | Data layout documentation; raw data are excluded |

## Installation

Python 3.11 is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

`requirements-benchmark.txt` lists optional pretrained-model dependencies. The official SAM 2 checkout, model checkpoints, and legacy TensorFlow SavedModels require separate setup and are not included here.

## Running the analysis

Place the 25 pilot images in `data/raw/images/` to run `notebooks/01_dataset_inspection.ipynb` from the project root with the project Python environment. The human-review notebooks `11_target_seed_selection.ipynb`, `12_dense_visual_event_refinement.ipynb`, and `13_targeted_audio_review.ipynb` document the targeted visual and audio workflow. Run them in order when reproducing the review, preserving existing human annotations.

For full raw snapshots and audio stored elsewhere, configure paths before invoking scripts that need them:

```bash
export SEED_GERMINATION_SNAPSHOT_ROOT="/path/to/raw/snapshots"
export SEED_GERMINATION_AUDIO_ROOT="/path/to/raw/audio"
```

For example, `python scripts/run_audio_cross_correlation.py` performs the full resumable waveform search and writes checkpoint batches and complete rankings locally. It is computationally substantial and requires the external WAV corpus. The compact published tables show the method and result summaries; see the scripts for exact inputs and outputs.

## Cross-correlation search

The reference is `er_file_2026_07_14_19_23_28.wav`, cropped from **0.0725 to 0.3725 seconds** around the previously observed transient. For each candidate WAV, the zero-centred reference slides across both original stereo channels. The maximum positive normalized correlation is ranked; the reference self-match is flagged and excluded from external rankings. Low-energy windows are checked numerically, and candidate local RMS is retained for manual review. No denoising, band-pass filtering, or learned audio representation is used in this primary search.

The best external score is high relative to the corpus distribution, but correlation is a measure of waveform similarity. Manual listening and contextual review remain necessary; no source identification or accuracy metric follows from this ranking.

## Limitations

- Each audio file covers only 2.75 seconds, with large unrecorded intervals.
- Environmental and mechanical sounds can resemble the reference transient.
- Both microphones observe a shared tray and surroundings; seed-level acoustic source localisation is unavailable.
- Detailed visual timing is available for only five proof-of-concept seeds. Ambiguous images widen the event interval.
- The current results do not establish causation, a seed-specific sound, or a validated automatic detector.

## Data availability and reproducibility

Raw images and WAVs are excluded because of size, storage constraints, and experimental-data management. They must be obtained through the project data custodian to rerun image or audio processing. Paths in the published code can be configured through the environment variables above; the local folder layout is documented in [data/README.md](data/README.md). Saved notebook outputs were removed from the committed notebooks; local backups preserve the original outputs.

The repository retains source, tests, parameters, compact review tables, and selected figures. Rerunning a stage that writes a human-review table requires checking its documented preservation and backup workflow first.
