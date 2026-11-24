# Local experimental data

Raw experiment data are intentionally absent from GitHub. The collection is too large for a source repository and is managed separately for storage and experimental-data reasons.

The fixed camera produced approximately 112,000–115,000 timestamped 1920×1080 snapshots at roughly 10-second cadence. The local scan used for this repository found 114,565 timestamped images from 2026-07-04 through 2026-07-18. The first day includes a substantial acquisition gap, so regular cadence must not be assumed. The 25-image pilot subset is a separate convenience copy.

The audio collection contains 120,311 stereo WAV files, each 2.75 seconds at 48 kHz. Recording was intermittent. Two microphones captured the shared tray and surrounding environment; their channels do not identify individual seeds. Gaps between files represent unrecorded time, not silence.

Suggested local layout:

```text
data/
  raw/
    images/       # optional 25-image visual pilot
    snapshots/    # optional full timestamped snapshot tree
    audio/        # optional full WAV tree
  processed/      # local derived review copies and intermediate files
  features/       # local derived feature tables
```

The full datasets may instead remain on an external drive. Configure their locations without changing source code:

```bash
export SEED_GERMINATION_SNAPSHOT_ROOT="/path/to/raw/snapshots"
export SEED_GERMINATION_AUDIO_ROOT="/path/to/raw/audio"
```

The defaults are `data/raw/snapshots/` and `data/raw/audio/` inside the project. Pilot notebooks expect the 25-image subset under `data/raw/images/`. Do not copy the full experiment into Git. Preserve original filenames because their parsed timestamps define the observation times.
