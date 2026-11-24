"""Copy every recorded WAV in seed_05's reviewed crack interval ±5 minutes.

Selection uses filename timestamps only. Raw WAV bytes are never changed.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import os
from pathlib import Path
import re
import shutil


PROJECT = Path(__file__).resolve().parents[1]
RAW_AUDIO = Path(os.environ.get("SEED_GERMINATION_AUDIO_ROOT", PROJECT / "data/raw/audio"))
OUTPUT = PROJECT / "data/processed/targeted_multimodal/seed05_plusminus5_all_audio"
TABLES = PROJECT / "results/tables/targeted_multimodal"
MANIFEST = TABLES / "seed05_plusminus5_all_audio_manifest.csv"
TIMELINE = TABLES / "seed05_plusminus5_timeline.txt"
PLAYLIST = OUTPUT / "seed05_plusminus5_playlist.m3u"

VISUAL_START = datetime(2026, 7, 14, 19, 22, 37)
VISUAL_END = datetime(2026, 7, 14, 19, 23, 58)
START = datetime(2026, 7, 14, 19, 17, 37)
END = datetime(2026, 7, 14, 19, 28, 58)
WAV_DURATION_SECONDS = 2.75
REFERENCE_FILENAME = "er_file_2026_07_14_19_23_28.wav"
STAMP = re.compile(
    r"^er_file_(20\d{2})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})\.wav$",
    re.IGNORECASE,
)
FIELDS = ("order", "filename", "source_path", "copied_path",
          "recording_timestamp", "seconds_from_visual_interval_start",
          "seconds_from_visual_interval_end", "inside_exact_visual_interval",
          "relative_position")


def filename_timestamp(name: str) -> datetime | None:
    match = STAMP.fullmatch(name)
    if match is None:
        return None
    try:
        return datetime(*map(int, match.groups()))
    except ValueError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(root: Path) -> list[tuple[datetime, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Mount the INTENSO drive before exporting: {root}")
    found = []
    for directory, _, names in os.walk(root):
        for name in names:
            if Path(name).suffix.lower() != ".wav":
                continue
            stamp = filename_timestamp(name)
            if stamp is not None and START <= stamp <= END:
                found.append((stamp, Path(directory) / name))
    return sorted(found, key=lambda item: (item[0], str(item[1])))


def position(stamp: datetime) -> str:
    # A visual event is bounded by (last definite negative, first definite positive].
    if stamp <= VISUAL_START:
        return "before_visual_interval"
    if stamp <= VISUAL_END:
        return "inside_visual_interval"
    return "after_visual_interval"


def export() -> list[dict]:
    sources = discover(RAW_AUDIO)
    if not sources:
        raise FileNotFoundError(f"No timestamped WAVs found within {START}–{END} under {RAW_AUDIO}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    width = max(4, len(str(len(sources))))
    rows = []
    for order, (stamp, source) in enumerate(sources, 1):
        copied = OUTPUT / f"{order:0{width}d}_{stamp:%H-%M-%S}_{source.name}"
        source_hash = sha256(source)
        if copied.exists():
            if sha256(copied) != source_hash:
                raise ValueError(f"Existing copy has different content: {copied}")
        else:
            shutil.copy2(source, copied)
        if sha256(source) != source_hash or sha256(copied) != source_hash:
            raise IOError(f"Source changed or copy failed verification: {source}")
        relative = position(stamp)
        rows.append({"order": order, "filename": source.name,
                     "source_path": str(source), "copied_path": str(copied),
                     "recording_timestamp": stamp.isoformat(sep=" "),
                     "seconds_from_visual_interval_start": (stamp - VISUAL_START).total_seconds(),
                     "seconds_from_visual_interval_end": (stamp - VISUAL_END).total_seconds(),
                     "inside_exact_visual_interval": relative == "inside_visual_interval",
                     "relative_position": relative})
    copied_wavs = {path.resolve() for path in OUTPUT.glob("*.wav")}
    expected_wavs = {Path(row["copied_path"]).resolve() for row in rows}
    if copied_wavs != expected_wavs:
        raise ValueError("Output folder has missing or extra WAVs; resolve before using the manifest")
    stamps = [datetime.fromisoformat(row["recording_timestamp"]) for row in rows]
    if any(stamps[i] > stamps[i + 1] for i in range(len(stamps) - 1)):
        raise ValueError("Export is not chronological")
    if any(not START <= stamp <= END for stamp in stamps):
        raise ValueError("An exported timestamp is outside the requested window")
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with TIMELINE.open("w", encoding="utf-8") as stream:
        stream.write("Seed_05 visual interval: (2026-07-14 19:22:37, 19:23:58]\n")
        stream.write("Recordings are intermittent; gaps are not silence.\n\n")
        for row in rows:
            label = row["relative_position"].replace("_visual_interval", "")
            marker = "  <-- previously identified crack-like transient" if row["filename"] == REFERENCE_FILENAME else ""
            stream.write(f'{row["recording_timestamp"]}  [{label}]  {row["filename"]}{marker}\n')
    with PLAYLIST.open("w", encoding="utf-8") as stream:
        stream.write("#EXTM3U\n")
        for row in rows:
            stream.write(Path(row["copied_path"]).name + "\n")
    counts = {name: sum(row["relative_position"] == name for row in rows)
              for name in ("before_visual_interval", "inside_visual_interval", "after_visual_interval")}
    elapsed = (END - START).total_seconds()
    nominal_audio = len(rows) * WAV_DURATION_SECONDS
    print(f"Total WAVs found and copied: {len(rows)}")
    print(f"First available WAV: {stamps[0]}")
    print(f"Last available WAV:  {stamps[-1]}")
    print(f"Before / inside / after visual interval: {counts['before_visual_interval']} / {counts['inside_visual_interval']} / {counts['after_visual_interval']}")
    print(f"Nominal recorded duration ({WAV_DURATION_SECONDS:g} s per WAV): {nominal_audio:.2f} s")
    print(f"Elapsed search window: {elapsed:.0f} s")
    print(f"Approximate audio coverage: {100 * nominal_audio / elapsed:.1f}%")
    print("Recording gaps are unknown, not silence.")
    print(f"Copies: {OUTPUT}\nManifest: {MANIFEST}\nPlaylist: {PLAYLIST}\nTimeline: {TIMELINE}")
    return rows


if __name__ == "__main__":
    export()
