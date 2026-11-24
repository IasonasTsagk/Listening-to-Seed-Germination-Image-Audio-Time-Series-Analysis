"""Read-only WAV discovery, timestamp parsing, header metadata, and coverage."""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import re, wave
import pandas as pd

_STAMP = re.compile(r"^er_file_(?P<year>20\d{2})_(?P<month>\d{1,2})_(?P<day>\d{1,2})_(?P<hour>\d{1,2})_(?P<minute>\d{1,2})_(?P<second>\d{1,2})\.wav$", re.I)

def parse_audio_timestamp(filename: str) -> datetime | None:
    match = _STAMP.match(Path(filename).name)
    if not match: return None
    try: return datetime(**{k:int(v) for k,v in match.groupdict().items()})
    except ValueError: return None

def discover_audio_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".wav" and not path.name.startswith("._"):
            yield path

def wav_header(path: Path) -> dict:
    try:
        with wave.open(str(path), "rb") as handle:
            rate, channels, frames, width = handle.getframerate(), handle.getnchannels(), handle.getnframes(), handle.getsampwidth()
        return {"readable": True, "sample_rate_hz": rate, "channels": channels, "frame_count": frames, "sample_width_bytes": width,
                "duration_seconds": frames / rate if rate else None, "error_message": ""}
    except Exception as error:
        return {"readable": False, "sample_rate_hz": None, "channels": None, "frame_count": None, "sample_width_bytes": None,
                "duration_seconds": None, "error_message": f"{type(error).__name__}: {error}"}

def inventory_rows(root: Path):
    for path in discover_audio_files(root):
        timestamp = parse_audio_timestamp(path.name); meta = wav_header(path)
        end = timestamp + timedelta(seconds=meta["duration_seconds"]) if timestamp and meta["duration_seconds"] is not None else None
        yield {"filename": path.name, "relative_path": str(path.relative_to(root)), "parent_directory": path.parent.name, "extension": path.suffix.lower(),
               "file_size_bytes": path.stat().st_size, "parsed_timestamp": timestamp, "timestamp_source": "filename" if timestamp else "", "start_timestamp": timestamp,
               "end_timestamp": end, **meta, "metadata_valid": bool(meta["readable"] and meta["duration_seconds"] is not None), "timestamp_valid": timestamp is not None}

def add_gap_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["start_timestamp", "relative_path"], na_position="last").copy()
    out["next_file_start_timestamp"] = out["start_timestamp"].shift(-1)
    out["gap_to_next_start_seconds"] = (out["next_file_start_timestamp"] - out["start_timestamp"]).dt.total_seconds()
    out["uncovered_gap_seconds"] = (out["next_file_start_timestamp"] - out["end_timestamp"]).dt.total_seconds()
    return out
