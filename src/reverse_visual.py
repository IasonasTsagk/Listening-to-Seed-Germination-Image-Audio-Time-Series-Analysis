"""Exploratory full-scene review around a known acoustic timestamp.

Image differences rank places to inspect; they do not identify a seed, sound
source, or biological event. All source images are read-only.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .io_utils import parse_timestamp_from_filename

ACOUSTIC_TIME = pd.Timestamp("2026-07-14 19:23:28.220")
PRIMARY_START = pd.Timestamp("2026-07-14 19:22:28")
PRIMARY_END = pd.Timestamp("2026-07-14 19:24:28")
WIDE_START = pd.Timestamp("2026-07-14 19:21:28")
WIDE_END = pd.Timestamp("2026-07-14 19:25:28")
CANDIDATE_COLUMNS = ("candidate_id", "source_type", "x_min", "y_min", "x_max", "y_max",
                     "basis_for_selection", "visual_change_near_event", "likely_seed_present",
                     "likely_event_type", "notes")


def window_inventory(index_path: Path, raw_root: Path, start: pd.Timestamp,
                     end: pd.Timestamp) -> pd.DataFrame:
    """Use the existing dense index, then verify each selected raw frame."""
    index = pd.read_csv(index_path, keep_default_na=False)
    index["indexed_timestamp"] = pd.to_datetime(index.timestamp)
    chosen = index.loc[index.indexed_timestamp.between(start, end, inclusive="both")].copy()
    if chosen.empty:
        raise FileNotFoundError(f"No indexed raw images in {start}–{end}")
    rows = []
    root = raw_root.resolve()
    for record in chosen.itertuples(index=False):
        stamp = parse_timestamp_from_filename(record.filename)
        if stamp is None or pd.Timestamp(stamp) != record.indexed_timestamp:
            raise ValueError(f"Filename/index timestamp mismatch: {record.filename}")
        path = Path(record.source_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"Indexed raw frame unavailable: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
            readable = True
        except (OSError, ValueError):
            readable = False
        rows.append({"filename": record.filename, "path": str(path),
                     "timestamp": pd.Timestamp(stamp),
                     "seconds_from_acoustic_event": (pd.Timestamp(stamp)-ACOUSTIC_TIME).total_seconds(),
                     "readable": readable})
    frame = pd.DataFrame(rows).sort_values(["timestamp", "filename"]).reset_index(drop=True)
    if frame.timestamp.duplicated().any() or not frame.readable.all():
        raise ValueError("Window has duplicate timestamps or unreadable images")
    return frame


def read_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def grayscale(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0].astype(np.float32)*.299 +
            rgb[..., 1].astype(np.float32)*.587 +
            rgb[..., 2].astype(np.float32)*.114)


def tile_change_summary(frames: list[np.ndarray], timestamps: list[pd.Timestamp],
                        columns: int = 8, rows: int = 6) -> tuple[pd.DataFrame, list[np.ndarray]]:
    if len(frames) < 2 or len(frames) != len(timestamps):
        raise ValueError("Need at least two ordered frames with matching timestamps")
    if any(timestamps[i] >= timestamps[i+1] for i in range(len(timestamps)-1)):
        raise ValueError("Timestamps must be strictly increasing")
    height, width = frames[0].shape[:2]
    if columns < 1 or rows < 1 or width < columns or height < rows:
        raise ValueError("Invalid grid shape")
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise ValueError("Image dimensions changed within review window")
    x_edges = np.linspace(0, width, columns+1, dtype=int)
    y_edges = np.linspace(0, height, rows+1, dtype=int)
    scores = np.zeros((len(frames)-1, rows, columns), dtype=np.float64)
    differences = []
    previous = grayscale(frames[0])
    for pair_index, image in enumerate(frames[1:]):
        current = grayscale(image)
        difference = np.abs(current-previous)
        differences.append(difference)
        for row in range(rows):
            for col in range(columns):
                tile = difference[y_edges[row]:y_edges[row+1], x_edges[col]:x_edges[col+1]]
                scores[pair_index, row, col] = float(tile.mean())
        previous = current
    records = []
    for row in range(rows):
        for col in range(columns):
            tile_scores = scores[:, row, col]
            strongest_pair = int(tile_scores.argmax())
            records.append({"tile_id": f"tile_r{row+1:02d}_c{col+1:02d}",
                            "row": row+1, "col": col+1,
                            "x_min": int(x_edges[col]), "y_min": int(y_edges[row]),
                            "x_max": int(x_edges[col+1]), "y_max": int(y_edges[row+1]),
                            "cumulative_change_score": float(tile_scores.sum()),
                            "max_pair_change_score": float(tile_scores.max()),
                            "frame_pair_of_max_change": f"{timestamps[strongest_pair]} → {timestamps[strongest_pair+1]}"})
    return (pd.DataFrame(records).sort_values(["cumulative_change_score", "tile_id"],
                                              ascending=[False, True]).reset_index(drop=True), differences)


def intersects(left, right) -> bool:
    return (int(left["x_min"]) < int(right["x_max"]) and int(left["x_max"]) > int(right["x_min"]) and
            int(left["y_min"]) < int(right["y_max"]) and int(left["y_max"]) > int(right["y_min"]))


def initial_candidates(top_tiles: pd.DataFrame, path: Path) -> pd.DataFrame:
    desired = pd.DataFrame([{
        "candidate_id": f"candidate_{index+1:02d}", "source_type": "grid_tile",
        **{key: int(tile[key]) for key in ("x_min", "y_min", "x_max", "y_max")},
        "basis_for_selection": "top cumulative frame-difference change",
        "visual_change_near_event": "", "likely_seed_present": "",
        "likely_event_type": "", "notes": "",
    } for index, tile in top_tiles.head(5).iterrows()], columns=CANDIDATE_COLUMNS)
    if len(desired) != 5:
        raise ValueError("Need five automatic tile candidates")
    if path.exists():
        existing = pd.read_csv(path, keep_default_na=False)
        if set(CANDIDATE_COLUMNS)-set(existing) or existing.candidate_id.duplicated().any():
            raise ValueError("Existing candidate table has invalid schema or IDs")
        for row in desired.itertuples(index=False):
            old = existing.loc[existing.candidate_id.eq(row.candidate_id)]
            if len(old) != 1 or old.iloc[0].source_type != "grid_tile" or any(
                int(old.iloc[0][key]) != getattr(row, key) for key in ("x_min", "y_min", "x_max", "y_max")
            ):
                raise ValueError("Automatic tile candidates changed; existing human review was not overwritten")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    desired.to_csv(path, index=False)
    return desired


def append_manual_candidate(path: Path, bounds: tuple[int, int, int, int],
                            image_size: tuple[int, int]) -> str:
    x0, y0, x1, y1 = map(int, bounds)
    width, height = image_size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Manual ROI must lie inside the full image")
    table = pd.read_csv(path, keep_default_na=False)
    if table.candidate_id.duplicated().any():
        raise ValueError("Duplicate candidate IDs")
    number = max(int(value.split("_")[-1]) for value in table.candidate_id) + 1
    candidate_id = f"candidate_{number:02d}"
    row = {"candidate_id": candidate_id, "source_type": "manual_roi",
           "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1,
           "basis_for_selection": "manual full-frame inspection",
           "visual_change_near_event": "", "likely_seed_present": "",
           "likely_event_type": "", "notes": ""}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    shutil.copy2(path, path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}"))
    pd.concat([table, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    return candidate_id


def annotated_preview(rgb: np.ndarray, label: str, max_size=(960, 540)) -> Image.Image:
    image = Image.fromarray(rgb).copy()
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image.width, image.height+28), "black")
    canvas.paste(image, (0, 28))
    ImageDraw.Draw(canvas).text((8, 7), label, fill="white")
    return canvas


def contact_sheet(images: list[Image.Image], path: Path, columns=4, background="#101820") -> None:
    if not images:
        raise ValueError("Cannot make an empty contact sheet")
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = (len(images)+columns-1)//columns
    sheet = Image.new("RGB", (columns*width, rows*height), background)
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns)*width, (index // columns)*height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def crop_timeline(frames: list[np.ndarray], timestamps: list[pd.Timestamp], bounds,
                  output: Path, label: str, max_size=(360, 280)) -> None:
    x0, y0, x1, y1 = [int(bounds[key]) for key in ("x_min", "y_min", "x_max", "y_max")]
    previews = []
    for image, stamp in zip(frames, timestamps):
        crop = image[y0:y1, x0:x1]
        previews.append(annotated_preview(crop, f"{label} · {stamp:%H:%M:%S}", max_size))
    contact_sheet(previews, output)
