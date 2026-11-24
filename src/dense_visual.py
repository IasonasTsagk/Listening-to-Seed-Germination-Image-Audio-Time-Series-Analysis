"""Human-led dense visual review for the five Phase B target seeds.

Raw snapshots are read-only inputs; no biological state is inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
from io import BytesIO
import shutil

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .io_utils import parse_timestamp_from_filename


@dataclass(frozen=True)
class DenseWindow:
    window_id: str
    start: str
    end: str
    seed_ids: tuple[str, ...]


WINDOWS = (
    DenseWindow("WINDOW_A", "2026-07-14 17:03:56", "2026-07-14 18:11:11", ("seed_01", "seed_03", "seed_04")),
    DenseWindow("WINDOW_B", "2026-07-14 19:01:36", "2026-07-14 20:01:57", ("seed_05",)),
    DenseWindow("WINDOW_C", "2026-07-14 20:01:57", "2026-07-14 21:00:07", ("seed_02",)),
)

INVENTORY_COLUMNS = ["window_id", "filename", "timestamp", "source_path", "file_size_bytes", "is_readable"]
MANIFEST_COLUMNS = ["seed_id", "window_id", "image_filename", "image_timestamp", "frame_index",
                    "x_min", "y_min", "x_max", "y_max", "roi_source", "review_stage",
                    "manual_state", "manual_confidence", "notes"]
EVENT_COLUMNS = ["seed_id", "event_type", "coarse_start_timestamp", "coarse_end_timestamp",
                 "stage1_last_negative_timestamp", "stage1_first_positive_timestamp",
                 "last_negative_timestamp", "first_positive_timestamp", "refined_interval_seconds",
                 "visual_confidence", "identity_confidence", "notes"]
STATES = {"intact", "possible_crack", "clear_crack", "radicle_visible", "uncertain", "occluded"}
EVENT_TYPES = {"crack", "radicle_emergence"}
REVIEW_KEY = ["seed_id", "review_stage", "image_timestamp"]
HUMAN_REVIEW_COLUMNS = ("manual_state", "manual_confidence", "notes")
HUMAN_EVENT_COLUMNS = ("event_type", "stage1_last_negative_timestamp",
                       "stage1_first_positive_timestamp", "last_negative_timestamp",
                       "first_positive_timestamp", "refined_interval_seconds",
                       "visual_confidence", "identity_confidence")
MANIFEST_INTEGER_COLUMNS = ("frame_index", "x_min", "y_min", "x_max", "y_max")
EVENT_NUMERIC_COLUMN = "refined_interval_seconds"


def _nullable_number(values: pd.Series, dtype: str, column: str) -> pd.Series:
    """Treat only empty cells as missing; reject nonnumeric entered data."""
    source = values.astype("string").str.strip().replace("", pd.NA)
    numeric = pd.to_numeric(source, errors="coerce")
    if (source.notna() & numeric.isna()).any():
        raise ValueError(f"Non-numeric value in {column}")
    return numeric.astype(dtype)


def event_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit text and nullable numeric dtypes for refined event records."""
    missing = set(EVENT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Event table is missing columns: {sorted(missing)}")
    result = frame.copy()
    for column in EVENT_COLUMNS:
        if column != EVENT_NUMERIC_COLUMN:
            result[column] = result[column].astype("string").fillna("")
    result[EVENT_NUMERIC_COLUMN] = _nullable_number(result[EVENT_NUMERIC_COLUMN], "Float64", EVENT_NUMERIC_COLUMN)
    return result


def manifest_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit text and nullable integer dtypes for the review manifest."""
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Review manifest is missing columns: {sorted(missing)}")
    result = frame.copy()
    for column in MANIFEST_COLUMNS:
        if column in MANIFEST_INTEGER_COLUMNS:
            result[column] = _nullable_number(result[column], "Int64", column)
        else:
            result[column] = result[column].astype("string").fillna("")
    return result


def load_events(path: Path) -> pd.DataFrame:
    return event_schema(pd.read_csv(path, dtype="string", keep_default_na=False))


def load_review_manifest(path: Path) -> pd.DataFrame:
    return manifest_schema(pd.read_csv(path, dtype="string", keep_default_na=False))


def _time(value):
    return pd.Timestamp(value).to_pydatetime()


def window_for_seed(seed_id: str) -> DenseWindow:
    matches = [w for w in WINDOWS if seed_id in w.seed_ids]
    if len(matches) != 1:
        raise ValueError(f"No unique dense window for {seed_id}")
    return matches[0]


def relevant_group_dirs(snapshot_root: Path, windows=WINDOWS) -> set[Path]:
    """Use recorder metadata to locate only groups touching requested times."""
    root = snapshot_root.resolve()
    bounds = [(_time(w.start), _time(w.end)) for w in windows]
    groups = set()
    for index in root.glob("snapshots_*/snapshots_*.json"):
        with index.open(encoding="utf-8") as handle:
            for line in handle:
                if '"image_file"' not in line:
                    continue
                try:
                    record = json.loads(line.rstrip().rstrip(","))
                except json.JSONDecodeError:
                    continue
                relative = record.get("image_file")
                if not relative:
                    continue
                stamp = parse_timestamp_from_filename(relative)
                if stamp is None or not any(start <= stamp <= end for start, end in bounds):
                    continue
                # Recorder paths are relative to raw/, not to snapshots/.
                group = (root.parent / relative).parent.resolve()
                if not group.is_relative_to(root):
                    raise ValueError(f"Recorder path leaves snapshot root: {relative}")
                groups.add(group)
    if not groups:
        raise FileNotFoundError(f"No recorder groups overlap requested windows under {root}")
    return groups


def discover_dense_images(snapshot_root: Path, windows=WINDOWS, groups=None, verify=True) -> pd.DataFrame:
    """Scan selected recorder groups once, retaining real timestamps and shared-boundary membership."""
    groups = relevant_group_dirs(snapshot_root, windows) if groups is None else set(groups)
    if any(not group.is_dir() for group in groups):
        raise FileNotFoundError("A selected raw snapshot group is missing")
    seen = set()
    rows = []
    for group in sorted(groups):
        with os.scandir(group) as entries:
            for entry in entries:
                if not entry.is_file() or Path(entry.name).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                if entry.name.startswith("._") or "-checkpoint" in entry.name.lower():
                    continue
                stamp = parse_timestamp_from_filename(entry.name)
                matching = [w.window_id for w in windows if stamp is not None and _time(w.start) <= stamp <= _time(w.end)]
                if not matching:
                    continue
                path = Path(entry.path)
                if path in seen:
                    continue
                seen.add(path)
                readable = True
                if verify:
                    try:
                        with Image.open(path) as image:
                            image.verify()
                    except (OSError, ValueError):
                        readable = False
                rows.append([";".join(matching), entry.name, stamp, str(path), entry.stat().st_size, readable])
    if not rows:
        raise FileNotFoundError("No raw images found in requested dense windows")
    inventory = pd.DataFrame(rows, columns=INVENTORY_COLUMNS).sort_values(["timestamp", "filename"]).reset_index(drop=True)
    for w in windows:
        subset = window_images(inventory, w.window_id)
        if subset.empty or subset.timestamp.min() != _time(w.start) or subset.timestamp.max() != _time(w.end):
            raise ValueError(f"Missing boundary image in {w.window_id}")
        if subset.timestamp.duplicated().any():
            raise ValueError(f"Duplicate timestamp in {w.window_id}")
    return inventory


def window_images(inventory: pd.DataFrame, window_id: str) -> pd.DataFrame:
    return inventory.loc[inventory.window_id.str.split(";").map(lambda ids: window_id in ids)].sort_values("timestamp").reset_index(drop=True)


def spacing_summary(inventory: pd.DataFrame) -> dict:
    result = {}
    for w in WINDOWS:
        frame = window_images(inventory, w.window_id)
        gaps = pd.to_datetime(frame.timestamp).diff().dropna().dt.total_seconds()
        result[w.window_id] = {"images": len(frame), "min_seconds": int(gaps.min()),
                               "median_seconds": float(gaps.median()), "max_seconds": int(gaps.max()),
                               "unreadable": int((~frame.is_readable.astype(bool)).sum())}
    return result


def sample_nearest(frame: pd.DataFrame, step_seconds=90) -> pd.DataFrame:
    """Nearest actual frames to a regular review grid; preserve endpoints."""
    if step_seconds <= 0 or frame.empty:
        raise ValueError("Sampling needs frames and a positive interval")
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    stamps = pd.to_datetime(ordered.timestamp)
    start, end = stamps.iloc[0], stamps.iloc[-1]
    selected = {0, len(ordered) - 1}
    target = start
    while target < end:
        selected.add(int((stamps - target).abs().argmin()))
        target += timedelta(seconds=step_seconds)
    return ordered.iloc[sorted(selected)].reset_index(drop=True)


def clip_bounds(bounds, width, height, padding=0):
    x0, y0, x1, y1 = [int(v) for v in bounds]
    if x0 >= x1 or y0 >= y1 or padding < 0:
        raise ValueError("Invalid ROI")
    clipped = max(0, x0-padding), max(0, y0-padding), min(width, x1+padding), min(height, y1+padding)
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError("ROI does not intersect image")
    return clipped


def load_rois(path: Path, frame_rois_path: Path | None = None) -> pd.DataFrame:
    rois = pd.read_csv(path, keep_default_na=False)
    required = {"seed_id", "padded_x_min", "padded_y_min", "padded_x_max", "padded_y_max"}
    if required - set(rois) or rois.seed_id.duplicated().any():
        raise ValueError("Phase A ROI table is incomplete or has duplicate seeds")
    if set(rois.seed_id) != {seed for w in WINDOWS for seed in w.seed_ids}:
        raise ValueError("Phase A ROI seeds do not match supplied dense windows")
    rois["dense_roi_source"] = "phase_a_padded"
    if frame_rois_path is not None and frame_rois_path.exists():
        frames = pd.read_csv(frame_rois_path, keep_default_na=False)
        required_frames = {"seed_id", "image_timestamp", "x_min", "y_min", "x_max", "y_max", "roi_source"}
        if required_frames - set(frames):
            raise ValueError("Phase A frame ROI table is missing required columns")
        for w in WINDOWS:
            for seed in w.seed_ids:
                matched = frames.loc[(frames.seed_id == seed) & (frames.image_timestamp == w.start)]
                if len(matched) > 1:
                    raise ValueError("Duplicate Phase A frame ROI at coarse window start")
                if len(matched) == 1 and matched.iloc[0].roi_source == "manual_correction":
                    for coord in ("x_min", "y_min", "x_max", "y_max"):
                        rois.loc[rois.seed_id == seed, f"padded_{coord}"] = int(matched.iloc[0][coord])
                    rois.loc[rois.seed_id == seed, "dense_roi_source"] = "phase_a_frame_correction"
    return rois


def make_manifest(inventory: pd.DataFrame, rois: pd.DataFrame, stage="stage1", step_seconds=90) -> pd.DataFrame:
    if stage not in {"stage1", "stage2"}:
        raise ValueError("Unknown review stage")
    rows = []
    for w in WINDOWS:
        images = window_images(inventory, w.window_id)
        selected = sample_nearest(images, step_seconds) if stage == "stage1" else images
        frame_indices = {path: i for i, path in enumerate(images.source_path)}
        for seed in w.seed_ids:
            roi = rois.loc[rois.seed_id == seed].iloc[0]
            bounds = [int(roi[f"padded_{coord}"]) for coord in ("x_min", "y_min", "x_max", "y_max")]
            for image in selected.itertuples(index=False):
                rows.append([seed, w.window_id, image.filename, image.timestamp,
                             frame_indices[image.source_path], *bounds, roi.dense_roi_source if "dense_roi_source" in rois else "phase_a_padded", stage, "", "", ""])
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def validate_interval(last_negative, first_positive, allowed_start=None, allowed_end=None) -> float:
    if pd.isna(last_negative) or not str(last_negative).strip():
        raise ValueError("Last-negative timestamp is required")
    if pd.isna(first_positive) or not str(first_positive).strip():
        raise ValueError("First-positive timestamp is required")
    negative, positive = _time(last_negative), _time(first_positive)
    if negative >= positive:
        raise ValueError("last-negative must precede first-positive")
    if allowed_start is not None and negative < _time(allowed_start):
        raise ValueError("last-negative precedes coarse window")
    if allowed_end is not None and positive > _time(allowed_end):
        raise ValueError("first-positive exceeds coarse window")
    return (positive - negative).total_seconds()


def initial_events() -> pd.DataFrame:
    frame = pd.DataFrame([[seed, "", w.start, w.end, "", "", "", "", pd.NA, "", "", ""]
                          for w in WINDOWS for seed in w.seed_ids], columns=EVENT_COLUMNS)
    return event_schema(frame.sort_values("seed_id").reset_index(drop=True))


def row_has_human_review(row) -> bool:
    """Recognize saved annotation text and manually corrected coordinates."""
    columns = (*HUMAN_REVIEW_COLUMNS, *HUMAN_EVENT_COLUMNS)
    has_text = any(column in row and pd.notna(row[column]) and str(row[column]).strip()
                   for column in columns)
    roi_source = str(row.get("roi_source", "")).strip()
    return bool(has_text or (roi_source and roi_source not in {"phase_a_padded", "phase_a_frame_correction", "reference"}))


def _backup_review_csv(path: Path) -> Path:
    """Make a uniquely named copy before replacing a human-review table."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def save_events(frame: pd.DataFrame, path: Path) -> None:
    """Save explicitly typed event rows, backing up an existing CSV first."""
    typed = event_schema(frame)
    if path.exists():
        _backup_review_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    typed.to_csv(path, index=False)


def save_review_manifest(frame: pd.DataFrame, path: Path) -> None:
    """Save explicitly typed review rows, backing up an existing CSV first."""
    typed = manifest_schema(frame)
    if typed.duplicated(REVIEW_KEY).any():
        raise ValueError("Review manifest has duplicate stable keys")
    if path.exists():
        _backup_review_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    typed.to_csv(path, index=False)


def save_review_csv(frame: pd.DataFrame, path: Path, key: list[str]) -> None:
    """Merge by stable key, retaining human input and backing up before changes.

    Only obsolete, unreviewed generated Stage 2 rows may disappear. Existing
    Stage 1 rows and reviewed Stage 2 rows remain byte-for-byte equivalent as
    records; event rows are not removed by this helper.
    """
    is_manifest = set(MANIFEST_COLUMNS).issubset(frame.columns)
    is_events = set(EVENT_COLUMNS).issubset(frame.columns)
    if is_manifest:
        frame = manifest_schema(frame)
    elif is_events:
        frame = event_schema(frame)
    if path.exists():
        old = load_review_manifest(path) if is_manifest else load_events(path) if is_events else pd.read_csv(path, keep_default_na=False)
        if set(old.columns) != set(frame.columns) or old.duplicated(key).any() or frame.duplicated(key).any():
            raise ValueError(f"Existing review table is incompatible: {path}")
        frame = frame[old.columns].copy()
        old_by_key = old.set_index(key, drop=False)
        new_positions = {tuple(record[column] for column in key): index
                         for index, record in frame.iterrows()}
        retained = []
        for stable_key, previous in old_by_key.iterrows():
            normalized_key = stable_key if isinstance(stable_key, tuple) else (stable_key,)
            if normalized_key in new_positions:
                position = new_positions[normalized_key]
                if previous.get("review_stage") == "stage1":
                    # Stage 1 is an existing human workflow, never regenerate it here.
                    for column in old.columns:
                        frame.at[position, column] = previous[column]
                else:
                    for column in (*HUMAN_REVIEW_COLUMNS, *HUMAN_EVENT_COLUMNS):
                        if column in frame:
                            frame.at[position, column] = previous[column]
                    if previous.get("roi_source") == "manual_correction":
                        for column in ("x_min", "y_min", "x_max", "y_max", "roi_source"):
                            frame.at[position, column] = previous[column]
            elif previous.get("review_stage") != "stage2" or row_has_human_review(previous):
                retained.append(previous[old.columns].to_dict())
        if retained:
            frame = pd.concat([frame, pd.DataFrame(retained)], ignore_index=True)
        if frame.duplicated(key).any():
            raise ValueError("Review merge produced duplicate stable keys")
        if frame.equals(old):
            return
        _backup_review_csv(path)
    elif frame.duplicated(key).any():
        raise ValueError("Review table has duplicate stable keys")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_inventory(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_crop(source_path: str, bounds, padding=30) -> Image.Image:
    """Decode only the requested frame; caller owns returned crop."""
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        return image.crop(clip_bounds(bounds, *image.size, padding))


def save_contact_sheet(manifest: pd.DataFrame, inventory: pd.DataFrame, destination: Path, columns=5):
    """Sequentially decode small crops, never an entire dense window."""
    lookup = inventory.set_index("filename").source_path.to_dict()
    tile_w, tile_h = 260, 230
    sheet = Image.new("RGB", (columns*tile_w, ((len(manifest)+columns-1)//columns)*tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(manifest.itertuples(index=False)):
        crop = load_crop(lookup[row.image_filename], (row.x_min, row.y_min, row.x_max, row.y_max))
        crop.thumbnail((tile_w-12, tile_h-50))
        x, y = (i % columns)*tile_w, (i // columns)*tile_h
        sheet.paste(crop, (x+(tile_w-crop.width)//2, y+36))
        draw.text((x+6, y+3), f"{row.seed_id} | frame {row.frame_index}", fill="black")
        draw.text((x+6, y+19), str(row.image_timestamp), fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def narrowed_frames(inventory: pd.DataFrame, window_id: str, last_negative, first_positive) -> pd.DataFrame:
    w = next((item for item in WINDOWS if item.window_id == window_id), None)
    if w is None:
        raise ValueError("Unknown window")
    validate_interval(last_negative, first_positive, w.start, w.end)
    frames = window_images(inventory, window_id)
    stamps = pd.to_datetime(frames.timestamp)
    return frames.loc[stamps.between(_time(last_negative), _time(first_positive), inclusive="both")].reset_index(drop=True)


def validate_final_event(row, inventory: pd.DataFrame) -> float:
    if row.event_type not in EVENT_TYPES:
        raise ValueError("A human-selected crack or radicle_emergence event_type is required")
    interval = validate_interval(row.last_negative_timestamp, row.first_positive_timestamp,
                                 row.coarse_start_timestamp, row.coarse_end_timestamp)
    if not row.stage1_last_negative_timestamp or not row.stage1_first_positive_timestamp:
        raise ValueError("Stage 1 boundaries are required")
    validate_interval(row.stage1_last_negative_timestamp, row.stage1_first_positive_timestamp,
                      row.coarse_start_timestamp, row.coarse_end_timestamp)
    if _time(row.last_negative_timestamp) < _time(row.stage1_last_negative_timestamp) or _time(row.first_positive_timestamp) > _time(row.stage1_first_positive_timestamp):
        raise ValueError("Final boundaries must lie inside Stage 1 interval")
    frames = narrowed_frames(inventory, window_for_seed(row.seed_id).window_id,
                             row.stage1_last_negative_timestamp, row.stage1_first_positive_timestamp)
    stamps = list(pd.to_datetime(frames.timestamp))
    if pd.Timestamp(row.last_negative_timestamp) not in stamps or pd.Timestamp(row.first_positive_timestamp) not in stamps:
        raise ValueError("Final timestamps must match real raw frames")
    return interval


def intermediate_frame_count(inventory: pd.DataFrame, seed_id: str,
                             last_negative: str, first_positive: str) -> int:
    """Count available frames strictly inside a human-marked uncertainty interval."""
    frames = window_images(inventory, window_for_seed(seed_id).window_id)
    stamps = pd.to_datetime(frames.timestamp)
    return int(((stamps > pd.Timestamp(last_negative)) &
                (stamps < pd.Timestamp(first_positive))).sum())


def save_stage2_interval(events_path: Path, inventory: pd.DataFrame, seed_id: str,
                         event_type: str, last_negative: str, first_positive: str,
                         visual_confidence: str = "", identity_confidence: str = "") -> float:
    """Persist human-marked definite bounds, including wider uncertain intervals."""
    events = load_events(events_path)
    mask = events.seed_id == seed_id
    if mask.sum() != 1:
        raise ValueError("Event row is not unique")
    previous = events.loc[mask].iloc[0]
    if previous.event_type and previous.event_type != event_type:
        raise ValueError("Event type differs from Stage 1; review this manually")
    candidate = previous.copy()
    candidate.event_type = event_type
    candidate.last_negative_timestamp = last_negative
    candidate.first_positive_timestamp = first_positive
    seconds = validate_final_event(candidate, inventory)
    events.loc[mask, "event_type"] = event_type
    events.loc[mask, "last_negative_timestamp"] = last_negative
    events.loc[mask, "first_positive_timestamp"] = first_positive
    events.loc[mask, EVENT_NUMERIC_COLUMN] = seconds
    events.loc[mask, "visual_confidence"] = visual_confidence
    events.loc[mask, "identity_confidence"] = identity_confidence
    save_events(events, events_path)
    return seconds


def save_event_figure(row, inventory: pd.DataFrame, roi, destination: Path,
                      manifest: pd.DataFrame | None = None):
    seconds = validate_final_event(row, inventory)
    frames = window_images(inventory, window_for_seed(row.seed_id).window_id)
    timestamps = list(pd.to_datetime(frames.timestamp))
    negative_index = timestamps.index(pd.Timestamp(row.last_negative_timestamp))
    positive_index = timestamps.index(pd.Timestamp(row.first_positive_timestamp))
    selected = sorted({i for i in (negative_index-1, negative_index,
                                   positive_index, positive_index+1) if 0 <= i < len(frames)})
    tile_w, tile_h = 320, 285
    sheet = Image.new("RGB", (len(selected)*tile_w, tile_h+45), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f"{row.seed_id} | {row.event_type} | ({row.last_negative_timestamp}, {row.first_positive_timestamp}] | {seconds:g} s", fill="black")
    for position, frame_index in enumerate(selected):
        frame = frames.iloc[frame_index]
        frame_roi = roi
        if manifest is not None:
            corrected = manifest.loc[(manifest.seed_id == row.seed_id) &
                                     (manifest.review_stage == "stage2") &
                                     (manifest.image_timestamp == str(frame.timestamp)) &
                                     (manifest.roi_source == "manual_correction")]
            if len(corrected) == 1:
                frame_roi = tuple(int(corrected.iloc[0][name]) for name in ("x_min", "y_min", "x_max", "y_max"))
        crop = load_crop(frame.source_path, frame_roi)
        crop.thumbnail((tile_w-12, tile_h-46))
        x = position*tile_w
        sheet.paste(crop, (x+(tile_w-crop.width)//2, 76))
        label = "last negative" if frame_index == negative_index else "first positive" if frame_index == positive_index else "context"
        draw.text((x+7, 46), f"{label}: {frame.timestamp}", fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def prepare_stage2(seed_id: str, inventory: pd.DataFrame, manifest_path: Path, events_path: Path) -> pd.DataFrame:
    """Add only narrowed native-rate rows after a human Stage 1 decision."""
    events = load_events(events_path)
    decision = events.loc[events.seed_id == seed_id]
    if len(decision) != 1:
        raise ValueError("Seed missing from event table")
    row = decision.iloc[0]
    if not row.stage1_last_negative_timestamp or not row.stage1_first_positive_timestamp:
        raise ValueError("Save the human Stage 1 interval first")
    w = window_for_seed(seed_id)
    frames = narrowed_frames(inventory, w.window_id, row.stage1_last_negative_timestamp, row.stage1_first_positive_timestamp)
    manifest = load_review_manifest(manifest_path)
    stage1 = manifest.loc[(manifest.seed_id == seed_id) & (manifest.review_stage == "stage1")]
    if stage1.empty:
        raise ValueError("Stage 1 manifest is missing")
    base = stage1.iloc[0]
    window = window_images(inventory, w.window_id)
    indices = {path: i for i, path in enumerate(window.source_path)}
    rows = []
    for frame in frames.itertuples(index=False):
        rows.append([seed_id, w.window_id, frame.filename, frame.timestamp, indices[frame.source_path],
                     base.x_min, base.y_min, base.x_max, base.y_max, base.roi_source, "stage2", "", "", ""])
    additions = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    additions["image_timestamp"] = additions["image_timestamp"].astype(str)
    # Regenerate the current window; save_review_csv merges old keys and
    # retains any reviewed rows that no longer fall inside this interval.
    existing = manifest.loc[~((manifest.seed_id == seed_id) & (manifest.review_stage == "stage2"))]
    merged = pd.concat([existing, additions], ignore_index=True)
    save_review_csv(merged, manifest_path, REVIEW_KEY)
    current_timestamps = set(additions.image_timestamp)
    saved = load_review_manifest(manifest_path)
    return saved.loc[(saved.seed_id == seed_id) & (saved.review_stage == "stage2") &
                     saved.image_timestamp.isin(current_timestamps)].sort_values("image_timestamp").reset_index(drop=True)


def review_widget(seed_id: str, stage: str, inventory: pd.DataFrame,
                  manifest_path: Path, events_path: Path):
    """VS Code Jupyter viewer. Buttons alone write human decisions."""
    import ipywidgets as widgets
    from IPython.display import display

    if stage not in {"stage1", "stage2"}:
        raise ValueError("Stage must be stage1 or stage2")
    manifest = load_review_manifest(manifest_path)
    if stage == "stage2":
        manifest = prepare_stage2(seed_id, inventory, manifest_path, events_path)
    rows = manifest.loc[(manifest.seed_id == seed_id) & (manifest.review_stage == stage)].sort_values("image_timestamp").reset_index(drop=True)
    if rows.empty:
        raise ValueError("No review rows for seed/stage")
    lookup = inventory.set_index("filename").source_path.to_dict()
    slider = widgets.IntSlider(value=0, min=0, max=len(rows)-1, description="Frame", continuous_update=False, layout=widgets.Layout(width="680px"))
    picture = widgets.Image(format="png", layout=widgets.Layout(max_width="700px"))
    title = widgets.HTML()
    state = widgets.Dropdown(options=[("(unreviewed)", "")] + [(x, x) for x in sorted(STATES)], description="State")
    confidence = widgets.Text(description="Confidence")
    notes = widgets.Text(description="Notes", layout=widgets.Layout(width="680px"))
    event_type = widgets.Dropdown(options=[("Select event", ""), ("crack", "crack"), ("radicle_emergence", "radicle_emergence")], description="Event")
    identity = widgets.Text(description="Identity")
    feedback = widgets.HTML()
    negative_label, positive_label = widgets.HTML(value="Last negative: —"), widgets.HTML(value="First positive: —")
    marks = {"negative": None, "positive": None}
    prev_button, next_button = widgets.Button(description="Previous"), widgets.Button(description="Next")
    save_frame = widgets.Button(description="Save frame review", button_style="info")
    roi_fields = {name: widgets.IntText(description=name, layout=widgets.Layout(width="160px"))
                  for name in ("x_min", "y_min", "x_max", "y_max")}
    save_roi = widgets.Button(description="Save ROI correction")
    mark_negative = widgets.Button(description="Mark last definite negative")
    mark_positive = widgets.Button(description="Mark first definite positive")
    save_interval = widgets.Button(description="Save reviewed interval", button_style="success")

    def render(*_):
        row = rows.iloc[slider.value]
        crop = load_crop(lookup[row.image_filename], (row.x_min, row.y_min, row.x_max, row.y_max))
        crop.thumbnail((680, 580))
        buffer = BytesIO()
        crop.save(buffer, format="PNG")
        picture.value = buffer.getvalue()
        title.value = f"<b>{seed_id} · {stage} · native frame {row.frame_index} · {row.image_timestamp}</b>"
        state.value = row.manual_state if row.manual_state in STATES else ""
        confidence.value = str(row.manual_confidence)
        notes.value = str(row.notes)
        for name, field in roi_fields.items():
            field.value = int(getattr(row, name))

    def save_current(_):
        try:
            row = rows.iloc[slider.value]
            table = load_review_manifest(manifest_path)
            mask = (table.seed_id == seed_id) & (table.review_stage == stage) & (table.image_timestamp == row.image_timestamp)
            if mask.sum() != 1:
                raise ValueError("Review row is not unique")
            table.loc[mask, ["manual_state", "manual_confidence", "notes"]] = [state.value, confidence.value, notes.value]
            save_review_manifest(table, manifest_path)
            rows.loc[slider.value, ["manual_state", "manual_confidence", "notes"]] = [state.value, confidence.value, notes.value]
            feedback.value = "Saved frame review."
        except Exception as error:
            feedback.value = f"<span style='color:red'>{error}</span>"

    def mark(which):
        marks[which] = str(rows.iloc[slider.value].image_timestamp)
        (negative_label if which == "negative" else positive_label).value = f"{which.title()}: {marks[which]}"

    def correct_roi(_):
        try:
            row = rows.iloc[slider.value]
            bounds = tuple(roi_fields[name].value for name in roi_fields)
            with Image.open(lookup[row.image_filename]) as image:
                clip_bounds(bounds, *image.size)
            table = load_review_manifest(manifest_path)
            mask = (table.seed_id == seed_id) & (table.review_stage == stage) & (table.image_timestamp == row.image_timestamp)
            if mask.sum() != 1:
                raise ValueError("ROI row is not unique")
            for name, value in zip(roi_fields, bounds):
                table.loc[mask, name] = value
                rows.loc[slider.value, name] = value
            table.loc[mask, "roi_source"] = "manual_correction"
            rows.loc[slider.value, "roi_source"] = "manual_correction"
            save_review_manifest(table, manifest_path)
            render()
            feedback.value = "Saved human ROI correction for this frame."
        except Exception as error:
            feedback.value = f"<span style='color:red'>{error}</span>"

    def save_pair(_):
        try:
            if not marks["negative"] or not marks["positive"] or not event_type.value:
                raise ValueError("Select event type and mark both frames")
            w = window_for_seed(seed_id)
            validate_interval(marks["negative"], marks["positive"], w.start, w.end)
            if stage == "stage1":
                events = load_events(events_path)
                mask = events.seed_id == seed_id
                if mask.sum() != 1:
                    raise ValueError("Event row is not unique")
                events.loc[mask, "event_type"] = event_type.value
                events.loc[mask, "stage1_last_negative_timestamp"] = marks["negative"]
                events.loc[mask, "stage1_first_positive_timestamp"] = marks["positive"]
                # New Stage 1 decision invalidates any older final pair.
                events.loc[mask, "last_negative_timestamp"] = ""
                events.loc[mask, "first_positive_timestamp"] = ""
                events.loc[mask, EVENT_NUMERIC_COLUMN] = pd.NA
                save_events(events, events_path)
            else:
                save_stage2_interval(events_path, inventory, seed_id, event_type.value,
                                     marks["negative"], marks["positive"],
                                     confidence.value, identity.value)
            feedback.value = f"Saved human-confirmed {stage} interval: ({marks['negative']}, {marks['positive']}]"
            if stage == "stage2":
                between = intermediate_frame_count(inventory, seed_id, marks["negative"], marks["positive"])
                if between:
                    feedback.value += f"<br>Final interval contains {between} intermediate frame(s); uncertainty is wider."
        except Exception as error:
            feedback.value = f"<span style='color:red'>{error}</span>"

    slider.observe(render, names="value")
    prev_button.on_click(lambda _: setattr(slider, "value", max(slider.min, slider.value-1)))
    next_button.on_click(lambda _: setattr(slider, "value", min(slider.max, slider.value+1)))
    save_frame.on_click(save_current)
    save_roi.on_click(correct_roi)
    mark_negative.on_click(lambda _: mark("negative"))
    mark_positive.on_click(lambda _: mark("positive"))
    save_interval.on_click(save_pair)
    render()
    display(widgets.VBox([title, picture, slider, widgets.HBox([prev_button, next_button]),
                          widgets.HBox([state, confidence]), notes, save_frame,
                          widgets.HBox(list(roi_fields.values())), save_roi, event_type,
                          widgets.HBox([mark_negative, mark_positive]), negative_label, positive_label,
                          identity, save_interval, feedback]))
