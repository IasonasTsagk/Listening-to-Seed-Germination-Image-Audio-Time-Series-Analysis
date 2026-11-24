"""Prepare focused, exploratory full-frame review near one acoustic timestamp."""
from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.reverse_visual import (ACOUSTIC_TIME, PRIMARY_START, PRIMARY_END,
    WIDE_START, WIDE_END, annotated_preview, contact_sheet, crop_timeline,
    initial_candidates, intersects, read_rgb, tile_change_summary, window_inventory)

RAW_ROOT = Path(os.environ.get("SEED_GERMINATION_SNAPSHOT_ROOT", PROJECT / "data/raw/snapshots"))
TABLES = PROJECT / "results/tables/targeted_multimodal"
FIGURES = PROJECT / "results/figures/targeted_multimodal/reverse_visual"
INDEX = TABLES / "dense_window_image_inventory.csv"


def gif(frames: list[np.ndarray], timestamps: list[pd.Timestamp], path: Path) -> None:
    previews = [annotated_preview(image, f"Full scene · {stamp:%Y-%m-%d %H:%M:%S}")
                for image, stamp in zip(frames, timestamps)]
    first, *rest = [preview.convert("P", palette=Image.Palette.ADAPTIVE) for preview in previews]
    first.save(path, save_all=True, append_images=rest, duration=450,
               loop=0, optimize=False, disposal=2)


def difference_figures(differences: list[np.ndarray], stamps: list[pd.Timestamp], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    previews = []
    for index, difference in enumerate(differences):
        upper = max(float(np.percentile(difference, 99)), 1.0)
        display_pixels = np.uint8(np.clip(difference / upper * 255, 0, 255))
        preview = annotated_preview(np.repeat(display_pixels[..., None], 3, axis=2),
            f"Absolute difference · {stamps[index]:%H:%M:%S} → {stamps[index+1]:%H:%M:%S} · display contrast scaled")
        preview.save(destination / f"difference_{index+1:02d}.png")
        previews.append(preview)
    contact_sheet(previews, destination / "absolute_difference_contact_sheet.png", columns=3)


def top_overlay(image: np.ndarray, tiles: pd.DataFrame, stamp: pd.Timestamp, path: Path) -> None:
    canvas = Image.fromarray(image).copy()
    draw = ImageDraw.Draw(canvas)
    for rank, tile in enumerate(tiles.head(5).itertuples(index=False), 1):
        bounds = (int(tile.x_min), int(tile.y_min), int(tile.x_max)-1, int(tile.y_max)-1)
        color = (255, 45+rank*30, 20)
        draw.rectangle(bounds, outline=color, width=5)
        draw.text((bounds[0]+5, bounds[1]+5), f"#{rank} {tile.tile_id}", fill=color,
                  stroke_width=2, stroke_fill="black")
    preview = annotated_preview(np.asarray(canvas),
        f"{stamp:%Y-%m-%d %H:%M:%S} · top five cumulative-change tiles; visual guide only")
    preview.save(path)


def main() -> None:
    primary = window_inventory(INDEX, RAW_ROOT, PRIMARY_START, PRIMARY_END)
    wide = window_inventory(INDEX, RAW_ROOT, WIDE_START, WIDE_END)
    if not set(primary.path).issubset(set(wide.path)):
        raise ValueError("Primary frames must be contained in the wide window")
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    primary.to_csv(TABLES / "reverse_visual_primary_inventory.csv", index=False)
    wide.to_csv(TABLES / "reverse_visual_wide_inventory.csv", index=False)
    wide_frames = [read_rgb(path) for path in wide.path]
    by_path = dict(zip(wide.path, wide_frames))
    frames = [by_path[path] for path in primary.path]
    stamps = primary.timestamp.tolist()
    wide_stamps = wide.timestamp.tolist()
    gif(frames, stamps, FIGURES / "reverse_primary_fullframe.gif")
    gif(wide_frames, wide_stamps, FIGURES / "reverse_wide_fullframe.gif")
    full_previews = [annotated_preview(image, f"Full scene · {stamp:%H:%M:%S}", (480, 270))
                     for image, stamp in zip(frames, stamps)]
    contact_sheet(full_previews, FIGURES / "reverse_primary_contact_sheet.png")
    summary, differences = tile_change_summary(frames, stamps)
    summary.to_csv(TABLES / "reverse_visual_tile_change_summary.csv", index=False)
    difference_figures(differences, stamps, FIGURES / "differences")
    representative = int(np.argmin([abs((stamp-ACOUSTIC_TIME).total_seconds()) for stamp in stamps]))
    Image.fromarray(frames[representative]).save(FIGURES / "representative_fullframe.png")
    top_overlay(frames[representative], summary, stamps[representative], FIGURES / "top_tiles_overlay.png")
    timeline_dir = FIGURES / "top_tile_timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    for rank, tile in enumerate(summary.head(5).to_dict("records"), 1):
        crop_timeline(frames, stamps, tile, timeline_dir / f"top_tile_{rank:02d}_timeline.png",
                      f"#{rank} {tile['tile_id']}")
    candidates = initial_candidates(summary, TABLES / "reverse_visual_candidate_regions.csv")
    seed05 = pd.read_csv(TABLES / "target_seed_rois.csv", keep_default_na=False)
    seed05 = seed05.loc[seed05.seed_id.eq("seed_05")].iloc[0]
    seed_bounds = {axis: int(seed05[f"padded_{axis}"]) for axis in ("x_min", "y_min", "x_max", "y_max")}
    height, width = frames[0].shape[:2]
    if not (0 <= seed_bounds["x_min"] < seed_bounds["x_max"] <= width and
            0 <= seed_bounds["y_min"] < seed_bounds["y_max"] <= height):
        raise ValueError("Seed_05 ROI is outside full-frame image bounds")
    crop_timeline(frames, stamps, seed_bounds, FIGURES / "seed05_comparison.png", "seed_05 ROI")
    overlap = [tile.tile_id for tile in summary.head(5).to_dict("records") if intersects(tile, seed_bounds)]
    required = [FIGURES / "reverse_primary_fullframe.gif", FIGURES / "reverse_wide_fullframe.gif",
                FIGURES / "reverse_primary_contact_sheet.png", FIGURES / "top_tiles_overlay.png",
                FIGURES / "seed05_comparison.png", FIGURES / "differences/absolute_difference_contact_sheet.png",
                TABLES / "reverse_visual_primary_inventory.csv", TABLES / "reverse_visual_wide_inventory.csv",
                TABLES / "reverse_visual_tile_change_summary.csv", TABLES / "reverse_visual_candidate_regions.csv"]
    required += [timeline_dir / f"top_tile_{rank:02d}_timeline.png" for rank in range(1, 6)]
    if not all(path.is_file() for path in required):
        raise IOError("A required reverse-visual output is missing")
    if any(tile.x_min < 0 or tile.y_min < 0 or tile.x_max > width or tile.y_max > height
           for tile in summary.itertuples(index=False)) or len(candidates) < 5:
        raise ValueError("Tile coordinates or candidate table failed validation")
    print(f"Primary frames: {len(primary)} ({stamps[0]} to {stamps[-1]})")
    print(f"Wide frames: {len(wide)} ({wide_stamps[0]} to {wide_stamps[-1]})")
    print("Top five tiles:", ", ".join(summary.head(5).tile_id))
    print("Top-five tiles intersecting seed_05 ROI:", ", ".join(overlap) if overlap else "none")
    print("Tile scores are exploratory visual-change guides; no sound source or biological event is assigned.")


if __name__ == "__main__":
    main()
