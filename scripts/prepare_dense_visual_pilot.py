"""Prepare read-only dense-window inventory and blank human review tables."""
from __future__ import annotations

from pathlib import Path
import os
import argparse
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.dense_visual import (WINDOWS, discover_dense_images, initial_events, load_rois,
                              make_manifest, save_contact_sheet, save_inventory,
                              save_review_csv, spacing_summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path,
                        default=Path(os.environ.get("SEED_GERMINATION_SNAPSHOT_ROOT", PROJECT / "data/raw/snapshots")))
    parser.add_argument("--refresh-inventory", action="store_true")
    args = parser.parse_args()
    tables = PROJECT / "results/tables/targeted_multimodal"
    figures = PROJECT / "results/figures/targeted_multimodal/dense_stage1"
    inventory_path = tables / "dense_window_image_inventory.csv"
    if inventory_path.exists() and not args.refresh_inventory:
        inventory = pd.read_csv(inventory_path, parse_dates=["timestamp"])
    else:
        inventory = discover_dense_images(args.raw_root)
        save_inventory(inventory, inventory_path)
    rois = load_rois(tables / "target_seed_rois.csv", tables / "target_seed_frame_rois.csv")
    manifest_path = tables / "dense_visual_review_manifest.csv"
    if not manifest_path.exists():
        manifest = make_manifest(inventory, rois)
        save_review_csv(manifest, manifest_path, ["seed_id", "review_stage", "image_timestamp"])
    else:
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
    events_path = tables / "refined_visual_events.csv"
    if not events_path.exists():
        save_review_csv(initial_events(), events_path, ["seed_id"])
    for seed in sorted(rois.seed_id):
        rows = manifest.loc[(manifest.seed_id == seed) & (manifest.review_stage == "stage1")]
        if not rows.empty:
            save_contact_sheet(rows, inventory, figures / f"{seed}_stage1.png")
    print(spacing_summary(inventory))
    print(f"Unique files: {len(inventory)}; Stage 1 rows: {len(manifest.loc[manifest.review_stage == 'stage1'])}")
    print(f"Human review tables: {manifest_path}, {events_path}")


if __name__ == "__main__":
    main()
