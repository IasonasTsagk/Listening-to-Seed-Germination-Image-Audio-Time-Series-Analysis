"""Run the Phase 2.7B PASS-gated all-25 SAM automatic diagnostics only."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_sam21_temporal_validation import OVERLAP_IOU_THRESHOLD, SAM_PARAMETERS, pairwise_overlap

TABLES = ROOT / "results/tables/pretrained_benchmark"
FIGURES = ROOT / "results/figures/pretrained_benchmark/sam21_temporal"
CHECKPOINT = ROOT / "results/models/sam2.1/sam2.1_hiera_small.pt"


def plot_all25(summary: pd.DataFrame) -> None:
    fields = [
        ("raw_mask_count", "Raw SAM mask count (not seed count)"),
        ("union_coverage_percentage", "Union coverage (%)"),
        ("median_mask_area", "Median mask area (pixels)"),
        ("mean_predicted_iou", "Mean predicted IoU (model score, not accuracy)"),
        ("mean_stability_score", "Mean stability score (not accuracy)"),
        ("runtime_seconds", "Runtime (seconds)"),
        ("high_overlap_mask_fraction", "High-overlap mask fraction"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(16, 16), constrained_layout=True)
    for axis, (column, label) in zip(axes.flat, fields):
        axis.plot(summary["timestamp"], summary[column], marker="o", linewidth=1.2)
        axis.set_title(label)
        axis.set_xlabel("Timestamp")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle("SAM 2.1 all-25 automatic-mask diagnostics (descriptive only)")
    fig.savefig(FIGURES / "sam21_all25_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    inventory = pd.read_csv(ROOT / "results/tables/dataset_inventory.csv", parse_dates=["timestamp"])
    frames = inventory.loc[inventory["is_readable"]].sort_values(["timestamp", "filename"]).reset_index(drop=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(CHECKPOINT), device=device,
                       apply_postprocessing=SAM_PARAMETERS["apply_postprocessing"])
    generator = SAM2AutomaticMaskGenerator(model, **{key: value for key, value in SAM_PARAMETERS.items() if key != "apply_postprocessing"})
    rows = []
    for index, rec in enumerate(frames.itertuples(index=False)):
        image = np.array(Image.open(rec.path).convert("RGB"), copy=True)
        start = time.perf_counter()
        masks = generator.generate(image)
        elapsed = time.perf_counter() - start
        areas = np.asarray([item["area"] for item in masks], dtype=float)
        ious = np.asarray([item.get("predicted_iou", np.nan) for item in masks], dtype=float)
        stability = np.asarray([item.get("stability_score", np.nan) for item in masks], dtype=float)
        union = np.zeros(image.shape[:2], dtype=bool)
        border_count = 0
        for item in masks:
            mask = item["segmentation"].astype(bool)
            union |= mask
            if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
                border_count += 1
        pair_count, overlap_fraction = pairwise_overlap(masks, OVERLAP_IOU_THRESHOLD)
        row = {
            "timestamp": rec.timestamp, "filename": rec.filename, "chronological_index": index,
            "device": device, "runtime_seconds": elapsed, "raw_mask_count": len(masks),
            "total_mask_area_pixels": int(areas.sum()), "union_area_pixels": int(union.sum()),
            "union_coverage_percentage": union.mean() * 100, "mean_mask_area": float(areas.mean()),
            "median_mask_area": float(np.median(areas)), "largest_mask_area": int(areas.max()),
            "smallest_mask_area": int(areas.min()), "mean_predicted_iou": float(np.nanmean(ious)),
            "median_predicted_iou": float(np.nanmedian(ious)), "mean_stability_score": float(np.nanmean(stability)),
            "median_stability_score": float(np.nanmedian(stability)), "border_touching_mask_count": border_count,
            "high_overlap_pair_count": pair_count, "high_overlap_mask_fraction": overlap_fraction,
            "overlap_iou_threshold": OVERLAP_IOU_THRESHOLD,
        }
        rows.append(row)
        print(json.dumps({"frame": index + 1, "of": len(frames), "filename": rec.filename,
                          "runtime_seconds": elapsed, "raw_mask_count": len(masks)}), flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "sam21_all25_summary.csv", index=False)
    plot_all25(summary)


if __name__ == "__main__":
    main()
