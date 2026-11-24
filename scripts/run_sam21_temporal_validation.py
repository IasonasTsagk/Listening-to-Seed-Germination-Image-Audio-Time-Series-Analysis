"""Phase 2.7B: fixed-parameter SAM 2.1 automatic temporal validation.

This script deliberately reports mask diagnostics only.  It never treats SAM
instances as seeds and does no cross-frame tracking or biological quantification.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
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

from src.germination_validation import (
    CandidateMaskConfig,
    compute_germination_candidate_masks,
    compute_seed_body_mask,
)

TABLES = ROOT / "results/tables/pretrained_benchmark"
FIGURES = ROOT / "results/figures/pretrained_benchmark/sam21_temporal"
MODELS = ROOT / "results/models/sam2.1"
CHECKPOINT = MODELS / "sam2.1_hiera_small.pt"
OVERLAP_IOU_THRESHOLD = 0.70
SAM_PARAMETERS = {
    "points_per_side": 16,
    "points_per_batch": 32,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.95,
    "stability_score_offset": 1.0,
    "box_nms_thresh": 0.70,
    "crop_n_layers": 0,
    "crop_nms_thresh": 0.7,
    "crop_overlap_ratio": 0.3413333333333333,
    "crop_n_points_downscale_factor": 1,
    "min_mask_region_area": 100,
    "output_mode": "binary_mask",
    "use_m2m": False,
    "multimask_output": True,
    "apply_postprocessing": True,
}


def pick_frames(inventory: pd.DataFrame) -> pd.DataFrame:
    frames = inventory.loc[inventory["is_readable"]].copy()
    frames["timestamp"] = pd.to_datetime(frames["timestamp"], errors="coerce")
    frames = frames.sort_values(["timestamp", "filename"], na_position="last").reset_index(drop=True)
    if len(frames) < 5:
        raise ValueError("Five readable chronological frames are required.")
    indices = np.rint(np.linspace(0, len(frames) - 1, 5)).astype(int)
    if len(set(indices)) != 5:
        raise ValueError("Could not choose five distinct chronological frames.")
    selected = frames.iloc[indices].copy()
    selected.insert(0, "stage_label", ["early", "early-middle", "middle", "middle-late", "late"])
    selected.insert(3, "chronological_index", indices)
    return selected[["stage_label", "timestamp", "filename", "chronological_index", "path"]]


def pairwise_overlap(masks: list[dict], threshold: float) -> tuple[int, float]:
    """Return high-IoU pair count and fraction of masks in one or more such pairs."""
    involved: set[int] = set()
    pairs = 0
    for left in range(len(masks)):
        ax, ay, aw, ah = (int(round(v)) for v in masks[left]["bbox"])
        a_right, a_bottom = ax + aw, ay + ah
        area_a = int(masks[left]["area"])
        for right in range(left + 1, len(masks)):
            bx, by, bw, bh = (int(round(v)) for v in masks[right]["bbox"])
            x0, y0 = max(ax, bx), max(ay, by)
            x1, y1 = min(a_right, bx + bw), min(a_bottom, by + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            intersection = int(
                np.logical_and(
                    masks[left]["segmentation"][y0:y1, x0:x1],
                    masks[right]["segmentation"][y0:y1, x0:x1],
                ).sum()
            )
            union = area_a + int(masks[right]["area"]) - intersection
            if union and intersection / union > threshold:
                pairs += 1
                involved.update((left, right))
    return pairs, len(involved) / len(masks) if masks else 0.0


def instance_visuals(image: np.ndarray, masks: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(271)
    overlay = image.astype(np.float32).copy()
    boundaries = image.copy()
    for item in masks:
        mask = item["segmentation"].astype(bool)
        color = rng.integers(40, 256, size=3, dtype=np.uint8)
        overlay[mask] = overlay[mask] * 0.55 + color.astype(np.float32) * 0.45
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundaries, contours, -1, tuple(int(v) for v in color), 2)
    return overlay.astype(np.uint8), boundaries


def plot_frame(stage: str, timestamp: pd.Timestamp, image: np.ndarray, otsu: np.ndarray,
               candidate: np.ndarray, sam_overlay: np.ndarray, sam_boundaries: np.ndarray) -> None:
    comparison = image.copy()
    contours, _ = cv2.findContours((otsu > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(comparison, contours, -1, (255, 210, 0), 2)
    comparison = cv2.addWeighted(comparison, 0.72, sam_boundaries, 0.28, 0)
    panels = [
        ("Original RGB", image, None),
        ("Otsu seed-body mask\n(components are not seeds)", otsu, "gray"),
        ("SAM automatic-mask overlay\n(masks are not seeds)", sam_overlay, None),
        ("SAM instance boundaries", sam_boundaries, None),
        ("Combined germination candidate\n(candidate is not radicle)", candidate, "gray"),
        ("Otsu contour (gold) + SAM boundaries", comparison, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for axis, (title, data, cmap) in zip(axes.flat, panels):
        axis.imshow(data, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"SAM 2.1 temporal validation — {stage}: {timestamp}", fontsize=15)
    fig.savefig(FIGURES / f"{stage.replace('-', '_')}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_area_histograms(records: list[dict]) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), constrained_layout=True)
    for axis, record in zip(axes, records):
        axis.hist(record["areas"], bins=20, color="#377eb8", edgecolor="white")
        axis.set_title(record["stage_label"])
        axis.set_xlabel("SAM mask area (pixels)")
        axis.set_ylabel("Mask count")
    fig.suptitle("SAM 2.1 automatic-mask area distributions (raw; no filtering)")
    fig.savefig(FIGURES / "sam21_mask_area_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_temporal(summary: pd.DataFrame) -> None:
    fields = [
        ("raw_mask_count", "Raw SAM mask count\n(not seed count)"),
        ("union_coverage_percentage", "Union coverage (%)"),
        ("median_mask_area", "Median mask area (pixels)"),
        ("mean_predicted_iou", "Mean predicted IoU\n(model score, not accuracy)"),
        ("mean_stability_score", "Mean stability score\n(not accuracy)"),
        ("runtime_seconds", "Runtime (seconds)"),
        ("high_overlap_mask_fraction", "High-overlap mask fraction"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(16, 16), constrained_layout=True)
    for axis, (column, title) in zip(axes.flat, fields):
        axis.plot(summary["timestamp"], summary[column], marker="o")
        axis.set_title(title)
        axis.set_xlabel("Timestamp")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle("SAM 2.1 five-frame temporal diagnostics (descriptive only)", fontsize=15)
    fig.savefig(FIGURES / "sam21_temporal_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(ROOT / "results/tables/dataset_inventory.csv", parse_dates=["timestamp"])
    selected = pick_frames(inventory)
    selected.drop(columns="path").to_csv(TABLES / "sam21_temporal_frames.csv", index=False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    parameter_report = {
        "source": "https://github.com/facebookresearch/sam2",
        "model": "SAM 2.1 Hiera Small",
        "checkpoint": str(CHECKPOINT),
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "device": device,
        "automatic_mask_generator": SAM_PARAMETERS,
        "overlap_iou_threshold": OVERLAP_IOU_THRESHOLD,
        "fixed_across_all_frames": True,
    }
    (TABLES / "sam21_parameters.json").write_text(json.dumps(parameter_report, indent=2))
    model = build_sam2(parameter_report["config"], str(CHECKPOINT), device=device,
                       apply_postprocessing=SAM_PARAMETERS["apply_postprocessing"])
    generator = SAM2AutomaticMaskGenerator(model, **{key: value for key, value in SAM_PARAMETERS.items() if key != "apply_postprocessing"})
    rows: list[dict] = []
    records: list[dict] = []
    for rec in selected.itertuples(index=False):
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
        sam_overlay, sam_boundaries = instance_visuals(image, masks)
        otsu = compute_seed_body_mask(image)
        candidates = compute_germination_candidate_masks(image, otsu, CandidateMaskConfig())
        plot_frame(rec.stage_label, rec.timestamp, image, otsu, candidates["combined_yellow_candidate"], sam_overlay, sam_boundaries)
        np.savez_compressed(FIGURES / f"{rec.stage_label.replace('-', '_')}_raw_masks.npz", **{f"mask_{i:03d}": item["segmentation"] for i, item in enumerate(masks, 1)})
        row = {
            "stage_label": rec.stage_label, "timestamp": rec.timestamp, "filename": rec.filename,
            "chronological_index": rec.chronological_index, "runtime_seconds": elapsed,
            "raw_mask_count": len(masks), "total_mask_area_pixels": int(areas.sum()),
            "union_area_pixels": int(union.sum()), "union_coverage_percentage": union.mean() * 100,
            "mean_mask_area": float(areas.mean()), "median_mask_area": float(np.median(areas)),
            "largest_mask_area": int(areas.max()), "smallest_mask_area": int(areas.min()),
            "mask_area_p10": float(np.percentile(areas, 10)), "mask_area_p25": float(np.percentile(areas, 25)),
            "mask_area_p50": float(np.percentile(areas, 50)), "mask_area_p75": float(np.percentile(areas, 75)),
            "mask_area_p90": float(np.percentile(areas, 90)), "mask_area_p95": float(np.percentile(areas, 95)),
            "mean_predicted_iou": float(np.nanmean(ious)), "median_predicted_iou": float(np.nanmedian(ious)),
            "mean_stability_score": float(np.nanmean(stability)), "median_stability_score": float(np.nanmedian(stability)),
            "border_touching_mask_count": border_count, "high_overlap_pair_count": pair_count,
            "high_overlap_mask_fraction": overlap_fraction, "overlap_iou_threshold": OVERLAP_IOU_THRESHOLD,
        }
        rows.append(row)
        records.append({"stage_label": rec.stage_label, "areas": areas})
        print(json.dumps({key: row[key] for key in ("stage_label", "raw_mask_count", "runtime_seconds", "union_coverage_percentage", "high_overlap_pair_count")}))
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "sam21_temporal_summary.csv", index=False)
    plot_area_histograms(records)
    plot_temporal(summary)


if __name__ == "__main__":
    main()
