"""One-image, reproducible Phase 2.7 SAM 2.1 automatic-mask smoke test."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "data/raw/images/recording__2026_07_15_0_04_10.png"
CHECKPOINT = ROOT / "results/models/sam2.1/sam2.1_hiera_small.pt"
FIGURES = ROOT / "results/figures/pretrained_benchmark"
TABLES = ROOT / "results/tables/pretrained_benchmark"


def choose_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run(device: str) -> tuple[list[dict], float]:
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml", str(CHECKPOINT), device=device
    )
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=16,
        points_per_batch=32,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.95,
        box_nms_thresh=0.70,
        crop_n_layers=0,
        min_mask_region_area=100,
        output_mode="binary_mask",
    )
    image = np.asarray(Image.open(IMAGE).convert("RGB"))
    start = time.perf_counter()
    masks = generator.generate(image)
    return masks, time.perf_counter() - start


def save_diagnostics(image: np.ndarray, masks: list[dict], runtime: float, device: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(27)
    overlay = image.astype(np.float32).copy()
    boundaries = image.copy()
    union = np.zeros(image.shape[:2], dtype=bool)
    rows = []
    for index, mask_data in enumerate(masks, start=1):
        mask = mask_data["segmentation"].astype(bool)
        union |= mask
        color = rng.integers(40, 256, size=3, dtype=np.uint8)
        overlay[mask] = overlay[mask] * 0.55 + color.astype(np.float32) * 0.45
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundaries, contours, -1, tuple(int(x) for x in color), 2)
        x, y, w, h = mask_data["bbox"]
        rows.append(
            {
                "mask_id": index,
                "area_pixels": int(mask_data["area"]),
                "bbox_x": float(x), "bbox_y": float(y), "bbox_width": float(w), "bbox_height": float(h),
                "predicted_iou": float(mask_data.get("predicted_iou", np.nan)),
                "stability_score": float(mask_data.get("stability_score", np.nan)),
                "crop_box": [float(v) for v in mask_data.get("crop_box", [])],
                "point_coords": np.asarray(mask_data.get("point_coords", [])).tolist(),
            }
        )
    cv2.imwrite(str(FIGURES / "sam21_smoke_original.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(FIGURES / "sam21_smoke_mask_overlay.png"), cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(FIGURES / "sam21_smoke_instance_boundaries.png"), cv2.cvtColor(boundaries, cv2.COLOR_RGB2BGR))
    np.savez_compressed(FIGURES / "sam21_smoke_masks.npz", **{f"mask_{i:03d}": m["segmentation"] for i, m in enumerate(masks, 1)})
    report = {
        "phase": "2.7",
        "source": "https://github.com/facebookresearch/sam2",
        "model": "SAM 2.1 Hiera Small",
        "checkpoint": str(CHECKPOINT),
        "image": IMAGE.name,
        "strategy": "official SAM2AutomaticMaskGenerator; points_per_side=16, points_per_batch=32, pred_iou_thresh=0.80, stability_score_thresh=0.95, box_nms_thresh=0.70, crop_n_layers=0, min_mask_region_area=100",
        "prompting": "none (automatic mask generation)",
        "device": device,
        "runtime_seconds": runtime,
        "mask_count": len(masks),
        "union_coverage_percent": float(union.mean() * 100),
        "mean_predicted_iou": float(np.mean([row["predicted_iou"] for row in rows])) if rows else None,
        "mean_stability_score": float(np.mean([row["stability_score"] for row in rows])) if rows else None,
        "caution": "SAM mask count is not a seed count; predicted_iou and stability_score are model outputs, not ground-truth accuracy metrics.",
        "artifacts": {
            "original": "results/figures/pretrained_benchmark/sam21_smoke_original.png",
            "mask_overlay": "results/figures/pretrained_benchmark/sam21_smoke_mask_overlay.png",
            "instance_boundaries": "results/figures/pretrained_benchmark/sam21_smoke_instance_boundaries.png",
            "raw_masks": "results/figures/pretrained_benchmark/sam21_smoke_masks.npz"
        },
        "masks": rows,
    }
    (TABLES / "sam21_smoke_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    image = np.asarray(Image.open(IMAGE).convert("RGB"))
    device = choose_device()
    try:
        masks, runtime = run(device)
    except Exception as error:
        if device != "mps":
            raise
        print(f"MPS automatic generation failed: {type(error).__name__}: {error}; retrying on CPU")
        device = "cpu"
        masks, runtime = run(device)
    save_diagnostics(image, masks, runtime, device)
    print(json.dumps({"device": device, "mask_count": len(masks), "runtime_seconds": runtime}, indent=2))
