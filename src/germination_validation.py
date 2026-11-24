"""Phase 2.5 validation of interpretable germination-candidate masks.

These utilities deliberately produce *candidate* masks, not radicle labels.
They provide traceable visual evidence for deciding whether later feature work is
appropriate, while retaining grayscale Otsu as the independent seed-body baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .image_utils import color_spaces, load_image, validate_image
from .preprocessing import SegmentationConfig, create_foreground_mask, validate_binary_mask


@dataclass(frozen=True)
class CandidateMaskConfig:
    """Transparent thresholds for exploratory yellow/bright candidate masks.

    Channels use OpenCV's uint8 representations. The boundary band is optional:
    it limits bright-background leakage by examining pixels immediately outside
    dark seed bodies, but may exclude detached structures and must be reviewed.
    """

    lab_b_min: int = 145
    lab_a_max: int = 140
    lightness_min: int = 135
    brightness_min: int = 175
    hsv_hue_min: int = 12
    hsv_hue_max: int = 48
    hsv_saturation_min: int = 35
    hsv_value_min: int = 110
    exclude_seed_body: bool = True
    restrict_to_boundary_band: bool = True
    boundary_kernel_size: int = 31


def _as_mask(condition: np.ndarray) -> np.ndarray:
    mask = np.where(condition, 255, 0).astype(np.uint8)
    validate_binary_mask(mask)
    return mask


def select_representative_frames(inventory: pd.DataFrame,
                                 manual_filenames: Iterable[str] | None = None) -> pd.DataFrame:
    """Select chronological early/middle/late readable frames or manual filenames.

    Manual selection preserves supplied filenames after validating that they are
    experimental frames present in the provided inventory.
    """
    required = {"filename", "timestamp", "path", "is_readable"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Inventory lacks required columns: {sorted(missing)}")
    frames = inventory.loc[inventory["is_readable"]].copy()
    frames["timestamp"] = pd.to_datetime(frames["timestamp"], errors="coerce")
    frames = frames.sort_values(["timestamp", "filename"], na_position="last").reset_index(drop=True)
    if frames.empty:
        raise ValueError("No readable experimental images are available.")
    if manual_filenames:
        requested = list(manual_filenames)
        selected = frames.set_index("filename", drop=False).reindex(requested)
        if selected["path"].isna().any():
            missing_names = selected.index[selected["path"].isna()].tolist()
            raise ValueError(f"Manual filenames absent from inventory: {missing_names}")
        return selected.reset_index(drop=True)
    indices = sorted(set((0, len(frames) // 2, len(frames) - 1)))
    selected = frames.iloc[indices].copy().reset_index(drop=True)
    selected.insert(0, "representative_period", ["early", "middle", "late"][:len(selected)])
    return selected


def compute_seed_body_mask(image: np.ndarray) -> np.ndarray:
    """Return the validated dark-seed baseline: grayscale Otsu without cleanup."""
    _, mask = create_foreground_mask(image, SegmentationConfig(method="otsu", dark_foreground=True))
    return mask


def boundary_band(seed_body_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Return pixels outside but within a configurable dilation of a seed-body mask."""
    validate_binary_mask(seed_body_mask)
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("boundary_kernel_size must be odd and at least 3.")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(seed_body_mask, kernel)
    return _as_mask((dilated == 255) & (seed_body_mask == 0))


def compute_germination_candidate_masks(image: np.ndarray, seed_body_mask: np.ndarray,
                                        config: CandidateMaskConfig = CandidateMaskConfig()) -> dict[str, np.ndarray]:
    """Build color/brightness candidate masks with explicit exclusions and thresholds."""
    validate_image(image)
    validate_binary_mask(seed_body_mask)
    spaces = color_spaces(image)
    lab_l, lab_a, lab_b = cv2.split(spaces["lab"])
    hue, saturation, value = cv2.split(spaces["hsv"])

    allowed = np.ones(seed_body_mask.shape, dtype=bool)
    if config.exclude_seed_body:
        allowed &= seed_body_mask == 0
    if config.restrict_to_boundary_band:
        allowed &= boundary_band(seed_body_mask, config.boundary_kernel_size) == 255

    lab_b_candidate = allowed & (lab_b >= config.lab_b_min) & (lab_l >= config.lightness_min)
    lab_a_candidate = allowed & (lab_a <= config.lab_a_max) & (lab_l >= config.lightness_min)
    hsv_candidate = (allowed & (hue >= config.hsv_hue_min) & (hue <= config.hsv_hue_max)
                     & (saturation >= config.hsv_saturation_min) & (value >= config.hsv_value_min))
    bright_non_seed_candidate = allowed & (lab_l >= config.brightness_min)
    combined_yellow_candidate = lab_b_candidate | hsv_candidate

    return {
        "lab_b_candidate": _as_mask(lab_b_candidate),
        "lab_a_candidate": _as_mask(lab_a_candidate),
        "hsv_candidate": _as_mask(hsv_candidate),
        "bright_non_seed_candidate": _as_mask(bright_non_seed_candidate),
        "combined_yellow_candidate": _as_mask(combined_yellow_candidate),
    }


def summarize_mask(mask: np.ndarray, seed_body_mask: np.ndarray) -> dict[str, float | int]:
    """Calculate non-biological mask/component diagnostics for one candidate."""
    validate_binary_mask(mask)
    validate_binary_mask(seed_body_mask)
    foreground = mask == 255
    seed = seed_body_mask == 255
    count = int(foreground.sum())
    overlap = int((foreground & seed).sum())
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    height, width = mask.shape
    border_count = 0
    for label, (x, y, component_width, component_height, _) in enumerate(stats[1:], start=1):
        if x == 0 or y == 0 or x + component_width == width or y + component_height == height:
            border_count += 1
    return {
        "foreground_pixel_count": count,
        "foreground_percentage": float(count / mask.size * 100),
        "overlap_seed_body_pixel_count": overlap,
        "non_overlap_pixel_count": count - overlap,
        "component_count": int(labels_count - 1),
        "mean_component_size_pixels": float(areas.mean()) if len(areas) else 0.0,
        "border_touching_component_count": border_count,
    }


def create_candidate_overlay(rgb_image: np.ndarray, mask: np.ndarray,
                             color: tuple[int, int, int] = (255, 0, 255), alpha: float = 0.55) -> np.ndarray:
    """Return a non-destructive RGB overlay highlighting candidate-mask pixels."""
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have three channels.")
    validate_binary_mask(mask)
    if rgb_image.shape[:2] != mask.shape:
        raise ValueError("RGB image and mask dimensions must match.")
    result = rgb_image.copy().astype(np.float32)
    candidate = mask == 255
    result[candidate] = (1 - alpha) * result[candidate] + alpha * np.array(color, dtype=np.float32)
    return result.astype(np.uint8)


def plot_germination_validation_diagnostic(image: np.ndarray, seed_body_mask: np.ndarray,
                                           candidates: dict[str, np.ndarray], title: str,
                                           destination: Path) -> None:
    """Save a labeled visual diagnostic for one representative frame."""
    rgb = color_spaces(image)["rgb"]
    combined = candidates["combined_yellow_candidate"]
    panels = [
        ("Original RGB", rgb, None),
        ("Seed-body mask\nOtsu grayscale", seed_body_mask, "gray"),
        ("LAB-b candidate", candidates["lab_b_candidate"], "gray"),
        ("LAB-a candidate", candidates["lab_a_candidate"], "gray"),
        ("HSV candidate", candidates["hsv_candidate"], "gray"),
        ("Bright non-seed candidate", candidates["bright_non_seed_candidate"], "gray"),
        ("Combined yellow candidate", combined, "gray"),
        ("Combined-candidate overlay", create_candidate_overlay(rgb, combined), None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for axis, (label, panel, cmap) in zip(axes.flat, panels):
        axis.imshow(panel, cmap=cmap)
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(title, fontsize=14)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _recommendation(config: CandidateMaskConfig) -> dict:
    """Create an intentionally cautious conclusion in the absence of annotations."""
    return {
        "phase": "2.5_germination_segmentation_validation",
        "seed_body_mask_default": "grayscale_otsu_dark_foreground_without_heavy_morphology",
        "provisional_germination_candidate": "combined_yellow_candidate",
        "candidate_rule": {
            "lab_b_min": config.lab_b_min,
            "hsv_hue_range": [config.hsv_hue_min, config.hsv_hue_max],
            "hsv_saturation_min": config.hsv_saturation_min,
            "boundary_band_restriction": config.restrict_to_boundary_band,
            "exclude_seed_body": config.exclude_seed_body,
        },
        "white_yellow_structure_capture_validated": False,
        "ready_for_phase_3": False,
        "reason": (
            "Candidate masks are interpretable visual hypotheses only. Review overlays against "
            "the RGB images and add annotations or a documented acceptance criterion before "
            "using any candidate as a radicle feature input."
        ),
    }


def run_phase_2_5_validation(inventory: pd.DataFrame, figures_dir: Path, tables_dir: Path,
                             config: CandidateMaskConfig = CandidateMaskConfig(),
                             manual_filenames: Iterable[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run Phase 2.5 diagnostics and write figures/tables under ``results/`` only."""
    selected = select_representative_frames(inventory, manual_filenames)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(tables_dir / "representative_frames.csv", index=False)

    summary_rows: list[dict] = []
    for record in selected.itertuples(index=False):
        image = load_image(Path(record.path), expected_size=(int(record.width), int(record.height)))
        if image is None:
            raise ValueError(f"Representative image became unreadable: {record.filename}")
        seed_mask = compute_seed_body_mask(image)
        candidates = compute_germination_candidate_masks(image, seed_mask, config)
        period = getattr(record, "representative_period", "manual")
        for method, mask in candidates.items():
            summary_rows.append({"representative_period": period, "timestamp": record.timestamp,
                                 "filename": record.filename, "method": method,
                                 **summarize_mask(mask, seed_mask)})
        timestamp = pd.Timestamp(record.timestamp)
        plot_germination_validation_diagnostic(
            image, seed_mask, candidates,
            f"Phase 2.5 germination-candidate validation — {period} — {timestamp:%Y-%m-%d %H:%M:%S}",
            figures_dir / f"germination_validation_{timestamp:%Y%m%d_%H%M%S}.png",
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables_dir / "germination_candidate_summary.csv", index=False)
    recommendation = _recommendation(config)
    with (tables_dir / "phase_2_5_recommendation.json").open("w", encoding="utf-8") as handle:
        json.dump(recommendation, handle, indent=2)
    return selected, summary, recommendation
