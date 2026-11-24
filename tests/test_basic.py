from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.image_utils import absolute_difference, color_spaces, difference_features, load_image, split_channels
from src.io_utils import build_inventory, discover_images, parse_timestamp_from_filename
from src.preprocessing import (MorphologyConfig, SegmentationConfig, apply_morphology,
                               convert_to_gray, create_foreground_mask, threshold_image)
from src.germination_validation import (CandidateMaskConfig, compute_germination_candidate_masks,
                                        create_candidate_overlay, select_representative_frames,
                                        summarize_mask)
from src.annotation_preparation import (calculate_tile_geometry, export_annotation_manifest,
                                        propose_spatial_split, select_annotation_frames,
                                        validate_spatial_split)
from src.cvat_utils import validate_cvat_export


def test_timestamp_parsing_accepts_single_digit_hour():
    assert parse_timestamp_from_filename("recording__2026_07_15_1_02_10.png") == datetime(2026, 7, 15, 1, 2, 10)
    assert parse_timestamp_from_filename("no_timestamp.png") is None


def test_discovery_and_inventory_handles_unreadable_image(tmp_path):
    image_dir = tmp_path / "nested"
    image_dir.mkdir()
    valid = image_dir / "recording__2026_07_14_12_00_00.png"
    cv2.imwrite(str(valid), np.zeros((4, 5, 3), dtype=np.uint8))
    (image_dir / "recording__2026_07_14_13_00_00.jpg").write_bytes(b"not an image")
    (image_dir / "ignore.txt").write_text("ignore")
    assert discover_images(tmp_path) == [valid, image_dir / "recording__2026_07_14_13_00_00.jpg"]
    inventory = build_inventory(tmp_path)
    assert inventory["is_readable"].tolist() == [True, False]
    assert inventory.loc[0, "width"] == 5
    assert load_image(valid) is not None


def test_discovery_excludes_notebook_checkpoints(tmp_path):
    real_image = tmp_path / "recording__2026_07_14_12_00_00.png"
    checkpoint_dir = tmp_path / ".ipynb_checkpoints"
    checkpoint_dir.mkdir()
    cv2.imwrite(str(real_image), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(checkpoint_dir / "recording__2026_07_14_13_00_00-checkpoint.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "recording__2026_07_14_14_00_00-checkpoint.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))
    assert discover_images(tmp_path) == [real_image]


def test_difference_features():
    diff = absolute_difference(np.zeros((2, 2), dtype=np.uint8), np.array([[0, 10], [20, 30]], dtype=np.uint8))
    features = difference_features(diff, (10.0,))
    assert features["mean_difference"] == 15.0
    assert features["percentage_changed_gt_10"] == 50.0


def test_color_conversion_and_grayscale_preserve_input():
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    original = image.copy()
    spaces = color_spaces(image)
    assert spaces["rgb"].shape == image.shape
    assert spaces["gray"].shape == image.shape[:2]
    assert set(split_channels(spaces["hsv"], "hsv")) == {"H", "S", "V"}
    assert set(split_channels(spaces["lab"], "lab")) == {"L", "a", "b"}
    assert np.array_equal(convert_to_gray(image), spaces["gray"])
    assert np.array_equal(image, original)


def test_thresholding_and_morphology_produce_binary_masks():
    gray = np.array([[20, 20, 220], [20, 20, 220], [220, 220, 220]], dtype=np.uint8)
    mask = threshold_image(gray, SegmentationConfig(method="fixed", fixed_threshold=100))
    assert set(np.unique(mask)) == {0, 255}
    cleaned = apply_morphology(mask, MorphologyConfig(kernel_size=3, closing_iterations=1))
    assert set(np.unique(cleaned)).issubset({0, 255})
    initial, final = create_foreground_mask(gray, SegmentationConfig(method="otsu"))
    assert initial.shape == gray.shape
    assert final.shape == gray.shape


def test_germination_candidate_masks_are_binary_and_exclude_seed_body():
    image = np.full((40, 40, 3), 245, dtype=np.uint8)
    image[18:22, 20:24] = (0, 255, 255)  # BGR yellow candidate beside the seed body.
    seed_body = np.zeros((40, 40), dtype=np.uint8)
    seed_body[18:22, 15:19] = 255
    candidates = compute_germination_candidate_masks(
        image, seed_body,
        CandidateMaskConfig(lab_b_min=130, lightness_min=100, hsv_saturation_min=10,
                            boundary_kernel_size=11),
    )
    assert set(candidates) == {"lab_b_candidate", "lab_a_candidate", "hsv_candidate",
                               "bright_non_seed_candidate", "combined_yellow_candidate"}
    assert all(set(np.unique(mask)).issubset({0, 255}) for mask in candidates.values())
    assert not np.any((candidates["combined_yellow_candidate"] == 255) & (seed_body == 255))
    assert candidates["combined_yellow_candidate"].sum() > 0


def test_mask_summary_overlay_and_representative_selection():
    seed_body = np.zeros((8, 8), dtype=np.uint8)
    candidate = np.zeros((8, 8), dtype=np.uint8)
    candidate[1:3, 1:3] = 255
    summary = summarize_mask(candidate, seed_body)
    assert summary["foreground_pixel_count"] == 4
    assert summary["component_count"] == 1
    overlay = create_candidate_overlay(np.zeros((8, 8, 3), dtype=np.uint8), candidate)
    assert overlay[1, 1].tolist() == [140, 0, 140]
    inventory = pd.DataFrame({
        "filename": ["late.png", "early.png", "middle.png"],
        "timestamp": pd.to_datetime(["2026-07-15", "2026-07-13", "2026-07-14"]),
        "path": ["late.png", "early.png", "middle.png"],
        "is_readable": [True, True, True],
    })
    representative = select_representative_frames(inventory)
    assert representative["filename"].tolist() == ["early.png", "middle.png", "late.png"]


def test_annotation_selection_and_spatial_split(tmp_path):
    timestamps = pd.to_datetime(["2026-07-14 12:01:32", "2026-07-14 18:11:11", "2026-07-15 00:04:10", "2026-07-15 06:13:59", "2026-07-15 12:06:59"])
    inventory = pd.DataFrame({"image_id": range(5), "timestamp": timestamps, "filename": [f"f{i}.png" for i in range(5)], "path": [str(tmp_path / f"f{i}.png") for i in range(5)], "width": [1920]*5, "height": [1080]*5, "is_readable": [True]*5})
    selected = select_annotation_frames(inventory)
    assert selected["stage_label"].tolist() == ["early", "early-middle", "middle", "middle-late", "late"]
    tiles = propose_spatial_split(calculate_tile_geometry(1920, 1080, 3, 2), 3, 2)
    validate_spatial_split(tiles, 1920, 1080)
    assert tiles.tile_id.tolist() == ["r00_c00", "r00_c01", "r00_c02", "r01_c00", "r01_c01", "r01_c02"]
    manifest = export_annotation_manifest(selected, tmp_path / "manifest.csv")
    assert manifest.number_of_reviewed_instances.tolist() == [0] * 5
    try:
        select_annotation_frames(inventory.iloc[:-1])
    except ValueError as error:
        assert "absent" in str(error)
    else:
        raise AssertionError("A missing required annotation frame should fail")


def test_cvat_export_rejects_invalid_polygon(tmp_path):
    export = {"images": [{"id": 1, "file_name": "f.png", "width": 10, "height": 10}], "categories": [{"id": 1, "name": "seed_body"}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[]]}]}
    path = tmp_path / "invalid.json"; path.write_text(__import__("json").dumps(export))
    try:
        validate_cvat_export(path)
    except ValueError as error:
        assert "polygon" in str(error)
    else:
        raise AssertionError("Invalid polygon should fail validation")


def test_cvat_export_rejects_invalid_attribute(tmp_path):
    export = {"images": [{"id": 1, "file_name": "f.png", "width": 10, "height": 10}], "categories": [{"id": 1, "name": "seed_body"}], "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0, 0, 5, 0, 5, 5]], "attributes": {"truncated": "maybe"}}]}
    path = tmp_path / "invalid_attr.json"; path.write_text(__import__("json").dumps(export))
    try:
        validate_cvat_export(path)
    except ValueError as error:
        assert "attribute" in str(error)
    else:
        raise AssertionError("Invalid attribute should fail validation")
