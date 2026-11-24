"""Create Phase 2.8A derived CVAT preparation artifacts; never annotation labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation_preparation import (build_cvat_task_spec, calculate_otsu_tile_coverage,
    calculate_tile_geometry, dataset_version_record, export_annotation_manifest, load_inventory,
    prepare_cvat_upload_folder, propose_spatial_split, select_annotation_frames, validate_spatial_split,
    verify_annotation_frames)


def plot_overview(frames: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), constrained_layout=True)
    for axis, row in zip(axes, frames.itertuples(index=False)):
        axis.imshow(cv2.cvtColor(cv2.imread(row.source_path), cv2.COLOR_BGR2RGB))
        axis.set_title(f"{row.stage_label}\n{row.timestamp}")
        axis.axis("off")
    fig.suptitle("Phase 2.8A full-frame annotation working copies")
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid_comparison(image_path: str, width: int, height: int, destination: Path) -> None:
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    for axis, (columns, rows) in zip(axes, [(3, 2), (4, 2)]):
        axis.imshow(image)
        for tile in calculate_tile_geometry(width, height, columns, rows).itertuples(index=False):
            axis.add_patch(plt.Rectangle((tile.x_min, tile.y_min), tile.width, tile.height,
                                         fill=False, edgecolor="yellow", linewidth=2))
            axis.text(tile.x_min + 12, tile.y_min + 28, tile.tile_id, color="black",
                      bbox={"facecolor": "yellow", "alpha": .8, "pad": 2})
        axis.set_title(f"{columns}×{rows} fixed spatial grid")
        axis.axis("off")
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_preview(image_path: str, tiles: pd.DataFrame, destination: Path) -> None:
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    colors = {"train": "#22aa55", "validation": "#f0a202", "test": "#d62828"}
    fig, axis = plt.subplots(figsize=(14, 8), constrained_layout=True)
    axis.imshow(image)
    for tile in tiles.itertuples(index=False):
        axis.add_patch(plt.Rectangle((tile.x_min, tile.y_min), tile.width, tile.height,
                                     facecolor=colors[tile.proposed_split], edgecolor="white", alpha=.24, linewidth=2))
        axis.text(tile.x_min + 12, tile.y_min + 30, f"{tile.tile_id}\n{tile.proposed_split}", color="white",
                  bbox={"facecolor": colors[tile.proposed_split], "alpha": .9, "pad": 2})
    axis.set_title("Fixed spatial split: regions, not timestamps, define train/validation/test")
    axis.axis("off")
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    annotation_root = ROOT / "data/annotations"
    upload = annotation_root / "cvat_upload"
    for directory in (annotation_root, upload, annotation_root / "cvat_preannotations",
                      annotation_root / "cvat_exports", annotation_root / "archive"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch(exist_ok=True)
    tables = ROOT / "results/tables/annotation_preparation"
    figures = ROOT / "results/figures/annotation_preparation"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    frames = verify_annotation_frames(select_annotation_frames(load_inventory(ROOT / "results/tables/dataset_inventory.csv")))
    frames = prepare_cvat_upload_folder(frames, upload).rename(columns={"path": "source_path"})
    frames[["stage_label", "timestamp", "filename", "width", "height", "source_path", "cvat_upload_path"]].to_csv(
        tables / "representative_annotation_frames.csv", index=False)
    export_annotation_manifest(frames, tables / "annotation_manifest.csv")
    (tables / "cvat_task_spec.json").write_text(json.dumps(build_cvat_task_spec(), indent=2))
    (tables / "dataset_version.json").write_text(json.dumps(dataset_version_record(frames), indent=2, default=str))
    evaluations = []
    selected_plan = None
    for columns, rows in ((3, 2), (4, 2)):
        tiles = propose_spatial_split(calculate_tile_geometry(1920, 1080, columns, rows), columns, rows)
        tiles["approximate_otsu_coverage"] = calculate_otsu_tile_coverage(frames["source_path"], tiles)
        validate_spatial_split(tiles, 1920, 1080)
        evaluations.append({"grid": f"{columns}x{rows}", "tile_size": [int(tiles.width.iloc[0]), int(tiles.height.iloc[0])],
                            "split_tile_counts": tiles.proposed_split.value_counts().to_dict(),
                            "coverage_by_split": tiles.groupby("proposed_split").approximate_otsu_coverage.mean().to_dict()})
        if (columns, rows) == (3, 2): selected_plan = tiles
    selected_plan.to_csv(tables / "spatial_tile_manifest.csv", index=False)
    plan = {"evaluated_grid_layouts": evaluations, "chosen_grid": "3x2", "rationale": "640x540 tiles retain more seed context than 4x2 and give 4/1/1 tiles (67%/17%/17%), close to the target spatial split.",
            "tile_to_split_mapping": selected_plan[["tile_id", "proposed_split"]].to_dict("records"),
            "limitations": "Within-experiment spatial holdout only: this single scene cannot demonstrate generalization to a new seed batch, experiment, plate, camera, lighting, or setup.",
            "future_boundary_policy": "Annotate full frames first. After human review, assign an instance to one tile by centroid or majority mask area; configure a boundary buffer and exclude near-boundary instances from validation/test where needed. Never duplicate one biological object across train and test."}
    (tables / "spatial_split_plan.json").write_text(json.dumps(plan, indent=2))
    plot_overview(frames, figures / "representative_frames_overview.png")
    plot_grid_comparison(frames.source_path.iloc[0], 1920, 1080, figures / "grid_comparison.png")
    plot_split_preview(frames.source_path.iloc[0], selected_plan, figures / "spatial_split_preview.png")
    status = {"phase": "2.8A", "cvat_ready": True, "selected_frame_count": len(frames), "label_schema": build_cvat_task_spec()["labels"],
              "chosen_spatial_grid": "3x2", "train_tiles": 4, "validation_tiles": 1, "test_tiles": 1,
              "sam_preannotation_available": False, "sam_preannotation_note": "Raw SAM masks are preserved, but no faithful CVAT proposal import was produced. Use SAM interactively or manually review proposals; no prediction is ground truth.",
              "annotation_protocol_path": "docs/annotation_protocol_v1.md", "cvat_upload_path": str(upload),
              "ready_for_manual_annotation": True, "ready_for_model_training": False, "ready_for_phase_3": False}
    (tables / "phase_2_8A_status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
