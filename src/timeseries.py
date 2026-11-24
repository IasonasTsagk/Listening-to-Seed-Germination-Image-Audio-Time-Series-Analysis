"""Sequential global image-change time-series analysis."""
from __future__ import annotations
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .config import ChangeConfig
from .image_utils import absolute_difference, difference_features, load_image
from .preprocessing import preprocess_for_analysis

LOGGER = logging.getLogger(__name__)


def _series_columns(config: ChangeConfig) -> list[str]:
    return ["timestamp", "previous_filename", "filename", "mean_difference", "median_difference",
            "std_difference", *[f"percentage_changed_gt_{threshold:g}" for threshold in config.thresholds]]


def global_change_timeseries(
    inventory: pd.DataFrame,
    config: ChangeConfig = ChangeConfig(),
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 500,
) -> pd.DataFrame:
    """Process consecutive readable frames one at a time and return global change features.

    Frames lacking timestamps are retained with ``NaT`` timestamps, but their filename order
    is used. Shape changes are logged and skipped because rescaling is an explicit choice.
    When ``checkpoint_path`` is supplied, a CSV is updated every ``checkpoint_every`` pairs.
    Re-running resumes after its most recently completed filename.
    """
    valid = inventory.loc[inventory["is_readable"]].copy()
    completed = pd.DataFrame(columns=_series_columns(config))
    if checkpoint_path and checkpoint_path.exists():
        completed = pd.read_csv(checkpoint_path, parse_dates=["timestamp"])
        available_filenames = set(valid["filename"])
        checkpoint_filenames = set(completed.get("filename", pd.Series(dtype=str)).dropna())
        checkpoint_predecessors = set(completed.get("previous_filename", pd.Series(dtype=str)).dropna())
        if not checkpoint_filenames.issubset(available_filenames) or not checkpoint_predecessors.issubset(available_filenames):
            LOGGER.warning("Checkpoint contains files absent from the current inventory; rebuilding it.")
            completed = pd.DataFrame(columns=_series_columns(config))
        if not completed.empty:
            last_filename = completed.iloc[-1]["filename"]
            positions = valid.index[valid["filename"] == last_filename].tolist()
            if positions:
                # Retain the last completed frame so the next comparison uses it as predecessor.
                valid = valid.loc[valid.index >= positions[-1]]
                LOGGER.info("Resuming global change analysis after %s", last_filename)
            else:
                LOGGER.warning("Checkpoint filename %s is absent from inventory; starting over.", last_filename)
                completed = pd.DataFrame(columns=_series_columns(config))
    rows, previous, previous_name = [], None, None
    for record in valid.itertuples(index=False):
        image = load_image(Path(record.path))
        if image is None:
            continue
        current = preprocess_for_analysis(image, config.preprocessing)
        if previous is not None:
            try:
                diff = absolute_difference(previous, current)
            except ValueError:
                LOGGER.warning("Skipping mismatched image dimensions: %s -> %s", previous_name, record.filename)
            else:
                rows.append({"timestamp": record.timestamp, "previous_filename": previous_name,
                             "filename": record.filename, **difference_features(diff, config.thresholds)})
                if checkpoint_path and len(rows) % checkpoint_every == 0:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.concat([completed, pd.DataFrame(rows, columns=_series_columns(config))], ignore_index=True).to_csv(checkpoint_path, index=False)
        previous, previous_name = current, record.filename
        del image
    result = pd.concat([completed, pd.DataFrame(rows, columns=_series_columns(config))], ignore_index=True)
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(checkpoint_path, index=False)
    return result


def plot_global_change(series: pd.DataFrame, destination: Path | None = None) -> None:
    """Plot selected global difference features and optionally save the figure."""
    if series.empty:
        raise ValueError("No consecutive readable image pairs were available to plot.")
    columns = [column for column in series if column.startswith("percentage_changed_gt_")]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(series["timestamp"], series["mean_difference"], marker="o", label="Mean difference")
    axes[0].plot(series["timestamp"], series["median_difference"], marker="o", label="Median difference")
    axes[0].set_ylabel("Intensity difference (0–255)")
    axes[0].legend()
    for column in columns:
        axes[1].plot(series["timestamp"], series[column], marker="o", label=column.replace("percentage_changed_", ""))
    axes[1].set_ylabel("Changed pixels (%)")
    axes[1].set_xlabel("Timestamp of later frame")
    axes[1].legend()
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.show()
