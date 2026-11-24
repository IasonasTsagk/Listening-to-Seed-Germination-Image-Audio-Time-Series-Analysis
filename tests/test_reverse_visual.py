from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reverse_visual import (append_manual_candidate, initial_candidates,
    intersects, tile_change_summary, window_inventory)


def test_window_inventory_uses_parsed_timestamps_and_raw_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    names = ["recording__2026_07_14_19_22_27.png",
             "recording__2026_07_14_19_22_37.png",
             "recording__2026_07_14_19_22_47.png"]
    for name in names:
        Image.new("RGB", (40, 30), "black").save(raw / name)
    index = pd.DataFrame({"filename": names[::-1],
                          "timestamp": ["2026-07-14 19:22:47", "2026-07-14 19:22:37", "2026-07-14 19:22:27"],
                          "source_path": [str(raw/name) for name in names[::-1]]})
    path = tmp_path / "index.csv"
    index.to_csv(path, index=False)
    loaded = window_inventory(path, raw, pd.Timestamp("2026-07-14 19:22:28"),
                              pd.Timestamp("2026-07-14 19:22:47"))
    assert loaded.filename.tolist() == names[1:]
    assert loaded.timestamp.is_monotonic_increasing
    assert loaded.readable.all()
    assert loaded.seconds_from_acoustic_event.iloc[0] == pytest.approx(-51.22)


def test_tile_change_ranking_and_bounds():
    first = np.zeros((60, 80, 3), dtype=np.uint8)
    second = first.copy()
    second[:30, :40] = 200
    stamps = [pd.Timestamp("2026-07-14 19:23:20"), pd.Timestamp("2026-07-14 19:23:30")]
    ranked, diffs = tile_change_summary([first, second], stamps, columns=2, rows=2)
    assert ranked.iloc[0].tile_id == "tile_r01_c01"
    assert ranked.iloc[0].cumulative_change_score == pytest.approx(200)
    assert len(diffs) == 1
    assert ranked.x_min.min() >= 0 and ranked.x_max.max() <= 80
    assert ranked.y_min.min() >= 0 and ranked.y_max.max() <= 60
    assert intersects(ranked.iloc[0], {"x_min": 10, "y_min": 10, "x_max": 20, "y_max": 20})
    with pytest.raises(ValueError):
        tile_change_summary([first, second], stamps[::-1], columns=2, rows=2)


def test_candidates_preserve_human_fields_and_append_without_overwrite(tmp_path):
    tiles = pd.DataFrame([{"tile_id": f"tile_r01_c{i+1:02d}",
                           "x_min": i*10, "y_min": 0, "x_max": (i+1)*10, "y_max": 10}
                          for i in range(5)])
    path = tmp_path / "candidates.csv"
    initial = initial_candidates(tiles, path)
    assert len(initial) == 5 and initial.likely_event_type.eq("").all()
    reviewed = pd.read_csv(path, keep_default_na=False)
    reviewed.loc[0, "notes"] = "human observation"
    reviewed.to_csv(path, index=False)
    assert initial_candidates(tiles, path).loc[0, "notes"] == "human observation"
    candidate_id = append_manual_candidate(path, (2, 3, 9, 12), (80, 60))
    saved = pd.read_csv(path, keep_default_na=False)
    assert candidate_id == "candidate_06"
    assert len(saved) == 6 and saved.loc[0, "notes"] == "human observation"
    assert saved.iloc[-1].source_type == "manual_roi"
    assert saved.iloc[-1].likely_seed_present == ""
    assert len(list(tmp_path.glob("candidates.backup_*.csv"))) == 1
    with pytest.raises(ValueError):
        append_manual_candidate(path, (-1, 0, 10, 10), (80, 60))
