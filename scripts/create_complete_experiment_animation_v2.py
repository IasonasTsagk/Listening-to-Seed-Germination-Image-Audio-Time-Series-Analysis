"""Adaptive, real-frame full-experiment presentation time-lapse (v2)."""
from __future__ import annotations

import bisect
import csv
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(os.environ.get('SEED_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / 'scripts'))
from create_presentation_animations import font, readable, render, RESAMPLE
from create_complete_experiment_animation import scan, valid_boundaries

RAW = Path(os.environ.get("SEED_GERMINATION_SNAPSHOT_ROOT", PROJECT / "data/raw/snapshots"))
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / 'results/figures/presentation_assets'
TABLES = PROJECT / 'results/tables/targeted_multimodal'
STEM = 'full_scene_complete_experiment_v2'
SIZE = (1280, 720)
FPS = 12
BASELINE_COUNT = 112
EXTRA_COUNT = 56
COARSE_STEP = timedelta(minutes=30)
THUMB_SIZE = (128, 72)
TOTAL_COUNT = BASELINE_COUNT + EXTRA_COUNT


def nearest_valid(rows: list[tuple[datetime, Path]], times: list[datetime],
                  target: datetime, used: set[Path], readable_cache: dict[Path, bool],
                  window: int = 120) -> tuple[int, bool] | None:
    pivot = bisect.bisect_left(times, target)
    indices = sorted(range(max(0, pivot-window), min(len(rows), pivot+window+1)),
                     key=lambda i: (abs((times[i]-target).total_seconds()), i))
    encountered_bad = False
    for index in indices:
        path = rows[index][1]
        if path in used:
            continue
        if path not in readable_cache:
            readable_cache[path] = readable(path)
        if readable_cache[path]:
            return index, encountered_bad
        encountered_bad = True
    return None


def coarse_activity(rows: list[tuple[datetime, Path]], times: list[datetime],
                    readable_cache: dict[Path, bool]) -> tuple[list[tuple[float, datetime]], int]:
    """Difference of consecutive real 30-minute thumbnails; presentation cue only."""
    start, end = times[0], times[-1]
    locations = []
    target = start
    bad = 0
    seen = set()
    while target < end:
        found = nearest_valid(rows, times, target, set(), readable_cache)
        if found is not None:
            index, skipped_bad = found
            bad += int(skipped_bad)
            if index not in seen:
                locations.append(index)
                seen.add(index)
        target += COARSE_STEP
    locations.append(len(rows)-1)
    seen.add(len(rows)-1)
    locations = sorted(seen)
    changes = []
    previous = None
    for number, index in enumerate(locations):
        timestamp, path = rows[index]
        try:
            with Image.open(path) as original:
                if original.size != (1920, 1080):
                    raise ValueError('unexpected dimensions')
                thumbnail = np.asarray(original.convert('L').resize(
                    THUMB_SIZE, Image.Resampling.BILINEAR), dtype=np.int16)
        except (OSError, ValueError):
            readable_cache[path] = False
            bad += 1
            continue
        if previous is not None:
            old_timestamp, old_thumbnail = previous
            elapsed = timestamp - old_timestamp
            if elapsed <= timedelta(hours=1):
                difference = float(np.mean(np.abs(thumbnail - old_thumbnail)))
                midpoint = old_timestamp + elapsed / 2
                changes.append((difference, midpoint))
        previous = (timestamp, thumbnail)
        if (number+1) % 150 == 0 or number+1 == len(locations):
            print(f'Coarse visual scan: {number+1}/{len(locations)} real frames', flush=True)
    return changes, bad


def select_adaptive(rows: list[tuple[datetime, Path]],
                    changes: list[tuple[float, datetime]], readable_cache: dict[Path, bool]) -> tuple[list[tuple[datetime, Path, str]], int]:
    start, end = rows[0][0], rows[-1][0]
    times = [row[0] for row in rows]
    selected: dict[Path, tuple[datetime, Path, str]] = {}
    encountered_bad = 0
    # Fixed-interval coverage guarantees that high activity cannot hide quiet days.
    for position in range(BASELINE_COUNT):
        target = start + (end-start) * (position/(BASELINE_COUNT-1))
        found = nearest_valid(rows, times, target, set(selected), readable_cache)
        if found is None:
            raise RuntimeError(f'Could not find readable baseline frame near {target}')
        index, skipped_bad = found
        encountered_bad += int(skipped_bad)
        timestamp, path = rows[index]
        reason = 'start/end anchor' if position in (0, BASELINE_COUNT-1) else 'baseline coverage'
        selected[path] = (timestamp, path, reason)
    # Use the largest thumbnail changes, but cap extra frames per day and prevent
    # near-duplicate frame times. This keeps the time-lapse spread across days.
    extras_by_day: dict[object, int] = {}
    extra_count = 0
    for score, target in sorted(changes, key=lambda item: (-item[0], item[1])):
        if extra_count == EXTRA_COUNT:
            break
        if extras_by_day.get(target.date(), 0) >= 10:
            continue
        found = nearest_valid(rows, times, target, set(selected), readable_cache)
        if found is None:
            continue
        index, skipped_bad = found
        encountered_bad += int(skipped_bad)
        timestamp, path = rows[index]
        if min(abs((timestamp-existing[0]).total_seconds()) for existing in selected.values()) < 15*60:
            continue
        selected[path] = (timestamp, path, 'high-change region')
        extras_by_day[target.date()] = extras_by_day.get(target.date(), 0) + 1
        extra_count += 1
    if extra_count != EXTRA_COUNT:
        raise RuntimeError(f'Only selected {extra_count}/{EXTRA_COUNT} high-change frames')
    ordered = sorted(selected.values(), key=lambda row: row[0])
    assert len(ordered) == TOTAL_COUNT
    assert ordered[0][0:2] == rows[0] and ordered[-1][0:2] == rows[-1]
    assert all(a[0] < b[0] for a,b in zip(ordered,ordered[1:]))
    return ordered, encountered_bad


def make_preview(selected: list[tuple[datetime, Path, str]], frames: list[Image.Image]) -> None:
    percentages = [0, 15, 30, 45, 60, 75, 90, 100]
    thumb = (280, 158)
    label_height = 56
    sheet = Image.new('RGB',(thumb[0]*len(percentages),thumb[1]+label_height),'#ededed')
    draw = ImageDraw.Draw(sheet)
    for column, percent in enumerate(percentages):
        index = round((len(selected)-1)*percent/100)
        stamp = selected[index][0]
        sheet.paste(frames[index].resize(thumb,RESAMPLE),(column*thumb[0],0))
        draw.text((column*thumb[0]+6,thumb[1]+4),f'{percent}%  {stamp:%Y-%m-%d}',
                  fill='#111111',font=font(16))
        draw.text((column*thumb[0]+6,thumb[1]+29),f'{stamp:%H:%M:%S}',
                  fill='#111111',font=font(16))
    sheet.save(OUTPUT/f'{STEM}_preview.png')


def render_outputs(selected: list[tuple[datetime, Path, str]]) -> None:
    start = selected[0][0]
    duration = len(selected)/FPS
    assert 12 <= duration <= 15
    video_path = OUTPUT/f'{STEM}.mp4'
    writer = cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'avc1'),FPS,SIZE)
    if not writer.isOpened():
        raise RuntimeError('H.264 MP4 writer did not open')
    frames = []
    original_stats = {}
    records = []
    try:
        for index,(timestamp,path,reason) in enumerate(selected):
            original_stats[path] = (path.stat().st_size,path.stat().st_mtime_ns)
            frame = render(path,timestamp,None,SIZE,'full_scene')
            writer.write(cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR))
            frames.append(frame)
            delta = (timestamp-start).total_seconds()
            records.append({'frame_index':index,'source_filename':path.name,
                            'source_timestamp':timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'relative_time_hours':f'{delta/3600:.6f}',
                            'relative_time_days':f'{delta/86400:.8f}',
                            'sampling_weight_or_reason':reason})
            if (index+1) % 35 == 0 or index+1 == len(selected):
                print(f'Rendered {index+1}/{len(selected)} real frames',flush=True)
    finally:
        writer.release()
    assert all((path.stat().st_size,path.stat().st_mtime_ns)==original
               for path,original in original_stats.items()),'A raw source image changed'
    make_preview(selected,frames)
    sample = Image.new('RGB',(SIZE[0],SIZE[1]*3))
    for index, source_index in enumerate((0,len(frames)//2,len(frames)-1)):
        sample.paste(frames[source_index],(0,index*SIZE[1]))
    palette = sample.quantize(colors=128,method=Image.Quantize.MEDIANCUT)
    gif_frames = [frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
    gif_path = OUTPUT/f'{STEM}.gif'
    # 12 fps is 83.333 ms; 80/80/90 ms cadence preserves the exact 14 s runtime.
    gif_durations = [90 if index % 3 == 2 else 80 for index in range(len(frames))]
    gif_frames[0].save(gif_path,save_all=True,append_images=gif_frames[1:],
                       duration=gif_durations,loop=0,optimize=True,disposal=2)
    with Image.open(gif_path) as gif:
        assert gif.n_frames == len(frames)
        assert gif.info.get('loop') == 0 and gif.size == SIZE
        gif_duration = sum(gif.seek(i) or gif.info['duration'] for i in range(gif.n_frames))/1000
        assert abs(gif_duration-duration) < .01,(gif_duration,duration)
    video = cv2.VideoCapture(str(video_path))
    assert video.isOpened()
    assert round(video.get(cv2.CAP_PROP_FRAME_COUNT))==len(frames)
    assert round(video.get(cv2.CAP_PROP_FPS))==FPS
    assert (round(video.get(cv2.CAP_PROP_FRAME_WIDTH)),round(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))==SIZE
    for index in (0,len(frames)//2,len(frames)-1):
        video.set(cv2.CAP_PROP_POS_FRAMES,index)
        assert video.read()[0],f'Could not decode MP4 frame {index}'
    codec = int(video.get(cv2.CAP_PROP_FOURCC))
    assert ''.join(chr((codec>>(8*i))&255) for i in range(4))=='h264'
    video.release()
    default_output = PROJECT/'results/figures/presentation_assets'
    table_dir = TABLES if OUTPUT == default_output else OUTPUT
    table_path = table_dir/'presentation_animation_complete_experiment_v2_frames.csv'
    with table_path.open('w',newline='') as stream:
        csv_writer = csv.DictWriter(stream,fieldnames=['frame_index','source_filename','source_timestamp',
                                                      'relative_time_hours','relative_time_days',
                                                      'sampling_weight_or_reason'])
        csv_writer.writeheader()
        csv_writer.writerows(records)
    print(f'Validated {len(frames)} distinct real frames at {FPS} fps, {duration:.2f}s, '
          f'{SIZE[0]}x{SIZE[1]}, H.264 MP4 and looping GIF',flush=True)
    print(f'Sampling reasons: baseline={sum(r[2]=="baseline coverage" for r in selected)}, '
          f'high-change={sum(r[2]=="high-change region" for r in selected)}, anchors=2',flush=True)
    print(f'Frame metadata: {table_path}',flush=True)


def main() -> None:
    OUTPUT.mkdir(parents=True,exist_ok=True)
    all_rows = scan()
    first,last,bad_boundaries = valid_boundaries(all_rows)
    rows = all_rows[first:last+1]
    biggest_gap,gap_start,gap_end = max(
        ((b[0]-a[0],a[0],b[0]) for a,b in zip(all_rows,all_rows[1:])),
        key=lambda item:item[0])
    print(f'Discovered {len(all_rows):,} timestamped raw PNGs; earliest readable '
          f'{rows[0][0]}; latest readable {rows[-1][0]}',flush=True)
    print(f'Experiment span {rows[-1][0]-rows[0][0]}; largest gap {biggest_gap} '
          f'({gap_start} to {gap_end})',flush=True)
    readable_cache = {rows[0][1]:True,rows[-1][1]:True}
    changes,bad_coarse = coarse_activity(rows,[row[0] for row in rows],readable_cache)
    print(f'Coarse visual intervals scored: {len(changes)}',flush=True)
    selected,bad_sampled = select_adaptive(rows,changes,readable_cache)
    print(f'Unreadable image attempts: boundaries={bad_boundaries}, '
          f'coarse={bad_coarse}, selected={bad_sampled}',flush=True)
    render_outputs(selected)


if __name__=='__main__':
    main()
