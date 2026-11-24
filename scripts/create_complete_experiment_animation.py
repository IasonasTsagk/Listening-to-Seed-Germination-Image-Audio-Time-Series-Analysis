"""Create a full-scene, full-experiment presentation time-lapse from raw PNGs."""
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

RAW = Path(os.environ.get("SEED_GERMINATION_SNAPSHOT_ROOT", PROJECT / "data/raw/snapshots"))
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / 'results/figures/presentation_assets'
TABLES = PROJECT / 'results/tables/targeted_multimodal'
STEM = 'full_scene_complete_experiment'
PATTERN = re.compile(r'^recording__(\d{4})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})\.png$')
SIZE = (1280, 720)
FPS = 10
UNIQUE_COUNT = 108
FIRST_EXTRA = 4    # 5/10 s total first-frame hold
LAST_EXTRA = 7     # 8/10 s total final-frame hold


def scan() -> list[tuple[datetime, Path]]:
    if not RAW.is_dir():
        raise FileNotFoundError(RAW)
    rows = []
    for directory, folders, names in os.walk(RAW):
        folders[:] = [name for name in folders if name != '.ipynb_checkpoints']
        for name in names:
            match = PATTERN.fullmatch(name)
            if match is None:
                continue
            try:
                timestamp = datetime(*map(int, match.groups()))
            except ValueError:
                continue
            rows.append((timestamp, Path(directory) / name))
    rows.sort(key=lambda row: (row[0], str(row[1])))
    if not rows:
        raise RuntimeError('No timestamped raw snapshot PNGs found')
    return rows


def valid_boundaries(rows: list[tuple[datetime, Path]]) -> tuple[int, int, int]:
    unreadable = 0
    first = None
    last = None
    for index, (_, path) in enumerate(rows):
        if readable(path):
            first = index
            break
        unreadable += 1
    for index in range(len(rows)-1, -1, -1):
        if readable(rows[index][1]):
            last = index
            break
        unreadable += 1
    if first is None or last is None or first >= last:
        raise RuntimeError('Cannot locate readable first and last images')
    return first, last, unreadable


def choose(rows: list[tuple[datetime, Path]], count: int) -> tuple[list[tuple[datetime, Path]], int]:
    start, end = rows[0][0], rows[-1][0]
    times = [row[0] for row in rows]
    chosen = []
    used = set()
    unreadable = 0
    for index in range(count):
        if index == 0:
            candidates = [0]
        elif index == count - 1:
            candidates = [len(rows)-1]
        else:
            target = start + timedelta(seconds=(end-start).total_seconds()*index/(count-1))
            position = bisect.bisect_left(times, target)
            candidates = sorted(range(max(0, position-30), min(len(rows), position+31)),
                                key=lambda i: (abs((times[i]-target).total_seconds()), i))
        for candidate in candidates:
            timestamp, path = rows[candidate]
            if path in used:
                continue
            if not readable(path):
                unreadable += 1
                continue
            chosen.append((timestamp, path))
            used.add(path)
            break
        else:
            raise RuntimeError(f'No readable unique source near sample index {index}')
    chosen.sort()
    assert len(chosen) == count
    assert chosen[0] == rows[0] and chosen[-1] == rows[-1]
    assert all(a[0] < b[0] for a, b in zip(chosen, chosen[1:]))
    return chosen, unreadable


def preview(selected: list[tuple[datetime, Path]], frames_by_path: dict[Path, Image.Image]) -> None:
    percentages = [0, 15, 30, 50, 70, 85, 100]
    thumb_size = (320, 180)
    label_height = 54
    sheet = Image.new('RGB', (thumb_size[0]*len(percentages), thumb_size[1]+label_height), '#eeeeee')
    draw = ImageDraw.Draw(sheet)
    for column, percent in enumerate(percentages):
        timestamp, path = selected[round((len(selected)-1)*percent/100)]
        thumb = frames_by_path[path].resize(thumb_size, RESAMPLE)
        sheet.paste(thumb, (column*thumb_size[0], 0))
        draw.text((column*thumb_size[0]+7, 183), f'{percent}%  {timestamp:%Y-%m-%d}',
                  fill='#111111', font=font(17))
        draw.text((column*thumb_size[0]+7, 208), f'{timestamp:%H:%M:%S}',
                  fill='#111111', font=font(16))
    sheet.save(OUTPUT / f'{STEM}_preview.png')


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows = scan()
    first_index, last_index, unreadable_boundaries = valid_boundaries(all_rows)
    rows = all_rows[first_index:last_index+1]
    chosen, unreadable_samples = choose(rows, UNIQUE_COUNT)
    start, end = chosen[0][0], chosen[-1][0]
    biggest_gap, gap_start, gap_end = max(
        ((b[0]-a[0], a[0], b[0]) for a,b in zip(all_rows,all_rows[1:])),
        key=lambda item: item[0])
    sequence = [chosen[0]]*FIRST_EXTRA + chosen + [chosen[-1]]*LAST_EXTRA
    duration = len(sequence)/FPS
    assert 10 <= duration <= 12 and len(sequence) <= 120
    print(f'Discovered {len(all_rows):,} timestamped raw images', flush=True)
    print(f'Earliest readable: {start}; latest readable: {end}', flush=True)
    print(f'Experiment span: {end-start}; largest raw-image timestamp gap: '
          f'{biggest_gap} ({gap_start} to {gap_end})', flush=True)
    print(f'Unreadable boundary attempts: {unreadable_boundaries}; '
          f'unreadable sampled attempts: {unreadable_samples}', flush=True)

    video_path = OUTPUT / f'{STEM}.mp4'
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'avc1'), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError('H.264 MP4 writer did not open')
    source_stats = {}
    frames = []
    selected_frames = {}
    metadata = []
    try:
        for frame_index, (timestamp, path) in enumerate(sequence):
            if path not in source_stats:
                source_stats[path] = (path.stat().st_size, path.stat().st_mtime_ns)
            if path not in selected_frames:
                selected_frames[path] = render(path, timestamp, None, SIZE, 'full_scene')
            frame = selected_frames[path]
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            frames.append(frame)
            elapsed = (timestamp-start).total_seconds()
            metadata.append({'frame_index':frame_index, 'source_filename':path.name,
                             'source_timestamp':timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                             'relative_time_hours':f'{elapsed/3600:.6f}',
                             'relative_time_days':f'{elapsed/86400:.8f}'})
            if frame_index % 25 == 0 or frame_index == len(sequence)-1:
                print(f'Rendered {frame_index+1}/{len(sequence)} playback frames', flush=True)
    finally:
        writer.release()
    assert all((path.stat().st_size,path.stat().st_mtime_ns)==original
               for path,original in source_stats.items()), 'A raw source file changed'
    assert all(a[0] <= b[0] for a,b in zip(sequence,sequence[1:]))
    preview(chosen, selected_frames)

    palette_sample = Image.new('RGB', (SIZE[0], SIZE[1]*3))
    for index, source_index in enumerate((0,len(frames)//2,len(frames)-1)):
        palette_sample.paste(frames[source_index], (0,index*SIZE[1]))
    palette = palette_sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    gif_frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    gif_path = OUTPUT / f'{STEM}.gif'
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=100, loop=0, optimize=True, disposal=2)
    with Image.open(gif_path) as gif:
        assert gif.n_frames == UNIQUE_COUNT, (gif.n_frames, UNIQUE_COUNT)
        assert gif.info.get('loop') == 0
        gif_duration = sum(gif.seek(i) or gif.info['duration'] for i in range(gif.n_frames))/1000
        assert abs(gif_duration-duration) < .01
        assert gif.size == SIZE
    capture = cv2.VideoCapture(str(video_path))
    assert capture.isOpened()
    assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(sequence)
    assert round(capture.get(cv2.CAP_PROP_FPS)) == FPS
    assert (round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == SIZE
    for index in (0,len(sequence)//2,len(sequence)-1):
        capture.set(cv2.CAP_PROP_POS_FRAMES,index)
        success,_ = capture.read()
        assert success, f'MP4 frame {index} could not be decoded'
    capture.release()
    table = (TABLES if OUTPUT == PROJECT/'results/figures/presentation_assets' else OUTPUT) / 'presentation_animation_complete_experiment_frames.csv'
    with table.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['frame_index','source_filename','source_timestamp',
                                                  'relative_time_hours','relative_time_days'])
        writer.writeheader()
        writer.writerows(metadata)
    print(f'Validated: {UNIQUE_COUNT} unique images, {len(sequence)} MP4 frames, '
          f'{FPS} fps, {duration:.2f} seconds, {SIZE[0]}x{SIZE[1]}, looping GIF', flush=True)
    print(f'Frame metadata: {table}', flush=True)


if __name__ == '__main__':
    run()
