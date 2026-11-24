"""Longer, evenly paced full-experiment presentation animation from real frames."""
from __future__ import annotations

import csv
from datetime import datetime
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(os.environ.get('SEED_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / 'scripts'))
from create_presentation_animations import font, render, RESAMPLE
from create_complete_experiment_animation import scan, valid_boundaries, choose

OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / 'results/figures/presentation_assets'
TABLES = PROJECT / 'results/tables/targeted_multimodal'
STEM = 'full_scene_complete_experiment_smooth'
SIZE = (1280, 720)
FRAME_COUNT = 240
FPS = 12
PREVIEW_PERCENTAGES = (0, 15, 30, 45, 60, 75, 90, 100)


def make_preview(selected: list[tuple[datetime, Path]],
                 images: dict[int, Image.Image]) -> None:
    tile = (280, 158)
    sheet = Image.new('RGB', (tile[0] * len(PREVIEW_PERCENTAGES), tile[1] + 56), '#ededed')
    draw = ImageDraw.Draw(sheet)
    for column, percent in enumerate(PREVIEW_PERCENTAGES):
        index = round((len(selected) - 1) * percent / 100)
        stamp = selected[index][0]
        sheet.paste(images[index].resize(tile, RESAMPLE), (column * tile[0], 0))
        draw.text((column * tile[0] + 6, tile[1] + 4), f'{percent}%  {stamp:%Y-%m-%d}',
                  fill='#111111', font=font(16))
        draw.text((column * tile[0] + 6, tile[1] + 29), f'{stamp:%H:%M:%S}',
                  fill='#111111', font=font(16))
    sheet.save(OUTPUT / f'{STEM}_preview.png')


def encode(selected: list[tuple[datetime, Path]]) -> None:
    first, last = selected[0][0], selected[-1][0]
    duration = len(selected) / FPS
    assert len(selected) == FRAME_COUNT and 15 <= duration <= 20
    video_path = OUTPUT / f'{STEM}.mp4'
    gif_path = OUTPUT / f'{STEM}.gif'
    preview_indices = {round((len(selected)-1) * value / 100) for value in PREVIEW_PERCENTAGES}
    # Use one fixed palette for all frames so GIF colours do not flicker.
    palette_source = Image.new('RGB', (SIZE[0], SIZE[1] * 3))
    for slot, index in enumerate((0, len(selected)//2, len(selected)-1)):
        timestamp, path = selected[index]
        palette_source.paste(render(path, timestamp, None, SIZE, 'full_scene'),
                             (0, slot * SIZE[1]))
    palette = palette_source.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'avc1'), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError('H.264 MP4 writer did not open')
    gif_frames = []
    preview_frames = {}
    source_stats = {}
    metadata = []
    try:
        for index, (timestamp, path) in enumerate(selected):
            source_stats[path] = (path.stat().st_size, path.stat().st_mtime_ns)
            frame = render(path, timestamp, None, SIZE, 'full_scene')
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            gif_frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
            if index in preview_indices:
                preview_frames[index] = frame
            elapsed = (timestamp-first).total_seconds()
            metadata.append({'frame_index': index, 'source_filename': path.name,
                             'source_timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                             'relative_time_hours': f'{elapsed/3600:.6f}',
                             'relative_time_days': f'{elapsed/86400:.8f}',
                             'selection_note': 'start/end anchor' if index in (0, FRAME_COUNT-1)
                             else 'baseline coverage'})
            if (index+1) % 40 == 0 or index+1 == len(selected):
                print(f'Rendered {index+1}/{len(selected)} real frames', flush=True)
    finally:
        writer.release()
    assert all((path.stat().st_size, path.stat().st_mtime_ns) == old
               for path, old in source_stats.items()), 'A raw source image changed'
    make_preview(selected, preview_frames)
    # 12 fps = 83.333 ms; 80,80,90 ms in the GIF has the same total duration.
    durations = [90 if index % 3 == 2 else 80 for index in range(len(gif_frames))]
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=durations, loop=0, optimize=True, disposal=2)
    with Image.open(gif_path) as gif:
        assert gif.size == SIZE and gif.n_frames == FRAME_COUNT and gif.info.get('loop') == 0
        gif_duration = sum(gif.seek(index) or gif.info['duration']
                           for index in range(gif.n_frames)) / 1000
        assert abs(gif_duration-duration) < .01, (gif_duration, duration)
    video = cv2.VideoCapture(str(video_path))
    assert video.isOpened()
    assert round(video.get(cv2.CAP_PROP_FRAME_COUNT)) == FRAME_COUNT
    assert round(video.get(cv2.CAP_PROP_FPS)) == FPS
    assert (round(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))) == SIZE
    codec = int(video.get(cv2.CAP_PROP_FOURCC))
    assert ''.join(chr((codec >> (8*i)) & 255) for i in range(4)) == 'h264'
    for index in (0, FRAME_COUNT//2, FRAME_COUNT-1):
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        assert video.read()[0], f'Could not decode MP4 frame {index}'
    video.release()
    output_table = (TABLES if OUTPUT == PROJECT/'results/figures/presentation_assets'
                    else OUTPUT) / f'{STEM}_frames.csv'
    with output_table.open('w', newline='') as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=['frame_index','source_filename',
            'source_timestamp','relative_time_hours','relative_time_days','selection_note'])
        csv_writer.writeheader()
        csv_writer.writerows(metadata)
    print(f'Validated {FRAME_COUNT} distinct real frames, {FPS} fps, {duration:.2f}s, '
          f'{SIZE[0]}x{SIZE[1]}, H.264 MP4 and looping GIF', flush=True)
    print(f'Frame range: {first} to {last}; metadata: {output_table}', flush=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows = scan()
    first_index, last_index, unreadable_boundaries = valid_boundaries(all_rows)
    rows = all_rows[first_index:last_index+1]
    selected, unreadable_samples = choose(rows, FRAME_COUNT)
    max_gap, gap_start, gap_end = max(
        ((b[0]-a[0], a[0], b[0]) for a,b in zip(all_rows,all_rows[1:])),
        key=lambda item:item[0])
    print(f'Discovered {len(all_rows):,} timestamped raw PNGs', flush=True)
    print(f'Earliest readable {rows[0][0]}; latest readable {rows[-1][0]}; '
          f'experiment span {rows[-1][0]-rows[0][0]}', flush=True)
    print(f'Largest raw timestamp gap {max_gap} ({gap_start} to {gap_end}); '
          f'unreadable boundary attempts {unreadable_boundaries}, '
          f'unreadable selected attempts {unreadable_samples}', flush=True)
    assert selected[0] == rows[0] and selected[-1] == rows[-1]
    assert all(a[0] < b[0] for a,b in zip(selected, selected[1:]))
    encode(selected)


if __name__ == '__main__':
    main()
