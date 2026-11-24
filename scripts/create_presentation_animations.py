"""Render faithful, timestamped visual-context animations from fixed-camera PNGs."""
from __future__ import annotations

import bisect
import csv
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("SEED_GERMINATION_SNAPSHOT_ROOT", PROJECT / "data/raw/snapshots"))
TABLES = PROJECT / 'results/tables/targeted_multimodal'
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / 'results/figures/presentation_assets'
START = datetime(2026, 7, 14, 12, 0, 0)
END = datetime(2026, 7, 15, 12, 10, 0)
PATTERN = re.compile(r'^recording__(\d{4})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{1,2})\.png$')
RESAMPLE = Image.Resampling.LANCZOS


def timestamp_from_name(name: str) -> datetime | None:
    match = PATTERN.fullmatch(name)
    return datetime(*map(int, match.groups())) if match else None


def discover() -> list[tuple[datetime, Path]]:
    if not RAW.is_dir():
        raise FileNotFoundError(RAW)
    rows = []
    for directory, _, filenames in os.walk(RAW):
        if '.ipynb_checkpoints' in Path(directory).parts:
            continue
        for filename in filenames:
            stamp = timestamp_from_name(filename)
            if stamp is not None and START <= stamp <= END:
                rows.append((stamp, Path(directory) / filename))
    rows.sort(key=lambda row: (row[0], str(row[1])))
    if not rows:
        raise RuntimeError('No source images in requested period')
    return rows


def readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            if image.size != (1920, 1080):
                return False
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def select(rows: list[tuple[datetime, Path]], count: int) -> list[tuple[datetime, Path]]:
    times = [row[0] for row in rows]
    selected = []
    used = set()
    total = (END - START).total_seconds()
    for index in range(count):
        target = START + timedelta(seconds=total * index / (count - 1))
        position = bisect.bisect_left(times, target)
        choices = sorted(range(max(0, position - 12), min(len(rows), position + 13)),
                         key=lambda i: (abs((times[i] - target).total_seconds()), i))
        for candidate in choices:
            stamp, path = rows[candidate]
            if path not in used and readable(path):
                selected.append((stamp, path))
                used.add(path)
                break
        else:
            raise RuntimeError(f'No readable nearby frame for target {target}')
    selected.sort()
    assert len(selected) == count
    assert all(a[0] < b[0] for a, b in zip(selected, selected[1:]))
    return selected


def seed05_crop() -> tuple[int, int, int, int]:
    with (TABLES / 'target_seed_rois.csv').open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['seed_id'] == 'seed_05']
    if len(rows) != 1:
        raise ValueError('Expected exactly one existing seed_05 ROI')
    row = rows[0]
    left, top, right, bottom = (int(row[key]) for key in
                                ('padded_x_min', 'padded_y_min', 'padded_x_max', 'padded_y_max'))
    if not (0 <= left < right <= 1920 and 0 <= top < bottom <= 1080):
        raise ValueError('Invalid existing seed_05 ROI')
    # Add 12% around the already-padded, human-selected fixed ROI.
    dx, dy = round((right - left) * .12), round((bottom - top) * .12)
    return (max(0, left - dx), max(0, top - dy), min(1920, right + dx), min(1080, bottom + dy))


def font(size: int):
    for path in ('/System/Library/Fonts/Supplemental/Arial.ttf',
                 '/System/Library/Fonts/Helvetica.ttc'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def overlay(image: Image.Image, lines: list[str], size: int) -> Image.Image:
    image = image.convert('RGBA')
    typeface = font(size)
    draw = ImageDraw.Draw(image)
    margin = max(12, round(image.width * .012))
    line_height = size + 6
    width = max(draw.textbbox((0, 0), line, font=typeface)[2] for line in lines)
    panel = Image.new('RGBA', image.size, (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rounded_rectangle((margin, margin, margin + width + 24,
                                  margin + len(lines) * line_height + 16),
                                 radius=8, fill=(0, 0, 0, 155))
    image = Image.alpha_composite(image, panel)
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((margin + 12, margin + 8 + index * line_height), line,
                  font=typeface, fill='white')
    return image.convert('RGB')


def render(path: Path, stamp: datetime, crop: tuple[int, int, int, int] | None,
           dimensions: tuple[int, int], name: str) -> Image.Image:
    with Image.open(path) as source:
        if source.size != (1920, 1080):
            raise ValueError(f'Unexpected source size: {path}')
        image = source.convert('RGB')
    if crop is not None:
        image = image.crop(crop)
    image = image.resize(dimensions, RESAMPLE)
    if name == 'full_scene':
        lines = [stamp.strftime('%Y-%m-%d %H:%M'), 'Fixed-camera time-lapse']
        return overlay(image, lines, 26)
    lines = [stamp.strftime('%m-%d %H:%M'), 'seed_05']
    return overlay(image, lines, 20)


def make_preview(frames: list[Image.Image], name: str) -> None:
    picks = [0, round((len(frames)-1)*.25), round((len(frames)-1)*.5),
             round((len(frames)-1)*.75), len(frames)-1]
    thumb_width = 320
    thumb_height = round(frames[0].height * thumb_width / frames[0].width)
    label_height = 28
    sheet = Image.new('RGB', (5*thumb_width, thumb_height+label_height), '#ededed')
    draw = ImageDraw.Draw(sheet)
    for column, index in enumerate(picks):
        thumb = frames[index].resize((thumb_width, thumb_height), RESAMPLE)
        sheet.paste(thumb, (column*thumb_width, 0))
        draw.text((column*thumb_width+8, thumb_height+5),
                  f'{[0,25,50,75,100][column]}%', fill='#111111', font=font(16))
    preview_name = 'full_scene_animation_preview.png' if name == 'full_scene' else 'seed05_animation_preview.png'
    sheet.save(OUTPUT / preview_name)


def encode(name: str, selected: list[tuple[datetime, Path]], fps: int,
           first_extra: int, last_extra: int, crop: tuple[int, int, int, int] | None,
           dimensions: tuple[int, int], metadata: list[dict]) -> None:
    sequence = [selected[0]]*first_extra + selected + [selected[-1]]*last_extra
    mp4_name = 'full_scene_germination_progression' if name == 'full_scene' else 'seed05_germination_progression'
    mp4_path = OUTPUT / f'{mp4_name}.mp4'
    gif_path = OUTPUT / f'{mp4_name}.gif'
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*'avc1'), fps, dimensions)
    if not writer.isOpened():
        raise RuntimeError('H.264 MP4 encoder could not open')
    frames = []
    source_stats = {}
    try:
        for index, (stamp, path) in enumerate(sequence):
            if path not in source_stats:
                source_stats[path] = (path.stat().st_size, path.stat().st_mtime_ns)
            image = render(path, stamp, crop, dimensions, name)
            writer.write(cv2.cvtColor(__import__('numpy').asarray(image), cv2.COLOR_RGB2BGR))
            frames.append(image)
            metadata.append({'animation': name, 'frame_index': index,
                             'source_filename': path.name,
                             'source_timestamp': stamp.strftime('%Y-%m-%d %H:%M:%S'),
                             'relative_time_hours': f'{(stamp-START).total_seconds()/3600:.6f}'})
            if index % 20 == 0 or index == len(sequence)-1:
                print(f'{name}: rendered {index+1}/{len(sequence)} frames', flush=True)
    finally:
        writer.release()
    assert all((p.stat().st_size, p.stat().st_mtime_ns) == stat for p, stat in source_stats.items())
    assert all(a[0] <= b[0] for a, b in zip(sequence, sequence[1:]))
    make_preview([render(path, stamp, crop, dimensions, name) for stamp, path in selected], name)

    # A common palette prevents per-frame palette changes from causing flicker.
    sample = Image.new('RGB', (dimensions[0], dimensions[1] * 3))
    for index, source_index in enumerate((0, len(frames)//2, len(frames)-1)):
        sample.paste(frames[source_index], (0, index*dimensions[1]))
    palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    # GIF timing has 10 ms precision; alternate 120/130 ms to represent 8 fps.
    durations = ([120 if index % 2 == 0 else 130 for index in range(len(sequence))]
                 if fps == 8 else [round(1000/fps)] * len(sequence))
    quantized[0].save(gif_path, save_all=True, append_images=quantized[1:],
                      duration=durations, loop=0, optimize=True, disposal=2)

    with Image.open(gif_path) as gif:
        # Pillow combines identical held frames while keeping their display time.
        assert len(selected) <= gif.n_frames <= len(sequence), (gif.n_frames, len(sequence))
        assert gif.info.get('loop') == 0
        gif_duration = sum(gif.seek(i) or gif.info['duration'] for i in range(gif.n_frames)) / 1000
    capture = cv2.VideoCapture(str(mp4_path))
    assert capture.isOpened(), 'MP4 could not be reopened'
    assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(sequence)
    assert round(capture.get(cv2.CAP_PROP_FPS)) == fps
    assert (round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == dimensions
    for index in (0, len(sequence)//2, len(sequence)-1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        okay, _ = capture.read()
        assert okay, f'Cannot decode MP4 frame {index}'
    capture.release()
    duration = len(sequence)/fps
    assert abs(gif_duration-duration) < .01 and duration <= 15
    print(f'{name}: {len(sequence)} frames, {fps} fps, {duration:.2f} s, {dimensions}, '
          f'{sequence[0][0]} to {sequence[-1][0]}, GIF {gif_duration:.2f} s', flush=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = discover()
    print(f'Found {len(rows)} timestamped source PNGs in requested period', flush=True)
    roi = seed05_crop()
    print(f'seed_05 fixed crop {roi}', flush=True)
    records = []
    full = select(rows, 84)
    encode('full_scene', full, 8, 3, 5, None, (1280, 720), records)
    seed = select(rows, 56)
    # 12% additional fixed padding around the human ROI, preserving aspect ratio.
    seed_width = 720
    seed_height = round((roi[3]-roi[1])*seed_width/(roi[2]-roi[0]))
    seed_height += seed_height % 2
    encode('seed_05', seed, 10, 4, 7, roi, (seed_width, seed_height), records)
    default_output = PROJECT / 'results/figures/presentation_assets'
    table = (TABLES if OUTPUT == default_output else OUTPUT) / 'presentation_animation_frames.csv'
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['animation', 'frame_index', 'source_filename',
                                                    'source_timestamp', 'relative_time_hours'])
        writer.writeheader()
        writer.writerows(records)
    print(f'Saved frame metadata: {table}', flush=True)


if __name__ == '__main__':
    main()
