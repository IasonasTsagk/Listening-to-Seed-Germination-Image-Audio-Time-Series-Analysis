"""Mouse-led ROI selection, safe persistence and fixed-scale visual diagnostics.

No model selection, biological labeling, registration, tracking or audio access.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .image_utils import load_image
from .targeted_visual import ROI_COLUMNS, validate_rois

COORDS = ("x_min", "y_min", "x_max", "y_max")
INTERACTIVE_COLUMNS = ["seed_id", "reference_image_filename", "reference_image_timestamp",
    *[f"manual_{c}" for c in COORDS], *[f"padded_{c}" for c in COORDS],
    "padding_pixels", "selection_status", "priority", "notes"]
FRAME_ROI_COLUMNS = ["seed_id", "frame_index", "image_filename", "image_timestamp",
    *COORDS, "roi_source", "identity_review"]
IDENTITY_REVIEWS = {"", "confirmed", "uncertain", "not_same_seed"}


def normalize_rectangle(rect, width, height):
    """Accept reverse-direction drags; round outwards and clip, upper-exclusive."""
    values = np.asarray(rect, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Rectangle needs four finite coordinates.")
    xa, ya, xb, yb = values
    x0, y0 = max(0, int(np.floor(min(xa, xb)))), max(0, int(np.floor(min(ya, yb))))
    x1, y1 = min(width, int(np.ceil(max(xa, xb)))), min(height, int(np.ceil(max(ya, yb))))
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Rectangle must intersect the image with positive width/height.")
    return x0, y0, x1, y1


def padded_rectangle(rect, width, height, padding=20):
    if not isinstance(padding, (int, np.integer)) or padding < 0:
        raise ValueError("Padding must be a nonnegative integer pixel margin.")
    x0, y0, x1, y1 = normalize_rectangle(rect, width, height)
    return max(0, x0-padding), max(0, y0-padding), min(width, x1+padding), min(height, y1+padding)


def recenter_rectangle(reference_roi, drawn_rect, width, height):
    """Mouse box supplies center only; preserve native reference crop width/height."""
    x0, y0, x1, y1 = normalize_rectangle(reference_roi, width, height)
    a, b, c, d = normalize_rectangle(drawn_rect, width, height)
    crop_w, crop_h = x1-x0, y1-y0
    new_x = min(max(0, round((a+c)/2 - crop_w/2)), width-crop_w)
    new_y = min(max(0, round((b+d)/2 - crop_h/2)), height-crop_h)
    return int(new_x), int(new_y), int(new_x+crop_w), int(new_y+crop_h)


def make_frame_rois(rois, frames, existing=None):
    """Build one explicit effective ROI per seed/frame; retain valid human edits."""
    if rois.seed_id.duplicated().any() or frames.pilot_frame_index.duplicated().any():
        raise ValueError("Duplicate seed or frame IDs are not allowed.")
    rows = []
    old = {}
    if existing is not None and not existing.empty:
        if list(existing.columns) != FRAME_ROI_COLUMNS or existing[["seed_id", "frame_index"]].duplicated().any():
            raise ValueError("Per-frame ROI table has invalid columns or duplicate seed/frame rows.")
        old = {(r.seed_id, int(r.frame_index)): r for r in existing.itertuples(index=False)}
        new_keys = {(r.seed_id, int(f.pilot_frame_index)) for r in rois.itertuples(index=False)
                    for f in frames.itertuples(index=False)}
        for key, previous in old.items():
            if key not in new_keys and (previous.roi_source == "manual_correction" or previous.identity_review):
                raise ValueError("Removing a reviewed/corrected seed or frame needs human reconciliation; original per-frame table retained.")
    for roi in rois.itertuples(index=False):
        base = tuple(int(getattr(roi,c)) for c in COORDS)
        for f in frames.itertuples(index=False):
            key = (roi.seed_id, int(f.pilot_frame_index))
            bounds, source, identity = base, "reference", ""
            if key in old:
                previous = old[key]
                if previous.image_filename != f.filename or str(previous.image_timestamp) != str(f.timestamp):
                    raise ValueError("Saved per-frame ROI timestamp/filename conflicts with inventory.")
                identity = previous.identity_review
                if previous.roi_source == "manual_correction":
                    bounds = tuple(int(getattr(previous,c)) for c in COORDS)
                    if bounds[2]-bounds[0] != base[2]-base[0] or bounds[3]-bounds[1] != base[3]-base[1]:
                        raise ValueError("Corrected ROI scale conflicts with edited reference ROI; reconcile manually.")
                    source = "manual_correction"
                elif previous.roi_source != "reference":
                    raise ValueError(f"Unknown ROI source: {previous.roi_source}")
                elif tuple(int(getattr(previous,c)) for c in COORDS) != base and identity:
                    raise ValueError("Changed reference ROI invalidates prior identity review; reconcile manually.")
            if identity not in IDENTITY_REVIEWS:
                raise ValueError(f"Unknown identity review: {identity}")
            if normalize_rectangle(bounds, int(f.width), int(f.height)) != bounds:
                raise ValueError("Per-frame ROI lies outside image bounds.")
            rows.append([roi.seed_id, int(f.pilot_frame_index), f.filename, str(f.timestamp),
                         *bounds, source, identity])
    return pd.DataFrame(rows, columns=FRAME_ROI_COLUMNS)


def selection_row(seed_id, rect, frame, padding=20, priority="", notes=""):
    manual = normalize_rectangle(rect, int(frame.width), int(frame.height))
    padded = padded_rectangle(manual, int(frame.width), int(frame.height), padding)
    return dict(zip(INTERACTIVE_COLUMNS, [seed_id, frame.filename, str(frame.timestamp),
        *manual, *padded, padding, "human_selected", priority, notes]))


def effective_rois(selections, width, height):
    """Adapt stored effective rectangles to the existing Phase A review format."""
    if list(selections.columns) == ROI_COLUMNS:
        return validate_rois(selections.replace({"": np.nan}), width, height)
    if set(INTERACTIVE_COLUMNS) - set(selections.columns):
        raise ValueError("Unrecognized ROI CSV schema.")
    rows = []
    for r in selections.to_dict("records"):
        if r["selection_status"] != "human_selected":
            raise ValueError("Selections must be explicitly human-selected.")
        manual = tuple(r[f"manual_{c}"] for c in COORDS)
        if any(float(v) != int(v) for v in manual):
            raise ValueError("Stored ROI coordinates must be integers.")
        normalized = normalize_rectangle(manual, width, height)
        if tuple(manual) != normalized:
            raise ValueError("Stored manual ROI is not normalized/clipped.")
        padding = r["padding_pixels"]
        if float(padding) != int(padding):
            raise ValueError("Padding must be integer pixels.")
        expected = padded_rectangle(manual, width, height, int(padding))
        actual = tuple(r[f"padded_{c}"] for c in COORDS)
        if actual != expected:
            raise ValueError("Padded ROI does not match manual ROI and padding.")
        rows.append([r["seed_id"], *expected, r["priority"] or np.nan, r["notes"]])
    return validate_rois(pd.DataFrame(rows, columns=ROI_COLUMNS), width, height)


def load_selections(path, frame):
    """Migrate legacy ROIs in memory only, retaining unknown reference provenance."""
    if not Path(path).exists():
        return pd.DataFrame(columns=INTERACTIVE_COLUMNS)
    table = pd.read_csv(path, keep_default_na=False)
    if list(table.columns) == ROI_COLUMNS:
        validate_rois(table.replace({"": np.nan}), int(frame.width), int(frame.height))
        rows = []
        for r in table.itertuples(index=False):
            row = selection_row(r.seed_id, (r.x_min, r.y_min, r.x_max, r.y_max), frame,
                                padding=0, priority=r.priority, notes=r.selection_notes)
            row["reference_image_filename"] = ""
            row["reference_image_timestamp"] = ""
            rows.append(row)
        return pd.DataFrame(rows, columns=INTERACTIVE_COLUMNS)
    effective_rois(table, int(frame.width), int(frame.height))
    return table[INTERACTIVE_COLUMNS].copy()


def backup_file(path):
    path = Path(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def save_selections(table, path, width, height, confirm_replace=False):
    effective_rois(table, width, height)
    if not 5 <= len(table) <= 10:
        raise ValueError("Save 5–10 human-selected seeds (prefer five clear examples).")
    path = Path(path)
    if path.exists() and not pd.read_csv(path).empty and not confirm_replace:
        raise ValueError("Explicit replacement confirmation is required for existing selections.")
    backup = backup_file(path) if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return backup


def estimate_shift(reference, image):
    """Phase correlation on high-pass grayscale; return shift and response, not tracking."""
    if reference.shape != image.shape:
        raise ValueError("Camera diagnostic requires equal image dimensions.")
    if reference.ndim == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    def prepare(a):
        a = a.astype(np.float32)
        return a - cv2.GaussianBlur(a, (0, 0), 5)
    a, b = prepare(reference), prepare(image)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    return float(dx), float(dy), float(response)


def estimate_orb_shift(reference, image):
    """Independent feature-matching fallback; reject weak/non-translational fits."""
    a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=2500)
    ka, da = detector.detectAndCompute(a, None)
    kb, db = detector.detectAndCompute(b, None)
    if da is None or db is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in pairs if m.distance < .7*n.distance]
    if len(good) < 25:
        return None
    source = np.float32([ka[m.queryIdx].pt for m in good])
    target = np.float32([kb[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                  ransacReprojThreshold=2., maxIters=2000)
    if matrix is None or inliers is None or int(inliers.sum()) < 20 or float(inliers.mean()) < .4:
        return None
    if abs(np.hypot(matrix[0,0], matrix[1,0])-1) > .005 or abs(np.arctan2(matrix[1,0], matrix[0,0])) > np.deg2rad(.5):
        return None
    return float(matrix[0,2]), float(matrix[1,2]), float(inliers.mean())


def camera_stability(frames, reference_index=0):
    ref = frames.iloc[reference_index]
    reference = load_image(Path(ref.path))
    if reference is None:
        raise ValueError(f"Cannot read {ref.path}")
    rows = []
    for f in frames.itertuples(index=False):
        image = load_image(Path(f.path), (int(ref.width), int(ref.height)))
        if image is None:
            raise ValueError(f"Cannot read {f.path}")
        dx, dy, response = estimate_shift(reference, image)
        usable = np.isfinite([dx, dy, response]).all() and response >= .15
        method = "high_pass_phase_correlation"
        if not usable:
            fallback = estimate_orb_shift(reference, image)
            if fallback is not None:
                dx, dy, response = fallback
                usable = True
                method = "ORB_RANSAC_translation"
        rows.append(dict(frame_index=f.pilot_frame_index, filename=f.filename, timestamp=f.timestamp,
                         reference_filename=ref.filename, x_drift_pixels=dx, y_drift_pixels=dy,
                         response=response, estimate_usable=bool(usable),
                         translation_applied=False, method=method,
                         interpretation="approximate global shift; lighting/growth may confound"))
    return pd.DataFrame(rows)


def stability_message(report, rois=None):
    reliable = report[report.estimate_usable.astype(str).str.lower().eq("true")]
    if reliable.empty:
        return "No reliable camera-shift estimates. Inspect fixed locations manually; no correction applied."
    dx = reliable.x_drift_pixels.abs().max()
    dy = reliable.y_drift_pixels.abs().max()
    threshold = 5.0
    if rois is not None and not rois.empty:
        threshold = min(threshold, .1 * min((rois.x_max-rois.x_min).min(), (rois.y_max-rois.y_min).min()))
    if len(reliable) < .8*len(report):
        status = "INCONCLUSIVE because too many frames have weak estimates; inspect fixed ROIs manually"
    else:
        status = "small relative to the diagnostic tolerance" if max(dx, dy) <= threshold else "potentially meaningful: inspect ROIs before review"
    return (f"Approximate max absolute drift x={dx:.2f}px, y={dy:.2f}px; {status}. "
            f"Tolerance {threshold:.1f}px. {len(report)-len(reliable)} weak estimates excluded. "
            "Fixed coordinates retained; no registration/correction applied.")


def chronological_crops(frames, roi, frame_rois=None, seed_id=None):
    by_frame = {} if frame_rois is None else {int(r.frame_index): tuple(int(getattr(r,c)) for c in COORDS)
        for r in frame_rois.loc[frame_rois.seed_id.eq(seed_id)].itertuples(index=False)}
    crops = []
    for f in frames.sort_values("timestamp").itertuples(index=False):
        current = by_frame.get(int(f.pilot_frame_index), roi)
        x0, y0, x1, y1 = map(int, current)
        image = load_image(Path(f.path), (int(frames.width.iloc[0]), int(frames.height.iloc[0])))
        if image is None:
            raise ValueError(f"Cannot read {f.path}")
        if normalize_rectangle(current, image.shape[1], image.shape[0]) != tuple(current):
            raise ValueError("Effective crop is not inside the frame.")
        crops.append((f, cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)))
    return crops


def diagnostic_font(size=16):
    for name in ["/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def labeled_crop(seed_id, frame, rgb):
    """Pad the canvas for text; never resize or distort the crop."""
    canvas = Image.new("RGB", (max(rgb.shape[1], 300), rgb.shape[0]+72), "white")
    canvas.paste(Image.fromarray(rgb), ((canvas.width-rgb.shape[1])//2, 72))
    ImageDraw.Draw(canvas).multiline_text((6, 3),
        f"{seed_id}  frame {frame.pilot_frame_index:02d}\n{frame.timestamp:%Y-%m-%d %H:%M:%S}",
        font=diagnostic_font(16), fill="black", spacing=3)
    return canvas


def save_gif(seed_id, roi, frames, destination, duration_ms=700, frame_rois=None):
    panels = [labeled_crop(seed_id, f, crop) for f, crop in chronological_crops(frames, roi, frame_rois, seed_id)]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    panels[0].save(destination, save_all=True, append_images=panels[1:], duration=duration_ms,
                   loop=0, disposal=2, optimize=False)


def save_selected_overview(selections, frame, destination):
    image = load_image(Path(frame.path))
    if image is None:
        raise ValueError(f"Cannot read {frame.path}")
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    for r in selections.to_dict("records"):
        manual = tuple(int(r[f"manual_{c}"]) for c in COORDS)
        padded = tuple(int(r[f"padded_{c}"]) for c in COORDS)
        draw.rectangle(padded, outline="cyan", width=3)
        draw.rectangle(manual, outline="red", width=3)
        draw.text((padded[0], max(0, padded[1]-24)), r["seed_id"], font=diagnostic_font(20), fill="red", stroke_width=1, stroke_fill="white")
    header = f"{frame.filename}  {frame.timestamp}  {frame.width}x{frame.height} | red: manual; cyan: padded"
    draw.rectangle((0, 0, canvas.width, 28), fill="white")
    draw.text((5, 3), header, font=diagnostic_font(17), fill="black")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def save_progression(rois, frames, destination, frame_rois=None):
    """First/middle/last, same native pixel scale for every crop of a seed."""
    panels = []
    stages = frames.iloc[[0, len(frames)//2, len(frames)-1]]
    for r in rois.itertuples(index=False):
        panels.append([labeled_crop(r.seed_id, f, crop) for f, crop in
                       chronological_crops(stages, (r.x_min, r.y_min, r.x_max, r.y_max), frame_rois, r.seed_id)])
    cell_width = max(p.width for row in panels for p in row)
    cell_height = max(p.height for row in panels for p in row)
    sheet = Image.new("RGB", (3*cell_width, len(panels)*cell_height), "#eeeeee")
    for y, row in enumerate(panels):
        for x, panel in enumerate(row):
            sheet.paste(panel, (x*cell_width, y*cell_height))
    sheet.save(destination)


def opencv_rectangle(frame):
    """Local desktop fallback. Selection coordinates map back to original pixels."""
    image = load_image(Path(frame.path))
    if image is None:
        raise ValueError(f"Cannot read {frame.path}")
    scale = min(1., 1280/image.shape[1], 800/image.shape[0])
    preview = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
    name = "Draw rectangle; Enter/Space accepts; C cancels"
    try:
        x, y, w, h = cv2.selectROI(name, preview, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyAllWindows()
    if not w or not h:
        return None
    sx, sy = image.shape[1]/preview.shape[1], image.shape[0]/preview.shape[0]
    return normalize_rectangle((x*sx, y*sy, (x+w)*sx, (y+h)*sy), int(frame.width), int(frame.height))


class SeedROIEditor:
    """Small ipywidgets UI; RectangleSelector when widget backend is available.

    Keep an instance alive in the notebook so Matplotlib retains the selector.
    """
    def __init__(self, root, frames):
        import ipywidgets as W
        import matplotlib.pyplot as plt
        from IPython.display import display

        self.root, self.frames = Path(root), frames
        self.path = self.root / "results/tables/targeted_multimodal/target_seed_rois.csv"
        self.reference = W.Dropdown(options=[(f"{f.pilot_frame_index:02d}: {f.timestamp} — {f.filename}", f.pilot_frame_index)
                                           for f in frames.itertuples(index=False)],
                                    value=len(frames)//2, description="Reference:", layout=W.Layout(width="95%"))
        self.table = load_selections(self.path, frames.iloc[self.reference.value])
        self.loaded_bytes = self.path.read_bytes() if self.path.exists() else None
        self.history, self.pending = [], None
        self.seed_id = W.Dropdown(options=[f"seed_{i:02d}" for i in range(1,11)], description="Seed ID:")
        self.padding = W.IntSlider(value=20, min=0, max=100, description="Padding px:")
        self.priority = W.Dropdown(options=["", "high", "medium", "exclude"], description="Priority:")
        self.notes = W.Text(value="", description="Notes:", layout=W.Layout(width="80%"))
        self.selected = W.Dropdown(options=[("None", "")], description="Saved in UI:")
        self.confirm = W.Checkbox(description="I confirm replacing the existing ROI CSV (backup will be made)", indent=False)
        self.reset_confirm = W.Checkbox(description="Confirm reset all in-memory rectangles", indent=False)
        self.output = W.Output()
        self.plot_output = W.Output()
        self.info = W.HTML()
        buttons = {}
        for label, callback in [("Add rectangle", self.add), ("Update selected", self.update_selected),
                ("Delete selected", self.delete), ("Undo", self.undo), ("Reset all", self.reset),
                ("OpenCV draw window", self.cv_draw), ("Save selections", self.save),
                ("Generate PNGs + GIFs", self.generate)]:
            button = W.Button(description=label, layout=W.Layout(width="170px"))
            button.on_click(self._handler(callback))
            buttons[label] = button
        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=(14, 8))
        backend = plt.get_backend().lower()
        self.widget_backend = "ipympl" in backend or "widget" in backend
        self.selector = None
        self.artists = []
        self.selected.observe(self.edit_fields, names="value")
        self.reference.observe(self.change_reference, names="value")
        self.ui = W.VBox([self.reference, self.info, self.plot_output,
            W.HBox([self.seed_id, self.padding, self.priority]), self.notes,
            W.HBox([buttons["Add rectangle"], buttons["OpenCV draw window"]]),
            W.HBox([self.selected, buttons["Update selected"], buttons["Delete selected"], buttons["Undo"]]),
            W.HBox([self.reset_confirm, buttons["Reset all"]]), self.confirm,
            W.HBox([buttons["Save selections"], buttons["Generate PNGs + GIFs"]]), self.output])
        self.draw_reference()
        self.refresh()
        display(self.ui)
        self.say("Loaded existing rectangles in memory. Nothing is saved until Save selections.")
        if not self.widget_backend:
            self.say("Static backend: use OpenCV draw window (local desktop). No coordinate typing needed.")

    def _handler(self, fn):
        def wrapped(_):
            try:
                fn()
            except Exception as error:
                self.say(f"{type(error).__name__}: {error}")
        return wrapped

    def say(self, message):
        with self.output:
            print(message)

    @property
    def frame(self):
        return self.frames.iloc[self.reference.value]

    def draw_reference(self):
        from IPython.display import display, clear_output
        from matplotlib.widgets import RectangleSelector
        image = load_image(Path(self.frame.path))
        if image is None:
            raise ValueError(f"Cannot read {self.frame.path}")
        if self.selector is not None:
            self.selector.disconnect_events()
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.ax.set_title(f"{self.frame.filename}\n{self.frame.timestamp} — {self.frame.width}×{self.frame.height}")
        self.ax.set_xlim(0, int(self.frame.width))
        self.ax.set_ylim(int(self.frame.height), 0)
        self.selector = RectangleSelector(self.ax, self.on_draw, useblit=False, button=[1],
                                          minspanx=2, minspany=2, spancoords="data", interactive=True)
        self.artists = []
        self.info.value = "<b>Drag around a seed body, then click Add rectangle.</b> Red=manual; cyan=padded. Toolbar pan/zoom must be off when drawing."
        with self.plot_output:
            clear_output(wait=True)
            display(self.fig.canvas if self.widget_backend else self.fig)

    def change_reference(self, _):
        self.pending = None
        self.draw_reference()
        self.refresh()

    def edit_fields(self, change):
        if change["new"]:
            row = self.table.loc[self.table.seed_id.eq(change["new"])].iloc[0]
            self.padding.value = int(row.padding_pixels)
            self.priority.value = row.priority
            self.notes.value = row.notes

    def on_draw(self, press, release):
        if None not in (press.xdata, press.ydata, release.xdata, release.ydata):
            self.pending = normalize_rectangle((press.xdata, press.ydata, release.xdata, release.ydata),
                                              int(self.frame.width), int(self.frame.height))
            self.say("Rectangle ready. Choose ID/padding and Add, or Update selected.")

    def cv_draw(self):
        self.say("A separate OpenCV window opens. Drag, then Enter/Space to accept; C cancels.")
        rect = opencv_rectangle(self.frame)
        if rect is not None:
            self.pending = rect
            self.say("Rectangle ready: click Add rectangle or Update selected.")

    def snapshot(self):
        self.history.append(self.table.copy(deep=True))

    def add(self):
        if self.pending is None:
            raise ValueError("Draw a rectangle first.")
        if self.seed_id.value in self.table.seed_id.values:
            raise ValueError("ID already exists. Use Update selected or choose a free ID.")
        row = selection_row(self.seed_id.value, self.pending, self.frame, self.padding.value,
                            self.priority.value, self.notes.value)
        self.snapshot()
        self.table = pd.concat([self.table, pd.DataFrame([row])], ignore_index=True)
        self.pending = None
        self.notes.value, self.priority.value = "", ""
        self.refresh()

    def update_selected(self):
        if not self.selected.value:
            raise ValueError("Choose a selected seed to update.")
        old = self.table.loc[self.table.seed_id.eq(self.selected.value)].iloc[0]
        rect = self.pending or tuple(old[f"manual_{c}"] for c in COORDS)
        row = selection_row(self.selected.value, rect, self.frame, self.padding.value,
                            self.priority.value, self.notes.value)
        self.snapshot()
        self.table.loc[self.table.seed_id.eq(self.selected.value), INTERACTIVE_COLUMNS] = list(row.values())
        self.pending = None
        self.refresh()

    def delete(self):
        if self.selected.value:
            self.snapshot()
            self.table = self.table.loc[~self.table.seed_id.eq(self.selected.value)].reset_index(drop=True)
            self.refresh()

    def undo(self):
        if self.history:
            self.table = self.history.pop()
            self.pending = None
            self.refresh()

    def reset(self):
        if not self.reset_confirm.value:
            raise ValueError("Check Confirm reset first. This only resets memory until you save.")
        self.snapshot()
        self.table = pd.DataFrame(columns=INTERACTIVE_COLUMNS)
        self.pending = None
        self.reset_confirm.value = False
        self.refresh()

    def refresh(self):
        from matplotlib.patches import Rectangle
        from IPython.display import display, clear_output
        for artist in self.artists:
            artist.remove()
        self.artists = []
        for r in self.table.to_dict("records"):
            for prefix, color in [("manual", "red"), ("padded", "cyan")]:
                x0, y0, x1, y1 = [r[f"{prefix}_{c}"] for c in COORDS]
                patch = Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor=color, linewidth=1.5)
                self.ax.add_patch(patch)
                self.artists.append(patch)
            self.artists.append(self.ax.text(r["padded_x_min"], r["padded_y_min"], r["seed_id"], color="red",
                                             bbox=dict(facecolor="white", alpha=.8, edgecolor="none")))
        ids = self.table.seed_id.tolist()
        self.selected.options = [("None", ""), *[(s,s) for s in ids]]
        free = [s for s in self.seed_id.options if s not in ids]
        if free:
            self.seed_id.value = free[0]
        self.fig.canvas.draw_idle()
        if not self.widget_backend:
            with self.plot_output:
                clear_output(wait=True)
                display(self.fig)
        self.say(f"{len(ids)} selected in memory. Save 5–10 when ready.")

    def save(self):
        current = self.path.read_bytes() if self.path.exists() else None
        if current != self.loaded_bytes:
            raise ValueError("ROI file changed outside this editor. Rerun the editor cell to load it before saving.")
        backup = save_selections(self.table, self.path, int(self.frame.width), int(self.frame.height),
                                 confirm_replace=self.confirm.value)
        self.loaded_bytes = self.path.read_bytes()
        self.confirm.value = False
        save_selected_overview(self.table, self.frame,
            self.root / "results/figures/targeted_multimodal/selected_seed_overview.png")
        self.say(f"Saved {self.path}. Previous file backup: {backup}. Now Generate PNGs + GIFs.")

    def generate(self):
        if self.path.read_bytes() != self.loaded_bytes:
            raise ValueError("ROI CSV changed. Reload the editor before generating.")
        saved = load_selections(self.path, self.frame)
        try:
            pd.testing.assert_frame_equal(saved, self.table, check_dtype=False)
        except AssertionError:
            raise ValueError("Unsaved changes: Save selections before generating.")
        from scripts.prepare_target_seed_pilot import main
        with self.output:
            main(root=self.root)


class FrameROICorrectionEditor:
    """Human-only per-frame recentering, with explicit identity review."""
    def __init__(self, root, frames):
        import ipywidgets as W
        import matplotlib.pyplot as plt
        from IPython.display import display

        self.root, self.frames = Path(root), frames
        tables = self.root / "results/tables/targeted_multimodal"
        self.path = tables / "target_seed_frame_rois.csv"
        self.rois = effective_rois(pd.read_csv(tables / "target_seed_rois.csv", keep_default_na=False),
                                   int(frames.width.iloc[0]), int(frames.height.iloc[0]))
        if self.rois.empty:
            raise ValueError("Select and save reference ROIs first.")
        saved = pd.read_csv(self.path, keep_default_na=False) if self.path.exists() else None
        self.table = make_frame_rois(self.rois, frames, saved)
        self.loaded_bytes = self.path.read_bytes() if self.path.exists() else None
        self.history, self.pending = [], None
        self.seed_id = W.Dropdown(options=self.rois.seed_id.tolist(), description="Seed:")
        self.frame_index = W.Dropdown(options=[(f"{f.pilot_frame_index:02d}: {f.timestamp}", int(f.pilot_frame_index))
                                                for f in frames.itertuples(index=False)], description="Frame:")
        self.confirm = W.Checkbox(description="I confirm replacing the existing per-frame CSV (backup will be made)", indent=False)
        self.output, self.plot_output = W.Output(), W.Output()
        buttons = {}
        for label, fn in [("Apply recentering", self.apply), ("Restore reference", self.restore),
                          ("Identity confirmed", lambda: self.set_identity("confirmed")),
                          ("Identity uncertain", lambda: self.set_identity("uncertain")),
                          ("Not same seed", lambda: self.set_identity("not_same_seed")),
                          ("Undo", self.undo), ("OpenCV draw window", self.cv_draw),
                          ("Save corrections", self.save), ("Regenerate timelines", self.generate)]:
            button = W.Button(description=label, layout=W.Layout(width="185px"))
            button.on_click(self._handler(fn))
            buttons[label] = button
        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
        backend = plt.get_backend().lower()
        self.widget_backend = "ipympl" in backend or "widget" in backend
        self.selector = None
        self.seed_id.observe(self.change_view, names="value")
        self.frame_index.observe(self.change_view, names="value")
        self.ui = W.VBox([W.HTML("<b>Inspect each seed through all 25 frames.</b> Select a seed/frame; the view zooms near its current ROI. Drag over the moved seed, then Apply recentering. A drawn rectangle sets the new center, while crop width/height stay unchanged. Reference-only rows do not assert biological identity."),
            W.HBox([self.seed_id, self.frame_index]), self.plot_output,
            W.HBox([buttons["Apply recentering"], buttons["OpenCV draw window"], buttons["Restore reference"], buttons["Undo"]]),
            W.HBox([buttons["Identity confirmed"], buttons["Identity uncertain"], buttons["Not same seed"]]),
            self.confirm, W.HBox([buttons["Save corrections"], buttons["Regenerate timelines"]]), self.output])
        self.draw_view()
        display(self.ui)
        self.say("No correction is saved until Save corrections. Use OpenCV draw window if the plot is static.")

    def _handler(self, fn):
        def wrapped(_):
            try:
                fn()
            except Exception as error:
                self.say(f"{type(error).__name__}: {error}")
        return wrapped

    def say(self, message):
        with self.output:
            print(message)

    @property
    def frame(self):
        return self.frames.loc[self.frames.pilot_frame_index.eq(self.frame_index.value)].iloc[0]

    @property
    def mask(self):
        return self.table.seed_id.eq(self.seed_id.value) & self.table.frame_index.eq(self.frame_index.value)

    def draw_view(self):
        from IPython.display import display, clear_output
        from matplotlib.widgets import RectangleSelector
        from matplotlib.patches import Rectangle
        if self.selector is not None:
            self.selector.disconnect_events()
        image = load_image(Path(self.frame.path))
        if image is None:
            raise ValueError(f"Cannot read {self.frame.path}")
        row = self.table.loc[self.mask].iloc[0]
        x0, y0, x1, y1 = [int(row[c]) for c in COORDS]
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                            fill=False, edgecolor="red", linewidth=2))
        self.ax.set_title(f"{self.seed_id.value} | frame {self.frame_index.value:02d} | {self.frame.timestamp}\n"
                          f"{self.frame.filename} | {self.frame.width}×{self.frame.height} | {row.roi_source} | identity: {row.identity_review or 'unreviewed'}")
        margin = max(100, 2*max(x1-x0, y1-y0))
        self.ax.set_xlim(max(0, x0-margin), min(int(self.frame.width), x1+margin))
        self.ax.set_ylim(min(int(self.frame.height), y1+margin), max(0, y0-margin))
        self.selector = RectangleSelector(self.ax, self.on_draw, useblit=False, button=[1],
                                          minspanx=2, minspany=2, spancoords="data", interactive=True)
        with self.plot_output:
            clear_output(wait=True)
            display(self.fig.canvas if self.widget_backend else self.fig)

    def change_view(self, _):
        self.pending = None
        self.draw_view()

    def on_draw(self, press, release):
        if None not in (press.xdata, press.ydata, release.xdata, release.ydata):
            self.pending = normalize_rectangle((press.xdata, press.ydata, release.xdata, release.ydata),
                                                int(self.frame.width), int(self.frame.height))
            self.say("New center ready. Click Apply recentering; nothing saved yet.")

    def cv_draw(self):
        self.say("A separate full-frame OpenCV window opens. Drag over the seed; Enter/Space accepts, C cancels.")
        self.pending = opencv_rectangle(self.frame)
        if self.pending is not None:
            self.say("New center ready. Click Apply recentering.")

    def snapshot(self):
        self.history.append(self.table.copy(deep=True))

    def apply(self):
        if self.pending is None:
            raise ValueError("Draw a new rectangle first.")
        base = self.rois.loc[self.rois.seed_id.eq(self.seed_id.value)].iloc[0]
        reference = tuple(int(base[c]) for c in COORDS)
        corrected = recenter_rectangle(reference, self.pending, int(self.frame.width), int(self.frame.height))
        self.snapshot()
        self.table.loc[self.mask, list(COORDS)] = corrected
        self.table.loc[self.mask, ["roi_source", "identity_review"]] = ["manual_correction", ""]
        self.pending = None
        self.draw_view()
        self.say("Correction in memory. Inspect identity and save explicitly.")

    def restore(self):
        base = self.rois.loc[self.rois.seed_id.eq(self.seed_id.value)].iloc[0]
        self.snapshot()
        self.table.loc[self.mask, list(COORDS)] = [int(base[c]) for c in COORDS]
        self.table.loc[self.mask, ["roi_source", "identity_review"]] = ["reference", ""]
        self.draw_view()

    def set_identity(self, status):
        self.snapshot()
        self.table.loc[self.mask, "identity_review"] = status
        self.draw_view()

    def undo(self):
        if self.history:
            self.table = self.history.pop()
            self.draw_view()

    def save(self):
        current = self.path.read_bytes() if self.path.exists() else None
        if current != self.loaded_bytes:
            raise ValueError("Per-frame ROI file changed externally. Rerun this editor cell before saving.")
        make_frame_rois(self.rois, self.frames, self.table)
        if self.path.exists() and not pd.read_csv(self.path).empty and not self.confirm.value:
            raise ValueError("Tick replacement confirmation before saving existing per-frame ROIs.")
        backup = backup_file(self.path) if self.path.exists() else None
        self.table.to_csv(self.path, index=False)
        self.loaded_bytes = self.path.read_bytes()
        self.confirm.value = False
        self.say(f"Saved per-frame ROIs. Previous file backup: {backup}. Click Regenerate timelines.")

    def generate(self):
        saved = pd.read_csv(self.path, keep_default_na=False)
        try:
            pd.testing.assert_frame_equal(saved, self.table, check_dtype=False)
        except AssertionError:
            raise ValueError("Unsaved changes: Save corrections before regenerating.")
        from scripts.prepare_target_seed_pilot import main
        with self.output:
            main(root=self.root)
