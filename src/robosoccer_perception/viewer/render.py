"""The viewer's four panels, drawn as RGB arrays with NumPy and OpenCV.

    camera         the image with the detector's boxes, labels and scores
    segmentation   the segmenter's field and line classes, with the extracted features
    map            the field from above: particles, estimate, ground truth, observations
    diagnostics    errors, confidence, effective sample size and latency over the run

Every panel is a pure function of one `ViewerFrame` (plus the run's `Timeline` for the plots
and the trails), so the window, the screenshots and the exported videos are the same picture.
No GUI toolkit is imported here.

Colour is never the only cue: boxes and junctions carry text, the estimate is a filled
triangle and the ground truth an open ring, and each panel has a legend.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping

import cv2
import numpy as np

from robosoccer_perception.datasets import schema
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.geometry.field import FieldGeometry, sample_arc
from robosoccer_perception.geometry.landmarks import LandmarkType
from robosoccer_perception.viewer.frame import (
    LANDMARK_TYPES, UNCLASSED, Timeline, ViewerFrame, landmark_label)
from robosoccer_perception.visualization.vision_overlay import (
    DETECTION_COLOURS, FIELD_COLOUR, LINE_COLOUR)

# ------------------------------------------------------------------------------- style (RGB)
SURFACE = (22, 24, 28)
BAR = (34, 37, 43)
TEXT = (232, 232, 228)
TEXT_DIM = (150, 152, 148)
GRID = (52, 55, 61)
# Two-series palette for the plots: validated for the dark surface (lightness band, chroma,
# colour-vision-deficiency separation, contrast) with the data-viz palette checker.
SERIES_A = (57, 135, 229)
SERIES_B = (217, 89, 38)
WARNING = (240, 180, 40)
GOOD = (70, 190, 120)

LINE_POINT = (255, 150, 30)
SEGMENT = (245, 245, 245)
CIRCLE = (190, 140, 255)
POST_FOOT = DETECTION_COLOURS[schema.GOAL_POST]
JUNCTION_COLOURS = {LandmarkType.L_INTERSECTION: (90, 200, 255),
                    LandmarkType.T_INTERSECTION: (190, 235, 80),
                    LandmarkType.X_INTERSECTION: (255, 110, 150)}
UNCLASSED_COLOUR = (170, 170, 170)

TURF = (26, 62, 38)
CARPET = (20, 44, 30)
PAINT = (214, 220, 214)
PARTICLE = (236, 222, 120)
ESTIMATE = (255, 140, 50)
TRUTH = (130, 205, 255)
OBSERVED = (90, 225, 235)

FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_BAR = 28
PANEL_SIZE = (960, 540)                 # width, height of a panel's content
POSITION_TARGET = 0.20                  # metres: the Phase 3 accuracy target
YAW_TARGET = 5.0                        # degrees


@dataclasses.dataclass
class Layers:
    """Which overlays are drawn. Every field is one checkbox in the window."""
    boxes: bool = True
    labels: bool = True
    robot_view: bool = True
    field_mask: bool = True
    line_mask: bool = True
    line_points: bool = True
    segments: bool = False
    junctions: bool = True
    circles: bool = True
    posts: bool = True
    particles: bool = True
    observations: bool = True
    uncertainty: bool = True
    trail: bool = True
    ground_truth: bool = True

    #: (attribute, panel, label) in the order the window lists them.
    CATALOGUE = (
        ("boxes", "Camera", "Detection boxes"),
        ("labels", "Camera", "Labels and scores"),
        ("robot_view", "Camera", "Robot view (3rd person)"),
        ("field_mask", "Segmentation", "Field mask"),
        ("line_mask", "Segmentation", "Line mask"),
        ("line_points", "Segmentation", "Line points"),
        ("segments", "Segmentation", "Fitted segments"),
        ("junctions", "Segmentation", "Junctions (L / T / X)"),
        ("circles", "Segmentation", "Centre circle"),
        ("posts", "Segmentation", "Goal-post feet"),
        ("particles", "Map", "Particles"),
        ("observations", "Map", "Observed features"),
        ("uncertainty", "Map", "Uncertainty (2 sigma)"),
        ("trail", "Map", "Trails"),
        ("ground_truth", "Map", "Ground truth"),
    )


# ------------------------------------------------------------------------------- primitives
def _text(image, text: str, x: float, y: float, colour=TEXT, scale: float = 0.45,
          thickness: int = 1) -> int:
    """Draw text with its baseline at ``y``; returns its width in pixels."""
    cv2.putText(image, text, (int(x), int(y)), FONT, scale, colour, thickness, cv2.LINE_AA)
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def _text_width(text: str, scale: float = 0.45) -> int:
    return cv2.getTextSize(text, FONT, scale, 1)[0][0]


def _tag(image, text: str, x: float, y: float, colour, scale: float = 0.45) -> None:
    """Text on a dark box whose top-left corner is near (x, y), kept inside the image."""
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x = int(np.clip(x, 0, max(0, image.shape[1] - width - 6)))
    y = int(np.clip(y, 0, max(0, image.shape[0] - height - baseline - 4)))
    cv2.rectangle(image, (x, y), (x + width + 6, y + height + baseline + 4), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 3, y + height + 2), FONT, scale, colour, 1, cv2.LINE_AA)


def _marker(image, kind: str, x: float, y: float, colour, size: int = 6) -> None:
    """A small symbol with a dark edge. kind: dot | square | cross | ring | diamond."""
    if not (math.isfinite(x) and math.isfinite(y)):
        return
    p = (int(round(x)), int(round(y)))
    if kind == "dot":
        cv2.circle(image, p, size, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, p, max(size - 1, 1), colour, -1, cv2.LINE_AA)
    elif kind == "square":
        cv2.rectangle(image, (p[0] - size, p[1] - size), (p[0] + size, p[1] + size),
                      (0, 0, 0), 4, cv2.LINE_AA)
        cv2.rectangle(image, (p[0] - size, p[1] - size), (p[0] + size, p[1] + size), colour, 2,
                      cv2.LINE_AA)
    elif kind == "cross":
        for thickness, rgb in ((4, (0, 0, 0)), (2, colour)):
            cv2.line(image, (p[0] - size, p[1] - size), (p[0] + size, p[1] + size), rgb,
                     thickness, cv2.LINE_AA)
            cv2.line(image, (p[0] - size, p[1] + size), (p[0] + size, p[1] - size), rgb,
                     thickness, cv2.LINE_AA)
    elif kind == "ring":
        cv2.circle(image, p, size, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(image, p, size, colour, 2, cv2.LINE_AA)
    elif kind == "diamond":
        points = np.array([[p[0], p[1] - size], [p[0] + size, p[1]], [p[0], p[1] + size],
                           [p[0] - size, p[1]]], np.int32)
        cv2.polylines(image, [points], True, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.polylines(image, [points], True, colour, 2, cv2.LINE_AA)
    else:
        raise ValueError(kind)


def _polyline(image, points: np.ndarray, colour, thickness: int = 2, closed: bool = False
              ) -> None:
    """A polyline that is broken wherever a point is NaN."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    finite = np.isfinite(points).all(axis=1)
    start = None
    for index in range(len(points) + 1):
        inside = index < len(points) and finite[index]
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            if index - start >= 2:
                run = np.round(points[start:index]).astype(np.int32)
                cv2.polylines(image, [run], closed and finite.all(), colour, thickness,
                              cv2.LINE_AA)
            start = None


def _legend(image, entries, x: int, y: int) -> None:
    """One row of ``(marker kind, colour, text)`` starting at (x, y)."""
    width = sum(26 + _text_width(text, 0.42) + 14 for _, _, text in entries)
    cv2.rectangle(image, (x - 6, y - 14), (x + width, y + 9), (0, 0, 0), -1)
    for kind, colour, text in entries:
        if kind == "line":
            cv2.line(image, (x, y - 3), (x + 16, y - 3), colour, 2, cv2.LINE_AA)
        elif kind == "triangle":
            cv2.fillPoly(image, [np.array([[x + 15, y - 3], [x + 2, y - 9], [x + 2, y + 3]],
                                          np.int32)], colour, cv2.LINE_AA)
        else:
            _marker(image, kind, x + 8, y - 3, colour, 4 if kind != "dot" else 3)
        x += 24
        x += _text(image, text, x, y + 1, TEXT, 0.42) + 16


def _fit(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, float, float]:
    """The image scaled into ``size`` (width, height) keeping its aspect ratio.
    Returns ``(canvas, scale, offset_x, offset_y)``."""
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    new_w, new_h = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if (new_w, new_h) == (width, height):
        return resized, scale, 0.0, 0.0
    ox, oy = (width - new_w) // 2, (height - new_h) // 2
    canvas = np.empty((height, width, 3), np.uint8)
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    for margin in (canvas[:oy], canvas[oy + new_h:], canvas[oy:oy + new_h, :ox],
                   canvas[oy:oy + new_h, ox + new_w:]):
        margin[...] = SURFACE                   # only the bands around the picture
    return canvas, scale, float(ox), float(oy)


def titled(content: np.ndarray, title: str, note: str = "", note_colour=TEXT_DIM
           ) -> np.ndarray:
    """A panel: a title bar above the content, with a short note on the right."""
    bar = np.full((TITLE_BAR, content.shape[1], 3), BAR, np.uint8)
    _text(bar, title, 10, 19, TEXT, 0.5)
    if note:
        _text(bar, note, content.shape[1] - _text_width(note, 0.45) - 10, 19, note_colour, 0.45)
    return np.vstack([bar, content])


def mosaic(panels: Mapping[str, np.ndarray], gap: int = 6) -> np.ndarray:
    """The four panels as one 2 x 2 picture (screenshots and videos)."""
    order = ("camera", "segmentation", "map", "diagnostics")
    height, width = panels[order[0]].shape[:2]
    canvas = np.full((2 * height + gap, 2 * width + gap, 3), SURFACE, np.uint8)
    for index, name in enumerate(order):
        row, column = divmod(index, 2)
        y, x = row * (height + gap), column * (width + gap)
        canvas[y:y + height, x:x + width] = panels[name]
    return canvas


# ------------------------------------------------------------------------------- the panels
class PanelRenderer:
    """Draws the panels of one run. Holds what does not change between frames: the field, the
    run's description and the map's background."""

    def __init__(self, field: FieldGeometry, meta: Mapping[str, Any],
                 size: tuple[int, int] = PANEL_SIZE):
        self.field, self.meta, self.size = field, dict(meta), (int(size[0]), int(size[1]))
        half_x, half_y = field.half_extent
        margin = 0.25
        width, height = self.size
        self._scale = min(width / (2.0 * (half_x + margin)),
                          (height - 30) / (2.0 * (half_y + margin)))
        self._centre = (width / 2.0, (height - 30) / 2.0 + 4.0)
        self._background = self._draw_field()

    # ---------------------------------------------------------------- camera
    def camera(self, frame: ViewerFrame, layers: Layers) -> np.ndarray:
        canvas, scale, ox, oy = _fit(frame.image, self.size)
        if layers.boxes:
            for detection in frame.detections:
                colour = DETECTION_COLOURS[detection.class_id]
                x0, y0, x1, y1 = (v * scale + o for v, o in
                                  zip(detection.bbox_xyxy, (ox, oy, ox, oy)))
                cv2.rectangle(canvas, (round(x0), round(y0)), (round(x1) - 1, round(y1) - 1),
                              colour, 2, cv2.LINE_AA)
                if layers.labels:
                    caption = f"{detection.class_name} {detection.confidence:.2f}"
                    _tag(canvas, caption, x0, y0 - 20 if y0 >= 20 else y1 + 2, colour)
        if layers.robot_view and frame.robot_view is not None:
            self._inset(canvas, frame.robot_view)
        counts = [sum(d.class_id == c for d in frame.detections)
                  for c in range(len(schema.DETECTION_CLASSES))]
        note = "  ".join(f"{name} {count}" for name, count in
                         zip(schema.DETECTION_CLASSES, counts))
        title = "CAMERA  |  detector"
        if self.meta.get("phase4"):
            title = (f"CAMERA  |  virtual ZED Mini on {self.meta.get('robot', '')}, mount "
                     f"{self.meta.get('mount', '')}")
        return titled(canvas, title, note)

    def _inset(self, canvas: np.ndarray, picture: np.ndarray) -> None:
        """The third-person picture of the robot in the panel's top-right corner."""
        width = max(160, int(0.34 * canvas.shape[1]))
        height = round(width * picture.shape[0] / picture.shape[1])
        small = cv2.resize(picture, (width, height), interpolation=cv2.INTER_AREA)
        x0, y0 = canvas.shape[1] - width - 8, 8
        canvas[y0:y0 + height, x0:x0 + width] = small
        cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + width, y0 + height), TEXT, 1)
        _tag(canvas, "robot, third person (markers: body axes, mount, view direction)",
             x0 + 4, y0 + height - 20, TEXT, 0.38)

    # ---------------------------------------------------------------- segmentation
    def segmentation(self, frame: ViewerFrame, layers: Layers) -> np.ndarray:
        # One multiply and one add per pixel, looked up by class: out = image * gain + tint.
        gain = np.ones(len(schema.SEMANTIC_CLASSES), np.float32)
        tint = np.zeros((len(schema.SEMANTIC_CLASSES), 3), np.float32)
        if layers.field_mask or layers.line_mask:
            gain[schema.BACKGROUND] = 0.55
        for enabled, class_id, colour, alpha in (
                (layers.field_mask, schema.FIELD, FIELD_COLOUR, 0.25),
                (layers.line_mask, schema.FIELD_LINE, LINE_COLOUR, 0.75)):
            if enabled:
                gain[class_id] = 1.0 - alpha
                tint[class_id] = alpha * np.asarray(colour, np.float32)
        mask = np.minimum(frame.mask, len(gain) - 1)
        shaded = frame.image * gain[mask][..., None] + tint[mask]
        canvas, scale, ox, oy = _fit(np.clip(shaded, 0, 255).astype(np.uint8), self.size)
        offset = np.array([ox, oy])

        def px(pixels) -> np.ndarray:
            # pixel-centre coordinates -> canvas coordinates
            return (np.asarray(pixels, dtype=float) + 0.5) * scale + offset

        if layers.segments:
            for start, end in px(frame.segment_pixels.reshape(-1, 2)).reshape(-1, 2, 2):
                _polyline(canvas, np.array([start, end]), SEGMENT, 1)
        if layers.circles:
            for outline, centre, valid in zip(frame.circle_outlines,
                                              px(frame.circle_center_pixels),
                                              frame.circle_valid):
                outline_px = px(outline)
                _polyline(canvas, outline_px, CIRCLE, 2 if valid else 1)
                if valid:
                    _marker(canvas, "cross", centre[0], centre[1], CIRCLE, 4)
                drawable = outline_px[np.isfinite(outline_px).all(axis=1)]
                inside = drawable[(drawable[:, 0] > 0) & (drawable[:, 0] < canvas.shape[1])
                                  & (drawable[:, 1] > 0) & (drawable[:, 1] < canvas.shape[0])]
                if len(inside):
                    anchor = centre if np.isfinite(centre).all() \
                        and 0 < centre[0] < canvas.shape[1] and 0 < centre[1] < canvas.shape[0] \
                        else inside[np.argmin(inside[:, 1])]
                    _tag(canvas, "circle" if valid else "circle candidate (rejected)",
                         anchor[0] + 8, anchor[1] + 6, CIRCLE, 0.42)
        if layers.line_points:
            for x, y in px(frame.line_pixels):
                _marker(canvas, "dot", x, y, LINE_POINT, 3)
        if layers.junctions:
            for (x, y), code in zip(px(frame.junction_pixels), frame.junction_codes):
                if not math.isfinite(x):
                    continue
                colour = UNCLASSED_COLOUR if code == UNCLASSED else \
                    JUNCTION_COLOURS.get(LANDMARK_TYPES[code], UNCLASSED_COLOUR)
                _marker(canvas, "square", x, y, colour, 7)
                _tag(canvas, landmark_label(int(code)), x + 10, y - 9, colour, 0.5)
        if layers.posts:
            for (x, y), valid in zip(px(frame.post_pixels), frame.post_range_valid):
                _marker(canvas, "cross" if valid else "ring", x, y, POST_FOOT, 7)
        _legend(canvas, [("dot", LINE_POINT, "line point"),
                         ("square", JUNCTION_COLOURS[LandmarkType.L_INTERSECTION], "junction"),
                         ("line", CIRCLE, "circle fit"),
                         ("cross", POST_FOOT, "post foot"),
                         ("ring", POST_FOOT, "post, bearing only")],
                12, canvas.shape[0] - 12)
        classed = int((frame.junction_codes != UNCLASSED).sum())
        circle = "circle" if frame.circle_valid.any() else \
            ("circle candidate" if len(frame.circle_valid) else "no circle")
        note = (f"{len(frame.line_points)} line points  {classed} junctions  {circle}  "
                f"{len(frame.posts)} posts")
        return titled(canvas, "SEGMENTATION  |  extracted features", note)

    # ---------------------------------------------------------------- map
    def _to_map(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        return np.column_stack([self._centre[0] + self._scale * points[:, 0],
                                self._centre[1] - self._scale * points[:, 1]])

    def _draw_field(self) -> np.ndarray:
        width, height = self.size
        field, scale = self.field, self._scale
        canvas = np.full((height, width, 3), SURFACE, np.uint8)
        half_x, half_y = field.half_extent
        corners = self._to_map([[-half_x, half_y], [half_x, -half_y]]).round().astype(int)
        cv2.rectangle(canvas, tuple(corners[0]), tuple(corners[1]), CARPET, -1)
        inner = self._to_map([[-field.half_length, field.half_width],
                              [field.half_length, -field.half_width]]).round().astype(int)
        cv2.rectangle(canvas, tuple(inner[0]), tuple(inner[1]), TURF, -1)
        for goal in field.goals:
            (x0, y0), (_, y1) = goal.posts
            back = x0 + goal.side * goal.dims.depth
            box = self._to_map([[x0, y0], [back, y0], [back, y1], [x0, y1]])
            _polyline(canvas, box, (120, 126, 120), 1)
        for segment in field.segments:
            ends = self._to_map([segment.start, segment.end]).round().astype(int)
            cv2.line(canvas, tuple(ends[0]), tuple(ends[1]), PAINT,
                     max(1, round(segment.width * scale)), cv2.LINE_AA)
        for arc in field.arcs:
            _polyline(canvas, self._to_map(sample_arc(arc, 0.03)), PAINT,
                      max(1, round(arc.width * scale)), closed=arc.is_full_circle)
        for goal in field.goals:
            for post in goal.posts:
                p = self._to_map([post])[0].round().astype(int)
                cv2.circle(canvas, tuple(p), max(3, round(0.5 * goal.dims.post_diameter * scale)
                                                 + 1), (250, 250, 250), -1, cv2.LINE_AA)
        return canvas

    def _pose_marker(self, canvas, pose, colour, filled: bool, length: float = 17.0) -> None:
        x, y = self._to_map([pose[:2]])[0]
        yaw = float(pose[2])
        forward = np.array([math.cos(yaw), -math.sin(yaw)])
        left = np.array([-forward[1], forward[0]])
        tip = np.array([x, y]) + forward * length
        base = np.array([x, y]) - forward * 0.45 * length
        triangle = np.round([tip, base + left * 0.6 * length,
                             base - left * 0.6 * length]).astype(np.int32)
        if filled:
            cv2.fillPoly(canvas, [triangle], colour, cv2.LINE_AA)
            cv2.polylines(canvas, [triangle], True, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [triangle], True, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.polylines(canvas, [triangle], True, colour, 2, cv2.LINE_AA)

    def _truth_marker(self, canvas, pose) -> None:
        x, y = self._to_map([pose[:2]])[0]
        yaw = float(pose[2])
        tip = (round(x + 26 * math.cos(yaw)), round(y - 26 * math.sin(yaw)))
        centre = (round(x), round(y))
        for thickness, colour in ((5, (0, 0, 0)), (2, TRUTH)):
            cv2.line(canvas, centre, tip, colour, thickness, cv2.LINE_AA)
            cv2.circle(canvas, centre, 11, colour, thickness, cv2.LINE_AA)

    def map(self, frame: ViewerFrame, layers: Layers, timeline: Timeline | None = None
            ) -> np.ndarray:
        canvas = self._background.copy()
        show_truth = layers.ground_truth and frame.truth is not None
        pose = frame.pose

        if layers.particles and len(frame.particles):
            points = self._to_map(frame.particles[:, :2]).round().astype(int)
            inside = (points[:, 0] >= 1) & (points[:, 0] < canvas.shape[1] - 1) \
                & (points[:, 1] >= 1) & (points[:, 1] < canvas.shape[0] - 1)
            for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)):
                canvas[points[inside, 1] + dy, points[inside, 0] + dx] = PARTICLE

        if layers.trail and timeline is not None and frame.step > 0:
            upto = slice(0, frame.step + 1)
            values = timeline.values
            if show_truth:
                # the robot was carried, not driven, across a kidnapping: break the line
                true_path = np.column_stack([values["true_x"][upto], values["true_y"][upto]])
                true_path[values["kidnapped"][upto] > 0.5] = np.nan
                _polyline(canvas, self._to_map(true_path), TRUTH, 1)
            _polyline(canvas, self._to_map(np.column_stack(
                [values["est_x"][upto], values["est_y"][upto]])), ESTIMATE, 1)

        if layers.observations:
            if len(frame.line_points):
                for x, y in self._to_map(tf.se2_transform_points(pose, frame.line_points)):
                    cv2.circle(canvas, (round(x), round(y)), 2, OBSERVED, -1, cv2.LINE_AA)
            if len(frame.landmarks):
                placed = self._to_map(tf.se2_transform_points(pose, frame.landmarks))
                for (x, y), code in zip(placed, frame.landmark_codes):
                    _marker(canvas, "diamond", x, y, OBSERVED, 6)
                    _tag(canvas, landmark_label(int(code)), x + 8, y - 18, OBSERVED, 0.42)

        centre = self._to_map([pose[:2]])[0]
        if layers.uncertainty:
            overlay = canvas.copy()
            spread = max(2.0 * frame.yaw_std_deg, 1.0)
            heading = -math.degrees(float(pose[2]))
            cv2.ellipse(overlay, (round(centre[0]), round(centre[1])), (46, 46), 0.0,
                        heading - spread, heading + spread, ESTIMATE, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0.0, canvas)
            values, vectors = np.linalg.eigh(frame.covariance[:2, :2])
            axes = 2.0 * np.sqrt(np.maximum(values, 0.0)) * self._scale
            major = vectors[:, 1]
            cv2.ellipse(canvas, (round(centre[0]), round(centre[1])),
                        (max(3, round(axes[1])), max(3, round(axes[0]))),
                        -math.degrees(math.atan2(major[1], major[0])), 0.0, 360.0, ESTIMATE, 1,
                        cv2.LINE_AA)

        # Other hypotheses the filter still holds (the mirrored pose, typically).
        for mode in frame.modes[1:4]:
            self._pose_marker(canvas, mode[:3], ESTIMATE, filled=False)
            x, y = self._to_map([mode[:2]])[0]
            _tag(canvas, f"{100.0 * mode[3]:.0f} %", x + 14, y + 8, ESTIMATE, 0.42)
        if show_truth:
            self._truth_marker(canvas, frame.truth.pose)
        self._pose_marker(canvas, pose, ESTIMATE, filled=True)
        if len(frame.modes) > 1:
            _tag(canvas, f"{100.0 * frame.modes[0][3]:.0f} %", centre[0] + 14, centre[1] + 8,
                 ESTIMATE, 0.42)

        # ---- text: the numbers, and the state in words
        lines = [(f"estimate  x {pose[0]:+.2f}  y {pose[1]:+.2f}  yaw {math.degrees(pose[2]):+6.1f} deg",
                  ESTIMATE),
                 (f"1 sigma   {frame.position_std:.3f} m   {frame.yaw_std_deg:.2f} deg", TEXT)]
        if show_truth:
            t = frame.truth
            lines += [(f"truth     x {t.pose[0]:+.2f}  y {t.pose[1]:+.2f}  yaw "
                       f"{math.degrees(t.pose[2]):+6.1f} deg", TRUTH),
                      (f"error     {t.position_error:.3f} m   {t.yaw_error_deg:.2f} deg", TEXT)]
        for index, (text, colour) in enumerate(lines):
            _tag(canvas, text, 8, 6 + 19 * index, colour, 0.42)

        if frame.ambiguous:
            state, colour = "AMBIGUOUS: the mirrored pose fits equally well", WARNING
        elif frame.converged:
            state, colour = "CONVERGED", GOOD
        else:
            state, colour = "NOT CONVERGED", WARNING
        if show_truth and frame.truth.kidnapped:
            state += "  |  robot moved (kidnapped)"
        if frame.odometry_lost:
            state += "  |  odometry lost"
        entries = [("triangle", ESTIMATE, "estimate")]
        if show_truth:
            entries.append(("ring", TRUTH, "ground truth"))
        if layers.particles:
            entries.append(("dot", PARTICLE, f"particles ({frame.num_particles})"))
        if layers.observations:
            entries += [("dot", OBSERVED, "observed line"),
                        ("diamond", OBSERVED, "observed landmark")]
        _legend(canvas, entries, 12, canvas.shape[0] - 12)
        return titled(canvas, "LOCALIZATION MAP  |  particle filter", state, colour)

    # ---------------------------------------------------------------- diagnostics
    def diagnostics(self, frame: ViewerFrame, layers: Layers, timeline: Timeline,
                    status: str = "", reveal_all: bool = True) -> np.ndarray:
        """``reveal_all`` plots everything recorded so far with a cursor at this frame (for
        scrubbing); False stops the curves at this frame (for videos)."""
        width, height = self.size
        canvas = np.full((height, width, 3), SURFACE, np.uint8)
        meta, timing = self.meta, frame.timing_ms
        steps = int(meta.get("steps", timeline.steps))
        dt = float(meta.get("dt", timeline.dt))
        realtime = frame.sim_fps * dt
        device = str(meta.get("device", ""))
        _text(canvas, f"{meta.get('scenario', '')}  |  {meta.get('title', '')}  |  seed "
                      f"{meta.get('seed', 0)}", 12, 22, TEXT, 0.52)
        _text(canvas, f"t = {frame.timestamp:6.2f} s    step {frame.step + 1} / {steps}    "
                      f"simulation {frame.sim_fps:4.1f} FPS ({realtime:.2f}x real time)",
              12, 44, TEXT, 0.47)
        _text(canvas, f"inference {frame.inference_ms:5.1f} ms   geometry "
                      f"{timing.get('geometry', 0.0):4.1f}   filter {timing.get('filter', 0.0):4.1f}"
                      f"   pipeline {timing.get('total', 0.0):5.1f} ms   render "
                      f"{timing.get('render', 0.0):4.1f} ms   {device}", 12, 64, TEXT_DIM, 0.44)

        show_truth = layers.ground_truth and frame.truth is not None
        v = timeline.values
        if not reveal_all:
            v = {name: np.where(np.arange(timeline.steps) <= frame.step, series, np.nan)
                 for name, series in v.items()}
        fusion = frame.fusion
        top = 78
        if fusion is not None:
            top = self._fusion_text(canvas, frame, show_truth, top)
        position = [("reported 1 sigma", v["position_std"], SERIES_B)]
        yaw = [("reported 1 sigma", v["yaw_std_deg"], SERIES_B)]
        if show_truth:
            position.insert(0, ("error", v["position_error"], SERIES_A))
            yaw.insert(0, ("error", v["yaw_error_deg"], SERIES_A))
        hidden = "" if show_truth else "  (error hidden: no ground truth shown)"
        charts = [
            ("Position error [m]" + hidden, position,
             POSITION_TARGET if show_truth else None, 0.3, None, False),
            ("Yaw error [deg]" + hidden, yaw, YAW_TARGET if show_truth else None, 6.0, None,
             False),
            ("Filter confidence and effective sample size [0 to 1]",
             [("confidence", v["confidence"], SERIES_A),
              ("ESS / particles", v["ess_fraction"], SERIES_B)], None, 1.0, 1.0, False),
            ("Latency per frame [ms]", [("inference (detector + segmenter)", v["inference_ms"],
                                         SERIES_A), ("whole pipeline", v["total_ms"],
                                                     SERIES_B)], None, 60.0, None, True),
        ]
        if fusion is not None:
            # Camera pitch: what the robot projects with against (display only) the truth.
            pitch = [("estimated", v["est_camera_pitch"], SERIES_B)]
            if show_truth:
                pitch.insert(0, ("true", v["true_camera_pitch"], SERIES_A))
            charts.insert(2, ("Camera pitch below the horizon [deg]"
                              + ("" if show_truth else "  (true pitch hidden)"), pitch, None,
                              0.0, None, "range"))
        bottom_margin = 22
        chart_height = (height - top - bottom_margin) // len(charts)
        left, right = 58, width - 16
        duration = max(dt * (steps - 1), dt)
        events = np.flatnonzero(v["kidnapped"] > 0.5) if show_truth else []
        for index, (title, series, target, floor, fixed_max, robust) in enumerate(charts):
            y0 = top + index * chart_height
            self._chart(canvas, (left, y0 + 18, right, y0 + chart_height - 8), title, series,
                        timeline, duration, frame.step, target, floor, fixed_max, events,
                        v["ambiguous"] if title.startswith("Filter") else None, robust)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            label = f"{fraction * duration:.0f} s"
            x = left + fraction * (right - left)
            _text(canvas, label, x - _text_width(label, 0.4) / 2, height - 6, TEXT_DIM, 0.4)
        return titled(canvas, "DIAGNOSTICS", status)

    def _chart(self, canvas, box, title: str, series, timeline: Timeline, duration: float,
               cursor: int, target, floor: float, fixed_max, events, ambiguous,
               robust: bool | str = False) -> None:
        """One strip chart. ``robust`` scales the axis to the 98th percentile, so that a
        single outlier (the first inference after loading takes over a second) does not
        flatten everything else; values above the axis are drawn at its top. ``"range"``
        fits the axis to the data's own range instead of starting at zero (for a quantity
        such as an angle that hovers around a non-zero value)."""
        x0, y0, x1, y1 = box
        seen = slice(0, max(timeline.last + 1, 1))
        peak, low = 0.0, 0.0
        collected = [values[seen][np.isfinite(values[seen])] for _, values, _ in series]
        collected = np.concatenate(collected) if collected else np.zeros(0)
        if robust == "range":
            if len(collected):
                low, peak = float(collected.min()), float(collected.max())
            centre, half = 0.5 * (low + peak), max(0.55 * (peak - low), 1.0)
            low, top = math.floor(centre - half), math.ceil(centre + half)
        else:
            if len(collected):
                peak = float(np.percentile(collected, 98.0)) * 1.25 if robust \
                    else float(collected.max())
            top = fixed_max if fixed_max is not None else _nice_ceiling(max(peak * 1.08, floor))
        time = timeline.time

        def to_xy(values: np.ndarray) -> np.ndarray:
            x = x0 + (x1 - x0) * time / duration
            y = y1 - (y1 - y0) * np.clip((values - low) / (top - low), 0.0, 1.0)
            out = np.column_stack([x, y])
            out[~np.isfinite(values)] = np.nan
            return out

        for fraction in (0.0, 0.5, 1.0):
            y = round(y1 - fraction * (y1 - y0))
            cv2.line(canvas, (x0, y), (x1, y), GRID, 1)
            tick = low + fraction * (top - low)
            label = f"{tick:.0f}" if top >= 20.0 else f"{tick:.3g}"
            _text(canvas, label, x0 - 8 - _text_width(label, 0.4), y + 4, TEXT_DIM, 0.4)
        if ambiguous is not None:
            flagged = np.flatnonzero(ambiguous[seen] > 0.5)
            for step in flagged:
                x = round(x0 + (x1 - x0) * time[step] / duration)
                cv2.line(canvas, (x, y0), (x, y0 + 5), WARNING, 2)
            if len(flagged):
                _text(canvas, "ambiguous", x1 - _text_width("ambiguous", 0.4) - 4, y0 + 18,
                      WARNING, 0.4)
        if target is not None and target < top:
            y = round(y1 - (y1 - y0) * target / top)
            for x in range(x0, x1, 12):
                cv2.line(canvas, (x, y), (min(x + 6, x1), y), TEXT_DIM, 1)
            label = f"target {target:g}"
            _text(canvas, label, x1 - _text_width(label, 0.4) - 4, y - 4, TEXT_DIM, 0.4)
        for step in events:
            x = round(x0 + (x1 - x0) * time[step] / duration)
            for y in range(y0, y1, 8):
                cv2.line(canvas, (x, y), (x, min(y + 4, y1)), WARNING, 1)
        for _, values, colour in series:
            _polyline(canvas, to_xy(values), colour, 2)
        cursor_x = round(x0 + (x1 - x0) * time[min(cursor, timeline.steps - 1)] / duration)
        cv2.line(canvas, (cursor_x, y0), (cursor_x, y1), TEXT, 1)
        # title on the left; the legend with the current values on the right
        _text(canvas, title, x0, y0 - 6, TEXT, 0.44)
        x = x1
        for name, values, colour in reversed(series):
            now = values[min(cursor, timeline.steps - 1)]
            label = f"{name} {now:.3g}" if math.isfinite(now) else name
            x -= _text_width(label, 0.42)
            _text(canvas, label, x, y0 - 6, TEXT, 0.42)
            x -= 24
            cv2.line(canvas, (x, y0 - 10), (x + 16, y0 - 10), colour, 2, cv2.LINE_AA)
            x -= 18

    def _fusion_text(self, canvas, frame: ViewerFrame, show_truth: bool, y: int) -> int:
        """The Phase 4 lines of the diagnostics panel: the robot's camera-pose estimate, the
        truth next to it (display only), and the mount. Returns where the charts may start."""
        fusion, meta = frame.fusion, self.meta

        def degrees(value: float) -> str:
            return f"{value:+5.1f}" if math.isfinite(value) else "  n/a"

        line = (f"camera (estimated)  pitch {degrees(fusion.camera_pitch_deg)}  roll "
                f"{degrees(fusion.camera_roll_deg)} deg  height {fusion.camera_height:.3f} m")
        _text(canvas, line, 12, y + 8, SERIES_B, 0.44)
        if show_truth and math.isfinite(frame.truth.camera_pitch_deg):
            t = frame.truth
            _text(canvas, f"camera (true)       pitch {degrees(t.camera_pitch_deg)}  roll "
                          f"{degrees(t.camera_roll_deg)} deg  height {t.camera_height:.3f} m"
                          f"     error  pitch {degrees(fusion.camera_pitch_deg - t.camera_pitch_deg)}"
                          f"  roll {degrees(fusion.camera_roll_deg - t.camera_roll_deg)} deg",
                  12, y + 27, SERIES_A, 0.44)
        std = f" +-{fusion.tilt_std_deg:.1f}" if math.isfinite(fusion.tilt_std_deg) else ""
        sync = ""
        if fusion.sync_method:
            sync = f", {fusion.sync_method}"
            if math.isfinite(fusion.sync_offset_ms):
                sync += f" ({abs(fusion.sync_offset_ms):.1f} ms to the nearest IMU sample)"
        body = (f"{meta.get('fusion_label', '')}: body tilt roll {degrees(fusion.body_roll_deg)}"
                f"  pitch {degrees(fusion.body_pitch_deg)}{std} deg  [{fusion.source}{sync}]")
        if show_truth and math.isfinite(frame.truth.body_pitch_deg):
            body += (f"   true roll {degrees(frame.truth.body_roll_deg)}  pitch "
                     f"{degrees(frame.truth.body_pitch_deg)}")
        _text(canvas, body, 12, y + 46, TEXT, 0.42)
        mount = str(meta.get("mount_description", ""))
        if fusion.monitor_status:
            bias = fusion.monitor_bias_deg
            mount += (f"   |   mount monitor: {fusion.monitor_status}, bias roll "
                      f"{degrees(bias[0])} pitch {degrees(bias[1])}, correcting "
                      f"{degrees(fusion.monitor_correction_deg[1])} pitch")
            colour = WARNING if fusion.monitor_status in ("bias_suspected", "correcting") \
                else TEXT_DIM
        else:
            colour = TEXT_DIM
        _text(canvas, f"mount {mount}", 12, y + 64, colour, 0.40)
        return y + 74

    # ---------------------------------------------------------------- all four
    def panels(self, frame: ViewerFrame, layers: Layers, timeline: Timeline,
               status: str = "", reveal_all: bool = True) -> dict[str, np.ndarray]:
        return {"camera": self.camera(frame, layers),
                "segmentation": self.segmentation(frame, layers),
                "map": self.map(frame, layers, timeline),
                "diagnostics": self.diagnostics(frame, layers, timeline, status, reveal_all)}


def _nice_ceiling(value: float) -> float:
    """The next 1, 2, 2.5 or 5 times a power of ten at or above ``value``."""
    if value <= 0.0 or not math.isfinite(value):
        return 1.0
    exponent = math.floor(math.log10(value))
    for mantissa in (1.0, 2.0, 2.5, 5.0, 10.0):
        if mantissa * 10.0 ** exponent >= value * (1.0 - 1e-9):
            return mantissa * 10.0 ** exponent
    return 10.0 ** (exponent + 1)
