"""Ground-truth labels from the simulator: object boxes and the semantic mask.

Method: MuJoCo's segmentation rendering gives the id of the geom seen at every pixel
(`MujocoScene.render_segmentation`). Every label is read off that image, so labels are by
construction what the camera sees:

* an object's box is the tight box around the pixels that show one of its geoms. Occlusion and
  clipping at the image border need no special handling: hidden or out-of-image parts simply
  have no pixels;
* each object is also rendered on its own to get its un-occluded silhouette, which yields the
  visible fraction used by the occlusion rule;
* the semantic mask maps turf to ``field``, line paint to ``field_line`` and everything else
  (floor outside the carpet, sky, walls, and every object standing on the field: ball, goals,
  robots) to ``background``. A pixel never says "line" where a robot hides the line.

The segmentation image is rendered ``supersample`` times finer than the camera image and
reduced by coverage, because colour images are anti-aliased while id images cannot be.

Known limits: goal nets are see-through in colour images and are ignored here, so objects and
lines behind a net are labelled as visible although the net dims them; the crossbar belongs to
no detection class (it is ``background``); a post seen in two pieces around an occluder gets
one box spanning both; shadows do not change labels. See docs/synthetic_data.md.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from robosoccer_perception.datasets import schema
from robosoccer_perception.datasets.schema import LabelPolicy, ObjectAnnotation
from robosoccer_perception.geometry import transforms as tf
from robosoccer_perception.simulation.mujoco_scene import LEFT_CAMERA, MujocoScene


@dataclasses.dataclass(frozen=True)
class _Instance:
    name: str
    class_id: int
    geom_ids: np.ndarray
    attributes: dict


@dataclasses.dataclass(frozen=True)
class FrameLabels:
    semantic: np.ndarray                    # (H, W) uint8, values index SEMANTIC_CLASSES
    objects: tuple[ObjectAnnotation, ...]   # every object of a detection class, labelled or not
    width: int
    height: int

    @property
    def labeled(self) -> tuple[ObjectAnnotation, ...]:
        return tuple(obj for obj in self.objects if obj.labeled)

    def yolo_text(self) -> str:
        lines = [schema.yolo_line(obj, self.width, self.height) for obj in self.labeled]
        return "\n".join(lines) + ("\n" if lines else "")


def reduce_semantic(fine: np.ndarray, scale: int, line_threshold: float) -> np.ndarray:
    """Reduce a class-id image by ``scale`` in each direction.

    A block becomes ``field_line`` when at least ``line_threshold`` of it is line, otherwise
    ``field`` or ``background``, whichever covers more of it (background on a tie).
    """
    if scale == 1:
        return fine.astype(np.uint8)
    height, width = fine.shape[0] // scale, fine.shape[1] // scale
    blocks = fine.reshape(height, scale, width, scale)
    line = (blocks == schema.FIELD_LINE).sum(axis=(1, 3))
    field = (blocks == schema.FIELD).sum(axis=(1, 3))
    background = scale * scale - line - field
    result = np.where(field > background, schema.FIELD, schema.BACKGROUND).astype(np.uint8)
    result[line >= line_threshold * scale * scale] = schema.FIELD_LINE
    return result


def mask_box(mask: np.ndarray, scale: int) -> tuple[float, float, float, float] | None:
    """Tight box (x0, y0, x1, y1) around the True pixels, in image pixel-edge coordinates."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return None
    return (cols[0] / scale, rows[0] / scale, (cols[-1] + 1) / scale, (rows[-1] + 1) / scale)


class SceneLabeler:
    """Produces `FrameLabels` for the current state of a `MujocoScene`."""

    def __init__(self, scene: MujocoScene, policy: LabelPolicy | None = None,
                 camera_name: str = LEFT_CAMERA):
        self.scene = scene
        self.policy = policy or LabelPolicy()
        self.camera_name = camera_name
        model = scene.model
        names = [model.geom(i).name for i in range(model.ngeom)]
        # Index with (geom id + 1): entry 0 is "no geom" (sky).
        self.semantic_lut = np.full(model.ngeom + 1, schema.BACKGROUND, dtype=np.uint8)
        for gid, name in enumerate(names):
            if name == "turf":
                self.semantic_lut[gid + 1] = schema.FIELD
            elif name.startswith("line/"):
                self.semantic_lut[gid + 1] = schema.FIELD_LINE
        self.instances: list[_Instance] = [
            _Instance("ball", schema.BALL, np.array([names.index("ball")]),
                      {"radius": float(scene.scene_cfg["ball"]["radius"])})]
        for goal in scene.field.goals:
            for label in ("pos_y", "neg_y"):
                name = f"{goal.name}/post_{label}"
                self.instances.append(_Instance(
                    name, schema.GOAL_POST, np.array([names.index(name)]),
                    {"goal": goal.name, "height": goal.dims.height,
                     "diameter": goal.dims.post_diameter}))
        robot_looks = scene.scene_cfg.get("robots") or []
        for index in range(scene.num_robots):
            self.instances.append(_Instance(
                f"robot{index}", schema.ROBOT, scene.robot_geom_ids(index),
                {"appearance_index": index,
                 "height": float((robot_looks[index] or {}).get("height", 0.8))}))
        self.instance_lut = np.full(model.ngeom + 1, -1, dtype=np.int16)
        for index, instance in enumerate(self.instances):
            self.instance_lut[instance.geom_ids + 1] = index

    # ------------------------------------------------------------------ helpers
    def _may_be_in_view(self, instance: _Instance, T_camera_field: np.ndarray) -> bool:
        """Cheap conservative test: could any part of the object project into the image?"""
        model, data = self.scene.model, self.scene.data
        k = self.scene.camera_cfg.intrinsics
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        corners = []
        for gid in instance.geom_ids:
            size = model.geom_size[gid]
            half = size if model.geom_type[gid] == 6 else (        # 6 = mjGEOM_BOX
                np.array([size[0]] * 3) if size[1] == 0 else
                np.array([size[0], size[0], size[1] + size[0]]))
            rotation = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
            corners.append(data.geom_xpos[gid] + (signs * half) @ rotation.T)
        points = tf.transform_points(T_camera_field, np.concatenate(corners))
        in_front = points[:, 2] > self.scene.camera_cfg.clip_near
        if not in_front.any():
            return False
        if not in_front.all():
            return True                      # straddles the camera plane: let the render decide
        pixels, _ = k.project(points)
        return bool(pixels[:, 0].max() >= -0.5 and pixels[:, 0].min() <= k.width - 0.5
                    and pixels[:, 1].max() >= -0.5 and pixels[:, 1].min() <= k.height - 0.5)

    def _reference_point(self, instance: _Instance) -> np.ndarray:
        data = self.scene.data
        if instance.class_id == schema.ROBOT:
            return np.array(data.body(instance.name).xpos)
        return np.array(data.geom_xpos[instance.geom_ids[0]])

    # ------------------------------------------------------------------ labels
    def annotate(self) -> FrameLabels:
        policy, scale = self.policy, self.policy.supersample
        k = self.scene.camera_cfg.intrinsics
        geom_image = self.scene.render_segmentation(self.camera_name, scale)
        semantic = reduce_semantic(self.semantic_lut[geom_image + 1], scale,
                                   policy.line_coverage_threshold)
        instance_image = self.instance_lut[geom_image + 1]
        counts = np.bincount(instance_image[instance_image >= 0].ravel(),
                             minlength=len(self.instances))
        T_field_camera = self.scene.camera_pose(self.camera_name)
        T_camera_field = tf.invert_transform(T_field_camera)
        camera_xy = T_field_camera[:2, 3]
        area = float(scale * scale)

        objects = []
        for index, instance in enumerate(self.instances):
            reference = self._reference_point(instance)
            parked = reference[2] < -1.0
            common = dict(
                instance=instance.name, class_id=instance.class_id,
                distance_m=float(np.hypot(*(reference[:2] - camera_xy))),
                position_field=tuple(float(v) for v in reference),
                attributes=dict(instance.attributes))
            visible = float(counts[index]) / area
            amodal_mask = None
            if not parked and (visible > 0 or self._may_be_in_view(instance, T_camera_field)):
                amodal_mask = self.scene.render_segmentation(
                    self.camera_name, scale, only_geoms=instance.geom_ids) >= 0
            amodal = 0.0 if amodal_mask is None else float(amodal_mask.sum()) / area
            if amodal <= 0.0:
                objects.append(ObjectAnnotation(
                    labeled=False, reason="hidden" if parked else "out_of_view", bbox_xyxy=None,
                    visible_pixels=0.0, amodal_pixels=0.0, visible_fraction=0.0,
                    truncated=False, **common))
                continue
            fraction = min(1.0, visible / amodal)
            truncated = bool(amodal_mask[0].any() or amodal_mask[-1].any()
                             or amodal_mask[:, 0].any() or amodal_mask[:, -1].any())
            box = mask_box(instance_image == index, scale) if visible > 0 else None
            if box is None or fraction < policy.min_visible_fraction:
                reason = "occluded"
            elif (visible < policy.min_visible_pixels
                  or min(box[2] - box[0], box[3] - box[1]) < policy.min_box_side_px):
                reason = "too_small"
            else:
                reason = "ok"
            objects.append(ObjectAnnotation(
                labeled=reason == "ok", reason=reason, bbox_xyxy=box, visible_pixels=visible,
                amodal_pixels=amodal, visible_fraction=fraction, truncated=truncated, **common))
        return FrameLabels(semantic, tuple(objects), k.width, k.height)
