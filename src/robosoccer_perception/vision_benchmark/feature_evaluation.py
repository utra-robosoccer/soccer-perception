"""Scoring geometric feature extraction on random views of the field, without localization.

Each frame is an independent random pose and head pan; the networks (or, for the diagnostic,
the rendered labels) are run, features are extracted with the robot's nominal camera estimate,
and `FeatureScorer` compares them with the simulator. This is the direct measurement of the
perception half of Phase 3; the benchmark scenarios measure it again along trajectories.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from robosoccer_perception.datasets import randomization
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.ground_projection import EstimatedCameraCalibration
from robosoccer_perception.geometry.landmarks import build_landmark_map
from robosoccer_perception.inference import profiling
from robosoccer_perception.perception.extractor import GeometricFeatureExtractor
from robosoccer_perception.vision_benchmark.camera_truth import (
    HeadJitter, MountError, TrueCameraModel)
from robosoccer_perception.vision_benchmark.feature_metrics import (
    FeatureScorer, FeatureStatistics, FrameTruth)
from robosoccer_perception.vision_benchmark.runner import BenchmarkResources
from robosoccer_perception.vision_benchmark.sim_camera import SimulatedCamera

PAN_RANGE_DEG = 50.0
BALL = (1.0, 0.5)


def evaluate_features(resources: BenchmarkResources, frames: int = 300, seed: int = 0, *,
                      field_name: str = "s_field", source: str = "vision",
                      mount_error: MountError | None = None, jitter: HeadJitter | None = None,
                      appearance: str | None = None, frames_per_scene: int = 25
                      ) -> dict[str, Any]:
    """Feature-extraction scores over ``frames`` random views.

    ``source`` is ``vision`` (the trained networks) or ``perfect_vision`` (rendered labels).
    ``appearance`` names a randomization preset; a new scene look is then drawn every
    ``frames_per_scene`` frames and the preset's photometric effects are applied.
    """
    if source not in ("vision", "perfect_vision"):
        raise ValueError("source must be 'vision' or 'perfect_vision'")
    field = FieldGeometry.load(field_name)
    landmarks = build_landmark_map(field)
    camera_cfg = resources.camera_config()
    calibration = EstimatedCameraCalibration.from_camera_config(
        camera_cfg, uncertainty=resources.noise.calibration)
    extractor = GeometricFeatureExtractor(field, resources.extraction, resources.noise)
    scorer = FeatureScorer(field, landmarks, camera_cfg)
    root = np.random.SeedSequence([int(seed), 3003])
    pose_rng, camera_rng, scene_rng, photo_rng = (np.random.default_rng(s) for s in root.spawn(4))
    true_camera = TrueCameraModel(camera_cfg, mount_error, jitter, camera_rng)
    preset = randomization.load_preset(appearance) if appearance else None
    statistics = FeatureStatistics()
    geometry_ms: list[float] = []
    half_x, half_y = field.half_length - 0.2, field.half_width - 0.2
    done = 0
    while done < frames:
        overrides = randomization.sample_scene_overrides(scene_rng, preset, field, 0) \
            if preset else None
        with SimulatedCamera(field, camera_cfg, overrides, source == "perfect_vision",
                             preset.get("image") if preset else None) as sim:
            for _ in range(min(frames_per_scene if preset else frames, frames - done)):
                pose = np.array([pose_rng.uniform(-half_x, half_x),
                                 pose_rng.uniform(-half_y, half_y),
                                 pose_rng.uniform(-math.pi, math.pi)])
                pan = math.radians(pose_rng.uniform(-PAN_RANGE_DEG, PAN_RANGE_DEG))
                truth = true_camera.sample(pan)
                image = sim.capture(pose, truth, BALL, (), photo_rng)
                predictions = resources.vision.run(image) if source == "vision" \
                    else sim.labels_as_vision()
                started = time.perf_counter()
                features = extractor.extract(predictions, calibration, calibration.pose(pan=pan))
                geometry_ms.append((time.perf_counter() - started) * 1000.0)
                occluders = np.array([[*BALL, sim.ball_radius, 2.0 * sim.ball_radius]])
                statistics.add(scorer.score(features, FrameTruth(pose, truth, occluders)))
                done += 1
    summary = statistics.summary()
    summary["settings"] = {
        "frames": frames, "seed": seed, "field": field_name, "source": source,
        "resolution": list(resources.resolution), "appearance": appearance or "default scene",
        "mount_error": (mount_error or MountError()).describe(),
        "head_jitter_deg": {"pitch": true_camera.jitter.pitch_std_deg,
                            "roll": true_camera.jitter.roll_std_deg}}
    summary["geometry_latency_ms"] = profiling.latency_stats(geometry_ms[5:] or geometry_ms)
    return summary
