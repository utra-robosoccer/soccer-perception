#!/usr/bin/env python3
"""Render a field with MuJoCo: a top-down overview and a stereo pair from the camera platform.

Examples:
    python scripts/render_field.py                                   # S field, default pose
    python scripts/render_field.py --field m_field --pose -4 -2 30
    python scripts/render_field.py --pose -2 -1 20 --pitch 35 --height 1.2 --overlay
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from robosoccer_perception.geometry.camera import StereoCameraConfig
from robosoccer_perception.geometry.field import FieldGeometry
from robosoccer_perception.geometry.landmarks import build_landmark_map
from robosoccer_perception.simulation.ground_truth import static_ground_truth
from robosoccer_perception.simulation.mujoco_scene import MujocoScene
from robosoccer_perception.simulation.stereo_camera import VirtualStereoCamera, save_stereo_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field", default="s_field", help="field name or YAML path")
    parser.add_argument("--camera", default="zed_mini_sim", help="camera name or YAML path")
    parser.add_argument("--pose", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
                        default=[-2.0, -1.0, 20.0], help="platform pose in the field frame")
    parser.add_argument("--height", type=float, help="camera height in metres")
    parser.add_argument("--pitch", type=float, help="camera pitch in degrees, positive = down")
    parser.add_argument("--pan", type=float, help="camera pan in degrees, positive = left")
    parser.add_argument("--ball", type=float, nargs=2, metavar=("X", "Y"), help="ball position")
    parser.add_argument("--overlay", action="store_true",
                        help="also save the left image with projected landmarks drawn on it")
    parser.add_argument("--output", default="outputs/render", help="output directory")
    args = parser.parse_args()

    import matplotlib.image as mpimg

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    field = FieldGeometry.load(args.field)
    landmarks = build_landmark_map(field)
    pose = [args.pose[0], args.pose[1], math.radians(args.pose[2])]
    camera_cfg = StereoCameraConfig.load(args.camera)
    with MujocoScene(field, camera_cfg, scenario_id="render_field") as scene:
        scene.teleport(pose)
        scene.set_rig(height=args.height,
                      pitch=None if args.pitch is None else math.radians(args.pitch),
                      pan=None if args.pan is None else math.radians(args.pan))
        if args.ball:
            scene.set_ball(args.ball)
        overview = output / f"{field.name}_overview.png"
        mpimg.imsave(overview, scene.render_overview())
        frame = VirtualStereoCamera(scene).capture()
        paths = save_stereo_frame(frame, output, stem=f"{field.name}_stereo")
        (output / f"{field.name}_static_ground_truth.json").write_text(
            json.dumps(static_ground_truth(field, landmarks), indent=2))
        written = [overview, *paths.values()]
        if args.overlay:
            from robosoccer_perception.visualization.camera_overlay import plot_camera_overlay
            figure = plot_camera_overlay(frame.left, scene.camera_model, landmarks, pose,
                                         scene.rig, title="Left image with projected landmarks")
            overlay = output / f"{field.name}_left_overlay.png"
            figure.savefig(overlay)
            written.append(overlay)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
