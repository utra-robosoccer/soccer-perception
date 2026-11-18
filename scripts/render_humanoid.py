#!/usr/bin/env python3
"""Pictures of humanoid_prototype_1 on the field with each virtual camera mount: a third-person
view with the body-frame and mount markers, the same view with the body drawn see-through (so
that the body frame's origin and the mount offset inside the torso show), and what the robot's
camera sees from there.

Every mount is an EXPERIMENTAL ASSUMPTION (the robot has no head); the mast in the pictures is
a marker, not a design. See docs/camera_mount.md.

Examples:
    python scripts/render_humanoid.py                         # every preset, outputs/humanoid/
    python scripts/render_humanoid.py head_mid --pitch 6 --roll -3     # a tilted body
    python scripts/render_humanoid.py --robot placeholder legacy_rig
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mounts", nargs="*", help="mount presets (default: all)")
    parser.add_argument("--robot", default="humanoid_prototype_1")
    parser.add_argument("--pose", type=float, nargs=3, default=[-1.6, -0.9, 0.45],
                        metavar=("X", "Y", "YAW"), help="footprint pose on the field")
    parser.add_argument("--roll", type=float, default=0.0, help="body roll in degrees")
    parser.add_argument("--pitch", type=float, default=0.0, help="body pitch in degrees")
    parser.add_argument("--output", default="outputs/humanoid")
    args = parser.parse_args()

    import cv2
    import numpy as np

    from robosoccer_perception.geometry import mount as mg
    from robosoccer_perception.geometry.camera import StereoCameraConfig
    from robosoccer_perception.geometry.field import FieldGeometry
    from robosoccer_perception.geometry.mount import CameraMount, available_mounts
    from robosoccer_perception.geometry.robot_description import RobotDescription
    from robosoccer_perception.simulation import mujoco_scene as ms
    from robosoccer_perception.simulation.humanoid import BodyPose

    robot = RobotDescription.load(args.robot)
    body = robot.body
    camera_cfg = StereoCameraConfig.load("zed_mini_sim").with_resolution(960, 540)
    scene_cfg = ms.load_scene_config("default", {"chase_size": [960, 540]})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    roll, pitch = math.radians(args.roll), math.radians(args.pitch)
    rows, summary = [], []
    for name in args.mounts or available_mounts():
        mount = CameraMount.load(name)
        with ms.MujocoScene(FieldGeometry.load("s_field"), camera_cfg, scene_cfg, robot=robot,
                            mount=mount) as scene:
            T_body = mg.T_footprint_body(body.standing_height, roll, pitch)
            rig = mg.rig_state_from_transform(
                mg.T_footprint_rig(mount, body, body.standing_height, roll, pitch))
            scene.teleport(args.pose, rig)
            joints = scene.humanoid.joints_for(BodyPose(T_body)) if scene.humanoid else None
            scene.set_body(T_body, joints)
            chase = scene.render_chase(show_markers=True)
            through = scene.render_chase(show_markers=True, body_alpha=0.3)
            view = scene.render(ms.LEFT_CAMERA, show_platform=True)
            height = float(scene.ground_truth().T_field_left_optical[2, 3])
        picture = np.hstack([chase, through, view])
        caption = (f"{name}{'  [EXPERIMENTAL]' if mount.experimental else ''}   camera "
                   f"{height:.3f} m above the ground   |   {mount.describe(body)}")
        bar = np.full((34, picture.shape[1], 3), 24, np.uint8)
        cv2.putText(bar, caption, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 230), 1,
                    cv2.LINE_AA)
        picture = np.vstack([bar, picture])
        path = output / f"mount_{args.robot}_{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
        rows.append(cv2.resize(picture, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))
        summary.append({"mount": name, "experimental": mount.experimental,
                        "camera_height_m": round(height, 4),
                        "position_in_body_m": [round(float(v), 4)
                                               for v in mount.position_in_body(body)],
                        "rpy_deg": [round(math.degrees(v), 3) for v in mount.rpy],
                        "picture": str(path)})
        print(f"{name:<12} camera {height:.3f} m   {path}")
    overview = output / f"mounts_{args.robot}.png"
    cv2.imwrite(str(overview), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    (output / f"mounts_{args.robot}.json").write_text(json.dumps(
        {"robot": args.robot, "body_geometry": body.to_dict(), "body_roll_deg": args.roll,
         "body_pitch_deg": args.pitch, "mounts": summary}, indent=2))
    print(f"overview: {overview}")
    print("left: third person; middle: the same with a see-through body; red / green / blue = "
          "body x / y / z, yellow = mount offset from the body origin, grey = virtual head "
          "mast, magenta = viewing direction. Right: the robot's camera.")


if __name__ == "__main__":
    main()
