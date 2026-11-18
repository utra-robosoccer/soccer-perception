#!/usr/bin/env python3
"""Import UTRA's humanoid_prototype_1 from the official soccer-software model into
``assets/robots/humanoid_prototype_1/`` as a kinematic, visual-only body.

The source directory is the model's folder in a soccer-software checkout (the one holding
``mjcf/robot.xml`` and ``meshes/``). It is only READ: nothing is written, cached or built
there. See docs/humanoid_integration.md for what is taken over and what is left out.

Examples:
    python scripts/import_humanoid_model.py \\
        --source ../../soccer-software/src/humanoid_description/robots/humanoid_prototype_1
    python scripts/import_humanoid_model.py --check      # compare the bundle with its manifest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="official model directory (holds mjcf/robot.xml)")
    parser.add_argument("--robot", default="humanoid_prototype_1",
                        help="robot name: configs/robots/<name>.yaml and assets/robots/<name>/")
    parser.add_argument("--check", action="store_true",
                        help="do not import; report what the installed bundle contains")
    args = parser.parse_args()

    from robosoccer_perception import config
    from robosoccer_perception.simulation import humanoid

    destination = config.assets_dir() / "robots" / args.robot
    if args.check:
        from robosoccer_perception.geometry.robot_description import RobotDescription

        description = RobotDescription.load(args.robot)
        manifest = humanoid.load_manifest(description)
        measured = humanoid.measure_body_geometry(description)
        print(json.dumps({"bundle": str(destination), "source": manifest["source"],
                          "triangles": manifest["triangles"],
                          "stored_geometry": description.body.to_dict(),
                          "measured_now": measured.to_dict()}, indent=2))
        return 0
    if not args.source:
        parser.error("--source is required (or use --check)")
    stance = config.load_config("robots", args.robot).get("stance")
    manifest = humanoid.import_model(Path(args.source), destination, args.robot, stance)
    geometry = manifest["body_geometry"]
    print(f"imported {args.robot} into {destination}")
    print(f"  source commit      {manifest['source']['git_commit']}"
          f"{' (modified)' if manifest['source']['git_dirty'] else ''}")
    print(f"  visual meshes      {len(manifest['meshes'])} ({manifest['triangles']} triangles)")
    print(f"  standing height    {geometry['standing_height_m']:.4f} m (pelvis above the ground)")
    print(f"  torso top          {geometry['torso_top_m']} m in the pelvis frame "
          f"({geometry['standing_height_m'] + geometry['torso_top_m'][2]:.4f} m above the ground)")
    if manifest["home_stance"] and stance and {
            k: round(v, 6) for k, v in manifest["home_stance"].items()} != {
            k: round(float(v), 6) for k, v in stance.items() if float(v) != 0.0}:
        print("  WARNING: the stance in configs/robots differs from the model's 'home' keyframe:"
              f"\n    model  {manifest['home_stance']}\n    config {stance}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
