#!/usr/bin/env python3
"""Interactive viewer for the vision-localization pipeline: camera, segmentation, map and
diagnostics of one scenario, frame by frame. See docs/viewer.md.

Examples:
    python scripts/visualize_pipeline.py                      # the window, h05_walk_rectangle:
                                                              # the humanoid walking a rectangle
    python scripts/visualize_pipeline.py h02 --fusion fixed   # sway, with the Phase 3 camera model
    python scripts/visualize_pipeline.py h05 --mount head_high --motion walking_rough
    python scripts/visualize_pipeline.py h05 --robot placeholder --mount legacy_rig
    python scripts/visualize_pipeline.py v13a_kidnapped --seed 2 --autoplay     # a Phase 3 scenario
    python scripts/visualize_pipeline.py --replay outputs/viewer/recordings/<file>.npz
    python scripts/visualize_pipeline.py --list

    # no window: record a run and export a video of the four panels
    python scripts/visualize_pipeline.py v04_rectangle --headless --record --video

Keys in the window: Space run / pause, Right / Left step, R reset, S screenshot,
G ground truth, Home / End first / last frame.

Everything written goes to ``outputs/viewer/`` (recordings/, videos/, screenshots/), which Git
ignores. The window needs the ``viewer`` extra (PySide6); ``--headless`` does not.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", nargs="?", default="h05_walk_rectangle",
                        help="scenario id or a prefix of one (see --list)")
    body = parser.add_argument_group(
        "Phase 4 scenarios (h...)", "the body, its virtual camera mount, its motion and the "
        "camera-pose estimator; each defaults to the scenario's own")
    body.add_argument("--robot", help="humanoid_prototype_1 or placeholder (configs/robots/)")
    body.add_argument("--mount", help="camera mount preset (configs/mounts/); experimental")
    body.add_argument("--motion", help="body motion profile (configs/motion/)")
    body.add_argument("--fusion", default="imu_uncertainty",
                      help="camera pose: fixed, imu, imu_uncertainty, imu_selfcal, imu_latest")
    body.add_argument("--no-robot-view", action="store_true",
                      help="do not render the third-person picture of the robot")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", default="1",
                        help="simulated seconds per second: 0.25, 0.5, 1, 2, 4 or max")
    parser.add_argument("--autoplay", action="store_true", help="start running at once")
    parser.add_argument("--replay", metavar="FILE", help="open a recording instead of simulating")
    parser.add_argument("--no-ground-truth", action="store_true",
                        help="start with the ground-truth display off")
    parser.add_argument("--particles", type=int, default=600,
                        help="how many of the filter's particles are drawn (default 600)")
    parser.add_argument("--max-steps", type=int, help="stop the run after this many frames")
    parser.add_argument("--settings", default="vision_localization",
                        help="benchmark settings naming the checkpoints (configs/evaluation/)")
    parser.add_argument("--device", help="cuda, cpu, ... (default: the settings')")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="camera image size (default: the settings' 640x360)")
    group = parser.add_argument_group("without a window")
    group.add_argument("--headless", action="store_true",
                       help="run the scenario to its end without opening a window")
    group.add_argument("--record", nargs="?", const="", metavar="FILE",
                       help="save a recording (default name under outputs/viewer/recordings/)")
    group.add_argument("--video", nargs="?", const="", metavar="FILE",
                       help="export a video (default name under outputs/viewer/videos/)")
    group.add_argument("--screenshots", type=int, default=0, metavar="N",
                       help="save N evenly spaced four-panel pictures")
    args = parser.parse_args()

    from robosoccer_perception.viewer import catalog
    from robosoccer_perception.viewer.humanoid_session import SessionOptions

    if args.list:
        for scenario in catalog.all_scenarios():
            print(f"{scenario.id:<32}{scenario.title}")
        return 0
    try:
        scenario = catalog.find_scenario(args.scenario)
        options = SessionOptions(robot=args.robot, mount=args.mount, motion=args.motion,
                                 fusion=args.fusion, robot_view=not args.no_robot_view)
    except (KeyError, ValueError) as error:
        parser.error(str(error.args[0]))
    speed = math.inf if args.speed == "max" else float(args.speed)

    from robosoccer_perception.vision_benchmark import setup

    overrides = {"device": args.device} if args.device else {}
    if args.resolution:
        overrides["resolution"] = list(args.resolution)
    settings = setup.load_settings(args.settings, overrides)

    def load_resources():
        return setup.load_resources(settings)

    if not args.replay:
        missing = setup.missing_checkpoints(settings)
        if missing:
            print("Trained checkpoints not found:\n  " + "\n  ".join(str(p) for p in missing)
                  + "\nTrain them first (README, 'Vision'), or replay a recording with --replay.",
                  file=sys.stderr)
            return 2

    if args.headless:
        from robosoccer_perception.viewer import headless
        from robosoccer_perception.viewer import recording as recording_module

        if args.replay:
            parser.error("--headless simulates a run; it cannot be combined with --replay")
        meta = {"scenario": scenario.id, "seed": args.seed}
        record = None if args.record is None else (
            args.record or recording_module.default_path("recordings", meta, ".npz"))
        video = None if args.video is None else (
            args.video or recording_module.default_path("videos", meta, ".mp4"))
        started = time.perf_counter()

        def progress(done: int, total: int) -> None:
            if done == total or done % 50 == 0:
                print(f"  frame {done} / {total}", flush=True)

        print(f"{scenario.id}: {scenario.title} (seed {args.seed})")
        from robosoccer_perception.viewer.render import Layers

        summary = headless.run_headless(
            load_resources(), scenario, args.seed, max_steps=args.max_steps, record=record,
            video=video, screenshots=args.screenshots,
            layers=Layers(ground_truth=not args.no_ground_truth),
            with_truth=True, max_particles=args.particles,
            video_speed=1.0 if math.isinf(speed) else speed, progress=progress,
            options=options)
        summary["wall_time_s"] = round(time.perf_counter() - started, 1)
        print(json.dumps(summary, indent=2))
        return 0

    from robosoccer_perception.viewer import qt_env

    try:
        qt_env.prepare()
    except qt_env.QtUnavailable as error:
        print(error, file=sys.stderr)
        return 2
    from robosoccer_perception.viewer.app import run_app

    return run_app(load_resources, scenario=scenario.id, seed=args.seed, speed=speed,
                   max_particles=args.particles, max_steps=args.max_steps,
                   show_truth=not args.no_ground_truth, autoplay=args.autoplay,
                   replay=args.replay, options=options)


if __name__ == "__main__":
    sys.exit(main())
