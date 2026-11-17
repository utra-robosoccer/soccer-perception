#!/usr/bin/env python3
"""Score geometric feature extraction against simulator ground truth on random views.

No localization: this measures line points, goal posts, junctions and the centre circle as
they come out of the networks and the geometry, frame by frame.

Examples:
    python scripts/evaluate_features.py                           # 300 views, trained models
    python scripts/evaluate_features.py --source perfect_vision   # rendered labels instead
    python scripts/evaluate_features.py --appearance strong       # strong domain randomization
    python scripts/evaluate_features.py --pitch-error 2           # camera really 2 deg lower

Writes ``outputs/vision_localization/features/<name>.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", default="vision", choices=("vision", "perfect_vision"))
    parser.add_argument("--appearance", help="randomization preset (none, mild, strong)")
    parser.add_argument("--pitch-error", type=float, default=0.0,
                        help="true camera pitch minus believed pitch, degrees")
    parser.add_argument("--height-error", type=float, default=0.0, help="the same, metres")
    parser.add_argument("--settings", default="vision_localization")
    parser.add_argument("--device")
    parser.add_argument("--name", help="output file stem")
    parser.add_argument("--output", default="outputs/vision_localization/features")
    args = parser.parse_args()

    from robosoccer_perception.vision_benchmark import setup
    from robosoccer_perception.vision_benchmark.camera_truth import MountError
    from robosoccer_perception.vision_benchmark.feature_evaluation import evaluate_features

    settings = setup.load_settings(args.settings, {"device": args.device} if args.device else None)
    resources = setup.load_resources(settings)
    summary = evaluate_features(
        resources, args.frames, args.seed, source=args.source, appearance=args.appearance,
        mount_error=MountError(height=args.height_error, pitch_deg=args.pitch_error))
    name = args.name or "_".join(filter(None, [
        args.source, args.appearance, f"pitch{args.pitch_error:+g}" if args.pitch_error else "",
        f"height{args.height_error:+g}" if args.height_error else ""]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(json.dumps(summary, indent=2))

    line, posts, circle = summary["line_points"], summary["goal_posts"], summary["center_circle"]
    print(f"{summary['frames']} frames, source: {args.source}")
    print(f"line points   {summary['observations_per_frame']['line_points']:.1f} per frame, error "
          f"mean {line['error_m']['mean']:.3f} m, P95 {line['error_m']['p95']:.3f} m, "
          f"{line['within_2_sigma'] * 100:.1f} % within 2 sigma")
    print(f"projection    {summary['projection']['valid_share'] * 100:.1f} % of candidate pixels "
          f"valid, rejected: {summary['projection']['rejected_by_reason']}")
    print(f"goal posts    {posts['with_range']} with range, {posts['bearing_only']} bearing only, "
          f"{posts['unmatched']} unmatched; ground error mean "
          f"{posts['ground_point_error_m']['mean']} m, bearing error mean "
          f"{posts['bearing_error_deg']['mean']} deg")
    for kind in ("L", "T", "X", "all"):
        j = summary["intersections"][kind]
        print(f"junctions {kind:<3} reported {j['reported']}, precision {j['precision']}, recall "
              f"{j['recall']} (arms in view: {j['recall_arms_in_view']}), error mean "
              f"{j['position_error_m']['mean']} m")
    print(f"centre circle found in {circle['found_when_expected']} of {circle['frames_expected']} "
          f"frames where enough of it was in view; {circle['false']} false; centre error mean "
          f"{circle['centre_error_m']['mean']} m")
    print(f"geometry      {summary['geometry_latency_ms']['mean_ms']:.1f} ms mean per frame")
    print(f"written: {output / (name + '.json')}")


if __name__ == "__main__":
    main()
