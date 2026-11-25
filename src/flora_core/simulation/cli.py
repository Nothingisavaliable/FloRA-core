#!/usr/bin/env python3
"""Replay one FloRA NPZ in MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .replay import ReplayConfig, replay_episode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--libero-root", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", choices=("replay", "topdown"), default="replay")
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument(
        "--collect-point-cloud",
        action="store_true",
        help="augment the replay with dynamic task-frame semantic point clouds",
    )
    parser.add_argument("--point-cloud-points", type=int, default=1024)
    parser.add_argument("--point-width", type=int, default=320)
    parser.add_argument("--point-height", type=int, default=240)
    parser.add_argument(
        "--point-cameras",
        default="replay,topdown",
        help="comma-separated MuJoCo camera names used for point-cloud fusion",
    )
    parser.add_argument("--point-cloud-seed", type=int, default=3407)
    args = parser.parse_args(argv)
    report = replay_episode(
        args.episode,
        output_dir=args.out_dir,
        video_path=args.video,
        libero_root=args.libero_root,
        config=ReplayConfig(
            width=args.width,
            height=args.height,
            camera=args.camera,
            render_stride=args.render_stride,
            collect_point_cloud=args.collect_point_cloud,
            point_cloud_points=args.point_cloud_points,
            point_width=args.point_width,
            point_height=args.point_height,
            point_cameras=tuple(
                value.strip()
                for value in args.point_cameras.split(",")
                if value.strip()
            ),
            point_cloud_seed=args.point_cloud_seed,
        ),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
