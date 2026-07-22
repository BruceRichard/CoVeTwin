#!/usr/bin/env python3
"""Run the complete four-stage CoVeTwin inference pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = ROOT / "pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run selected CoVeTwin geometry-to-simulation stages.",
    )
    parser.add_argument("--demo-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", type=int, choices=(1, 2, 3, 4), default=(1, 2, 3, 4))
    parser.add_argument("--ckpt", default="./pretrain/vlm")
    parser.add_argument("--processor-ckpt", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("-K", "--candidate-count", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--simplify", type=float, default=0.5)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--remove-bg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-part-ply", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-stage1", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(command: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    python = sys.executable
    stages = set(args.stages)
    if args.force_stage1 and stages.intersection({2, 3, 4}):
        existing_fine_assets = list(args.output_path.glob("*/sample.glb"))
        if existing_fine_assets:
            raise RuntimeError(
                "--force-stage1 would replace coarse voxels while stage 2 would skip "
                "existing sample.glb files. Use a new --output-path, or remove the "
                "specific downstream assets explicitly before rerunning."
            )
    if 1 in stages:
        command = [
            python,
            str(PIPELINE_ROOT / "1_geometry_reasoning.py"),
            "--demo-path", str(args.demo_path),
            "--output-path", str(args.output_path),
            "--ckpt", args.ckpt,
            "--processor-ckpt", args.processor_ckpt,
            "--candidate-count", str(args.candidate_count),
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--seed", str(args.seed),
            "--save-part-ply" if args.save_part_ply else "--no-save-part-ply",
            "--remove-bg" if args.remove_bg else "--no-remove-bg",
            "--verify-candidates" if args.verify_candidates else "--no-verify-candidates",
        ]
        if args.only:
            command.extend(("--only", *args.only))
        if args.force_stage1:
            command.append("--force")
        _run(command, args.dry_run)
    if 2 in stages:
        command = [
            python,
            str(PIPELINE_ROOT / "2_flow_reconstruction.py"),
            "--demo_path", str(args.demo_path),
            "--input_paths", str(args.output_path),
            "--seed", str(args.seed),
            "--simplify", str(args.simplify),
            "--texture_size", str(args.texture_size),
        ]
        if args.only:
            command.extend(("--only", *args.only))
        _run(command, args.dry_run)
    if 3 in stages:
        _run(
            [
                python,
                str(PIPELINE_ROOT / "3_part_segmentation.py"),
                "--basepaths",
                str(args.output_path),
            ],
            args.dry_run,
        )
    if 4 in stages:
        _run(
            [
                python,
                str(PIPELINE_ROOT / "4_simulation_export.py"),
                "--basepath",
                str(args.output_path),
            ],
            args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
