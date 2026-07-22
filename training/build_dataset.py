#!/usr/bin/env python3
"""Build exact CoVeTwin two-turn fine-tuning records from voxel GT files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from covetwin.ablation_codecs import REPRESENTATIONS, encode_geometry, representation_prompt
from covetwin.geometry_codec import encode_relative_shape_spans


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create Qwen-VL conversations using CoVeTwin relative shape spans.",
    )
    parser.add_argument("--voxel-root", type=Path, default=Path("dataset/tmp_mobility/partseg"))
    parser.add_argument("--structure-root", type=Path, default=Path("dataset/txt_rep_32_finetune_mobility_all"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset_toolkits/renders_all"))
    parser.add_argument("--global-prompt", type=Path, default=Path("dataset/overall_prompt.txt"))
    parser.add_argument("--output", type=Path, default=Path("dataset/covetwin_training/conversations.json"))
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default="relative_span",
        help="Geometry target; the default is the full CoVeTwin representation.",
    )
    parser.add_argument("--views-per-object", type=int, default=25, help="0 means every available view")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all objects after --start")
    return parser.parse_args()


def _conversation(human: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"from": "human", "value": human},
        {"from": "gpt", "value": assistant},
    ]


def _part_files(voxel_dir: Path) -> list[Path]:
    result: list[Path] = []
    index = 0
    while (voxel_dir / f"ind_{index}.npy").is_file():
        result.append(voxel_dir / f"ind_{index}.npy")
        index += 1
    return result


def main() -> int:
    args = parse_args()
    for required in (args.voxel_root, args.structure_root, args.image_root):
        if not required.is_dir():
            raise FileNotFoundError(required)
    if not args.global_prompt.is_file():
        raise FileNotFoundError(args.global_prompt)
    global_prompt = args.global_prompt.read_text(encoding="utf-8")
    selected = set(args.only)
    object_ids = sorted(
        path.stem
        for path in args.structure_root.glob("*.txt")
        if not selected or path.stem in selected
    )[args.start :]
    if args.limit:
        object_ids = object_ids[: args.limit]

    records: list[dict] = []
    skipped: list[dict] = []
    for object_id in object_ids:
        structure_file = args.structure_root / f"{object_id}.txt"
        voxel_dir = args.voxel_root / object_id / str(args.grid_size)
        image_dir = args.image_root / object_id
        parts = _part_files(voxel_dir)
        images = sorted(
            path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if image_dir.is_dir() else []
        if args.views_per_object:
            images = images[: args.views_per_object]
        if not parts or not images:
            skipped.append({"object_id": object_id, "parts": len(parts), "views": len(images)})
            continue
        global_answer = structure_file.read_text(encoding="utf-8").strip()
        for part_index, part_file in enumerate(parts):
            voxels = np.load(part_file)
            relative = encode_relative_shape_spans(voxels, args.grid_size)
            geometry_answer = encode_geometry(voxels, args.representation, args.grid_size)
            for image_path in images:
                relative_image = image_path.relative_to(args.image_root).as_posix()
                conversations = _conversation(f"<image>\n{global_prompt}", global_answer)
                conversations.extend(
                    _conversation(
                        representation_prompt(part_index, args.representation, args.grid_size),
                        geometry_answer,
                    )
                )
                records.append(
                    {
                        "id": f"{object_id}_{image_path.stem}_part{part_index}",
                        "image": relative_image,
                        "conversations": conversations,
                        "data_source": "covetwin",
                        "object_id": object_id,
                        "part_index": part_index,
                        "voxel_count": relative.voxel_count,
                        "span_count": len(relative.spans),
                        "codec": args.representation,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "output": str(args.output.resolve()),
        "records": len(records),
        "objects_requested": len(object_ids),
        "objects_emitted": len({record["object_id"] for record in records}),
        "skipped": skipped,
        "grid_size": args.grid_size,
        "representation": args.representation,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
