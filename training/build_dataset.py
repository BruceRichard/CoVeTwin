#!/usr/bin/env python3
"""Build exact CoVeTwin two-turn fine-tuning records from voxel GT files.

The object-disjoint PhysX-Mobility split (1636 train / 388 test objects) is
enforced by default: test objects are excluded from the emitted records unless
``--allow-test-leak`` is passed.  The canonical split is loaded from
``dataset/splits/trainingset.npy`` and ``dataset/splits/testset.npy`` (or from
``--split-json``).  If no split files exist, a deterministic one can be
generated with ``--make-split-json`` (fixed ``--split-seed``, default 42).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from covetwin.ablation_codecs import REPRESENTATIONS, encode_geometry, representation_prompt
from covetwin.geometry_codec import encode_relative_shape_spans


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_SPLIT_SEED = 42
DEFAULT_TEST_COUNT = 388


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
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="train",
        help=(
            "Object-disjoint split to emit. 'train' (default) and 'test' restrict "
            "objects to the canonical split; 'all' requires --allow-test-leak."
        ),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "splits",
        help="Directory with the canonical trainingset.npy / testset.npy split files.",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help='Optional JSON split override with {"train": [...], "test": [...]} object IDs.',
    )
    parser.add_argument(
        "--allow-test-leak",
        action="store_true",
        help=(
            "Disable split enforcement and allow test objects in the output. "
            "Records built this way must NOT be used to train checkpoints that "
            "claim official held-out test results."
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Seed for deterministic record shuffling; unset keeps the sorted object order.",
    )
    parser.add_argument(
        "--make-split-json",
        type=Path,
        default=None,
        help=(
            "Write a deterministic split JSON from the discovered structure files "
            "(--split-seed, --test-count) and exit. Fallback for when no canonical "
            "split files exist."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--test-count", type=int, default=DEFAULT_TEST_COUNT)
    return parser.parse_args()


def discover_object_ids(structure_root: Path) -> list[str]:
    return sorted(path.stem for path in structure_root.glob("*.txt"))


def load_split_ids(split_dir: Path, split_json: Path | None = None) -> dict[str, set[str]]:
    """Load the canonical object-disjoint split (train/test object ID sets)."""
    if split_json is not None:
        payload = json.loads(split_json.read_text(encoding="utf-8"))
        return {
            "train": {str(item) for item in payload.get("train", [])},
            "test": {str(item) for item in payload.get("test", [])},
        }
    split: dict[str, set[str]] = {}
    for name, filename in (("train", "trainingset.npy"), ("test", "testset.npy")):
        path = split_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"canonical split file not found: {path}. Provide --split-json, or "
                "generate a deterministic split with --make-split-json, or pass "
                "--allow-test-leak to build without split enforcement."
            )
        split[name] = {str(item) for item in np.load(path, allow_pickle=True).tolist()}
    overlap = split["train"] & split["test"]
    if overlap:
        raise ValueError(f"split files are not object-disjoint: {len(overlap)} shared IDs")
    return split


def generate_split(
    object_ids: list[str],
    test_count: int = DEFAULT_TEST_COUNT,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, list[str]]:
    """Deterministic object-level split from sorted IDs with a fixed seed."""
    shuffled = list(object_ids)
    random.Random(seed).shuffle(shuffled)
    test_count = min(max(0, int(test_count)), len(shuffled))
    test = sorted(shuffled[:test_count])
    train = sorted(shuffled[test_count:])
    return {"train": train, "test": test}


def select_object_ids(
    discovered: list[str],
    only: list[str],
    split_ids: dict[str, set[str]] | None,
    split: str,
    allow_test_leak: bool,
) -> tuple[list[str], dict[str, object]]:
    """Apply the object-disjoint split to the discovered object IDs.

    Returns the selected IDs (sorted) plus provenance counts for the summary.
    """
    selected = set(only)
    candidates = [item for item in discovered if not selected or item in selected]
    info: dict[str, object] = {"split": split, "allow_test_leak": bool(allow_test_leak)}
    if split == "all" or allow_test_leak or split_ids is None:
        info["excluded_test_objects"] = 0
        info["excluded_unknown_objects"] = 0
        return candidates, info
    keep = split_ids["train"] if split == "train" else split_ids["test"]
    other = split_ids["test"] if split == "train" else split_ids["train"]
    excluded_test = sorted(set(candidates) & other)
    unknown = sorted(set(candidates) - keep - other)
    result = [item for item in candidates if item in keep]
    info["excluded_test_objects"] = len(excluded_test)
    info["excluded_unknown_objects"] = len(unknown)
    if excluded_test:
        info["excluded_test_ids_sample"] = excluded_test[:10]
    if unknown:
        info["excluded_unknown_ids_sample"] = unknown[:10]
    return result, info


def shuffle_records(records: list[dict], seed: int | None) -> list[dict]:
    """Deterministically shuffle records when a seed is supplied."""
    if seed is None:
        return records
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


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

    discovered = discover_object_ids(args.structure_root)
    if args.make_split_json is not None:
        split = generate_split(discovered, args.test_count, args.split_seed)
        args.make_split_json.parent.mkdir(parents=True, exist_ok=True)
        args.make_split_json.write_text(
            json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"Wrote deterministic split ({len(split['train'])} train / "
            f"{len(split['test'])} test, seed={args.split_seed}) to {args.make_split_json}"
        )
        return 0

    if args.split == "all" and not args.allow_test_leak:
        raise SystemExit(
            "--split all would emit training records for test objects; pass "
            "--allow-test-leak explicitly if this is intentional."
        )
    split_ids = None
    split_source = "none (split enforcement disabled)"
    if not args.allow_test_leak:
        split_ids = load_split_ids(args.split_dir, args.split_json)
        split_source = str(args.split_json) if args.split_json else str(args.split_dir)
    else:
        print(
            "[WARNING] --allow-test-leak is set: test objects may be included in "
            "the emitted records. Do NOT train paper checkpoints from this file "
            "or claim official held-out test results with it.",
            file=sys.stderr,
        )

    global_prompt = args.global_prompt.read_text(encoding="utf-8")
    object_ids, split_info = select_object_ids(
        discovered, args.only, split_ids, args.split, args.allow_test_leak
    )
    if split_info.get("excluded_test_objects"):
        print(
            f"Excluded {split_info['excluded_test_objects']} test-split objects "
            f"from the '{args.split}' split (object-disjoint protocol enforced).",
            file=sys.stderr,
        )
    if split_info.get("excluded_unknown_objects"):
        print(
            f"Excluded {split_info['excluded_unknown_objects']} objects absent from "
            "the canonical split files.",
            file=sys.stderr,
        )
    object_ids = object_ids[args.start :]
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
    records = shuffle_records(records, args.shuffle_seed)
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
        "split": args.split,
        "split_source": split_source,
        "allow_test_leak": bool(args.allow_test_leak),
        "shuffle_seed": args.shuffle_seed,
        **split_info,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
