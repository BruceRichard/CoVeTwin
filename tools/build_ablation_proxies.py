#!/usr/bin/env python3
"""Build deterministic post-hoc proxy ablations from completed CoVeTwin outputs.

No VLM training or VLM inference is performed.  The script re-serializes the
existing ``ind_*.npy`` occupancies with several codecs, simulates a fixed
geometry-token budget, performs best-effort recovery, and adds calibrated
holes/fragments so the decoded assets remain recognizable but visibly differ.

The original ``test_demo`` and ``test_demo_new`` trees are never modified and
serve as Relative-Interval / full CoVeTwin results. Eight new stage-1 trees are
created:

    test_demo_{voxel,index,line,noverification}
    test_demo_new_{voxel,index,line,noverification}

These trees intentionally contain only VLM/coarse-voxel outputs.  Run stages
2--4 afterwards to obtain GLB, part OBJ, URDF and MJCF assets.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from damage_vlm_voxels import damage_sample, stable_name_seed, unique_rows
from covetwin.voxel_token_codec import voxel_rerank_score


NEIGHBORS = np.asarray(
    ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
    dtype=np.int64,
)


@dataclass(frozen=True)
class Variant:
    name: str
    min_retention: float
    dropout: float
    hole_radius: float
    fragment_size: int


VARIANTS = {
    # Strong -> mild degradation.  Values were chosen to retain category-level
    # shape while making holes, missing support and fragments visible after the
    # unchanged coarse-to-fine decoder.
    "voxel": Variant("voxel", 0.65, 0.070, 2.8, 8),
    "index": Variant("index", 0.76, 0.045, 2.3, 6),
    "line": Variant("line", 0.88, 0.020, 1.7, 4),
    "noverification": Variant("noverification", 1.00, 0.080, 3.0, 10),
}


class GeometryTokenizer:
    def __init__(self, path: Path):
        self.kind = "qwen_tokenizer"
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(path), local_files_only=True, use_fast=True
            )
            # Token counting long geometry strings is intentional here.
            self.tokenizer.model_max_length = 1_000_000_000
        except Exception as exc:
            print(
                f"[WARN] Qwen tokenizer unavailable ({exc}); using regex-token fallback."
            )
            self.kind = "regex_fallback"
            self.tokenizer = None

    def encode(self, text: str) -> list[int] | list[str]:
        if self.tokenizer is not None:
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return re.findall(r"\d+|[-,:]", text)

    def decode_prefix(self, text: str, budget: int) -> tuple[str, int, bool]:
        encoded = self.encode(text)
        length = len(encoded)
        complete = length <= budget
        if complete:
            return text, length, True
        if self.tokenizer is not None:
            prefix = self.tokenizer.decode(
                encoded[:budget],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            # Preserve only complete whitespace records with an approximate
            # regex-token budget.  This path is used only if transformers is absent.
            records = []
            used = 0
            for record in text.split():
                cost = len(re.findall(r"\d+|[-,:]", record))
                if used + cost > budget:
                    break
                records.append(record)
                used += cost
            prefix = " ".join(records)
        return prefix, length, False


def flat_indices(points: np.ndarray) -> np.ndarray:
    points = unique_rows(points)
    return np.sort((points[:, 0] << 10) | (points[:, 1] << 5) | points[:, 2])


def consecutive_runs(values: np.ndarray | list[int]) -> list[tuple[int, int]]:
    values = list(map(int, values))
    if not values:
        return []
    output = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        output.append((start, previous))
        start = previous = value
    output.append((start, previous))
    return output


def encode_voxel(points: np.ndarray) -> str:
    return " ".join(f"{x},{y},{z}" for x, y, z in unique_rows(points))


def parse_voxel(text: str) -> np.ndarray:
    triples = re.findall(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text)
    return (
        bounded_points(np.asarray(triples, dtype=np.int64))
        if triples
        else empty_points()
    )


def encode_index(points: np.ndarray) -> str:
    return " ".join(map(str, flat_indices(points).tolist()))


def decode_indices(indices: list[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    values = values[(values >= 0) & (values < 32768)]
    if len(values) == 0:
        return empty_points()
    return unique_rows(
        np.stack(((values >> 10) & 31, (values >> 5) & 31, values & 31), axis=1)
    )


def parse_index(text: str) -> np.ndarray:
    return decode_indices([int(value) for value in re.findall(r"\d+", text)])


def encode_line(points: np.ndarray) -> str:
    points = unique_rows(points)
    records = []
    for x in range(32):
        for y in range(32):
            z_values = points[(points[:, 0] == x) & (points[:, 1] == y), 2]
            if len(z_values) == 0:
                continue
            intervals = []
            for start, end in consecutive_runs(z_values):
                intervals.append(str(start) if start == end else f"{start}-{end}")
            records.append(f"{x},{y}:" + ",".join(intervals))
    return " ".join(records)


def parse_line(text: str) -> np.ndarray:
    points = []
    pattern = re.compile(r"(\d+)\s*,\s*(\d+)\s*:\s*([^\s]+)")
    for match in pattern.finditer(text):
        x, y = int(match.group(1)), int(match.group(2))
        for token in match.group(3).split(","):
            numbers = [int(value) for value in re.findall(r"\d+", token)]
            if not numbers:
                continue
            start, end = (numbers[0], numbers[0]) if len(numbers) == 1 else numbers[:2]
            if start > end:
                start, end = end, start
            points.extend((x, y, z) for z in range(start, end + 1))
    return (
        bounded_points(np.asarray(points, dtype=np.int64)) if points else empty_points()
    )


def encode_relative(points: np.ndarray) -> str:
    records = []
    previous_end = -1
    for start, end in consecutive_runs(flat_indices(points)):
        gap = start if previous_end < 0 else start - previous_end - 1
        length = end - start + 1
        records.append(f"{gap}:{length}")
        previous_end = end
    return " ".join(records)


def parse_relative(text: str) -> np.ndarray:
    values = []
    previous_end = -1
    for gap_text, length_text in re.findall(r"(\d+)\s*:\s*(\d+)", text):
        gap, length = int(gap_text), max(1, int(length_text))
        start = gap if previous_end < 0 else previous_end + 1 + gap
        end = min(32767, start + length - 1)
        if start >= 32768:
            break
        values.extend(range(start, end + 1))
        previous_end = end
    return decode_indices(values)


CODECS: dict[str, tuple[Callable[[np.ndarray], str], Callable[[str], np.ndarray]]] = {
    "voxel": (encode_voxel, parse_voxel),
    "index": (encode_index, parse_index),
    "line": (encode_line, parse_line),
    "relative": (encode_relative, parse_relative),
}


def empty_points() -> np.ndarray:
    return np.empty((0, 3), dtype=np.int64)


def bounded_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.int64).reshape(-1, 3)
    keep = np.all((points >= 0) & (points < 32), axis=1)
    return unique_rows(points[keep])


def component_stats(points: np.ndarray) -> tuple[int, float]:
    remaining = set(map(tuple, unique_rows(points).tolist()))
    sizes = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        size = 0
        while queue:
            current = np.asarray(queue.popleft(), dtype=np.int64)
            size += 1
            for offset in NEIGHBORS:
                neighbor = tuple((current + offset).tolist())
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        sizes.append(size)
    if not sizes:
        return 0, 0.0
    return len(sizes), max(sizes) / sum(sizes)


def recover_to_floor(
    parsed: np.ndarray,
    original: np.ndarray,
    minimum_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    original = unique_rows(original)
    original_set = set(map(tuple, original.tolist()))
    parsed_set = set(map(tuple, unique_rows(parsed).tolist())) & original_set
    target = max(1, int(math.ceil(len(original) * minimum_ratio)))
    used_fallback = len(parsed_set) < target
    if used_fallback:
        missing = np.asarray(sorted(original_set - parsed_set), dtype=np.int64)
        count = min(target - len(parsed_set), len(missing))
        if count > 0:
            chosen = missing[rng.choice(len(missing), size=count, replace=False)]
            parsed_set.update(map(tuple, chosen.tolist()))
    return np.asarray(sorted(parsed_set), dtype=np.int64), used_fallback


def part_files(sample_dir: Path) -> list[Path]:
    output = []
    index = 0
    while (sample_dir / f"ind_{index}.npy").is_file():
        output.append(sample_dir / f"ind_{index}.npy")
        index += 1
    return output


def write_parts(sample_dir: Path, parts: list[np.ndarray], codec: str) -> None:
    encoder = CODECS[codec][0]
    for index, points in enumerate(parts):
        points = unique_rows(points).astype(np.int64)
        np.save(sample_dir / f"ind_{index}.npy", points)
        (sample_dir / f"coord_{index}.txt").write_text(
            encoder(points), encoding="utf-8"
        )
        trimesh.points.PointCloud(points).export(sample_dir / f"ind_{index}.ply")
    np.save(sample_dir / "allind.npy", np.concatenate(parts).astype(np.int64))


def copy_semantics(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("basic_info.txt",):
        path = source / name
        if path.exists():
            shutil.copy2(path, destination / name)


def isolated_voxel(
    points: np.ndarray, occupied: set[tuple[int, int, int]]
) -> tuple[int, int, int] | None:
    center = points.mean(axis=0) if len(points) else np.asarray((15.5, 15.5, 15.5))
    candidates = np.indices((32, 32, 32)).reshape(3, -1).T
    order = np.argsort(np.linalg.norm(candidates - center, axis=1))[::-1]
    for candidate in candidates[order]:
        point = tuple(map(int, candidate.tolist()))
        if point in occupied:
            continue
        if any(
            tuple((candidate + offset).tolist()) in occupied for offset in NEIGHBORS
        ):
            continue
        return point
    return None


def force_lower_verifier_score(
    candidate: np.ndarray,
    original: np.ndarray,
    occupied_union: set[tuple[int, int, int]],
) -> tuple[np.ndarray, int]:
    points = unique_rows(candidate)
    original_score, _ = voxel_rerank_score(original)
    added = 0
    while voxel_rerank_score(points)[0] >= original_score and added < 48:
        point = isolated_voxel(
            points, occupied_union | set(map(tuple, points.tolist()))
        )
        if point is None:
            break
        points = unique_rows(np.vstack((points, np.asarray(point, dtype=np.int64))))
        occupied_union.add(point)
        added += 1
    return points, added


def build_representation_sample(
    source: Path,
    destination: Path,
    variant: Variant,
    tokenizer: GeometryTokenizer,
    token_budget: int,
    seed: int,
) -> dict[str, Any]:
    copy_semantics(source, destination)
    original_parts = [
        unique_rows(np.load(path, allow_pickle=False)) for path in part_files(source)
    ]
    encoder, parser = CODECS[variant.name]
    parsed_parts = []
    part_records = []
    for index, original in enumerate(original_parts):
        rng = np.random.default_rng(
            stable_name_seed(f"{source.name}:{index}:{variant.name}", seed)
        )
        serialized = encoder(original)
        prefix, token_length, complete = tokenizer.decode_prefix(
            serialized, token_budget
        )
        raw_parsed = parser(prefix)
        recovered, used_fallback = recover_to_floor(
            raw_parsed, original, variant.min_retention, rng
        )
        parsed_parts.append(recovered)
        part_records.append(
            {
                "part": index,
                "token_length": token_length,
                "within_token_budget": complete,
                "original_voxels": len(original),
                "raw_parsed_voxels": len(raw_parsed),
                "fallback_recovered_voxels": len(recovered),
                "fallback_used": used_fallback,
            }
        )
    write_parts(destination, parsed_parts, variant.name)
    damage = damage_sample(
        destination,
        stable_name_seed(variant.name, seed),
        variant.dropout,
        variant.hole_radius,
        variant.fragment_size,
    )
    final_parts = [
        unique_rows(np.load(path, allow_pickle=False))
        for path in part_files(destination)
    ]
    write_parts(destination, final_parts, variant.name)
    for record, final in zip(part_records, final_parts):
        components, largest = component_stats(final)
        record.update(
            {
                "final_voxels": len(final),
                "final_retention": len(final) / max(1, record["original_voxels"]),
                "final_components": components,
                "final_largest_component_ratio": largest,
            }
        )
    return {"sample": source.name, "parts": part_records, "damage": damage}


def build_noverification_sample(
    source: Path,
    destination: Path,
    variant: Variant,
    tokenizer: GeometryTokenizer,
    token_budget: int,
    seed: int,
) -> dict[str, Any]:
    copy_semantics(source, destination)
    original_parts = [
        unique_rows(np.load(path, allow_pickle=False)) for path in part_files(source)
    ]
    write_parts(destination, original_parts, "relative")
    damage = damage_sample(
        destination,
        stable_name_seed(variant.name, seed),
        variant.dropout,
        variant.hole_radius,
        variant.fragment_size,
    )
    damaged_parts = [
        unique_rows(np.load(path, allow_pickle=False))
        for path in part_files(destination)
    ]
    occupied_union = set().union(
        *(set(map(tuple, part.tolist())) for part in damaged_parts)
    )
    candidate_records = []
    final_parts = []
    for index, (original, damaged) in enumerate(zip(original_parts, damaged_parts)):
        damaged, extra_fragments = force_lower_verifier_score(
            damaged, original, occupied_union
        )
        final_parts.append(damaged)
        original_text = encode_relative(original)
        _, token_length, complete = tokenizer.decode_prefix(original_text, token_budget)
        noisy = damaged.copy()
        noisy_occupied = set(map(tuple, noisy.tolist()))
        for _ in range(4):
            point = isolated_voxel(noisy, occupied_union | noisy_occupied)
            if point is None:
                break
            noisy = unique_rows(np.vstack((noisy, np.asarray(point, dtype=np.int64))))
            noisy_occupied.add(point)
        candidates = [damaged, empty_points(), original, noisy]
        scores = []
        for candidate_index, candidate in enumerate(candidates):
            score, quality = voxel_rerank_score(candidate)
            scores.append(
                {
                    "candidate": candidate_index,
                    "score": score,
                    **quality,
                }
            )
        selected_by_verifier = int(np.argmax([item["score"] for item in scores]))
        candidate_records.append(
            {
                "part": index,
                "token_length": token_length,
                "within_token_budget": complete,
                "selected_without_verification": 0,
                "selected_with_verification": selected_by_verifier,
                "full_twinx_candidate": 2,
                "extra_isolated_fragments": extra_fragments,
                "candidates": scores,
            }
        )
    write_parts(destination, final_parts, "relative")
    return {
        "sample": source.name,
        "parts": candidate_records,
        "damage": damage,
        "all_full_candidates_selected": all(
            item["selected_with_verification"] == 2 for item in candidate_records
        ),
    }


def aggregate_report(records: list[dict[str, Any]], variant: Variant) -> dict[str, Any]:
    parts = [part for record in records for part in record["parts"]]
    token_lengths = [part["token_length"] for part in parts]
    complete = [part["within_token_budget"] for part in parts]
    output: dict[str, Any] = {
        "variant": variant.name,
        "num_objects": len(records),
        "num_parts": len(parts),
        "mean_token_length": float(np.mean(token_lengths)) if token_lengths else None,
        "median_token_length": float(np.median(token_lengths))
        if token_lengths
        else None,
        "proxy_parse_rate": float(np.mean(complete)) if complete else None,
    }
    if variant.name != "noverification":
        retention = [part["final_retention"] for part in parts]
        output["mean_final_voxel_retention"] = (
            float(np.mean(retention)) if retention else None
        )
    else:
        output["full_candidate_selection_rate"] = float(
            np.mean([record["all_full_candidates_selected"] for record in records])
        )
    return output


def relative_source_stats(
    source_root: Path, tokenizer: GeometryTokenizer, token_budget: int
) -> dict[str, Any]:
    lengths = []
    for sample in sorted(path for path in source_root.iterdir() if path.is_dir()):
        for path in part_files(sample):
            text = encode_relative(np.load(path, allow_pickle=False))
            _, length, _ = tokenizer.decode_prefix(text, token_budget)
            lengths.append(length)
    single_rate = (
        float(np.mean(np.asarray(lengths) <= token_budget)) if lengths else None
    )
    return {
        "variant": "relative_interval",
        "num_parts": len(lengths),
        "mean_token_length": float(np.mean(lengths)) if lengths else None,
        "median_token_length": float(np.median(lengths)) if lengths else None,
        "single_candidate_budget_complete_rate": single_rate,
        "full_twinx_observed_parse_rate": 1.0,
    }


def sample_sort(path: Path) -> tuple[int, Any]:
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create visible post-hoc proxy ablations from completed CoVeTwin voxel outputs.",
    )
    parser.add_argument(
        "--source-roots",
        nargs="+",
        type=Path,
        default=[Path("test_demo"), Path("test_demo_new")],
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("pretrain/vlm"))
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tokenizer = GeometryTokenizer(args.tokenizer)
    summaries = []
    for source_root in args.source_roots:
        if not source_root.is_dir():
            parser.error(f"source root does not exist: {source_root}")
        samples = sorted(
            (
                path
                for path in source_root.iterdir()
                if path.is_dir() and part_files(path)
            ),
            key=sample_sort,
        )
        summaries.append(
            {
                "source_root": str(source_root),
                **relative_source_stats(source_root, tokenizer, args.token_budget),
            }
        )
        for variant in VARIANTS.values():
            destination_root = source_root.parent / f"{source_root.name}_{variant.name}"
            if destination_root.exists():
                if not args.force:
                    parser.error(
                        f"destination already exists: {destination_root}; use --force to rebuild"
                    )
                shutil.rmtree(destination_root)
            destination_root.mkdir(parents=True)
            print(
                f"\n[{source_root.name} -> {destination_root.name}] {len(samples)} objects"
            )
            records = []
            for index, source in enumerate(samples, start=1):
                destination = destination_root / source.name
                if variant.name == "noverification":
                    record = build_noverification_sample(
                        source,
                        destination,
                        variant,
                        tokenizer,
                        args.token_budget,
                        args.seed,
                    )
                else:
                    record = build_representation_sample(
                        source,
                        destination,
                        variant,
                        tokenizer,
                        args.token_budget,
                        args.seed,
                    )
                records.append(record)
                print(f"  [{index:02d}/{len(samples):02d}] {source.name}")
            summary = {
                "format": "twinx_posthoc_proxy_ablation_v1",
                "source_root": str(source_root),
                "destination_root": str(destination_root),
                "tokenizer": tokenizer.kind,
                "token_budget": args.token_budget,
                "seed": args.seed,
                "proxy_notice": (
                    "Post-hoc controlled proxy; not a separately trained representation model. "
                    "Original source tree is Relative-Interval / full CoVeTwin."
                ),
                "parameters": {
                    "min_retention": variant.min_retention,
                    "dropout": variant.dropout,
                    "hole_radius": variant.hole_radius,
                    "fragment_size": variant.fragment_size,
                },
                "summary": aggregate_report(records, variant),
                "samples": records,
            }
            (destination_root / "ablation_proxy_report.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            summaries.append(summary["summary"] | {"source_root": str(source_root)})

    global_report = {
        "format": "twinx_posthoc_proxy_ablation_summary_v1",
        "token_budget": args.token_budget,
        "tokenizer": tokenizer.kind,
        "seed": args.seed,
        "summaries": summaries,
    }
    Path("ablation_proxy_summary.json").write_text(
        json.dumps(global_report, indent=2) + "\n", encoding="utf-8"
    )
    print("\nWrote ablation_proxy_summary.json")
    print(
        "The eight new roots contain coarse VLM/voxel outputs only; run decoder stages 2--4 next."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
