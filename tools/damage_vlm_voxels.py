#!/usr/bin/env python3
"""Apply deterministic, reversible holes and small fragments to VLM voxels."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import trimesh


NEIGHBORS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int64,
)


def stable_name_seed(name: str, seed: int) -> int:
    value = seed & 0xFFFFFFFF
    for byte in name.encode("utf-8"):
        value = ((value * 16777619) ^ byte) & 0xFFFFFFFF
    return value


def unique_rows(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.asarray(points, dtype=np.int64).reshape(-1, 3), axis=0)


def component_sizes(points: np.ndarray) -> list[int]:
    remaining = set(map(tuple, unique_rows(points).tolist()))
    sizes: list[int] = []
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        size = 0
        while queue:
            point = np.asarray(queue.popleft())
            size += 1
            for offset in NEIGHBORS:
                neighbor = tuple((point + offset).tolist())
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def encode_coord_text(points: np.ndarray) -> str:
    points = unique_rows(points)
    encoded = np.sort((points[:, 0] << 10) | (points[:, 1] << 5) | points[:, 2])
    if len(encoded) == 0:
        return ""
    tokens: list[str] = []
    start = previous = int(encoded[0])
    for value_np in encoded[1:]:
        value = int(value_np)
        if value == previous + 1:
            previous = value
            continue
        tokens.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = value
    tokens.append(f"{start}-{previous}" if start != previous else str(start))
    return " ".join(tokens)


def carve_global_holes(
    union: np.ndarray,
    rng: np.random.Generator,
    radius: float,
    max_holes: int,
) -> tuple[set[tuple[int, int, int]], list[dict[str, object]]]:
    remaining = unique_rows(union)
    removed: set[tuple[int, int, int]] = set()
    holes: list[dict[str, object]] = []
    radius_squared = radius**2
    for _ in range(max_holes):
        if len(remaining) < 16:
            break
        sample_count = min(256, len(remaining))
        candidate_ids = rng.choice(len(remaining), size=sample_count, replace=False)
        candidates = remaining[candidate_ids]
        # Prefer a locally dense center so each hole is visible rather than a
        # single missing extremity voxel.
        best_center = None
        best_mask = None
        best_count = 0
        for center in candidates:
            mask = np.sum((remaining - center) ** 2, axis=1) <= radius_squared
            count = int(mask.sum())
            if count > best_count:
                best_center, best_mask, best_count = center, mask, count
        if best_center is None or best_count < 3:
            break
        removed.update(map(tuple, remaining[best_mask].tolist()))
        holes.append(
            {"center": best_center.astype(int).tolist(), "radius": radius, "removed": best_count}
        )
        remaining = remaining[~best_mask]
    return removed, holes


def grow_cluster(
    points: set[tuple[int, int, int]],
    seed: tuple[int, int, int],
    size: int,
) -> list[tuple[int, int, int]]:
    queue = deque([seed])
    seen = {seed}
    cluster: list[tuple[int, int, int]] = []
    while queue and len(cluster) < size:
        current = queue.popleft()
        cluster.append(current)
        point = np.asarray(current)
        for offset in NEIGHBORS:
            neighbor = tuple((point + offset).tolist())
            if neighbor in points and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return cluster


def detach_fragment(
    parts: list[np.ndarray],
    rng: np.random.Generator,
    fragment_size: int,
) -> dict[str, object] | None:
    eligible = [index for index, part in enumerate(parts) if len(part) >= max(80, fragment_size * 4)]
    if not eligible:
        return None
    weights = np.array([len(parts[index]) for index in eligible], dtype=np.float64)
    part_index = int(rng.choice(eligible, p=weights / weights.sum()))
    part_set = set(map(tuple, parts[part_index].tolist()))
    all_union = set().union(*(set(map(tuple, part.tolist())) for part in parts))
    center = parts[part_index].mean(axis=0)
    distances = np.linalg.norm(parts[part_index] - center, axis=1)
    seed_candidates = parts[part_index][np.argsort(distances)[::-1][: min(128, len(parts[part_index]))]]

    for seed_array in seed_candidates[rng.permutation(len(seed_candidates))]:
        seed = tuple(seed_array.tolist())
        cluster = grow_cluster(part_set, seed, fragment_size)
        if len(cluster) < max(3, fragment_size // 2):
            continue
        cluster_array = np.asarray(cluster, dtype=np.int64)
        direction_order = np.argsort(np.abs(seed_array - center))[::-1]
        directions: list[np.ndarray] = []
        for axis in direction_order:
            sign = 1 if seed_array[axis] >= center[axis] else -1
            for signed in (sign, -sign):
                direction = np.zeros(3, dtype=np.int64)
                direction[axis] = signed
                directions.append(direction)
        remaining_union = all_union - set(cluster)
        for direction in directions:
            for distance in (3, 4, 5, 2):
                shifted = cluster_array + direction * distance
                if np.any(shifted < 0) or np.any(shifted > 31):
                    continue
                shifted_set = set(map(tuple, shifted.tolist()))
                if shifted_set & remaining_union:
                    continue
                touches = False
                for point in shifted:
                    if any(tuple((point + offset).tolist()) in remaining_union for offset in NEIGHBORS):
                        touches = True
                        break
                if touches:
                    continue
                new_part = (part_set - set(cluster)) | shifted_set
                parts[part_index] = np.asarray(sorted(new_part), dtype=np.int64)
                return {
                    "part": part_index,
                    "source_seed": list(seed),
                    "voxel_count": len(cluster),
                    "shift": (direction * distance).astype(int).tolist(),
                }
    return None


def clear_derived_outputs(root: Path) -> dict[str, int]:
    file_names = {"sample.glb", "basic.urdf", "basic.xml", "basic_info.json", "desert.png"}
    removed_files = 0
    removed_dirs = 0
    for sample_dir in [path for path in root.iterdir() if path.is_dir()]:
        for name in file_names:
            path = sample_dir / name
            if path.exists():
                path.unlink()
                removed_files += 1
        objs = sample_dir / "objs"
        if objs.exists():
            shutil.rmtree(objs)
            removed_dirs += 1
    return {"removed_derived_files": removed_files, "removed_objs_directories": removed_dirs}


def damage_sample(
    sample_dir: Path,
    base_seed: int,
    dropout: float,
    hole_radius: float,
    fragment_size: int,
) -> dict[str, object]:
    part_paths: list[Path] = []
    index = 0
    while (sample_dir / f"ind_{index}.npy").is_file():
        part_paths.append(sample_dir / f"ind_{index}.npy")
        index += 1
    parts = [unique_rows(np.load(path, allow_pickle=False)) for path in part_paths]
    rng = np.random.default_rng(stable_name_seed(sample_dir.name, base_seed))
    original_union = unique_rows(np.concatenate(parts))
    before_components = component_sizes(original_union)

    hole_count = 1 + int(len(original_union) >= 1200) + int(len(original_union) >= 2800)
    hole_removed, holes = carve_global_holes(original_union, rng, hole_radius, hole_count)
    damaged_parts: list[np.ndarray] = []
    dropout_removed = 0
    for part in parts:
        kept = np.asarray([point for point in part.tolist() if tuple(point) not in hole_removed], dtype=np.int64)
        if len(kept) == 0:
            # Never delete an entire semantic part.
            kept = part[[int(rng.integers(len(part)))]]
        remove_count = min(int(round(len(kept) * dropout)), max(0, len(kept) - 4))
        if remove_count > 0:
            remove_ids = rng.choice(len(kept), size=remove_count, replace=False)
            mask = np.ones(len(kept), dtype=bool)
            mask[remove_ids] = False
            kept = kept[mask]
            dropout_removed += remove_count
        damaged_parts.append(unique_rows(kept))

    fragments: list[dict[str, object]] = []
    fragment_count = 1 + int(len(original_union) >= 2500)
    for _ in range(fragment_count):
        fragment = detach_fragment(damaged_parts, rng, fragment_size)
        if fragment is not None:
            fragments.append(fragment)

    for part_index, (path, points) in enumerate(zip(part_paths, damaged_parts)):
        np.save(path, points.astype(np.int64))
        (sample_dir / f"coord_{part_index}.txt").write_text(
            encode_coord_text(points), encoding="utf-8"
        )
        trimesh.points.PointCloud(points).export(sample_dir / f"ind_{part_index}.ply")

    allind = np.concatenate(damaged_parts).astype(np.int64)
    np.save(sample_dir / "allind.npy", allind)
    damaged_union = unique_rows(allind)
    after_components = component_sizes(damaged_union)
    return {
        "sample": sample_dir.name,
        "parts": len(parts),
        "original_union_voxels": len(original_union),
        "damaged_union_voxels": len(damaged_union),
        "union_change_percent": 100.0 * (len(damaged_union) - len(original_union)) / len(original_union),
        "holes": holes,
        "hole_voxels_selected": len(hole_removed),
        "dropout_voxels_removed_across_parts": dropout_removed,
        "fragments": fragments,
        "components_before": len(before_components),
        "components_after": len(after_components),
        "largest_component_ratio_before": before_components[0] / len(original_union),
        "largest_component_ratio_after": after_components[0] / len(damaged_union),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("test_demo_bad"))
    parser.add_argument(
        "--backup", type=Path, default=Path("test_demo_bad_before_voxel_damage")
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--dropout", type=float, default=0.035)
    parser.add_argument("--hole-radius", type=float, default=2.6)
    parser.add_argument("--fragment-size", type=int, default=6)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"missing root directory: {args.root}")
    if args.backup.exists():
        parser.error(f"backup already exists: {args.backup}")
    if not 0.0 <= args.dropout < 0.5:
        parser.error("--dropout must be in [0, 0.5)")

    print(f"Backing up complete current results: {args.root} -> {args.backup}")
    shutil.copytree(args.root, args.backup)
    derived = clear_derived_outputs(args.root)

    sample_dirs = sorted(
        [path for path in args.root.iterdir() if path.is_dir() and (path / "allind.npy").is_file()],
        key=lambda path: int(path.name) if path.name.isdigit() else path.name,
    )
    records = []
    for sample_dir in sample_dirs:
        record = damage_sample(
            sample_dir, args.seed, args.dropout, args.hole_radius, args.fragment_size
        )
        records.append(record)
        print(
            f"{sample_dir.name}: union {record['original_union_voxels']} -> "
            f"{record['damaged_union_voxels']}, components "
            f"{record['components_before']} -> {record['components_after']}, "
            f"fragments={len(record['fragments'])}"
        )

    report = {
        "format": "physx_anything_vlm_voxel_damage_v1",
        "root": str(args.root),
        "backup": str(args.backup),
        "parameters": {
            "seed": args.seed,
            "dropout": args.dropout,
            "hole_radius": args.hole_radius,
            "fragment_size": args.fragment_size,
        },
        "derived_cleanup": derived,
        "samples": records,
    }
    (args.root / "vlm_damage_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote report: {args.root / 'vlm_damage_report.json'}")
    print("Old decoded/split/URDF outputs were removed from the working tree; rerun stages 2-4.")


if __name__ == "__main__":
    main()
