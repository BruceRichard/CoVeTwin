"""Deterministic geometry serializations used by the CoVeTwin ablation."""

from __future__ import annotations

import re

import numpy as np

from .geometry_codec import (
    decode_relative_shape_spans,
    encode_relative_shape_spans,
    flatten_voxels,
    serialize_relative_shape_spans,
    unflatten_indices,
)


REPRESENTATIONS = ("voxel", "index", "absolute_span", "relative_span")


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    result = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        value = int(raw)
        if value == previous + 1:
            previous = value
            continue
        result.append((start, previous - start + 1))
        start = previous = value
    result.append((start, previous - start + 1))
    return result


def encode_geometry(voxels: np.ndarray, representation: str, grid_size: int = 32) -> str:
    """Encode one occupancy with one of the four paper ablation formats."""

    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}")
    indices = flatten_voxels(voxels, grid_size)
    if len(indices) == 0:
        raise ValueError("cannot encode an empty occupied-voxel set")
    normalized = unflatten_indices(indices, grid_size)
    if representation == "voxel":
        return "vox " + " ".join(f"{x},{y},{z}" for x, y, z in normalized)
    if representation == "index":
        return "idx " + " ".join(str(int(value)) for value in indices)
    if representation == "absolute_span":
        return "asp " + " ".join(f"{start}:{length}" for start, length in _runs(indices))
    return serialize_relative_shape_spans(encode_relative_shape_spans(normalized, grid_size))


def representation_prompt(part_index: int, representation: str, grid_size: int = 32) -> str:
    base = f"Predict only the occupied geometry of l_{part_index} on a {grid_size}^3 voxel grid. "
    if representation == "voxel":
        return base + "Return exactly `vox x,y,z x,y,z ...` with sorted unique integer coordinates and no explanation."
    if representation == "index":
        return base + f"Flatten q=x*{grid_size}*{grid_size}+y*{grid_size}+z and return exactly `idx q q ...` with sorted unique indices and no explanation."
    if representation == "absolute_span":
        return base + f"Flatten q=x*{grid_size}*{grid_size}+y*{grid_size}+z, merge maximal runs, and return exactly `asp start:length start:length ...` with no explanation."
    if representation == "relative_span":
        from .prompts import part_geometry_prompt

        return part_geometry_prompt(part_index, grid_size)
    raise ValueError(f"unknown representation {representation!r}")


def _marked_payload(text: str, marker: str) -> str:
    matches = list(re.finditer(rf"(?i)(?<![A-Za-z0-9_]){re.escape(marker)}(?![A-Za-z0-9_])", text))
    if not matches:
        raise ValueError(f"missing `{marker}` marker")
    return text[matches[-1].end() :]


def decode_geometry(text: str, representation: str, grid_size: int = 32) -> np.ndarray:
    """Strict parser used for representation-specific parse-rate evaluation."""

    if representation == "relative_span":
        return decode_relative_shape_spans(text, grid_size)
    if representation == "voxel":
        payload = _marked_payload(text, "vox")
        matches = list(re.finditer(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", payload))
        if not matches:
            raise ValueError("no voxel coordinates found")
        voxels = np.asarray([[int(value) for value in match.groups()] for match in matches], dtype=np.int64)
        # Reusing the exact flatten/unflatten validation also enforces bounds.
        return unflatten_indices(flatten_voxels(voxels, grid_size), grid_size)
    if representation == "index":
        payload = _marked_payload(text, "idx")
        tokens = re.findall(r"\d+", payload)
        if not tokens:
            raise ValueError("no flattened indices found")
        return unflatten_indices(np.asarray([int(value) for value in tokens]), grid_size)
    if representation == "absolute_span":
        payload = _marked_payload(text, "asp")
        pairs = [(int(a), int(length)) for a, length in re.findall(r"(\d+)\s*:\s*(\d+)", payload)]
        if not pairs:
            raise ValueError("no absolute spans found")
        upper = grid_size**3
        previous_end = -2
        indices: list[int] = []
        for start, length in pairs:
            if length <= 0 or start <= previous_end + 1 or start + length > upper:
                raise ValueError("absolute spans must be positive, in range, ordered, and maximal")
            indices.extend(range(start, start + length))
            previous_end = start + length - 1
        return unflatten_indices(np.asarray(indices), grid_size)
    raise ValueError(f"unknown representation {representation!r}")
