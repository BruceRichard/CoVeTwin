"""Relative shape-span compression from Eqs. (7)--(14) of CoVeTwin.

The canonical wire format contains only ordinary text tokens::

    rss 184 0:1 14:1 15:18 46:8

``184`` is the part-local absolute reference ``b``.  Every following token is
``delta:length``.  A span is reconstructed as ``[b + delta, b + delta +
length - 1]``.  The first delta must be zero and spans must be sorted, maximal,
and non-overlapping.  These invariants make malformed VLM output detectable
before it reaches the fine decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import re
from typing import Iterable, Sequence

import numpy as np


CODEC_NAME = "rss"


@dataclass(frozen=True)
class RelativeShapeSpans:
    """A part-local reference and ``(relative start, length)`` spans."""

    base: int
    spans: tuple[tuple[int, int], ...]
    grid_size: int = 32

    @property
    def voxel_count(self) -> int:
        return sum(length for _, length in self.spans)

    def validate(self) -> "RelativeShapeSpans":
        _validate_grid_size(self.grid_size)
        upper = self.grid_size**3
        if not isinstance(self.base, Integral) or isinstance(self.base, bool):
            raise ValueError("base must be an integer")
        if not 0 <= self.base < upper:
            raise ValueError(f"base must be in [0, {upper}), got {self.base}")
        if not self.spans:
            raise ValueError("relative shape-span sequence is empty")
        if self.spans[0][0] != 0:
            raise ValueError("the first relative offset must be 0 (b = s_1)")

        previous_end = -2
        for position, pair in enumerate(self.spans):
            if len(pair) != 2:
                raise ValueError(f"span {position} must contain delta and length")
            delta, length = pair
            if (
                not isinstance(delta, Integral)
                or isinstance(delta, bool)
                or not isinstance(length, Integral)
                or isinstance(length, bool)
            ):
                raise ValueError(f"span {position} must contain integer values")
            if delta < 0:
                raise ValueError(f"span {position} has a negative offset")
            if length <= 0:
                raise ValueError(f"span {position} has a non-positive length")
            start = self.base + delta
            end = start + length - 1
            if start < 0 or end >= upper:
                raise ValueError(
                    f"span {position} reconstructs outside [0, {upper}): "
                    f"[{start}, {end}]"
                )
            if start <= previous_end + 1:
                raise ValueError(
                    "spans must be strictly ordered, disjoint, and maximal"
                )
            previous_end = end
        if self.voxel_count > upper:
            raise ValueError("expanded voxel count exceeds grid capacity")
        return self


def _validate_grid_size(grid_size: int) -> None:
    if not isinstance(grid_size, Integral) or isinstance(grid_size, bool) or grid_size <= 0:
        raise ValueError(f"grid_size must be a positive integer, got {grid_size!r}")


def _normalize_voxels(voxels: np.ndarray, grid_size: int) -> np.ndarray:
    _validate_grid_size(grid_size)
    array = np.asarray(voxels)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("voxels must have shape (N, 3)")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError("voxel coordinates must be finite integers")
    array = array.astype(np.int64, copy=False)
    if (array < 0).any() or (array >= grid_size).any():
        raise ValueError(f"voxel coordinates must lie in [0, {grid_size})")
    return np.unique(array, axis=0)


def flatten_voxels(voxels: np.ndarray, grid_size: int = 32) -> np.ndarray:
    """Apply ``q = x R^2 + y R + z`` and return sorted unique indices."""

    array = _normalize_voxels(voxels, grid_size)
    if len(array) == 0:
        return np.empty(0, dtype=np.int64)
    indices = (
        array[:, 0] * grid_size * grid_size
        + array[:, 1] * grid_size
        + array[:, 2]
    )
    return np.sort(indices.astype(np.int64, copy=False))


def unflatten_indices(indices: Iterable[int], grid_size: int = 32) -> np.ndarray:
    """Invert the CoVeTwin flattening map without silently clipping errors."""

    _validate_grid_size(grid_size)
    array = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError("flattened indices must be finite integers")
    array = array.astype(np.int64, copy=False).reshape(-1)
    upper = grid_size**3
    if (array < 0).any() or (array >= upper).any():
        raise ValueError(f"flattened indices must lie in [0, {upper})")
    array = np.unique(array)
    x = array // (grid_size * grid_size)
    remainder = array % (grid_size * grid_size)
    y = remainder // grid_size
    z = remainder % grid_size
    return np.stack((x, y, z), axis=1).astype(np.int64, copy=False)


def _absolute_spans(indices: Sequence[int]) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    result: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == previous + 1:
            previous = value
            continue
        result.append((start, previous))
        start = previous = value
    result.append((start, previous))
    return result


def encode_relative_shape_spans(
    voxels: np.ndarray, grid_size: int = 32
) -> RelativeShapeSpans:
    """Encode occupied voxels exactly as CoVeTwin Eqs. (8)--(13)."""

    indices = flatten_voxels(voxels, grid_size)
    if len(indices) == 0:
        raise ValueError("cannot encode an empty occupied-voxel set")
    absolute = _absolute_spans(indices)
    base = absolute[0][0]
    spans = tuple((start - base, end - start + 1) for start, end in absolute)
    return RelativeShapeSpans(base=base, spans=spans, grid_size=grid_size).validate()


def serialize_relative_shape_spans(value: RelativeShapeSpans) -> str:
    """Serialize a validated representation to the canonical compact syntax."""

    value.validate()
    pairs = " ".join(f"{delta}:{length}" for delta, length in value.spans)
    return f"{CODEC_NAME} {value.base} {pairs}"


def _payload_from_model_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("candidate output must be text")
    normalized = text.strip()
    if not normalized:
        raise ValueError("candidate output is empty")

    # Prefer the last explicit codec marker.  This avoids parsing numeric text
    # from explanations that a VLM may place before its final answer.
    matches = list(re.finditer(r"(?i)(?<![A-Za-z0-9_])rss(?![A-Za-z0-9_])", normalized))
    if matches:
        return normalized[matches[-1].end() :]

    # A documented verbose fallback is accepted for easier manual inspection.
    base_match = re.search(r"(?i)\bb(?:ase)?\s*=\s*(\d+)", normalized)
    if base_match:
        tail = normalized[base_match.end() :]
        return f"{base_match.group(1)} {tail}"
    raise ValueError("missing `rss` codec marker")


def parse_relative_shape_spans(
    text: str, grid_size: int = 32
) -> RelativeShapeSpans:
    """Parse a VLM response and enforce all representation invariants."""

    _validate_grid_size(grid_size)
    payload = _payload_from_model_text(text)
    base_match = re.match(r"\s*(\d+)", payload)
    if not base_match:
        raise ValueError("missing absolute base immediately after `rss`")
    base = int(base_match.group(1))
    tail = payload[base_match.end() :]

    # Canonical ``delta:length`` plus a tolerant ``(delta,length)`` form for
    # model outputs.  Any stray integer after removing pairs is rejected.
    pair_pattern = re.compile(r"(?:\(?\s*)(\d+)\s*[:,]\s*(\d+)(?:\s*\)?)")
    matches = list(pair_pattern.finditer(tail))
    if not matches:
        raise ValueError("no relative delta:length spans found")
    spans = tuple((int(match.group(1)), int(match.group(2))) for match in matches)

    residue_parts: list[str] = []
    cursor = 0
    for match in matches:
        residue_parts.append(tail[cursor : match.start()])
        cursor = match.end()
    residue_parts.append(tail[cursor:])
    residue = " ".join(residue_parts)
    residue = re.sub(
        r"(?i)\b(?:spans?|delta|length|offsets?)\b|[\s,;:{}\[\]()`'\"=\-]+",
        "",
        residue,
    )
    if residue:
        raise ValueError(f"unexpected content in shape-span payload: {residue!r}")

    return RelativeShapeSpans(base=base, spans=spans, grid_size=grid_size).validate()


def decode_relative_shape_spans(
    value: RelativeShapeSpans | str, grid_size: int | None = None
) -> np.ndarray:
    """Decode Eq. (14) and invert Eq. (8) into an ``(N,3)`` array."""

    if isinstance(value, str):
        value = parse_relative_shape_spans(value, grid_size or 32)
    elif grid_size is not None and value.grid_size != grid_size:
        raise ValueError(
            f"representation uses grid {value.grid_size}, requested grid {grid_size}"
        )
    value.validate()
    expanded: list[int] = []
    for delta, length in value.spans:
        start = value.base + delta
        expanded.extend(range(start, start + length))
    return unflatten_indices(np.asarray(expanded, dtype=np.int64), value.grid_size)
