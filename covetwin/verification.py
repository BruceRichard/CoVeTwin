"""Structure-verified voxel candidate refinement (CoVeTwin Eqs. 15--16)."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np

from .geometry_codec import decode_relative_shape_spans


@dataclass(frozen=True)
class CandidateEvaluation:
    index: int
    valid: bool
    voxel_count: int = 0
    component_count: int = 0
    largest_component_size: int = 0
    largest_component_ratio: float = 0.0
    score: float | None = None
    error: str | None = None
    raw_text: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CandidateSelection:
    selected_index: int
    voxels: np.ndarray
    evaluations: tuple[CandidateEvaluation, ...]

    @property
    def selected(self) -> CandidateEvaluation:
        return self.evaluations[self.selected_index]

    def to_dict(self) -> dict:
        return {
            "selected_index": self.selected_index,
            "selected_score": self.selected.score,
            "candidates": [item.to_dict() for item in self.evaluations],
        }


def connected_component_sizes(voxels: np.ndarray, grid_size: int = 32) -> list[int]:
    """Return sizes of all 6-connected occupied components."""

    points = np.asarray(voxels, dtype=np.int64)
    if points.size == 0:
        return []
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("voxels must have shape (N, 3)")
    if (points < 0).any() or (points >= grid_size).any():
        raise ValueError(f"voxel coordinates must lie in [0, {grid_size})")
    remaining = {tuple(point) for point in np.unique(points, axis=0).tolist()}
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    sizes: list[int] = []
    while remaining:
        start = remaining.pop()
        queue = deque((start,))
        size = 0
        while queue:
            x, y, z = queue.popleft()
            size += 1
            for dx, dy, dz in offsets:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def quality_score(voxels: np.ndarray, grid_size: int = 32) -> tuple[float, int, int, float]:
    """Compute ``Q = 100 rho - 2 c + min(n,R^3)/R^3`` exactly."""

    unique = np.unique(np.asarray(voxels, dtype=np.int64), axis=0)
    n_voxels = len(unique)
    if n_voxels == 0:
        raise ValueError("empty candidates are invalid")
    sizes = connected_component_sizes(unique, grid_size)
    component_count = len(sizes)
    largest_size = sizes[0]
    ratio = largest_size / n_voxels
    score = 100.0 * ratio - 2.0 * component_count + min(n_voxels, grid_size**3) / (grid_size**3)
    return float(score), component_count, largest_size, float(ratio)


def evaluate_candidate(
    candidate: str | np.ndarray, index: int = 0, grid_size: int = 32
) -> tuple[CandidateEvaluation, np.ndarray | None]:
    """Parse and score one candidate; invalid candidates become report rows."""

    raw_text = candidate if isinstance(candidate, str) else None
    try:
        voxels = (
            decode_relative_shape_spans(candidate, grid_size)
            if isinstance(candidate, str)
            else np.asarray(candidate, dtype=np.int64)
        )
        if voxels.size == 0:
            raise ValueError("empty candidates are invalid")
        if voxels.ndim != 2 or voxels.shape[1] != 3:
            raise ValueError("voxels must have shape (N, 3)")
        if (voxels < 0).any() or (voxels >= grid_size).any():
            raise ValueError(f"voxel coordinates must lie in [0, {grid_size})")
        voxels = np.unique(voxels, axis=0)
        score, components, largest, ratio = quality_score(voxels, grid_size)
        return (
            CandidateEvaluation(
                index=index,
                valid=True,
                voxel_count=len(voxels),
                component_count=components,
                largest_component_size=largest,
                largest_component_ratio=ratio,
                score=score,
                raw_text=raw_text,
            ),
            voxels,
        )
    except (TypeError, ValueError, OverflowError) as error:
        return (
            CandidateEvaluation(
                index=index,
                valid=False,
                score=None,
                error=str(error),
                raw_text=raw_text,
            ),
            None,
        )


def select_best_candidate(
    candidates: Sequence[str | np.ndarray], grid_size: int = 32
) -> CandidateSelection:
    """Select the highest-Q valid candidate, breaking exact ties by order."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    evaluations: list[CandidateEvaluation] = []
    decoded: dict[int, np.ndarray] = {}
    for index, candidate in enumerate(candidates):
        evaluation, voxels = evaluate_candidate(candidate, index, grid_size)
        evaluations.append(evaluation)
        if voxels is not None:
            decoded[index] = voxels
    if not decoded:
        errors = "; ".join(
            f"candidate {item.index}: {item.error}" for item in evaluations
        )
        raise ValueError(f"all geometry candidates are invalid ({errors})")
    selected_index = max(
        decoded,
        key=lambda index: (
            -math.inf if evaluations[index].score is None else evaluations[index].score,
            -index,
        ),
    )
    return CandidateSelection(
        selected_index=selected_index,
        voxels=decoded[selected_index],
        evaluations=tuple(evaluations),
    )
