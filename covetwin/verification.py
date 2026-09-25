"""Structure-verified voxel candidate refinement (CoVeTwin Eqs. 15--16).

Candidate validity rules
------------------------
Every sampled candidate is checked against an explicit, ordered rule set.  The
first violated rule is recorded as a named reason string in
:attr:`CandidateEvaluation.reason`:

1. ``parse_failure`` -- the text does not contain an ``rss`` payload with an
   integer base and at least one ``delta:length`` pair (or an ndarray
   candidate does not have shape ``(N, 3)``), or the strict codec decoder
   rejects text that passed the loose checks above.
2. ``empty_voxel_set`` -- the candidate decodes to zero occupied voxels (for
   example ``rss 7``, a base with no spans, or an empty ndarray).
3. ``non_positive_length`` -- a span has ``length <= 0``.
4. ``non_monotonic_span_order`` -- span offsets are not strictly increasing.
5. ``overlapping_or_duplicate_spans`` -- a span starts at or before the end of
   a previous span (for ndarray candidates: duplicate voxel coordinates).
6. ``out_of_grid_indices`` -- the base or a reconstructed span lies outside
   ``[0, R^3)`` for the ``R^3`` grid (``R = 32`` by default).

Selection rules
---------------
:func:`select_candidate` supports six explicit candidate-selection rules:

* ``connectivity`` (default) -- the paper's verified selection: among valid
  candidates, rank by the tuple ``(rho, -c, n)`` compared *lexicographically*,
  where ``rho`` is the largest-component ratio, ``c`` the number of
  6-connected components, and ``n`` the occupied-voxel count.  Only exact ties
  in ``rho`` fall through to ``c``, and only exact ties in both fall through
  to ``n``; final ties keep the earliest generation index.  No scalar weights
  are tuned.
* ``connectivity_weighted`` -- legacy variant kept for backward
  compatibility: rank by the weighted score
  ``Q = 100*rho - 2*c + min(n,R^3)/R^3`` (former Eq. 16), ties broken by
  earliest index.  Not the default; the constants 100/2 are hand-tuned.
* ``first`` -- the first sampled candidate, valid or not.  This is the exact
  definition of the paper's "w/o Verif." ablation.
* ``first_valid`` -- the first candidate that passes all validity rules.
* ``random_valid`` -- a uniform draw among valid candidates, seeded through
  the ``seed`` argument for determinism.
* ``likelihood`` -- the valid candidate with the highest VLM log-probability
  (``scores`` argument); when no scores are provided this falls back to
  ``first_valid``.

All-invalid fallback policy
---------------------------
When the chosen rule yields no decodable candidate (for validity-aware rules:
no candidate passes validity), selection never crashes.  Instead it:

1. selects the *parseable* candidate with the most occupied voxels (spans are
   expanded liberally and clipped to the grid; ties break to the lowest
   index), marking the selection with ``fallback=True`` and
   ``fallback_reason=FALLBACK_MOST_OCCUPIED``; or
2. when no candidate is parseable at all, returns an empty-geometry sentinel
   (``selected_index == -1``, zero voxels) with
   ``fallback_reason=FALLBACK_EMPTY_SENTINEL`` so the pipeline can skip the
   part gracefully.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import random
import re
from typing import Sequence

import numpy as np

from .geometry_codec import decode_relative_shape_spans, unflatten_indices


# Named validity-rule violation reasons.
REASON_PARSE_FAILURE = "parse_failure"
REASON_EMPTY_VOXEL_SET = "empty_voxel_set"
REASON_OVERLAPPING_SPANS = "overlapping_or_duplicate_spans"
REASON_NON_MONOTONIC_SPANS = "non_monotonic_span_order"
REASON_OUT_OF_GRID = "out_of_grid_indices"
REASON_NON_POSITIVE_LENGTH = "non_positive_length"

VALIDITY_REASONS = (
    REASON_PARSE_FAILURE,
    REASON_EMPTY_VOXEL_SET,
    REASON_OVERLAPPING_SPANS,
    REASON_NON_MONOTONIC_SPANS,
    REASON_OUT_OF_GRID,
    REASON_NON_POSITIVE_LENGTH,
)

SELECTION_RULES = (
    "connectivity",
    "connectivity_weighted",
    "first",
    "first_valid",
    "random_valid",
    "likelihood",
)

FALLBACK_MOST_OCCUPIED = (
    "all candidates failed validity; selected the parseable candidate with the "
    "most occupied voxels (ties broken by lowest index)"
)
FALLBACK_EMPTY_SENTINEL = (
    "no candidate was parseable; returned the empty-geometry sentinel so the "
    "pipeline can skip this part"
)
FALLBACK_NO_CANDIDATES = (
    "no candidates were provided; returned the empty-geometry sentinel"
)


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
    reason: str | None = None
    logprob: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CandidateSelection:
    selected_index: int
    voxels: np.ndarray
    evaluations: tuple[CandidateEvaluation, ...]
    selection_rule: str = "connectivity"
    fallback: bool = False
    fallback_reason: str | None = None
    seed: int | None = None

    @property
    def selected(self) -> CandidateEvaluation | None:
        if 0 <= self.selected_index < len(self.evaluations):
            return self.evaluations[self.selected_index]
        return None

    def to_dict(self) -> dict:
        selected = self.selected
        return {
            "selected_index": self.selected_index,
            "selected_score": selected.score if selected is not None else None,
            "selection_rule": self.selection_rule,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "seed": self.seed,
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


def _salvage_from_spans(
    base: int, spans: Sequence[tuple[int, int]], grid_size: int
) -> np.ndarray | None:
    """Expand spans liberally, clipping to the grid, for fallback use."""

    upper = grid_size**3
    indices: list[int] = []
    for delta, length in spans:
        start = base + delta
        end = start + max(int(length), 0)
        low, high = max(start, 0), min(end, upper)
        if low < high:
            indices.extend(range(low, high))
    if not indices:
        return None
    return unflatten_indices(np.asarray(indices, dtype=np.int64), grid_size)


def _salvage_from_array(array: np.ndarray, grid_size: int) -> np.ndarray | None:
    if array.size == 0 or array.ndim != 2 or array.shape[1] != 3:
        return None
    clipped = np.clip(array, 0, grid_size - 1)
    unique = np.unique(clipped, axis=0)
    return unique if len(unique) else None


def _classify_text_candidate(
    text: str, grid_size: int
) -> tuple[str | None, str | None, np.ndarray | None]:
    """Apply the named validity rules to text without strict decoding.

    Returns ``(reason, message, salvage_voxels)``; ``reason`` is ``None`` when
    every loose rule passes and strict decoding should succeed.
    """

    if not isinstance(text, str):
        return REASON_PARSE_FAILURE, "candidate output must be text", None
    normalized = text.strip()
    if not normalized:
        return REASON_PARSE_FAILURE, "candidate output is empty", None
    matches = list(re.finditer(r"(?i)(?<![A-Za-z0-9_])rss(?![A-Za-z0-9_])", normalized))
    if matches:
        payload = normalized[matches[-1].end() :]
    else:
        base_match = re.search(r"(?i)\bb(?:ase)?\s*=\s*(-?\d+)", normalized)
        if not base_match:
            return REASON_PARSE_FAILURE, "missing `rss` codec marker", None
        payload = f"{base_match.group(1)} {normalized[base_match.end():]}"

    base_match = re.match(r"\s*(-?\d+)", payload)
    if not base_match:
        return REASON_PARSE_FAILURE, "missing absolute base immediately after `rss`", None
    base = int(base_match.group(1))
    tail = payload[base_match.end() :]
    spans = [
        (int(delta), int(length))
        for delta, length in re.findall(r"\(?\s*(-?\d+)\s*[:,]\s*(-?\d+)\s*\)?", tail)
    ]
    salvage = _salvage_from_spans(base, spans, grid_size) if spans else None
    if not spans:
        if tail.strip():
            return REASON_PARSE_FAILURE, "no relative delta:length spans found", None
        return (
            REASON_EMPTY_VOXEL_SET,
            "base given without spans; decoded voxel set is empty",
            None,
        )
    for position, (_, length) in enumerate(spans):
        if length <= 0:
            return (
                REASON_NON_POSITIVE_LENGTH,
                f"span {position} has a non-positive length",
                salvage,
            )
    deltas = [delta for delta, _ in spans]
    if any(later <= earlier for earlier, later in zip(deltas, deltas[1:])):
        return (
            REASON_NON_MONOTONIC_SPANS,
            "span offsets are not strictly increasing",
            salvage,
        )
    previous_end = -2
    for position, (delta, length) in enumerate(spans):
        start = base + delta
        end = start + length - 1
        if start <= previous_end:
            return (
                REASON_OVERLAPPING_SPANS,
                f"span {position} overlaps or duplicates a previous span",
                salvage,
            )
        previous_end = end
    upper = grid_size**3
    if not 0 <= base < upper:
        return REASON_OUT_OF_GRID, f"base {base} lies outside [0, {upper})", salvage
    for position, (delta, length) in enumerate(spans):
        start = base + delta
        end = start + length - 1
        if start < 0 or end >= upper:
            return (
                REASON_OUT_OF_GRID,
                f"span {position} reconstructs outside [0, {upper}): [{start}, {end}]",
                salvage,
            )
    if sum(length for _, length in spans) == 0:
        return REASON_EMPTY_VOXEL_SET, "decoded voxel set is empty", salvage
    return None, None, salvage


def _evaluate(
    candidate: str | np.ndarray,
    index: int,
    grid_size: int,
    logprob: float | None = None,
) -> tuple[CandidateEvaluation, np.ndarray | None, np.ndarray | None]:
    """Evaluate one candidate, returning (report row, voxels, salvage voxels)."""

    raw_text = candidate if isinstance(candidate, str) else None

    def invalid(
        reason: str, message: str, salvage: np.ndarray | None
    ) -> tuple[CandidateEvaluation, None, np.ndarray | None]:
        return (
            CandidateEvaluation(
                index=index,
                valid=False,
                score=None,
                error=message,
                raw_text=raw_text,
                reason=reason,
                logprob=logprob,
            ),
            None,
            salvage,
        )

    if isinstance(candidate, str):
        reason, message, salvage = _classify_text_candidate(candidate, grid_size)
        if reason is not None:
            return invalid(reason, message or reason, salvage)
        try:
            voxels = decode_relative_shape_spans(candidate, grid_size)
        except (TypeError, ValueError, OverflowError) as error:
            return invalid(REASON_PARSE_FAILURE, str(error), salvage)
    else:
        try:
            array = np.asarray(candidate, dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as error:
            return invalid(REASON_PARSE_FAILURE, str(error), None)
        salvage = _salvage_from_array(array, grid_size)
        if array.size == 0:
            return invalid(REASON_EMPTY_VOXEL_SET, "empty candidates are invalid", None)
        if array.ndim != 2 or array.shape[1] != 3:
            return invalid(REASON_PARSE_FAILURE, "voxels must have shape (N, 3)", None)
        if (array < 0).any() or (array >= grid_size).any():
            return invalid(
                REASON_OUT_OF_GRID,
                f"voxel coordinates must lie in [0, {grid_size})",
                salvage,
            )
        if len(np.unique(array, axis=0)) != len(array):
            return invalid(
                REASON_OVERLAPPING_SPANS,
                "duplicate voxel coordinates",
                salvage,
            )
        voxels = np.unique(array, axis=0)

    if voxels.size == 0:
        return invalid(REASON_EMPTY_VOXEL_SET, "empty candidates are invalid", None)
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
            logprob=logprob,
        ),
        voxels,
        voxels,
    )


def evaluate_candidate(
    candidate: str | np.ndarray, index: int = 0, grid_size: int = 32
) -> tuple[CandidateEvaluation, np.ndarray | None]:
    """Parse and score one candidate; invalid candidates become report rows."""

    evaluation, voxels, _ = _evaluate(candidate, index, grid_size)
    return evaluation, voxels


def _empty_sentinel(
    evaluations: Sequence[CandidateEvaluation],
    selection_rule: str,
    seed: int | None,
    fallback_reason: str,
) -> CandidateSelection:
    return CandidateSelection(
        selected_index=-1,
        voxels=np.empty((0, 3), dtype=np.int64),
        evaluations=tuple(evaluations),
        selection_rule=selection_rule,
        fallback=True,
        fallback_reason=fallback_reason,
        seed=seed,
    )


def select_candidate(
    candidates: Sequence[str | np.ndarray],
    grid_size: int = 32,
    selection_rule: str = "connectivity",
    seed: int | None = None,
    scores: Sequence[float | None] | None = None,
) -> CandidateSelection:
    """Select one candidate with an explicit rule; never raises on bad input.

    See the module docstring for the validity rules, the selection-rule
    semantics, and the exact all-invalid fallback policy.
    """

    if selection_rule not in SELECTION_RULES:
        raise ValueError(
            f"unknown selection rule {selection_rule!r}; expected one of {SELECTION_RULES}"
        )
    if scores is not None and len(scores) != len(candidates):
        raise ValueError("scores must have the same length as candidates")
    if not candidates:
        return _empty_sentinel((), selection_rule, seed, FALLBACK_NO_CANDIDATES)

    score_list = list(scores) if scores is not None else [None] * len(candidates)
    evaluations: list[CandidateEvaluation] = []
    decoded: dict[int, np.ndarray] = {}
    salvage: dict[int, np.ndarray] = {}
    for index, candidate in enumerate(candidates):
        evaluation, voxels, salvaged = _evaluate(
            candidate, index, grid_size, score_list[index]
        )
        evaluations.append(evaluation)
        if voxels is not None:
            decoded[index] = voxels
        if salvaged is not None and len(salvaged):
            salvage[index] = salvaged

    valid = [index for index, evaluation in enumerate(evaluations) if evaluation.valid]

    chosen: int | None = None
    if selection_rule == "first":
        # The "w/o Verif." ablation: the first sampled candidate, valid or not.
        if 0 in decoded:
            chosen = 0
    elif selection_rule == "first_valid":
        if valid:
            chosen = valid[0]
    elif selection_rule == "random_valid":
        if valid:
            chosen = random.Random(seed).choice(valid)
    elif selection_rule == "likelihood":
        scored = [index for index in valid if score_list[index] is not None]
        if scored:
            chosen = max(scored, key=lambda index: (score_list[index], -index))
        elif valid:
            # No VLM log-probabilities available: documented first_valid fallback.
            chosen = valid[0]
    elif selection_rule == "connectivity_weighted":
        # Legacy weighted ranking Q = 100*rho - 2*c + min(n,R^3)/R^3.
        if valid:
            chosen = max(
                valid, key=lambda index: (evaluations[index].score, -index)
            )
    else:  # connectivity: true lexicographic ranking on (rho, -c, n)
        if valid:
            chosen = max(
                valid,
                key=lambda index: (
                    evaluations[index].largest_component_ratio,
                    -evaluations[index].component_count,
                    evaluations[index].voxel_count,
                    -index,
                ),
            )

    if chosen is not None:
        return CandidateSelection(
            selected_index=chosen,
            voxels=decoded[chosen],
            evaluations=tuple(evaluations),
            selection_rule=selection_rule,
            seed=seed,
        )

    # All-invalid fallback: never crash.  Prefer the parseable candidate with
    # the most occupied voxels; otherwise emit the empty-geometry sentinel.
    if salvage:
        chosen = max(salvage, key=lambda index: (len(salvage[index]), -index))
        return CandidateSelection(
            selected_index=chosen,
            voxels=salvage[chosen],
            evaluations=tuple(evaluations),
            selection_rule=selection_rule,
            fallback=True,
            fallback_reason=FALLBACK_MOST_OCCUPIED,
            seed=seed,
        )
    return _empty_sentinel(evaluations, selection_rule, seed, FALLBACK_EMPTY_SENTINEL)


def select_best_candidate(
    candidates: Sequence[str | np.ndarray], grid_size: int = 32
) -> CandidateSelection:
    """Select the highest-Q valid candidate, breaking exact ties by order.

    Legacy strict wrapper: identical ranking to
    ``select_candidate(..., selection_rule="connectivity_weighted")`` but
    raises when every candidate is invalid instead of applying the fallback
    policy.  For the paper's lexicographic structure priority, use
    ``selection_rule="connectivity"``.
    """

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
