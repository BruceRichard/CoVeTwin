"""Core CoVeTwin geometry representation and verification utilities.

The package is deliberately independent of the heavyweight VLM and TRELLIS
dependencies.  Importing :mod:`covetwin` is therefore safe in data-generation,
evaluation, and unit-test environments.
"""

from .geometry_codec import (
    CODEC_NAME,
    RelativeShapeSpans,
    decode_relative_shape_spans,
    encode_relative_shape_spans,
    flatten_voxels,
    parse_relative_shape_spans,
    serialize_relative_shape_spans,
    unflatten_indices,
)
from .verification import (
    CandidateEvaluation,
    CandidateSelection,
    evaluate_candidate,
    select_best_candidate,
)

__all__ = [
    "CODEC_NAME",
    "RelativeShapeSpans",
    "CandidateEvaluation",
    "CandidateSelection",
    "decode_relative_shape_spans",
    "encode_relative_shape_spans",
    "evaluate_candidate",
    "flatten_voxels",
    "parse_relative_shape_spans",
    "select_best_candidate",
    "serialize_relative_shape_spans",
    "unflatten_indices",
]
