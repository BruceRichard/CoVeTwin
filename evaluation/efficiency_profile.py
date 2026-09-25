#!/usr/bin/env python3
"""Token, latency and memory profiling for CoVeTwin geometry strings.

(a) Token accounting for the serialized geometry formats produced by
``covetwin/ablation_codecs.py`` (``rss`` relative spans, ``vox``, ``idx``
and ``asp``).  Counts come from a HuggingFace tokenizer when one is named
(lazy ``transformers`` import; a clear error is raised when it is missing or
the tokenizer cannot be loaded) or from a deterministic
whitespace/punctuation fallback that needs no third-party package.  Per-part
(each input file is one part's geometry string) and aggregate
mean/median/P95/max statistics are reported.

(b) Timing/memory helpers: the ``profile_run`` context manager captures wall
time and, when torch with CUDA is present, ``torch.cuda.max_memory_allocated``;
it degrades to wall-time-only on CPU-only hosts.

The CLI profiles a directory of geometry text files without any model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


# Integers/decimals, identifier-ish words, and individual punctuation marks.
# This mirrors how subword tokenizers split the codec formats (digits glued
# within a number, `:`/`,`/`,` separators as their own tokens) closely enough
# for a tokenizer-free estimate, and is fully deterministic.
FALLBACK_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[A-Za-z_]+|[^\sA-Za-z0-9_]")

FORMAT_MARKERS = {
    "rss": "relative_span",
    "vox": "voxel",
    "idx": "index",
    "asp": "absolute_span",
}


def fallback_token_count(text: str) -> int:
    """Tokenizer-free count: numbers, words and punctuation each count once."""
    return len(FALLBACK_TOKEN_RE.findall(text))


def load_hf_tokenizer(name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace token counting requires `transformers`, which is not "
            "installed. Install it (for example `pip install --user "
            "transformers`) or omit --tokenizer to use the deterministic "
            "whitespace/punctuation fallback."
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load HuggingFace tokenizer {name!r}: {exc}"
        ) from exc


def count_tokens(text: str, tokenizer: Any = None) -> int:
    if tokenizer is None:
        return fallback_token_count(text)
    return int(len(tokenizer.encode(text, add_special_tokens=False)))


def detect_format(text: str) -> str:
    match = re.search(r"(?i)(?<![A-Za-z0-9_])(rss|vox|idx|asp)(?![A-Za-z0-9_])", text)
    return FORMAT_MARKERS[match.group(1).lower()] if match else "unknown"


def token_stats(counts: Sequence[int]) -> dict[str, Any]:
    if not counts:
        return {"n": 0}
    array = np.asarray(counts, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": int(array.max()),
        "min": int(array.min()),
        "total": int(array.sum()),
    }


def profile_texts(
    texts: Sequence[str], names: Sequence[str], tokenizer: Any = None
) -> dict[str, Any]:
    per_part = []
    for name, text in zip(names, texts):
        per_part.append(
            {
                "name": name,
                "format": detect_format(text),
                "tokens": count_tokens(text, tokenizer),
                "counter": "huggingface" if tokenizer is not None else "fallback_regex",
            }
        )
    by_format: dict[str, list[int]] = {}
    for item in per_part:
        by_format.setdefault(item["format"], []).append(item["tokens"])
    return {
        "counter": "huggingface" if tokenizer is not None else "fallback_regex",
        "aggregate": token_stats([item["tokens"] for item in per_part]),
        "per_format": {
            fmt: token_stats(counts) for fmt, counts in sorted(by_format.items())
        },
        "per_part": per_part,
    }


def profile_directory(
    paths: Sequence[Path], pattern: str = "*.txt", tokenizer: Any = None
) -> dict[str, Any]:
    files: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.extend(sorted(path.glob(pattern)))
        elif path.exists():
            files.append(path)
        else:
            raise FileNotFoundError(f"input does not exist: {path}")
    texts = [path.read_text(encoding="utf-8") for path in files]
    names = [str(path) for path in files]
    return profile_texts(texts, names, tokenizer)


class profile_run:
    """Context manager capturing wall time and peak CUDA memory.

    ``peak_cuda_memory_bytes`` is ``None`` when torch or CUDA is unavailable,
    in which case the record is wall-time-only by design.
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.wall_time_s: float | None = None
        self.peak_cuda_memory_bytes: int | None = None
        self.cuda_available = False
        self._torch = None

    def __enter__(self) -> "profile_run":
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.cuda.is_available():
            self._torch = torch
            self.cuda_available = True
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._torch is not None:
            self._torch.cuda.synchronize()
            self.peak_cuda_memory_bytes = int(self._torch.cuda.max_memory_allocated())
        self.wall_time_s = time.perf_counter() - self._start
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "wall_time_s": self.wall_time_s,
            "peak_cuda_memory_bytes": self.peak_cuda_memory_bytes,
            "cuda_available": self.cuda_available,
        }


def profile_callable(fn: Callable, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run ``fn`` under ``profile_run`` and return result plus measurements."""
    with profile_run(getattr(fn, "__name__", "callable")) as profile:
        result = fn(*args, **kwargs)
    record = profile.as_dict()
    record["result"] = result
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Geometry text files or directories of them (one file = one part).",
    )
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HuggingFace tokenizer name/path; omit for the deterministic fallback counter.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = None
    if args.tokenizer:
        try:
            tokenizer = load_hf_tokenizer(args.tokenizer)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    try:
        report = profile_directory(args.inputs, args.pattern, tokenizer)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.tokenizer:
        report["tokenizer"] = args.tokenizer
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    aggregate = report["aggregate"]
    if aggregate.get("n"):
        print(
            f"parts={aggregate['n']} mean={aggregate['mean']:.1f} "
            f"median={aggregate['median']:.1f} p95={aggregate['p95']:.1f} "
            f"max={aggregate['max']} (counter={report['counter']})"
        )
        for fmt, stats in report["per_format"].items():
            print(
                f"  {fmt:15s} n={stats['n']:4d} mean={stats['mean']:.1f} p95={stats['p95']:.1f}"
            )
    else:
        print("no input files matched", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
