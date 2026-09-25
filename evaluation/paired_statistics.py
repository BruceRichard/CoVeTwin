#!/usr/bin/env python3
"""Auditable paired statistics for CoVeTwin per-object metric CSVs.

Given one or more per-object CSVs in the ``evaluate_metrics.py`` long format
(columns ``prediction_method``, ``gt_id``, ``status`` plus dotted metric
columns), this tool compares two methods on the objects both completed:

- per-method mean and mean paired difference per metric,
- paired bootstrap 95% CI (objects resampled with replacement),
- two-sided paired sign-flip permutation test with the Monte Carlo
  convention ``p = (exceedances + 1) / (B + 1)``,
- Holm step-down correction across the metric family.

All randomness flows from a single ``--seed`` through
``numpy.random.SeedSequence`` spawning, so identical inputs and seed produce
identical statistics.  The JSON report embeds SHA256 digests of the input
files, the seed, the UTC timestamp and the paired object count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


# The eight headline metrics of the paper table, with the direction in which
# a method is better.  Used when --metrics is not given.
DEFAULT_METRICS: list[tuple[str, str]] = [
    ("geometry.chamfer_l2_x1e3", "lower"),
    ("geometry.fscore", "higher"),
    ("scale.absolute_scale_error_cm", "lower"),
    ("articulation.joint_type_accuracy", "higher"),
    ("articulation.axis_error_deg", "lower"),
    ("articulation.origin_error_m", "lower"),
    ("articulation.motion_range_error", "lower"),
    ("executability.physics_engine_executable", "higher"),
]

DIRECTIONS = ("higher", "lower")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_per_object_csvs(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
    return rows


def paired_values(
    rows: Sequence[dict[str, str]],
    method_a: str,
    method_b: str,
    metric: str,
) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    """Inner-join two methods on gt_id over rows with status ``ok``.

    Duplicate (method, gt_id) rows (for example when several seed-run CSVs
    are passed together) are averaged; the number of collapsed duplicates is
    returned for the audit trail.
    """
    values: dict[str, dict[str, list[float]]] = {method_a: {}, method_b: {}}
    for row in rows:
        method = (row.get("prediction_method") or "").strip()
        if method not in values:
            continue
        if (row.get("status") or "").strip() != "ok":
            continue
        raw = (row.get(metric) or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        gt_id = (row.get("gt_id") or "").strip()
        values[method].setdefault(gt_id, []).append(value)

    duplicates = sum(
        len(items) - 1
        for method in values
        for items in values[method].values()
        if len(items) > 1
    )
    common = sorted(set(values[method_a]).intersection(values[method_b]))
    a = np.asarray([float(np.mean(values[method_a][g])) for g in common])
    b = np.asarray([float(np.mean(values[method_b][g])) for g in common])
    return common, a, b, duplicates


def paired_bootstrap_ci(
    diffs: np.ndarray,
    resamples: int = 10_000,
    seed: int | np.random.SeedSequence = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean paired difference."""
    diffs = np.asarray(diffs, dtype=np.float64)
    if diffs.size == 0:
        raise ValueError("bootstrap requires at least one paired difference")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diffs.size, size=(int(resamples), diffs.size))
    boot = diffs[indices].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def paired_sign_flip_permutation(
    diffs: np.ndarray,
    resamples: int = 10_000,
    seed: int | np.random.SeedSequence = 0,
) -> dict[str, Any]:
    """Two-sided paired sign-flip permutation test on the mean difference.

    Under the null each paired difference is symmetric about zero, so signs
    are flipped i.i.d.  The Monte Carlo p-value uses the conservative
    convention ``p = (exceedances + 1) / (B + 1)``; the raw value is never
    rounded so that downstream Holm correction stays exact.
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    if diffs.size == 0:
        raise ValueError("permutation test requires at least one paired difference")
    resamples = int(resamples)
    observed = float(np.abs(diffs.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(resamples, diffs.size)) * 2 - 1
    stats = np.abs((signs * diffs).mean(axis=1))
    exceedances = int(np.count_nonzero(stats >= observed))
    return {
        "observed_abs_mean_diff": observed,
        "exceedances": exceedances,
        "resamples": resamples,
        "raw_p": (exceedances + 1) / (resamples + 1),
    }


def holm_adjust(pvals: Sequence[float]) -> tuple[list[float], list[int]]:
    """Standard Holm step-down adjustment.

    Returns ``(adjusted_pvalues, ranks)`` in the original order, where rank 1
    is the smallest raw p-value.  Adjusted values are capped at 1 and made
    monotone along the sorted order (step-down cumulative maximum).
    """
    m = len(pvals)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: (pvals[i], i))
    adjusted = [0.0] * m
    ranks = [0] * m
    running = 0.0
    for position, index in enumerate(order):
        ranks[index] = position + 1
        candidate = min(1.0, (m - position) * float(pvals[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted, ranks


def parse_metric_specs(specs: Sequence[str] | None) -> list[tuple[str, str]]:
    if not specs:
        return list(DEFAULT_METRICS)
    default_directions = dict(DEFAULT_METRICS)
    parsed = []
    for spec in specs:
        if ":" in spec:
            name, direction = spec.rsplit(":", 1)
        else:
            name, direction = spec, default_directions.get(spec, "")
        if direction not in DIRECTIONS:
            raise ValueError(
                f"metric {spec!r} needs an explicit direction "
                f"({':higher'}/{':lower'})"
            )
        parsed.append((name, direction))
    return parsed


def winner(method_a: str, method_b: str, mean_diff: float, direction: str) -> str:
    if mean_diff == 0.0:
        return "tie"
    a_better = mean_diff > 0.0 if direction == "higher" else mean_diff < 0.0
    return method_a if a_better else method_b


def compare_methods(
    rows: Sequence[dict[str, str]],
    method_a: str,
    method_b: str,
    metrics: Sequence[tuple[str, str]],
    seed: int,
    permutation_resamples: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    per_metric: list[dict[str, Any]] = []
    for metric_index, (metric, direction) in enumerate(metrics):
        gt_ids, a, b, duplicates = paired_values(rows, method_a, method_b, metric)
        if not gt_ids:
            per_metric.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "error": "no paired objects with status ok",
                }
            )
            continue
        diffs = a - b
        # Independent, reproducible streams per metric and procedure.
        perm_seed, boot_seed = np.random.SeedSequence([seed, metric_index]).spawn(2)
        permutation = paired_sign_flip_permutation(
            diffs, resamples=permutation_resamples, seed=perm_seed
        )
        ci_lo, ci_hi = paired_bootstrap_ci(
            diffs, resamples=bootstrap_resamples, seed=boot_seed
        )
        per_metric.append(
            {
                "metric": metric,
                "direction": direction,
                "n_paired": len(gt_ids),
                "duplicates_averaged": duplicates,
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "mean_diff_a_minus_b": float(diffs.mean()),
                "bootstrap_ci95": [ci_lo, ci_hi],
                "bootstrap_resamples": int(bootstrap_resamples),
                "permutation": permutation,
                "raw_p": permutation["raw_p"],
                "winner": winner(method_a, method_b, float(diffs.mean()), direction),
            }
        )

    testable = [item for item in per_metric if "raw_p" in item]
    adjusted, ranks = holm_adjust([item["raw_p"] for item in testable])
    for item, adj, rank in zip(testable, adjusted, ranks):
        item["holm_adjusted_p"] = adj
        item["holm_rank"] = rank
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metrics": per_metric,
        "holm_family_size": len(testable),
    }


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def latex_table_fragment(result: dict[str, Any]) -> str:
    lines = [
        "% Auto-generated by evaluation/paired_statistics.py; IEEE style, no booktabs.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Paired comparison of "
        f"{latex_escape(result['method_a'])} vs.\\ {latex_escape(result['method_b'])}. "
        "$\\Delta$ is the mean paired difference (a $-$ b); $p$ from a two-sided "
        "sign-flip permutation test, $p_{\\mathrm{Holm}}$ after Holm correction.}",
        "\\label{tab:paired_stats}",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Metric & $\\Delta$ & 95\\% CI & $p$ & $p_{\\mathrm{Holm}}$ \\\\",
        "\\hline",
    ]
    for item in result["metrics"]:
        if "raw_p" not in item:
            lines.append(f"{latex_escape(item['metric'])} & \\multicolumn{{4}}{{c}}{{n/a}} \\\\")
            continue
        ci = item["bootstrap_ci95"]
        lines.append(
            f"{latex_escape(item['metric'])} & {item['mean_diff_a_minus_b']:.4g} & "
            f"[{ci[0]:.4g}, {ci[1]:.4g}] & {item['raw_p']:.4g} & "
            f"{item['holm_adjusted_p']:.4g} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def write_csv(path: Path, result: dict[str, Any]) -> None:
    fields = [
        "metric",
        "direction",
        "n_paired",
        "mean_a",
        "mean_b",
        "mean_diff_a_minus_b",
        "ci95_lo",
        "ci95_hi",
        "exceedances",
        "permutation_resamples",
        "raw_p",
        "holm_rank",
        "holm_adjusted_p",
        "winner",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in result["metrics"]:
            row = {key: item.get(key) for key in fields}
            if "bootstrap_ci95" in item:
                row["ci95_lo"], row["ci95_hi"] = item["bootstrap_ci95"]
            if "permutation" in item:
                row["exceedances"] = item["permutation"]["exceedances"]
                row["permutation_resamples"] = item["permutation"]["resamples"]
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        type=Path,
        help="Per-object CSV(s); repeat for multi-seed inputs.",
    )
    parser.add_argument("--method-a", required=True)
    parser.add_argument("--method-b", required=True)
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metric columns as name or name:higher/name:lower. Defaults to the eight paper metrics.",
    )
    parser.add_argument(
        "--mct",
        type=int,
        default=10_000,
        help="Monte Carlo sign-flip permutation resamples per metric.",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help="Output prefix; writes <prefix>.json, <prefix>.csv and (with --emit-latex) <prefix>.tex.",
    )
    parser.add_argument("--emit-latex", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in args.csv:
        if not path.exists():
            print(f"input CSV does not exist: {path}", file=sys.stderr)
            return 2
    try:
        metrics = parse_metric_specs(args.metrics)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rows = load_per_object_csvs(args.csv)
    result = compare_methods(
        rows,
        args.method_a,
        args.method_b,
        metrics,
        seed=args.seed,
        permutation_resamples=args.mct,
        bootstrap_resamples=args.bootstrap,
    )
    result["audit"] = {
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.csv
        ],
        "seed": args.seed,
        "permutation_resamples": args.mct,
        "bootstrap_resamples": args.bootstrap,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "p_value_convention": "two-sided sign-flip, p = (exceedances + 1) / (B + 1), unrounded",
        "multiple_testing": "Holm step-down across the metric family",
    }

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out_prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path = args.out_prefix.with_suffix(".csv")
    write_csv(csv_path, result)
    if args.emit_latex:
        args.out_prefix.with_suffix(".tex").write_text(
            latex_table_fragment(result), encoding="utf-8"
        )

    header = f"{'metric':45s} {'mean_a':>10s} {'mean_b':>10s} {'diff':>10s} {'raw_p':>10s} {'holm_p':>10s}"
    print(header)
    print("-" * len(header))
    for item in result["metrics"]:
        if "raw_p" not in item:
            print(f"{item['metric']:45s} {'n/a':>10s}")
            continue
        print(
            f"{item['metric']:45s} {item['mean_a']:10.4g} {item['mean_b']:10.4g} "
            f"{item['mean_diff_a_minus_b']:10.4g} {item['raw_p']:10.4g} "
            f"{item['holm_adjusted_p']:10.4g}"
        )
    print(f"\nWrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
