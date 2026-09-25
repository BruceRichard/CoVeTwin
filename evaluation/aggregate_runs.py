#!/usr/bin/env python3
"""Aggregate per-object metric CSVs from repeated runs (e.g. seeds 2026-2028).

Each input is a run: a per-object CSV in the ``evaluate_metrics.py`` format,
or a directory containing one (``per_object.csv`` is preferred).  For every
method and metric this tool computes the per-run mean over objects with
status ``ok`` and then aggregates across runs with mean and sample standard
deviation (``ddof=1``), so run-level variance is reported honestly instead
of a single pooled number.

Outputs a JSON report (with per-input SHA256 digests for auditability), an
aggregate CSV, a run-level CSV, and optionally an IEEE-style LaTeX table
fragment (``--emit-latex``, no booktabs).
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
from typing import Any, Sequence

import numpy as np

try:
    from evaluation.paired_statistics import DEFAULT_METRICS
except ImportError:  # allow running as a loose script from any cwd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.paired_statistics import DEFAULT_METRICS

ID_COLUMNS = {"prediction_method", "prediction_set", "sample_id", "gt_id", "status"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_inputs(inputs: Sequence[Path]) -> list[Path]:
    """Expand directories to the CSV(s) they contain."""
    resolved: list[Path] = []
    for path in inputs:
        path = Path(path)
        if path.is_dir():
            preferred = path / "per_object.csv"
            if preferred.exists():
                resolved.append(preferred)
            else:
                found = sorted(path.glob("*.csv"))
                if not found:
                    raise FileNotFoundError(f"no CSV files under directory {path}")
                resolved.extend(found)
        elif path.exists():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"input does not exist: {path}")
    return resolved


def load_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric_columns(rows: Sequence[dict[str, str]]) -> list[str]:
    """Columns, other than identifiers, that hold at least one finite float."""
    if not rows:
        return []
    candidates = [c for c in rows[0].keys() if c not in ID_COLUMNS]
    result = []
    for column in candidates:
        for row in rows:
            raw = (row.get(column) or "").strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                break
            if math.isfinite(value):
                result.append(column)
                break
    return result


def run_level_means(
    rows: Sequence[dict[str, str]], metrics: Sequence[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per method, per metric: mean over rows with status ``ok``."""
    methods = sorted({(row.get("prediction_method") or "").strip() for row in rows})
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for method in methods:
        if not method:
            continue
        method_rows = [
            row
            for row in rows
            if (row.get("prediction_method") or "").strip() == method
            and (row.get("status") or "").strip() == "ok"
        ]
        table[method] = {}
        for metric in metrics:
            values = []
            for row in method_rows:
                raw = (row.get(metric) or "").strip()
                if not raw:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if math.isfinite(value):
                    values.append(value)
            table[method][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "n_objects": len(values),
            }
    return table


def aggregate_across_runs(
    run_tables: Sequence[dict[str, dict[str, dict[str, Any]]]],
    metrics: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Mean and sample std (ddof=1) of the per-run means."""
    methods = sorted({method for table in run_tables for method in table})
    aggregate: dict[str, dict[str, dict[str, Any]]] = {}
    for method in methods:
        aggregate[method] = {}
        for metric in metrics:
            run_means = [
                table[method][metric]["mean"]
                for table in run_tables
                if method in table
                and metric in table[method]
                and table[method][metric]["mean"] is not None
            ]
            n = len(run_means)
            aggregate[method][metric] = {
                "mean": float(np.mean(run_means)) if n else None,
                "std_ddof1": float(np.std(run_means, ddof=1)) if n >= 2 else None,
                "n_runs": n,
                "run_means": [float(v) for v in run_means],
            }
    return aggregate


def aggregate(
    csv_paths: Sequence[Path],
    labels: Sequence[str],
    metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    all_rows = [load_rows(path) for path in csv_paths]
    if metrics is None:
        wanted = [name for name, _ in DEFAULT_METRICS]
        discovered: list[str] = []
        for rows in all_rows:
            for column in numeric_columns(rows):
                if column not in discovered:
                    discovered.append(column)
        metrics = [m for m in wanted if m in discovered] + [
            m for m in discovered if m not in wanted
        ]
    metrics = list(metrics)

    run_tables = [run_level_means(rows, metrics) for rows in all_rows]
    run_level = []
    for label, path, table in zip(labels, csv_paths, run_tables):
        for method in sorted(table):
            for metric in metrics:
                entry = table[method][metric]
                run_level.append(
                    {
                        "run": label,
                        "prediction_method": method,
                        "metric": metric,
                        "mean": entry["mean"],
                        "n_objects": entry["n_objects"],
                    }
                )
    return {
        "audit": {
            "inputs": [
                {"run": label, "path": str(Path(path).resolve()), "sha256": sha256_file(path)}
                for label, path in zip(labels, csv_paths)
            ],
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "aggregation": "per-run mean over status-ok objects; across-run mean and sample std (ddof=1) of run means",
        },
        "metrics": metrics,
        "run_level": run_level,
        "aggregate": aggregate_across_runs(run_tables, metrics),
    }


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def latex_table_fragment(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    methods = sorted(report["aggregate"])
    header = " & ".join(
        ["Method"] + [f"\\rotatebox{{90}}{{{latex_escape(m)}}}" for m in metrics]
    )
    lines = [
        "% Auto-generated by evaluation/aggregate_runs.py; IEEE style, no booktabs.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Mean $\\pm$ sample standard deviation across runs "
        "(per-run means over objects; std with ddof=1).}",
        "\\label{tab:aggregate_runs}",
        "\\begin{tabular}{l" + "r" * len(metrics) + "}",
        "\\hline",
        header + " \\\\",
        "\\hline",
    ]
    for method in methods:
        cells = [latex_escape(method)]
        for metric in metrics:
            entry = report["aggregate"][method][metric]
            if entry["mean"] is None:
                cells.append("--")
            elif entry["std_ddof1"] is None:
                cells.append(f"{entry['mean']:.4g}")
            else:
                cells.append(f"{entry['mean']:.4g} $\\pm$ {entry['std_ddof1']:.3g}")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def write_csvs(prefix: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    aggregate_path = prefix.with_suffix(".csv")
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["prediction_method", "metric", "mean", "std_ddof1", "n_runs", "run_means"]
        )
        for method in sorted(report["aggregate"]):
            for metric in report["metrics"]:
                entry = report["aggregate"][method][metric]
                writer.writerow(
                    [
                        method,
                        metric,
                        entry["mean"],
                        entry["std_ddof1"],
                        entry["n_runs"],
                        ";".join(format(v, ".17g") for v in entry["run_means"]),
                    ]
                )
    run_path = prefix.with_name(prefix.name + "_run_level.csv")
    with run_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["run", "prediction_method", "metric", "mean", "n_objects"]
        )
        writer.writeheader()
        writer.writerows(report["run_level"])
    return aggregate_path, run_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="Per-run per-object CSVs or directories containing them.",
    )
    parser.add_argument(
        "--run-names",
        nargs="*",
        default=None,
        help="Optional run labels; defaults to file stem / directory name.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metric columns to aggregate; defaults to auto-detected numeric columns.",
    )
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--emit-latex", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        csv_paths = resolve_inputs(args.inputs)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.run_names:
        if len(args.run_names) != len(csv_paths):
            print(
                f"--run-names count ({len(args.run_names)}) does not match resolved inputs ({len(csv_paths)})",
                file=sys.stderr,
            )
            return 2
        labels = [str(name) for name in args.run_names]
    else:
        labels = [path.parent.name if path.name == "per_object.csv" else path.stem for path in csv_paths]

    report = aggregate(csv_paths, labels, args.metrics)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out_prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    aggregate_path, run_path = write_csvs(args.out_prefix, report)
    if args.emit_latex:
        args.out_prefix.with_suffix(".tex").write_text(
            latex_table_fragment(report), encoding="utf-8"
        )

    for method in sorted(report["aggregate"]):
        print(f"{method}:")
        for metric in report["metrics"]:
            entry = report["aggregate"][method][metric]
            if entry["mean"] is None:
                continue
            std = "n/a" if entry["std_ddof1"] is None else f"{entry['std_ddof1']:.4g}"
            print(
                f"  {metric:45s} mean={entry['mean']:.6g} std={std} n_runs={entry['n_runs']}"
            )
    print(f"\nWrote {json_path}, {aggregate_path} and {run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
