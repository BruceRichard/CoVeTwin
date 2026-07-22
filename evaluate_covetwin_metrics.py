#!/usr/bin/env python3
"""Repository-level entry point for the unified CoVeTwin evaluator."""

import sys

from evaluation.evaluate_metrics import main


if __name__ == "__main__":
    if not any(
        argument == "--output-dir" or argument.startswith("--output-dir=")
        for argument in sys.argv[1:]
    ):
        sys.argv.extend(("--output-dir", "evaluation_results/covetwin_metrics"))
    raise SystemExit(main())
