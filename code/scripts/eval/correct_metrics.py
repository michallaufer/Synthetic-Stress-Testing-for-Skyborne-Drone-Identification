#!/usr/bin/env python3
"""Point to the corrected final-clean metrics CSV (GitHub-published).

The Colab evaluator originally wrote an uncorrected summary that overestimated
YOLO recall. Use the CORRECTED file under results/combined_metrics/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORRECTED = (
    PROJECT_ROOT
    / "results"
    / "combined_metrics"
    / "summary_by_model_by_eval_subset_CORRECTED.csv"
)
UNCORRECTED = (
    PROJECT_ROOT
    / "results"
    / "combined_metrics"
    / "summary_by_model_by_eval_subset.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show / copy the corrected final-clean metrics table."
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the corrected CSV to stdout.",
    )
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="Optional destination path to copy the corrected CSV.",
    )
    args = parser.parse_args()

    if not CORRECTED.is_file():
        print(f"Missing corrected metrics: {CORRECTED}", file=sys.stderr)
        print(
            "Import Colab zips or copy the CSV into results/combined_metrics/.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Authoritative metrics: {CORRECTED}")
    if UNCORRECTED.is_file():
        print(f"Do NOT report from: {UNCORRECTED}")

    if args.print:
        sys.stdout.write(CORRECTED.read_text(encoding="utf-8"))
    if args.copy_to is not None:
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CORRECTED, args.copy_to)
        print(f"Copied to {args.copy_to}")


if __name__ == "__main__":
    main()
