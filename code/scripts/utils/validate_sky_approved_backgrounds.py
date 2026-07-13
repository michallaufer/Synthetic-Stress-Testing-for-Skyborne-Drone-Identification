#!/usr/bin/env python3
"""Validate exported sky-approved background pack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.background_qa import print_sky_validation, validate_sky_approved_backgrounds

DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/backgrounds_sky_approved"
DEFAULT_REPORT = PROJECT_ROOT / "outputs/reports/sky_approved_backgrounds_validation.txt"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate backgrounds_sky_approved pack.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = validate_sky_approved_backgrounds(
        _resolve(args.output_dir),
        min_count=args.min_count,
        report_path=_resolve(args.report),
    )
    print_sky_validation(report)
    if args.report:
        print(f"\nWrote report: {_resolve(args.report)}")
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
