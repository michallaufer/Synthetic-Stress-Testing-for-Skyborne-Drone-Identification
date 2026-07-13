#!/usr/bin/env python3
"""Validate canonical final asset pack folders and metadata consistency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.final_asset_pack import print_validation_report, validate_final_asset_pack


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate assets_final / backgrounds_final.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data/processed",
        help="Root containing assets_final/ and backgrounds_final/",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/final_asset_pack_validation.txt",
        help="Optional text report output path",
    )
    args = parser.parse_args()

    report = validate_final_asset_pack(
        _resolve(args.processed_root),
        report_path=_resolve(args.report),
    )
    print_validation_report(report)
    if args.report:
        print(f"\nWrote report: {_resolve(args.report)}")
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
