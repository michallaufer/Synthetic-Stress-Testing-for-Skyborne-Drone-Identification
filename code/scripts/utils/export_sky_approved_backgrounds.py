#!/usr/bin/env python3
"""Export qa_final_label=accept backgrounds to backgrounds_sky_approved/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.background_qa import export_sky_approved_backgrounds, print_sky_export_stats

DEFAULT_METADATA = PROJECT_ROOT / "data/processed/backgrounds_curated/background_metadata_curated.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/backgrounds_sky_approved"
DEFAULT_BG_DIR = PROJECT_ROOT / "data/processed/backgrounds_final/images"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sky-approved backgrounds.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BG_DIR)
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Also export qa_final_label=review (default: accept only)",
    )
    args = parser.parse_args()

    stats = export_sky_approved_backgrounds(
        _resolve(args.metadata),
        _resolve(args.output_dir),
        source_dir=_resolve(args.background_dir),
        include_review=args.include_review,
    )
    print_sky_export_stats(stats)


if __name__ == "__main__":
    main()
