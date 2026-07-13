#!/usr/bin/env python3
"""Generate paginated contact sheets for background QA review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.background_qa import make_background_qa_contact_sheets

DEFAULT_METADATA = PROJECT_ROOT / "data/processed/backgrounds_curated/background_metadata_curated.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/contact_sheets"
DEFAULT_BG_DIR = PROJECT_ROOT / "data/processed/backgrounds_final/images"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build background QA contact sheets.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tiles-per-page", type=int, default=48)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument(
        "--split-by-label",
        action="store_true",
        help="Also write accept/review/reject candidate sheets",
    )
    args = parser.parse_args()

    paths = make_background_qa_contact_sheets(
        _resolve(args.metadata),
        _resolve(args.output_dir),
        tiles_per_page=args.tiles_per_page,
        cols=args.cols,
        source_dir=_resolve(args.background_dir),
        split_by_label=args.split_by_label,
    )
    print(f"\nWrote {len(paths)} contact sheet(s):")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
