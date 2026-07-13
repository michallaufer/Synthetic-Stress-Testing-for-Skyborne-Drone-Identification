#!/usr/bin/env python3
"""Initialize background QA metadata with optional lightweight heuristics.

See README.md — Background QA workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.background_qa import init_background_qa_metadata, print_background_qa_report

DEFAULT_BG_DIR = PROJECT_ROOT / "data/processed/backgrounds_final/images"
DEFAULT_METADATA = PROJECT_ROOT / "data/processed/backgrounds_curated/background_metadata_curated.csv"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize background QA metadata CSV.")
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BG_DIR)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--source-group", default="manual_curated")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite manual qa_final_label / qa_final_notes / background_category",
    )
    parser.add_argument(
        "--no-heuristics",
        action="store_true",
        help="Do not run auto heuristics; default all new rows to review",
    )
    args = parser.parse_args()

    bg_dir = _resolve(args.background_dir)
    if not bg_dir.is_dir():
        fallback = Path(r"C:\datasets\backgrounds")
        if fallback.is_dir():
            print(f"Note: using fallback background dir {fallback}")
            bg_dir = fallback
        else:
            raise FileNotFoundError(f"Background dir not found: {bg_dir}")

    report = init_background_qa_metadata(
        bg_dir,
        _resolve(args.metadata_out),
        source_group=args.source_group,
        overwrite_manual=args.overwrite,
        recompute_auto=not args.no_heuristics,
    )
    print_background_qa_report(report)
    print(
        "\nManual QA: edit qa_final_label in the metadata CSV, then re-run "
        "make_background_qa_contact_sheets.py and export_sky_approved_backgrounds.py "
        "without --overwrite."
    )


if __name__ == "__main__":
    main()
