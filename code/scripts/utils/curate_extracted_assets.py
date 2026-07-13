#!/usr/bin/env python3
"""Automated QA curation for SAM2-extracted foreground assets.

Splits assets into accept / review / reject using metadata heuristics and
alpha-channel analysis. Preserves manual qa_final_label unless --overwrite.

See README.md — Asset QA curation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.asset_curation import curate_extracted_assets, print_curation_report


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate SAM2-extracted assets into accept/review/reject pools."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Source asset_metadata.csv from SAM2 extraction",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Curated output root (accept/review/reject subfolders)",
    )
    parser.add_argument(
        "--asset-type",
        choices=("drone", "bird", "airplane"),
        required=True,
        help="Asset class for class-specific QA rules",
    )
    parser.add_argument(
        "--short-assets-root",
        type=Path,
        default=None,
        help="Optional short PNG root, e.g. C:\\datasets\\assets_full\\drone",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing qa_final_label with new auto labels",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink instead of copy into curated folders (default: copy)",
    )
    parser.add_argument(
        "--make-contact-sheets",
        action="store_true",
        help="Write contact_sheet_{accept,review,reject}.png",
    )
    parser.add_argument(
        "--contact-sheet-samples",
        type=int,
        default=60,
        help="Max tiles per contact sheet (default: 60)",
    )
    args = parser.parse_args()

    metadata_path = _resolve(args.metadata)
    output_dir = _resolve(args.output_dir)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"--metadata not found: {metadata_path}")

    short_root = _resolve(args.short_assets_root) if args.short_assets_root else None

    print("Asset QA curation")
    print(f"  asset_type: {args.asset_type}")
    print(f"  metadata:   {metadata_path}")
    print(f"  output:     {output_dir}")
    if short_root:
        print(f"  short_assets_root: {short_root}")

    stats = curate_extracted_assets(
        metadata_path,
        output_dir,
        asset_type=args.asset_type,
        overwrite_final=args.overwrite,
        make_contact_sheets=args.make_contact_sheets,
        short_assets_root=short_root,
        use_symlink=args.symlink,
        contact_sheet_samples=args.contact_sheet_samples,
    )
    print_curation_report(stats)
    print(
        "\nManual override: edit qa_final_label in asset_metadata_curated.csv, "
        "then re-run without --overwrite to refresh auto columns and copies."
    )


if __name__ == "__main__":
    main()
