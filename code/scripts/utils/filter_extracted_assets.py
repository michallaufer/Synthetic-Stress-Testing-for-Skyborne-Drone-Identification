#!/usr/bin/env python3
"""Filter extracted assets by quality_label for generation-ready pools.

Reads asset_metadata.csv, keeps rows where quality_label is not reject (and save
did not fail), optionally copies approved PNGs to a subfolder.

Manual workflow:
  1. Run extract_manual_assets_sam2.py
  2. Review qa_contact_sheet.png
  3. Set quality_label=reject on bad rows in asset_metadata.csv
  4. Run this script to summarize / export approved-only list
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.assets import load_approved_assets_from_metadata


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter extracted assets by quality_label for generation."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="asset_metadata.csv from SAM2 extraction",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="PNG folder (default: parent of metadata / images)",
    )
    parser.add_argument(
        "--exclude-labels",
        nargs="+",
        default=["reject"],
        help="quality_label values to exclude (default: reject)",
    )
    parser.add_argument(
        "--export-approved-csv",
        type=Path,
        default=None,
        help="Write approved-only metadata CSV (default: <metadata-dir>/asset_metadata_approved.csv)",
    )
    parser.add_argument(
        "--copy-approved-to",
        type=Path,
        default=None,
        help="Optional folder to copy approved PNGs into",
    )
    args = parser.parse_args()

    metadata_path = _resolve(args.metadata)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"--metadata not found: {metadata_path}")

    images_dir = args.images_dir
    if images_dir is None:
        images_dir = metadata_path.parent / "images"
    else:
        images_dir = _resolve(images_dir)

    approved_paths = load_approved_assets_from_metadata(
        images_dir,
        metadata_path,
        exclude_quality_labels=frozenset(args.exclude_labels),
    )

    df = pd.read_csv(metadata_path, keep_default_na=False)
    label_col = "quality_label" if "quality_label" in df.columns else "mask_quality_label"
    total = len(df)
    rejected = int((df[label_col].astype(str).str.lower() == "reject").sum()) if label_col in df.columns else 0
    failed = int(df.get("extraction_failed", pd.Series([False] * total)).astype(bool).sum())

    export_csv = args.export-approved_csv
    if export_csv is None:
        export_csv = metadata_path.parent / "asset_metadata_approved.csv"
    else:
        export_csv = _resolve(export_csv)

    approved_df = df[df["output_path"].astype(str).isin([str(p) for p in approved_paths])]
    approved_df.to_csv(export_csv, index=False)

    if args.copy_approved_to:
        dest_root = _resolve(args.copy_approved_to)
        dest_root.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src in approved_paths:
            dest = dest_root / src.name
            shutil.copy2(src, dest)
            copied += 1
        print(f"Copied {copied} approved PNGs to {dest_root}")

    print("\nAsset filter summary")
    print(f"  total rows: {total}")
    print(f"  rejected (quality_label): {rejected}")
    print(f"  extraction_failed: {failed}")
    print(f"  approved for generation: {len(approved_paths)}")
    print(f"  approved metadata: {export_csv}")


if __name__ == "__main__":
    main()
