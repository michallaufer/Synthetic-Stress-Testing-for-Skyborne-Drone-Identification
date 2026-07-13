#!/usr/bin/env python3
"""Batch rembg alpha conversion for opaque Gemini drone extractions.

See README.md — Gemini drone alpha fix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.gemini_drone_alpha import (
    ConversionOptions,
    convert_gemini_drone_folder,
    print_conversion_summary,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert opaque Gemini drone PNGs to transparent RGBA using rembg."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder of opaque Gemini drone PNGs/JPGs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder for transparent RGBA PNG outputs",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        required=True,
        help="Metadata CSV path",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        required=True,
        help="Text report path",
    )
    parser.add_argument(
        "--make-contact-sheet",
        action="store_true",
        help="Write side-by-side original vs checkerboard QA sheet",
    )
    parser.add_argument(
        "--contact-sheet-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/contact_sheets/gemini_drones_alpha_fix_qa.png",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reconvert even when output PNG already exists",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip outputs that already exist (default: true)",
    )
    parser.add_argument(
        "--postprocess-alpha",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply alpha threshold cleanup after rembg (default: true)",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=8,
        help="Zero alpha values below this threshold after rembg (default: 8)",
    )
    parser.add_argument(
        "--min-alpha-fraction",
        type=float,
        default=0.002,
        help="Warn when nonzero alpha fraction is below this (default: 0.002)",
    )
    parser.add_argument(
        "--contact-sheet-samples",
        type=int,
        default=48,
        help="Max tiles on contact sheet (default: 48)",
    )
    args = parser.parse_args()

    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    metadata_out = _resolve(args.metadata_out)
    report_out = _resolve(args.report_out)
    contact_sheet_output = _resolve(args.contact_sheet_output)

    options = ConversionOptions(
        postprocess_alpha=args.postprocess_alpha,
        alpha_threshold=args.alpha_threshold,
        min_alpha_fraction=args.min_alpha_fraction,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing and not args.overwrite,
    )

    report = convert_gemini_drone_folder(
        input_dir,
        output_dir,
        metadata_out=metadata_out,
        report_out=report_out,
        options=options,
        max_images=args.max_images,
        make_contact_sheet=args.make_contact_sheet,
        contact_sheet_output=contact_sheet_output if args.make_contact_sheet else None,
        contact_sheet_samples=args.contact_sheet_samples,
    )

    print_conversion_summary(report)
    print(f"\nMetadata: {metadata_out}")
    print(f"Report:   {report_out}")
    if args.make_contact_sheet:
        print(f"Contact:  {contact_sheet_output}")
    print(
        "\nNext: inspect the contact sheet, then re-export final drones with:\n"
        f"  python scripts/utils/export_final_assets.py "
        f'--drones-source "{output_dir}"'
    )


if __name__ == "__main__":
    main()
