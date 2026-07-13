#!/usr/bin/env python3
"""Extract RGBA foreground assets for the synthetic stress-test pipeline.

Two modes:

  1. Folder mode (toy debugging): threshold / RGBA conversion from raw class folders.
  2. Annotation mode (bridge): bbox crops or SAM2 masks from dataset annotation CSVs.

Optional SAM2 (--use-sam2 on annotate): box-prompted alpha masks with bbox fallback.
See src/drone_stress/extract.py and README.md — Asset extraction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.extract import (
    ANNOTATION_MODE_BANNER,
    ANNOTATION_SAM2_BANNER,
    EXTRACTION_METHOD_BBOX_CROP_FALLBACK,
    EXTRACTION_METHOD_SAM2,
    Sam2ExtractConfig,
    WARN_BANNER,
    build_extraction_contact_sheet,
    build_sam2_qa_contact_sheets,
    extract_assets,
    extract_assets_from_annotations,
    extraction_save_summary,
    resolve_use_class_subdirs,
    write_asset_metadata_csv,
    write_extraction_qa_report,
)

DEFAULT_CONTACT_SHEET = PROJECT_ROOT / "outputs" / "contact_sheets" / "extracted_assets_contact_sheet.png"
DEFAULT_SAM2_QA_SHEET_DIR = PROJECT_ROOT / "outputs" / "contact_sheets"
DEFAULT_QA_REPORT = PROJECT_ROOT / "outputs" / "reports" / "sam2_extraction_qa_report.txt"


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _add_folder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Source folder (e.g. data/raw/drones or data/raw/distractors)",
    )
    parser.add_argument(
        "--no-remove-bg",
        action="store_true",
        help="Only convert to RGBA; skip threshold background removal",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=240,
        help="RGB threshold for near-white background removal (default: 240)",
    )
    parser.add_argument(
        "--uniform-tolerance",
        type=int,
        default=18,
        help="Max per-channel delta from corner color for uniform-bg removal (default: 18)",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write all assets directly under output-dir (no class subfolders)",
    )


def _add_annotation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="CSV with one row per object bbox",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Root directory containing source images referenced by filename column",
    )
    parser.add_argument(
        "--class-column",
        default="class_name",
        help="CSV column for object class label (default: class_name)",
    )
    parser.add_argument(
        "--filename-column",
        default="filename",
        help="CSV column for image path relative to --image-root (default: filename)",
    )
    parser.add_argument(
        "--bbox-format",
        choices=("xywh", "xyxy"),
        default="xywh",
        help="BBox coordinate format (default: xywh)",
    )
    parser.add_argument(
        "--bbox-columns",
        nargs=4,
        metavar=("COL", "COL", "COL", "COL"),
        default=None,
        help=(
            "Four CSV column names for bbox. Default: x y w h (xywh) or x1 y1 x2 y2 (xyxy)"
        ),
    )
    parser.add_argument(
        "--source-dataset",
        default="",
        help="Optional dataset name applied to all rows in asset_metadata.csv",
    )
    parser.add_argument(
        "--source-dataset-column",
        default=None,
        help="Optional CSV column for per-row source_dataset (overrides --source-dataset)",
    )
    parser.add_argument(
        "--class-subdirs",
        action="store_true",
        help="Write drones into output-dir/<class>/ (default: flat for drone asset-type)",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write all assets flat under output-dir (overrides distractor class subfolders)",
    )
    parser.add_argument(
        "--use-sam2",
        action="store_true",
        help="Use SAM2 box prompts for segmentation masks (requires requirements-sam2.txt)",
    )
    parser.add_argument(
        "--sam2-model-size",
        choices=("tiny", "small", "base", "large"),
        default="tiny",
        help="SAM2 model size when --use-sam2 (default: tiny)",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=None,
        help="Optional local SAM2 checkpoint .pt (default: download from Hugging Face)",
    )
    parser.add_argument(
        "--sam2-device",
        default="auto",
        help="Device for SAM2: auto, cpu, cuda, or mps (default: auto)",
    )
    parser.add_argument(
        "--sam2-expand-box-ratio",
        type=float,
        default=0.10,
        help="Expand bbox before SAM2 box prompt by this fraction (default: 0.10)",
    )
    parser.add_argument(
        "--sam2-max-rows",
        type=int,
        default=None,
        help="Optional cap on annotation rows processed (pilot / dry run)",
    )
    parser.add_argument(
        "--make-contact-sheet",
        action="store_true",
        help="Save checkerboard QA contact sheet of extracted assets",
    )
    parser.add_argument(
        "--contact-sheet-output",
        type=Path,
        default=DEFAULT_CONTACT_SHEET,
        help="Contact sheet PNG path (default: outputs/contact_sheets/extracted_assets_contact_sheet.png)",
    )
    parser.add_argument(
        "--contact-sheet-samples",
        type=int,
        default=24,
        help="Number of assets on the contact sheet (default: 24)",
    )
    parser.add_argument(
        "--contact-sheet-seed",
        type=int,
        default=42,
        help="Random seed for contact sheet sampling (default: 42)",
    )
    parser.add_argument(
        "--sam2-qa-sheet-dir",
        type=Path,
        default=DEFAULT_SAM2_QA_SHEET_DIR,
        help="Directory for accept/review/reject SAM2 QA contact sheets",
    )
    parser.add_argument(
        "--qa-report-output",
        type=Path,
        default=DEFAULT_QA_REPORT,
        help="Path for SAM2 extraction QA text report",
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--asset-type",
        choices=("drone", "distractor"),
        required=True,
        help="Asset kind for metadata and default output layout",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination folder (e.g. data/processed/assets_drone)",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Path for asset_metadata.csv (default: <output-dir>/asset_metadata.csv)",
    )
    parser.add_argument(
        "--short-output-root",
        type=Path,
        default=None,
        help=(
            "Optional short path for PNG assets (metadata still under --output-dir). "
            "Use on Windows to avoid MAX_PATH errors, e.g. C:\\datasets\\coco2017\\assets"
        ),
    )


def run_folder_mode(args: argparse.Namespace) -> None:
    input_dir = _resolve_path(args.input_dir)
    output_dir = _resolve_path(args.output_dir)
    metadata_csv = args.metadata_csv or (output_dir / "asset_metadata.csv")
    metadata_csv = _resolve_path(metadata_csv) if not metadata_csv.is_absolute() else metadata_csv

    use_class_subdirs = resolve_use_class_subdirs(
        args.asset_type, input_dir, flat_output=args.flat_output
    )
    short_root = _resolve_path(args.short_output_root) if args.short_output_root else None

    print(WARN_BANNER)
    print("Mode: folder (threshold baseline)")
    print(f"Asset type: {args.asset_type}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    if short_root:
        print(f"PNG assets: {short_root} (short-output-root)")
    print(f"Layout: {'class subfolders' if use_class_subdirs else 'flat'}")

    records = extract_assets(
        input_dir,
        output_dir,
        remove_background=not args.no_remove_bg,
        white_threshold=args.white_threshold,
        uniform_tolerance=args.uniform_tolerance,
        use_class_subdirs=use_class_subdirs,
        asset_type=args.asset_type,
        short_output_root=short_root,
    )
    write_asset_metadata_csv(records, metadata_csv)
    _print_summary(records, metadata_csv)


def run_annotation_mode(args: argparse.Namespace) -> None:
    annotations = _resolve_path(args.annotations)
    image_root = _resolve_path(args.image_root)
    output_dir = _resolve_path(args.output_dir)
    metadata_csv = args.metadata_csv or (output_dir / "asset_metadata.csv")
    metadata_csv = _resolve_path(metadata_csv) if not metadata_csv.is_absolute() else metadata_csv

    if args.bbox_columns is None:
        bbox_columns = (
            ["x", "y", "w", "h"]
            if args.bbox_format == "xywh"
            else ["x1", "y1", "x2", "y2"]
        )
    else:
        bbox_columns = list(args.bbox_columns)

    use_class_subdirs = resolve_use_class_subdirs(
        args.asset_type,
        flat_output=args.flat_output,
        class_subdirs=args.class_subdirs,
    )
    short_root = _resolve_path(args.short_output_root) if args.short_output_root else None

    sam2_config = None
    if args.use_sam2:
        from drone_stress.sam2_extract import ensure_sam2_dependencies, resolve_sam2_device

        ensure_sam2_dependencies()
        checkpoint = _resolve_path(args.sam2_checkpoint) if args.sam2_checkpoint else None
        sam2_config = Sam2ExtractConfig(
            model_size=args.sam2_model_size,
            checkpoint=checkpoint,
            device=args.sam2_device,
            expand_box_ratio=args.sam2_expand_box_ratio,
            max_rows=args.sam2_max_rows,
        )
        print(ANNOTATION_SAM2_BANNER)
        resolved_device = resolve_sam2_device(args.sam2_device)
        print(
            f"Mode: annotation SAM2 mask (extraction_method=sam2_mask, "
            f"model={args.sam2_model_size}, device={resolved_device})"
        )
    else:
        print(ANNOTATION_MODE_BANNER)
        print("Mode: annotation bbox crop (extraction_method=bbox_crop)")

    print(f"Asset type: {args.asset_type}")
    print(f"Annotations: {annotations}")
    print(f"Image root: {image_root}")
    print(f"Output: {output_dir}")
    if short_root:
        print(f"PNG assets: {short_root} (short-output-root)")
    print(f"Layout: {'class subfolders' if use_class_subdirs else 'flat'}")
    print(f"BBox: {args.bbox_format} columns {bbox_columns}")
    if sam2_config and sam2_config.max_rows is not None:
        print(f"SAM2 max rows: {sam2_config.max_rows}")

    records = extract_assets_from_annotations(
        annotations,
        image_root,
        output_dir,
        asset_type=args.asset_type,
        class_column=args.class_column,
        filename_column=args.filename_column,
        bbox_format=args.bbox_format,
        bbox_columns=bbox_columns,
        source_dataset=args.source_dataset,
        source_dataset_column=args.source_dataset_column,
        use_class_subdirs=use_class_subdirs,
        flat_output=args.flat_output,
        class_subdirs=args.class_subdirs,
        sam2_config=sam2_config,
        short_output_root=short_root,
    )
    write_asset_metadata_csv(records, metadata_csv)
    _print_summary(records, metadata_csv, use_sam2=args.use_sam2)

    if args.use_sam2:
        qa_sheet_dir = _resolve_path(args.sam2_qa_sheet_dir)
        sheet_paths = build_sam2_qa_contact_sheets(
            records,
            qa_sheet_dir,
            num_samples=args.contact_sheet_samples,
            seed=args.contact_sheet_seed,
        )
        for label, path in sheet_paths.items():
            if path is not None:
                print(f"Saved SAM2 QA contact sheet ({label}): {path}")

        report_path = _resolve_path(args.qa_report_output)
        write_extraction_qa_report(records, report_path)
        print(f"Wrote QA report: {report_path}")
        print("\nReview SAM2 contact sheets before using assets in synthetic generation.")
        return

    make_sheet = args.make_contact_sheet
    if make_sheet:
        sheet_path = _resolve_path(args.contact_sheet_output)
        build_extraction_contact_sheet(
            records,
            sheet_path,
            num_samples=args.contact_sheet_samples,
            seed=args.contact_sheet_seed,
        )
        print(f"Saved contact sheet: {sheet_path}")

    print(
        "\nNext step: add --use-sam2 for SAM2 mask extraction "
        "(extraction_method=sam2_mask)."
    )


def _print_summary(records: list, metadata_csv: Path, use_sam2: bool = False) -> None:
    save_stats = extraction_save_summary(records)
    print(f"\nExtracted (save ok): {save_stats['extracted']}")
    print(f"Extraction save failed: {save_stats['extraction_failed']}")
    if save_stats["errors_by_type"]:
        print("Save failures by error:")
        for err, count in sorted(save_stats["errors_by_type"].items(), key=lambda x: -x[1]):
            print(f"  [{count}] {err}")
    print(f"Total metadata rows: {len(records)}")
    print(f"Wrote metadata: {metadata_csv}")
    with_alpha = sum(1 for r in records if r.has_alpha)
    needs_sam2 = sum(1 for r in records if r.needs_sam2_refinement)
    print(f"Assets with transparency (has_alpha=true): {with_alpha}/{len(records)}")
    if use_sam2:
        sam2_ok = sum(1 for r in records if r.sam2_used and r.extraction_method == EXTRACTION_METHOD_SAM2)
        fallback = sum(
            1 for r in records if r.extraction_method == EXTRACTION_METHOD_BBOX_CROP_FALLBACK
        )
        review = sum(1 for r in records if r.needs_manual_review)
        for label in ("accept", "review", "reject"):
            count = sum(1 for r in records if r.mask_quality_label == label)
            print(f"mask_quality_label={label}: {count}/{len(records)}")
        print(f"SAM2 masks: {sam2_ok}/{len(records)}")
        print(f"SAM2 fallback (bbox_crop_fallback): {fallback}")
        print(f"Needs manual review: {review}/{len(records)}")
    elif needs_sam2:
        print(f"Assets flagged for SAM2 refinement: {needs_sam2}/{len(records)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract RGBA PNG foreground assets. "
            "Use 'folder' for toy threshold extraction or 'annotate' for dataset bbox crops."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    folder_parser = subparsers.add_parser(
        "folder",
        help="Extract from raw image folders (threshold baseline, toy debugging)",
    )
    _add_common_args(folder_parser)
    _add_folder_args(folder_parser)

    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Extract bbox crops from annotation CSV (bridge toward real datasets)",
    )
    _add_common_args(annotate_parser)
    _add_annotation_args(annotate_parser)

    # Backward-compatible top-level flags (no subcommand)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--asset-type", choices=("drone", "distractor"), default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--class-column", default="class_name")
    parser.add_argument("--filename-column", default="filename")
    parser.add_argument("--bbox-format", choices=("xywh", "xyxy"), default="xywh")
    parser.add_argument("--bbox-columns", nargs=4, default=None)
    parser.add_argument("--source-dataset", default="")
    parser.add_argument("--source-dataset-column", default=None)
    parser.add_argument("--class-subdirs", action="store_true")
    parser.add_argument("--flat-output", action="store_true")
    parser.add_argument("--use-sam2", action="store_true")
    parser.add_argument("--sam2-model-size", choices=("tiny", "small", "base", "large"), default="tiny")
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--sam2-device", default="auto")
    parser.add_argument("--sam2-expand-box-ratio", type=float, default=0.10)
    parser.add_argument("--sam2-max-rows", type=int, default=None)
    parser.add_argument("--make-contact-sheet", action="store_true")
    parser.add_argument("--contact-sheet-output", type=Path, default=DEFAULT_CONTACT_SHEET)
    parser.add_argument("--contact-sheet-samples", type=int, default=24)
    parser.add_argument("--contact-sheet-seed", type=int, default=42)
    parser.add_argument("--sam2-qa-sheet-dir", type=Path, default=DEFAULT_SAM2_QA_SHEET_DIR)
    parser.add_argument("--qa-report-output", type=Path, default=DEFAULT_QA_REPORT)
    parser.add_argument("--short-output-root", type=Path, default=None)
    parser.add_argument("--no-remove-bg", action="store_true")
    parser.add_argument("--white-threshold", type=int, default=240)
    parser.add_argument("--uniform-tolerance", type=int, default=18)

    args = parser.parse_args()

    # Explicit subcommand
    if args.command == "folder":
        run_folder_mode(args)
        return
    if args.command == "annotate":
        run_annotation_mode(args)
        return

    # Legacy / shorthand: infer mode from flags
    if args.annotations is not None:
        if not args.image_root or not args.output_dir or not args.asset_type:
            parser.error(
                "Annotation mode requires --annotations, --image-root, --output-dir, --asset-type"
            )
        run_annotation_mode(args)
        return

    if args.input_dir is not None:
        if not args.output_dir or not args.asset_type:
            parser.error("Folder mode requires --input-dir, --output-dir, --asset-type")
        run_folder_mode(args)
        return

    parser.print_help()
    print(
        "\nExamples:\n"
        "  Folder:    python scripts/02_extract_assets.py folder --asset-type drone "
        "--input-dir data/raw/drones --output-dir data/processed/assets_drone\n"
        "  Annotate:  python scripts/02_extract_assets.py annotate --asset-type distractor "
        "--annotations data/annotations/distractors.csv --image-root data/source/images "
        "--output-dir data/processed/assets_distractors --bbox-format xywh\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
