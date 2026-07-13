#!/usr/bin/env python3
"""SAM2 extraction from unlabeled manual foreground folders (drones / birds / airplanes).

Proposes a bounding box per image (YOLO pretrained when ultralytics is installed,
otherwise a centered heuristic), runs SAM2 mask extraction, and writes transparent PNGs
with QA metadata and contact sheets.

See README.md — Manual SAM2 asset extraction (Mode C).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.bbox_proposal import YoloProposalService
from drone_stress.extract import Sam2ExtractConfig
from drone_stress.manual_sam2_extract import run_manual_extraction_pipeline
from drone_stress.sam2_extract import ensure_sam2_dependencies, resolve_sam2_device


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SAM2 RGBA assets from manual raw object folders."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root, e.g. data/processed/assets_pilot/drones_sam2",
    )
    parser.add_argument(
        "--asset-type",
        choices=("drone", "bird", "airplane"),
        required=True,
    )
    parser.add_argument("--use-sam2", action="store_true")
    parser.add_argument(
        "--sam2-model-size",
        choices=("tiny", "small", "base", "large"),
        default="tiny",
    )
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--sam2-device", default="auto")
    parser.add_argument("--sam2-expand-box-ratio", type=float, default=0.10)
    parser.add_argument("--source-dataset", default="manual_curated")
    parser.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help="Cap images processed (pilot runs, e.g. 10)",
    )
    parser.add_argument("--min-mask-area", type=int, default=0)
    parser.add_argument("--min-yolo-conf", type=float, default=0.15)
    parser.add_argument("--heuristic-coverage", type=float, default=0.72)
    parser.add_argument("--make-contact-sheet", action="store_true")
    parser.add_argument("--contact-sheet-output", type=Path, default=None)
    parser.add_argument("--contact-sheet-samples", type=int, default=48)
    parser.add_argument("--qa-report-output", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Write per-image debug panels under <output-dir>/debug/",
    )
    parser.add_argument(
        "--short-output-root",
        type=Path,
        default=None,
        help=(
            "Optional short path for PNG assets (Windows MAX_PATH). "
            "Metadata/contact sheets stay under --output-dir. "
            "Example: C:\\datasets\\assets_pilot"
        ),
    )
    args = parser.parse_args()

    if not args.use_sam2:
        parser.error("This script requires --use-sam2 (SAM2 mask extraction).")

    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input-dir not found: {input_dir}")

    ensure_sam2_dependencies()
    device = resolve_sam2_device(args.sam2_device)
    checkpoint = _resolve(args.sam2_checkpoint) if args.sam2_checkpoint else None

    sam2_config = Sam2ExtractConfig(
        model_size=args.sam2_model_size,
        checkpoint=checkpoint,
        device=args.sam2_device,
        expand_box_ratio=args.sam2_expand_box_ratio,
    )

    yolo_pre = YoloProposalService.startup_status()
    print("Manual SAM2 asset extraction")
    print(f"  asset_type: {args.asset_type}")
    print(f"  input:  {input_dir}")
    print(f"  output: {output_dir}")
    print(f"  SAM2:   {args.sam2_model_size} on {device}")
    print(f"  YOLO installed: {'yes' if yolo_pre.installed else 'no'}")
    if args.max_assets:
        print(f"  max_assets: {args.max_assets}")
    if args.short_output_root:
        print(f"  short_output_root: {_resolve(args.short_output_root)}")

    stats = run_manual_extraction_pipeline(
        input_dir,
        output_dir,
        asset_type=args.asset_type,
        sam2_config=sam2_config,
        metadata_csv=_resolve(args.metadata_csv) if args.metadata_csv else None,
        contact_sheet_output=_resolve(args.contact_sheet_output)
        if args.contact_sheet_output
        else None,
        qa_report_output=_resolve(args.qa_report_output) if args.qa_report_output else None,
        make_contact_sheet=args.make_contact_sheet,
        contact_sheet_samples=args.contact_sheet_samples,
        source_dataset=args.source_dataset,
        max_assets=args.max_assets,
        min_mask_area=args.min_mask_area,
        min_yolo_conf=args.min_yolo_conf,
        heuristic_coverage=args.heuristic_coverage,
        save_debug=args.save_debug,
        short_output_root=_resolve(args.short_output_root) if args.short_output_root else None,
    )

    print(f"\nExtracted: {stats['total']} assets ({stats['saved_ok']} saved ok)")
    print(f"Save failures: {stats['save_failed']}")
    print(f"YOLO installed: {stats.get('yolo_installed')}")
    print(f"YOLO model loaded: {stats.get('yolo_model_loaded')}")
    print("Quality labels:")
    for label, count in stats["quality_counts"].items():
        print(f"  {label}: {count}")
    print(f"\nImages:   {stats['images_dir']}")
    print(f"Metadata: {stats['metadata_csv']}")
    if stats["contact_sheet"]:
        print(f"Contact:  {stats['contact_sheet']}")
    print(f"Report:   {stats['qa_report']}")


if __name__ == "__main__":
    main()
