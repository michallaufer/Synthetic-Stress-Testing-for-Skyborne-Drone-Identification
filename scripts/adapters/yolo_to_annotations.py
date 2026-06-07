#!/usr/bin/env python3
"""Convert a multi-class YOLO dataset (e.g. BirdvsDrone2) to internal annotation CSV.

Reads class names from data.yaml and exports all selected classes into one CSV
for scripts/02_extract_assets.py annotate.

Does not download data or run SAM2. See README.md - BirdvsDrone2 ingestion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drone_stress.yolo_adapter import (
    convert_yolo_dataset,
    parse_data_yaml_names,
    resolve_class_filter,
    slugify_dataset_name,
    write_conversion_report,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "annotations"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a multi-class YOLO dataset (data.yaml + labels) to internal "
            "annotation CSV for bbox crop extraction."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of the YOLO dataset containing data.yaml (not downloaded automatically)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output annotation CSV (default: data/processed/annotations/<slug>_annotations.csv)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Conversion report path (default: outputs/reports/<slug>_annotation_conversion_report.txt)",
    )
    parser.add_argument(
        "--source-dataset",
        required=True,
        help="Value for source_dataset column (e.g. BirdvsDrone2)",
    )
    parser.add_argument(
        "--class-filter",
        nargs="+",
        default=None,
        help="Optional class names to export (must match data.yaml names), e.g. bird drone",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "val", "test"],
        help="Split folder names to scan (default: train valid val test)",
    )
    parser.add_argument(
        "--min-bbox-px",
        type=int,
        default=4,
        help="Drop boxes whose max(w,h) is below this (default: 4)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap exported rows for pilot testing",
    )
    args = parser.parse_args()

    dataset_root = _resolve_path(args.dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"--dataset-root not found: {dataset_root}")

    slug = slugify_dataset_name(args.source_dataset)
    output_csv = _resolve_path(
        args.output_csv or (DEFAULT_OUTPUT_DIR / f"{slug}_annotations.csv")
    )
    report_path = _resolve_path(
        args.report or (DEFAULT_REPORT_DIR / f"{slug}_annotation_conversion_report.txt")
    )

    all_names = parse_data_yaml_names(dataset_root)
    class_id_to_name = resolve_class_filter(all_names, args.class_filter)

    df, stats = convert_yolo_dataset(
        dataset_root=dataset_root,
        splits=args.splits,
        class_id_to_name=class_id_to_name,
        source_dataset=args.source_dataset,
        min_bbox_px=args.min_bbox_px,
        max_rows=args.max_rows,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    write_conversion_report(
        stats,
        report_path,
        dataset_root,
        output_csv,
        title=f"{args.source_dataset} YOLO annotation conversion report",
        class_id_to_name=class_id_to_name,
    )

    print(f"Wrote {len(df)} annotations to {output_csv}")
    print(f"Wrote report: {report_path}")
    print(f"Classes exported: {', '.join(sorted(class_id_to_name.values()))}")
    print("\nNext - filter CSV by class_name, then bbox crop (not final assets until SAM2):")
    print(
        "  # drones\n"
        f"  python scripts/02_extract_assets.py annotate \\\n"
        f"    --asset-type drone \\\n"
        f"    --annotations <drone_only.csv> \\\n"
        f"    --image-root {dataset_root} \\\n"
        f"    --output-dir data/processed/assets_drone_{slug} \\\n"
        f"    --source-dataset-column source_dataset \\\n"
        f"    --bbox-format xywh --bbox-columns x y w h\n"
        "  # birds (distractors)\n"
        f"  python scripts/02_extract_assets.py annotate \\\n"
        f"    --asset-type distractor \\\n"
        f"    --annotations <bird_only.csv> \\\n"
        f"    --image-root {dataset_root} \\\n"
        f"    --output-dir data/processed/assets_distractors_{slug} \\\n"
        f"    --source-dataset-column source_dataset \\\n"
        f"    --bbox-format xywh --bbox-columns x y w h"
    )


if __name__ == "__main__":
    main()
