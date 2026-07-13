#!/usr/bin/env python3
"""Convert a YOLO detection dataset (e.g. Seraphim) to internal annotation CSV.

Single-class convenience wrapper around shared YOLO adapter logic.
For multi-class datasets (e.g. BirdvsDrone2), use yolo_to_annotations.py instead.

Does not download data or run SAM2. See README.md - Seraphim ingestion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.yolo_adapter import convert_yolo_dataset, write_conversion_report

DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "seraphim_annotation_conversion_report.txt"
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "processed" / "annotations" / "seraphim_drone_annotations.csv"
)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert YOLO labels (Seraphim-style, single class) to internal annotation CSV."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--class-name", default="drone")
    parser.add_argument("--source-dataset", default="Seraphim")
    parser.add_argument("--target-class-id", type=int, default=0)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "valid", "test"],
    )
    parser.add_argument("--min-bbox-px", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    dataset_root = _resolve_path(args.dataset_root)
    output_csv = _resolve_path(args.output_csv)
    report_path = _resolve_path(args.report)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"--dataset-root not found: {dataset_root}")

    class_id_to_name = {args.target_class_id: args.class_name}
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
        title="Seraphim / YOLO annotation conversion report",
        class_id_to_name=class_id_to_name,
    )

    print(f"Wrote {len(df)} annotations to {output_csv}")
    print(f"Wrote report: {report_path}")
    print("\nNext step - bbox crop extraction (not final assets until SAM2):")
    print(
        f"  python scripts/02_extract_assets.py annotate \\\n"
        f"    --asset-type drone \\\n"
        f"    --annotations {output_csv.relative_to(PROJECT_ROOT)} \\\n"
        f"    --image-root {dataset_root.relative_to(PROJECT_ROOT)} \\\n"
        f"    --output-dir data/processed/assets_drone_seraphim \\\n"
        f"    --source-dataset-column source_dataset \\\n"
        f"    --bbox-format xywh --bbox-columns x y w h"
    )


if __name__ == "__main__":
    main()
