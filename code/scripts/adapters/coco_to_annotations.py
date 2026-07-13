#!/usr/bin/env python3
"""Convert COCO instance annotations to internal annotation CSV.

Exports selected categories (e.g. bird, airplane, kite) for distractor asset
extraction via scripts/02_extract_assets.py annotate.

Does not download COCO, run SAM2, or generate synthetic images.
See README.md — COCO val2017 distractor adapter.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "processed" / "annotations" / "coco_val2017_distractors_annotations.csv"
)
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "coco_val2017_annotation_conversion_report.txt"

OUTPUT_COLUMNS = [
    "filename",
    "image_path",
    "source_dataset",
    "split",
    "class_name",
    "class_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_format",
    "image_width",
    "image_height",
    "area",
    "iscrowd",
    "annotation_id",
    "image_id",
    "supercategory",
]

# COCO 2017 val — common distractor ids (overridden by JSON lookup)
COCO_CATEGORY_ALIASES: dict[str, str] = {
    "airplane": "airplane",
    "aeroplane": "airplane",
    "bird": "bird",
    "kite": "kite",
}


@dataclass
class CocoConversionStats:
    images_scanned: int = 0
    annotations_scanned: int = 0
    annotations_exported: int = 0
    skipped_image_missing: int = 0
    skipped_bbox_too_small: int = 0
    skipped_bbox_too_large: int = 0
    skipped_iscrowd: int = 0
    skipped_category_not_requested: int = 0
    skipped_invalid_bbox: int = 0
    skipped_other: int = 0
    by_class: dict[str, int] = field(default_factory=dict)
    bbox_max_dims: list[float] = field(default_factory=list)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize_class_name(name: str) -> str:
    key = name.strip().lower()
    return COCO_CATEGORY_ALIASES.get(key, key)


def _resolve_categories(
    categories: list[dict],
    requested_names: list[str],
) -> dict[int, dict[str, str]]:
    """Map COCO category_id -> {class_name, supercategory} for requested names."""
    wanted = {_normalize_class_name(n) for n in requested_names}
    selected: dict[int, dict[str, str]] = {}
    by_normalized: dict[str, list[dict]] = {}

    for cat in categories:
        raw_name = str(cat.get("name", ""))
        norm = _normalize_class_name(raw_name)
        by_normalized.setdefault(norm, []).append(cat)

    missing = wanted - set(by_normalized)
    if missing:
        available = sorted({_normalize_class_name(c["name"]) for c in categories})
        raise ValueError(
            f"Unknown --categories: {sorted(missing)}. "
            f"Available in JSON: {available}"
        )

    for norm in sorted(wanted):
        # Prefer exact name match when multiple COCO categories normalize the same
        cats = by_normalized[norm]
        cat = next((c for c in cats if _normalize_class_name(c["name"]) == norm), cats[0])
        cid = int(cat["id"])
        selected[cid] = {
            "class_name": norm,
            "supercategory": str(cat.get("supercategory", "")),
        }
    return selected


def _image_path_relative(file_name: str) -> str:
    return Path(file_name).as_posix()


def convert_coco_instances(
    annotation_json: Path,
    image_root: Path,
    *,
    category_names: list[str],
    source_dataset: str,
    split: str,
    min_bbox_px: int,
    max_bbox_px: int | None,
    max_rows: int | None,
    include_iscrowd: bool,
) -> tuple[pd.DataFrame, CocoConversionStats]:
    with annotation_json.open(encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    selected_cats = _resolve_categories(categories, category_names)
    selected_ids = set(selected_cats)

    image_by_id: dict[int, dict] = {int(img["id"]): img for img in images}
    stats = CocoConversionStats()
    stats.images_scanned = len(images)

    rows: list[dict] = []

    for ann in annotations:
        if max_rows is not None and stats.annotations_exported >= max_rows:
            break

        category_id = int(ann.get("category_id", -1))
        if category_id not in selected_ids:
            stats.skipped_category_not_requested += 1
            continue

        stats.annotations_scanned += 1
        cat_info = selected_cats[category_id]
        class_name = cat_info["class_name"]

        iscrowd = int(ann.get("iscrowd", 0))
        if iscrowd != 0 and not include_iscrowd:
            stats.skipped_iscrowd += 1
            continue

        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            stats.skipped_invalid_bbox += 1
            continue

        x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if w <= 0 or h <= 0:
            stats.skipped_invalid_bbox += 1
            continue

        max_dim = max(w, h)
        if max_dim < min_bbox_px:
            stats.skipped_bbox_too_small += 1
            continue
        if max_bbox_px is not None and max_dim > max_bbox_px:
            stats.skipped_bbox_too_large += 1
            continue

        image_id = int(ann["image_id"])
        img_rec = image_by_id.get(image_id)
        if img_rec is None:
            stats.skipped_image_missing += 1
            continue

        file_name = str(img_rec["file_name"])
        rel_path = _image_path_relative(file_name)
        abs_image = image_root / file_name
        if not abs_image.is_file():
            stats.skipped_image_missing += 1
            continue

        img_w = int(img_rec.get("width", 0))
        img_h = int(img_rec.get("height", 0))
        if img_w <= 0 or img_h <= 0:
            stats.skipped_other += 1
            continue

        if x < 0 or y < 0 or x + w > img_w + 0.5 or y + h > img_h + 0.5:
            stats.skipped_invalid_bbox += 1
            continue

        rows.append(
            {
                "filename": rel_path,
                "image_path": rel_path,
                "source_dataset": source_dataset,
                "split": split,
                "class_name": class_name,
                "class_id": category_id,
                "bbox_x": int(round(x)),
                "bbox_y": int(round(y)),
                "bbox_w": int(round(w)),
                "bbox_h": int(round(h)),
                "bbox_format": "xywh",
                "image_width": img_w,
                "image_height": img_h,
                "area": float(ann.get("area", w * h)),
                "iscrowd": iscrowd,
                "annotation_id": int(ann["id"]),
                "image_id": image_id,
                "supercategory": cat_info["supercategory"],
            }
        )
        stats.annotations_exported += 1
        stats.by_class[class_name] = stats.by_class.get(class_name, 0) + 1
        stats.bbox_max_dims.append(max_dim)

    if not rows:
        raise ValueError(
            "No annotations exported. Check --categories, --image-root, "
            "--min-bbox-px, and that val images are present."
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), stats


def write_coco_report(
    stats: CocoConversionStats,
    report_path: Path,
    *,
    annotation_json: Path,
    image_root: Path,
    output_csv: Path,
    category_names: list[str],
    source_dataset: str,
    split: str,
    min_bbox_px: int,
    max_bbox_px: int | None,
) -> None:
    dims = stats.bbox_max_dims
    if dims:
        min_dim = int(min(dims))
        max_dim = int(max(dims))
        median_dim = int(statistics.median(dims))
    else:
        min_dim = max_dim = median_dim = 0

    lines = [
        "COCO annotation conversion report",
        f"annotation_json: {annotation_json.resolve()}",
        f"image_root: {image_root.resolve()}",
        f"output_csv: {output_csv.resolve()}",
        f"source_dataset: {source_dataset}",
        f"split: {split}",
        f"categories: {', '.join(category_names)}",
        f"min_bbox_px: {min_bbox_px}",
        f"max_bbox_px: {max_bbox_px if max_bbox_px is not None else '(none)'}",
        "",
        f"images_scanned: {stats.images_scanned}",
        f"annotations_scanned: {stats.annotations_scanned}",
        f"annotations_exported: {stats.annotations_exported}",
        "",
        "exported_by_class_name:",
    ]
    for cls in sorted(stats.by_class):
        lines.append(f"  {cls}: {stats.by_class[cls]}")
    lines.extend(
        [
            "",
            "skipped:",
            f"  image_missing: {stats.skipped_image_missing}",
            f"  bbox_too_small: {stats.skipped_bbox_too_small}",
            f"  bbox_too_large: {stats.skipped_bbox_too_large}",
            f"  iscrowd: {stats.skipped_iscrowd}",
            f"  invalid_bbox: {stats.skipped_invalid_bbox}",
            f"  category_not_requested: {stats.skipped_category_not_requested}",
            f"  other: {stats.skipped_other}",
            "",
            "bbox max(width,height) px - min / median / max:",
            f"  {min_dim} / {median_dim} / {max_dim}",
        ]
    )
    if stats.skipped_image_missing or stats.skipped_bbox_too_small:
        lines.append("")
        lines.append(
            "Note: skipped counts are per-annotation; one image can contribute "
            "multiple rows."
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(stats: CocoConversionStats, category_names: list[str]) -> None:
    print(f"images scanned: {stats.images_scanned}")
    print(f"annotations scanned: {stats.annotations_scanned}")
    print(f"annotations exported: {stats.annotations_exported}")
    print("\nexported by class_name:")
    for name in category_names:
        norm = _normalize_class_name(name)
        count = stats.by_class.get(norm, 0)
        if count:
            print(f"  {norm}: {count}")
    for cls in sorted(stats.by_class):
        if _normalize_class_name(cls) not in {_normalize_class_name(n) for n in category_names}:
            print(f"  {cls}: {stats.by_class[cls]}")
    print(f"\nskipped image missing: {stats.skipped_image_missing}")
    print(f"skipped bbox too small: {stats.skipped_bbox_too_small}")
    if stats.skipped_bbox_too_large:
        print(f"skipped bbox too large: {stats.skipped_bbox_too_large}")
    print(f"skipped iscrowd: {stats.skipped_iscrowd}")

    dims = stats.bbox_max_dims
    if dims:
        print(
            "\nbbox max(width,height) px — min / median / max: "
            f"{int(min(dims))} / {int(statistics.median(dims))} / {int(max(dims))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert COCO instance JSON to internal annotation CSV (distractors)."
    )
    parser.add_argument(
        "--annotation-json",
        type=Path,
        required=True,
        help="COCO instances JSON (e.g. instances_val2017.json)",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Folder containing COCO images (e.g. val2017/)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output annotation CSV",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["bird", "airplane", "kite"],
        help="COCO category names to export (default: bird airplane kite)",
    )
    parser.add_argument(
        "--source-dataset",
        default="COCO_val2017",
        help="Value for source_dataset column (default: COCO_val2017)",
    )
    parser.add_argument(
        "--split",
        default="val2017",
        help="Value for split column (default: val2017)",
    )
    parser.add_argument(
        "--min-bbox-px",
        type=int,
        default=8,
        help="Drop boxes whose max(w,h) is below this (default: 8)",
    )
    parser.add_argument(
        "--max-bbox-px",
        type=int,
        default=None,
        help="Optional: drop boxes whose max(w,h) exceeds this",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap exported rows for pilot testing",
    )
    parser.add_argument(
        "--include-iscrowd",
        action="store_true",
        help="Include iscrowd=1 annotations (default: skip crowd segments)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Conversion report path",
    )
    parser.add_argument(
        "--make-report",
        action="store_true",
        help="Write conversion report (default report path if --report omitted)",
    )
    args = parser.parse_args()

    annotation_json = _resolve_path(args.annotation_json)
    image_root = _resolve_path(args.image_root)
    output_csv = _resolve_path(args.output_csv)
    report_path = _resolve_path(args.report)

    if not annotation_json.is_file():
        raise FileNotFoundError(f"--annotation-json not found: {annotation_json}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"--image-root not found: {image_root}")

    df, stats = convert_coco_instances(
        annotation_json,
        image_root,
        category_names=args.categories,
        source_dataset=args.source_dataset,
        split=args.split,
        min_bbox_px=args.min_bbox_px,
        max_bbox_px=args.max_bbox_px,
        max_rows=args.max_rows,
        include_iscrowd=args.include_iscrowd,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if args.make_report:
        write_coco_report(
            stats,
            report_path,
            annotation_json=annotation_json,
            image_root=image_root,
            output_csv=output_csv,
            category_names=args.categories,
            source_dataset=args.source_dataset,
            split=args.split,
            min_bbox_px=args.min_bbox_px,
            max_bbox_px=args.max_bbox_px,
        )
        print(f"Wrote report: {report_path}")

    print(f"\nWrote {len(df)} annotations to {output_csv}")
    print_summary(stats, args.categories)

    print(
        "\nNext — optional scene filter, then bbox/SAM2 extraction (distractors):\n"
        "  python scripts/06_filter_flying_object_scenes.py \\\n"
        f"    --annotations {output_csv} \\\n"
        f"    --image-root {image_root} \\\n"
        "    --output-csv data/processed/annotations/coco_val2017_distractors_filtered.csv \\\n"
        "    --filter-purpose asset_extraction\n"
        "  python scripts/02_extract_assets.py annotate \\\n"
        "    --asset-type distractor \\\n"
        f"    --annotations {output_csv} \\\n"
        f"    --image-root {image_root} \\\n"
        "    --output-dir data/processed/assets_distractors_coco_val2017 \\\n"
        "    --class-column class_name \\\n"
        "    --filename-column image_path \\\n"
        "    --source-dataset-column source_dataset \\\n"
        "    --bbox-format xywh --bbox-columns bbox_x bbox_y bbox_w bbox_h"
    )


if __name__ == "__main__":
    main()
