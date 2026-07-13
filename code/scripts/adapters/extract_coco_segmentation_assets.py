#!/usr/bin/env python3
"""Extract COCO distractor RGBA assets from native instance segmentation masks.

Preferred over SAM2 for COCO val2017 bird / airplane / kite — COCO provides
polygon/RLE masks; SAM2 box prompts are weak on small or cluttered objects.

Does not modify drone extraction or SAM2 scripts. See README.md §2a-iv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.coco_segment_extract import (
    METADATA_COLUMNS,
    extract_coco_segmentation_assets,
    write_extraction_report,
    write_qa_report,
)
from drone_stress.extract import rgba_on_checkerboard

DEFAULT_OUTPUT_DIR = Path(r"C:\datasets\coco2017\assets\coco_segmentation_distractors")
DEFAULT_EXTRACTION_REPORT = (
    PROJECT_ROOT / "outputs" / "reports" / "coco_segmentation_asset_extraction_report.txt"
)
DEFAULT_QA_REPORT = PROJECT_ROOT / "outputs" / "reports" / "coco_segmentation_asset_qa_report.txt"
DEFAULT_SHEET_DIR = PROJECT_ROOT / "outputs" / "contact_sheets"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _tile_title(row: pd.Series) -> str:
    reasons = str(row.get("qa_reasons", ""))[:40]
    return "\n".join(
        [
            str(row.get("asset_id", "?")),
            f"{row.get('class_name', '?')} | {row.get('mask_quality_label', '?')}",
            f"area={float(row.get('mask_area_ratio_in_crop', 0)):.3f} "
            f"borders={int(row.get('touches_border_count', 0))}",
            reasons or "-",
        ]
    )


def _save_contact_sheet(
    df: pd.DataFrame,
    output_path: Path,
    *,
    max_tiles: int = 24,
    seed: int = 42,
    cols: int = 6,
) -> Path | None:
    ok = df[~df["extraction_failed"].astype(bool)]
    ok = ok[ok["output_path"].astype(str).str.len() > 0]
    ok = ok[ok["output_path"].astype(str).apply(lambda p: Path(p).is_file())]
    if ok.empty:
        return None

    subset = ok if len(ok) <= max_tiles else ok.sample(n=max_tiles, random_state=seed)
    n = len(subset)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.0))
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    elif rows == 1:
        axes_flat = list(axes)
    elif cols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i >= n:
            ax.set_visible(False)
            continue
        row = subset.iloc[i]
        path = Path(str(row["output_path"]))
        try:
            with Image.open(path) as img:
                rgba = np.array(img.convert("RGBA"))
            ax.imshow(rgba_on_checkerboard(rgba))
        except OSError:
            ax.set_title(f"unreadable\n{row.get('asset_id')}", fontsize=7)
            continue
        ax.set_title(_tile_title(row), fontsize=6, loc="left", pad=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_contact_sheets(df: pd.DataFrame, sheet_dir: Path, *, max_tiles: int, seed: int) -> None:
    sheet_dir.mkdir(parents=True, exist_ok=True)
    ok = df[~df["extraction_failed"].astype(bool)]

    for class_name in sorted(ok["class_name"].unique()):
        class_df = ok[ok["class_name"] == class_name]
        path = sheet_dir / f"coco_seg_{class_name}_assets.png"
        saved = _save_contact_sheet(class_df, path, max_tiles=max_tiles, seed=seed)
        if saved:
            print(f"Saved contact sheet: {saved}")

    for label, filename in (
        ("accept", "coco_seg_accept_sample.png"),
        ("review", "coco_seg_review_sample.png"),
        ("reject", "coco_seg_reject_sample.png"),
    ):
        label_df = ok[ok["mask_quality_label"] == label]
        if label_df.empty:
            continue
        saved = _save_contact_sheet(
            label_df, sheet_dir / filename, max_tiles=max_tiles, seed=seed
        )
        if saved:
            print(f"Saved contact sheet: {saved}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract COCO distractor RGBA assets from instance segmentation masks."
    )
    parser.add_argument("--annotation-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Asset PNG root (use a short path on Windows)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["bird", "airplane", "kite"],
    )
    parser.add_argument("--source-dataset", default="COCO_val2017")
    parser.add_argument("--min-bbox-px", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--expand-box-ratio", type=float, default=0.05)
    parser.add_argument(
        "--class-subdirs",
        action="store_true",
        help="Write output-dir/<class>/ assets (default unless --flat-output)",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write all PNGs directly under output-dir",
    )
    parser.add_argument("--make-contact-sheets", action="store_true")
    parser.add_argument("--contact-sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--max-sheet-tiles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extraction-report",
        type=Path,
        default=DEFAULT_EXTRACTION_REPORT,
    )
    parser.add_argument("--qa-report-output", type=Path, default=DEFAULT_QA_REPORT)
    args = parser.parse_args()

    annotation_json = _resolve(args.annotation_json)
    image_root = _resolve(args.image_root)
    output_dir = _resolve(args.output_dir)
    class_subdirs = not args.flat_output

    if not annotation_json.is_file():
        raise FileNotFoundError(f"--annotation-json not found: {annotation_json}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"--image-root not found: {image_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df, stats = extract_coco_segmentation_assets(
        annotation_json,
        image_root,
        output_dir,
        category_names=args.categories,
        source_dataset=args.source_dataset,
        min_bbox_px=args.min_bbox_px,
        max_rows=args.max_rows,
        expand_box_ratio=args.expand_box_ratio,
        class_subdirs=class_subdirs,
    )

    metadata_path = output_dir / "asset_metadata.csv"
    df[METADATA_COLUMNS].to_csv(metadata_path, index=False)
    print(f"Wrote metadata: {metadata_path}")

    extraction_report = _resolve(args.extraction_report)
    write_extraction_report(
        stats,
        extraction_report,
        annotation_json=annotation_json,
        image_root=image_root,
        output_dir=output_dir,
        category_names=args.categories,
    )
    print(f"Wrote extraction report: {extraction_report}")

    qa_report = _resolve(args.qa_report_output)
    write_qa_report(df, qa_report)
    print(f"Wrote QA report: {qa_report}")

    print(f"\nannotations scanned: {stats.annotations_scanned}")
    print(f"exported (save ok): {stats.exported}")
    print(f"extraction_failed: {stats.extraction_failed}")
    print("exported by class:")
    for cls in sorted(stats.by_class):
        print(f"  {cls}: {stats.by_class[cls]}")
    print("QA totals:")
    for label in ("accept", "review", "reject"):
        print(f"  {label}: {stats.qa_totals.get(label, 0)}")

    if args.make_contact_sheets:
        _write_contact_sheets(
            df,
            _resolve(args.contact_sheet_dir),
            max_tiles=args.max_sheet_tiles,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
