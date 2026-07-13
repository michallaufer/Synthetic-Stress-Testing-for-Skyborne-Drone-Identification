#!/usr/bin/env python3
"""Filter COCO segmentation distractor assets by semantic usefulness.

Builds approved/review subsets for synthetic hard-negative compositing.
Does not run SAM2 or generate synthetic images.

See README.md — COCO distractor semantic filter (§2a-v).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.coco_distractor_filter import (
    DEFAULT_MAX_PER_CLASS,
    enrich_with_semantic_labels,
    filter_input_rows,
    resolve_source_asset_path,
    select_diverse_approved,
    write_filter_report,
)
from drone_stress.extract import rgba_on_checkerboard

DEFAULT_APPROVED_ROOT = Path(r"C:\datasets\coco2017\assets\coco_distractors_approved")
DEFAULT_REVIEW_ROOT = Path(r"C:\datasets\coco2017\assets\coco_distractors_review")
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "coco_distractor_filter_report.txt"
DEFAULT_SHEET_DIR = PROJECT_ROOT / "outputs" / "contact_sheets"

EXPORT_COLUMNS_EXTRA = [
    "final_distractor_status",
    "semantic_filter_reason",
    "crop_width",
    "crop_height",
    "crop_aspect_ratio",
    "bbox_aspect_ratio",
    "source_asset_path",
    "approved_asset_path",
]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_max_per_class(values: list[str] | None) -> dict[str, int]:
    limits = dict(DEFAULT_MAX_PER_CLASS)
    if not values:
        return limits
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected class=count, got {item!r}")
        cls, count = item.split("=", 1)
        limits[cls.strip().lower()] = int(count.strip())
    return limits


def _tile_title(row: pd.Series) -> str:
    return "\n".join(
        [
            str(row.get("asset_id", "?")),
            f"{row.get('class_name', '?')} | {row.get('final_distractor_status', '?')}",
            str(row.get("semantic_filter_reason", ""))[:48],
            (
                f"aspect={float(row.get('crop_aspect_ratio', 0)):.2f} "
                f"area={float(row.get('mask_area_ratio_in_crop', 0)):.3f} "
                f"borders={int(row.get('touches_border_count', 0))}"
            ),
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
    ok = df[df["approved_asset_path"].astype(str).str.len() > 0]
    ok = ok[ok["approved_asset_path"].astype(str).apply(lambda p: Path(p).is_file())]
    if ok.empty:
        return None

    subset = ok if len(ok) <= max_tiles else ok.sample(n=max_tiles, random_state=seed)
    n = len(subset)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.6, rows_n * 3.0))
    if rows_n == 1 and cols == 1:
        axes_flat = [axes]
    elif rows_n == 1:
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
        path = Path(str(row["approved_asset_path"]))
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


def _prepare_export_frame(
    df: pd.DataFrame,
    assets_root: Path,
    dest_root: Path,
    *,
    do_copy: bool,
) -> pd.DataFrame:
    export = df.copy()
    source_paths: list[str] = []
    dest_paths: list[str] = []
    for _, row in export.iterrows():
        src = resolve_source_asset_path(row, assets_root)
        if src is None:
            source_paths.append("")
            dest_paths.append("")
            continue
        source_paths.append(str(src.resolve()))
        dest = dest_root / str(row["class_name"]) / src.name
        dest_paths.append(str(dest.resolve()) if do_copy else "")
        if do_copy:
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dest)
            except OSError:
                dest_paths[-1] = ""

    export["source_asset_path"] = source_paths
    export["approved_asset_path"] = dest_paths
    return export


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter COCO segmentation distractors by semantic usefulness."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_APPROVED_ROOT,
        help="Approved asset root (default: C:\\datasets\\coco2017\\assets\\coco_distractors_approved)",
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        default=None,
        help="Review asset root (default: sibling coco_distractors_review)",
    )
    parser.add_argument("--copy", action="store_true", help="Copy PNG files to output roots")
    parser.add_argument("--make-contact-sheets", action="store_true")
    parser.add_argument("--contact-sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--max-sheet-tiles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-class",
        nargs="*",
        default=None,
        metavar="CLASS=N",
        help="Cap approved assets per class (default: bird=100 airplane=60 kite=100)",
    )
    parser.add_argument(
        "--technical-labels",
        nargs="+",
        default=["accept", "review"],
        help="Keep rows with these mask_quality_label values (default: accept review)",
    )
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    metadata_path = _resolve(args.metadata)
    assets_root = _resolve(args.assets_root)
    approved_root = _resolve(args.output_root)
    review_root = (
        _resolve(args.review_root)
        if args.review_root
        else approved_root.parent / "coco_distractors_review"
    )
    max_per_class = _parse_max_per_class(args.max_per_class)
    technical_labels = {lbl.lower() for lbl in args.technical_labels}

    if not metadata_path.is_file():
        raise FileNotFoundError(f"--metadata not found: {metadata_path}")
    if not assets_root.is_dir():
        raise FileNotFoundError(f"--assets-root not found: {assets_root}")

    df = pd.read_csv(metadata_path)
    filtered = filter_input_rows(df, technical_labels=technical_labels)
    if filtered.empty:
        raise ValueError("No rows passed technical pre-filter.")

    enriched = enrich_with_semantic_labels(filtered)

    input_by_class = enriched["class_name"].value_counts().to_dict()
    status_by_class: dict[str, dict[str, int]] = {}
    for cls in enriched["class_name"].unique():
        sub = enriched[enriched["class_name"] == cls]
        status_by_class[str(cls)] = sub["final_distractor_status"].value_counts().to_dict()

    reason_counts = enriched["semantic_filter_reason"].value_counts().to_dict()

    approved_all = enriched[enriched["final_distractor_status"] == "approved"].copy()
    review_all = enriched[enriched["final_distractor_status"] == "review"].copy()
    approved_selected = select_diverse_approved(
        approved_all, max_per_class=max_per_class, seed=args.seed
    )

    approved_root.mkdir(parents=True, exist_ok=True)
    review_root.mkdir(parents=True, exist_ok=True)

    approved_export = _prepare_export_frame(
        approved_selected, assets_root, approved_root, do_copy=args.copy
    )
    review_export = _prepare_export_frame(
        review_all, assets_root, review_root, do_copy=args.copy
    )

    drop_cols = [c for c in ("_priority",) if c in approved_export.columns]
    if drop_cols:
        approved_export = approved_export.drop(columns=drop_cols)
        review_export = review_export.drop(columns=[c for c in drop_cols if c in review_export.columns])

    approved_csv = approved_root / "distractor_metadata_approved.csv"
    review_csv = review_root / "distractor_metadata_review.csv"
    approved_export.to_csv(approved_csv, index=False)
    review_export.to_csv(review_csv, index=False)

    report_path = _resolve(args.report_output)
    write_filter_report(
        report_path,
        input_by_class={str(k): int(v) for k, v in input_by_class.items()},
        status_by_class=status_by_class,
        reason_counts={str(k): int(v) for k, v in reason_counts.items()},
        approved_root=approved_root,
        review_root=review_root,
        approved_csv=approved_csv,
        review_csv=review_csv,
    )

    print(f"Input (technical pre-filter): {len(enriched)}")
    print(f"Approved (before cap): {len(approved_all)}")
    print(f"Approved (exported): {len(approved_export)}")
    print(f"Review (exported): {len(review_export)}")
    print(f"Reject: {int((enriched['final_distractor_status'] == 'reject').sum())}")
    print("\nApproved by class:")
    for cls in sorted(approved_export["class_name"].unique()):
        print(f"  {cls}: {int((approved_export['class_name'] == cls).sum())}")
    print("\nReview by class:")
    for cls in sorted(review_export["class_name"].unique()):
        print(f"  {cls}: {int((review_export['class_name'] == cls).sum())}")
    print(f"\nWrote approved metadata: {approved_csv}")
    print(f"Wrote review metadata: {review_csv}")
    print(f"Wrote report: {report_path}")

    if args.make_contact_sheets:
        sheet_dir = _resolve(args.contact_sheet_dir)
        for cls in ("bird", "airplane", "kite"):
            app_df = approved_export[approved_export["class_name"] == cls]
            if not app_df.empty:
                path = sheet_dir / f"coco_distractors_approved_{cls}.png"
                saved = _save_contact_sheet(
                    app_df, path, max_tiles=args.max_sheet_tiles, seed=args.seed
                )
                if saved:
                    print(f"Saved contact sheet: {saved}")
            rev_df = review_export[review_export["class_name"] == cls]
            if not rev_df.empty:
                path = sheet_dir / f"coco_distractors_review_{cls}.png"
                saved = _save_contact_sheet(
                    rev_df, path, max_tiles=args.max_sheet_tiles, seed=args.seed
                )
                if saved:
                    print(f"Saved contact sheet: {saved}")


if __name__ == "__main__":
    main()
