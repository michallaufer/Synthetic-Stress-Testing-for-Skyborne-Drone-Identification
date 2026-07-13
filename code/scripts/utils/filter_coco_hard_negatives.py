#!/usr/bin/env python3
"""Filter full-image COCO hard negatives into a stricter approved subset.

Metadata-first heuristics with optional sky hint — no SAM2, no cutouts.

See README.md — COCO hard-negative strict filter (§2a-viii).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.coco_hard_negative_filter import (
    enrich_hard_negative_rows,
    parse_max_per_category,
    resolve_image_path,
    select_approved_subset,
    write_hard_negative_filter_report,
)

DEFAULT_METADATA = Path(r"C:\datasets\coco2017\hard_negatives\coco_hard_negative_metadata.csv")
DEFAULT_IMAGE_ROOT = Path(r"C:\datasets\coco2017\hard_negatives")
DEFAULT_STRICT_ROOT = Path(r"C:\datasets\coco2017\hard_negatives_strict")
DEFAULT_REVIEW_ROOT = Path(r"C:\datasets\coco2017\hard_negatives_review")
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "coco_hard_negative_filter_report.txt"
DEFAULT_SHEET_DIR = PROJECT_ROOT / "outputs" / "contact_sheets"
CATEGORIES = ("bird", "airplane", "kite")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _tile_title(row: pd.Series) -> str:
    name = Path(str(row.get("strict_output_path") or row.get("output_path", ""))).name
    if not name:
        name = Path(str(row.get("original_image_path", ""))).name
    max_dim = int(row.get("max_bbox_px", 0))
    area = float(row.get("bbox_area_ratio", 0))
    return "\n".join(
        [
            name,
            f"{row.get('dominant_distractor_type', '?')} | {row.get('final_hard_negative_status', '?')}",
            str(row.get("hard_negative_filter_reason", ""))[:52],
            (
                f"objects={int(row.get('num_requested_objects', 0))} "
                f"max_bbox={max_dim}px area={area:.3f}"
            ),
        ]
    )


def _save_contact_sheet(
    df: pd.DataFrame,
    output_path: Path,
    path_column: str,
    *,
    max_tiles: int = 24,
    seed: int = 42,
    cols: int = 6,
) -> Path | None:
    ok = df[df[path_column].astype(str).str.len() > 0]
    ok = ok[ok[path_column].astype(str).apply(lambda p: Path(p).is_file())]
    if ok.empty:
        return None

    subset = ok if len(ok) <= max_tiles else ok.sample(n=max_tiles, random_state=seed)
    n = len(subset)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.8, rows_n * 2.9))
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
        path = Path(str(row[path_column]))
        try:
            with Image.open(path) as img:
                ax.imshow(img.convert("RGB"))
        except OSError:
            ax.set_title(f"unreadable\n{row.get('hard_negative_id')}", fontsize=7)
            continue
        ax.set_title(_tile_title(row), fontsize=6, loc="left", pad=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _copy_rows(
    df: pd.DataFrame,
    dest_root: Path,
    image_root: Path,
    *,
    do_copy: bool,
    path_column: str,
) -> pd.DataFrame:
    export = df.copy()
    dest_paths: list[str] = []
    for _, row in export.iterrows():
        src = resolve_image_path(row, image_root)
        if src is None:
            dest_paths.append("")
            continue
        cat = str(row.get("dominant_distractor_type", "unknown"))
        dest = dest_root / cat / src.name
        dest_str = str(dest.resolve())
        if do_copy:
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dest)
                dest_paths.append(dest_str)
            except OSError:
                dest_paths.append("")
        else:
            dest_paths.append(dest_str)
    export[path_column] = dest_paths
    return export


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter COCO full-image hard negatives into strict approved/review subsets."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STRICT_ROOT)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--make-contact-sheets", action="store_true")
    parser.add_argument("--contact-sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--max-sheet-tiles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-category",
        nargs="*",
        default=None,
        metavar="CLASS=N",
        help="Cap approved images per category (default: bird=70 airplane=70 kite=80)",
    )
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    metadata_path = _resolve(args.metadata)
    image_root = _resolve(args.image_root)
    strict_root = _resolve(args.output_dir)
    review_root = _resolve(args.review_dir)
    max_per_category = parse_max_per_category(args.max_per_category)

    if not metadata_path.is_file():
        raise FileNotFoundError(f"--metadata not found: {metadata_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"--image-root not found: {image_root}")

    df = pd.read_csv(metadata_path)
    if df.empty:
        raise ValueError("Metadata CSV is empty.")

    enriched = enrich_hard_negative_rows(df, image_root=image_root)

    input_by_category = (
        enriched["dominant_distractor_type"].value_counts().astype(int).to_dict()
    )
    status_by_category: dict[str, dict[str, int]] = {}
    for cat in enriched["dominant_distractor_type"].unique():
        sub = enriched[enriched["dominant_distractor_type"] == cat]
        status_by_category[str(cat)] = (
            sub["final_hard_negative_status"].value_counts().astype(int).to_dict()
        )
    reason_counts = (
        enriched["hard_negative_filter_reason"].value_counts().astype(int).to_dict()
    )

    approved_all = enriched[enriched["final_hard_negative_status"] == "approved"].copy()
    review_all = enriched[enriched["final_hard_negative_status"] == "review"].copy()
    approved_selected = select_approved_subset(
        approved_all, max_per_category=max_per_category, seed=args.seed
    )

    strict_root.mkdir(parents=True, exist_ok=True)
    review_root.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        (strict_root / cat).mkdir(parents=True, exist_ok=True)
        (review_root / cat).mkdir(parents=True, exist_ok=True)

    strict_export = _copy_rows(
        approved_selected,
        strict_root,
        image_root,
        do_copy=args.copy,
        path_column="strict_output_path",
    )
    review_export = _copy_rows(
        review_all,
        review_root,
        image_root,
        do_copy=args.copy,
        path_column="strict_output_path",
    )

    drop_cols = [c for c in ("_priority",) if c in strict_export.columns]
    if drop_cols:
        strict_export = strict_export.drop(columns=drop_cols)
        review_export = review_export.drop(
            columns=[c for c in drop_cols if c in review_export.columns]
        )

    strict_csv = strict_root / "coco_hard_negative_metadata_strict.csv"
    review_csv = review_root / "coco_hard_negative_metadata_review.csv"
    strict_export.to_csv(strict_csv, index=False)
    review_export.to_csv(review_csv, index=False)

    report_path = _resolve(args.report_output)
    write_hard_negative_filter_report(
        report_path,
        input_by_category={str(k): int(v) for k, v in input_by_category.items()},
        status_by_category=status_by_category,
        reason_counts={str(k): int(v) for k, v in reason_counts.items()},
        strict_root=strict_root,
        review_root=review_root,
        strict_csv=strict_csv,
        review_csv=review_csv,
        final_selected_total=len(strict_export),
    )

    reject_n = int((enriched["final_hard_negative_status"] == "reject").sum())
    print(f"Input: {len(enriched)}")
    print(f"Approved (before cap): {len(approved_all)}")
    print(f"Approved (exported): {len(strict_export)}")
    print(f"Review (exported): {len(review_export)}")
    print(f"Reject: {reject_n}")
    print("\nApproved by category:")
    for cat in CATEGORIES:
        n = int((strict_export["dominant_distractor_type"] == cat).sum())
        if n:
            print(f"  {cat}: {n}")
    print("\nReview by category:")
    for cat in CATEGORIES:
        n = int((review_export["dominant_distractor_type"] == cat).sum())
        if n:
            print(f"  {cat}: {n}")
    print(f"\nWrote strict metadata: {strict_csv}")
    print(f"Wrote review metadata: {review_csv}")
    print(f"Wrote report: {report_path}")

    if args.make_contact_sheets:
        sheet_dir = _resolve(args.contact_sheet_dir)
        for cat in CATEGORIES:
            strict_df = strict_export[strict_export["dominant_distractor_type"] == cat]
            if not strict_df.empty:
                path = sheet_dir / f"coco_hard_negative_strict_{cat}.png"
                saved = _save_contact_sheet(
                    strict_df,
                    path,
                    "strict_output_path",
                    max_tiles=args.max_sheet_tiles,
                    seed=args.seed,
                )
                if saved:
                    print(f"Saved contact sheet: {saved}")
            rev_df = review_export[review_export["dominant_distractor_type"] == cat]
            if not rev_df.empty:
                path = sheet_dir / f"coco_hard_negative_review_{cat}.png"
                saved = _save_contact_sheet(
                    rev_df,
                    path,
                    "strict_output_path",
                    max_tiles=args.max_sheet_tiles,
                    seed=args.seed,
                )
                if saved:
                    print(f"Saved contact sheet: {saved}")


if __name__ == "__main__":
    main()
