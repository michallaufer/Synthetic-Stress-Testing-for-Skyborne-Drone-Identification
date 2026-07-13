#!/usr/bin/env python3
"""Build real hard-negative image sets from full COCO val2017 frames.

Uses original images containing bird / airplane / kite — no masks, no cutouts,
no SAM2. COCO isolated masks are poor compositing assets; full frames are
better real hard negatives for the skyborne drone benchmark.

See README.md — COCO hard-negative images (§2a-vii).
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

from drone_stress.coco_hard_negative import (
    METADATA_COLUMNS,
    build_image_candidates,
    candidate_to_row,
    parse_max_per_category,
    select_candidates,
    write_hard_negative_report,
)

DEFAULT_OUTPUT_DIR = Path(r"C:\datasets\coco2017\hard_negatives")
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "coco_hard_negative_images_report.txt"
DEFAULT_SHEET_DIR = PROJECT_ROOT / "outputs" / "contact_sheets"
CATEGORIES = ("bird", "airplane", "kite")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _unique_dest(dest_dir: Path, file_name: str, image_id: int) -> Path:
    dest = dest_dir / file_name
    if not dest.exists():
        return dest
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    return dest_dir / f"{stem}_{image_id:012d}{suffix}"


def _tile_title(row: pd.Series) -> str:
    name = Path(str(row.get("output_path", ""))).name or Path(
        str(row.get("original_image_path", ""))
    ).name
    max_dim = max(
        int(row.get("largest_distractor_bbox_w", 0)),
        int(row.get("largest_distractor_bbox_h", 0)),
    )
    return "\n".join(
        [
            name,
            str(row.get("dominant_distractor_type", "?")),
            f"objects={int(row.get('num_requested_objects', 0))} max_bbox={max_dim}px",
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
    ok = df[df["output_path"].astype(str).apply(lambda p: bool(p) and Path(p).is_file())]
    if ok.empty:
        return None

    subset = ok if len(ok) <= max_tiles else ok.sample(n=max_tiles, random_state=seed)
    n = len(subset)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.8, rows_n * 2.8))
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
        path = Path(str(row["output_path"]))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy full COCO val images as real hard negatives (bird/airplane/kite)."
    )
    parser.add_argument("--annotation-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORIES),
    )
    parser.add_argument("--source-dataset", default="COCO_val2017")
    parser.add_argument("--min-object-px", type=int, default=20)
    parser.add_argument(
        "--max-images-per-category",
        nargs="*",
        default=None,
        metavar="CLASS=N",
        help="Cap selected images per category (default: bird=150 airplane=100 kite=100)",
    )
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--make-contact-sheets", action="store_true")
    parser.add_argument("--contact-sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--max-sheet-tiles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    annotation_json = _resolve(args.annotation_json)
    image_root = _resolve(args.image_root)
    output_dir = _resolve(args.output_dir)
    max_per_category = parse_max_per_category(args.max_images_per_category)

    if not annotation_json.is_file():
        raise FileNotFoundError(f"--annotation-json not found: {annotation_json}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"--image-root not found: {image_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        (output_dir / cat).mkdir(parents=True, exist_ok=True)

    candidates, stats = build_image_candidates(
        annotation_json,
        image_root,
        category_names=args.categories,
        min_object_px=args.min_object_px,
    )
    if not candidates:
        raise ValueError("No candidate images found. Check categories and min-object-px.")

    selected = select_candidates(
        candidates,
        max_per_category=max_per_category,
        seed=args.seed,
    )

    rows: list[dict] = []
    copy_ok = 0
    copy_fail = 0

    for cand in selected:
        dest_dir = output_dir / cand.dominant_distractor_type
        dest_path = _unique_dest(dest_dir, cand.file_name, cand.image_id)
        output_path_str = ""

        if args.copy:
            try:
                shutil.copy2(cand.original_image_path, dest_path)
                output_path_str = str(dest_path.resolve())
                copy_ok += 1
            except OSError:
                copy_fail += 1
        else:
            output_path_str = str(dest_path.resolve())

        rows.append(
            candidate_to_row(
                cand,
                source_dataset=args.source_dataset,
                output_path=output_path_str,
            )
        )
        stats.selected_by_category[cand.dominant_distractor_type] = (
            stats.selected_by_category.get(cand.dominant_distractor_type, 0) + 1
        )

    df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    metadata_path = output_dir / "coco_hard_negative_metadata.csv"
    df.to_csv(metadata_path, index=False)

    report_path = _resolve(args.report_output)
    write_hard_negative_report(
        report_path,
        stats,
        annotation_json=annotation_json,
        image_root=image_root,
        output_dir=output_dir,
        metadata_csv=metadata_path,
    )

    print(f"Candidates: {len(candidates)}")
    print(f"Selected: {stats.selected_total}")
    if args.copy:
        print(f"Copied: {copy_ok}")
        if copy_fail:
            print(f"Copy failed: {copy_fail}")
    print("\nCandidate images by dominant category:")
    for cat in sorted(stats.candidates_by_category):
        print(f"  {cat}: {stats.candidates_by_category[cat]}")
    print("\nSelected by category:")
    for cat in sorted(stats.selected_by_category):
        print(f"  {cat}: {stats.selected_by_category[cat]}")
    print(f"\nSkipped (all objects too small): {stats.skipped_small_object}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote report: {report_path}")

    if args.make_contact_sheets:
        sheet_dir = _resolve(args.contact_sheet_dir)
        for cat in CATEGORIES:
            cat_df = df[df["dominant_distractor_type"] == cat]
            if cat_df.empty:
                continue
            path = sheet_dir / f"coco_hard_negative_{cat}.png"
            saved = _save_contact_sheet(
                cat_df, path, max_tiles=args.max_sheet_tiles, seed=args.seed
            )
            if saved:
                print(f"Saved contact sheet: {saved}")


if __name__ == "__main__":
    main()
