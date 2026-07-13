#!/usr/bin/env python3
"""Prepare high-resolution Places365-Standard validation backgrounds for compositing.

Uses val_large high-resolution images only — NOT val_256 or easyformat splits.

See README.md — Places365 high-res backgrounds (§2a-xi).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.places365_backgrounds import (
    CONTACT_SHEET_CATEGORIES,
    MAPPED_REJECT,
    MAPPED_REVIEW,
    METADATA_COLUMNS,
    prepare_places365_candidates,
    write_places365_report,
)

DEFAULT_PLACES_ROOT = Path(r"C:\datasets\places365")
DEFAULT_IMAGES_ROOT = Path(r"C:\datasets\places365\extracted")
DEFAULT_CATEGORIES = DEFAULT_PLACES_ROOT / "categories_places365.txt"
DEFAULT_FILELIST = DEFAULT_PLACES_ROOT / "places365_val.txt"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_highres_candidates"
)
DEFAULT_METADATA = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_highres_candidates_metadata.csv"
)
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "places365_highres_backgrounds_report.txt"
DEFAULT_SHEET_DIR = PROJECT_ROOT / "outputs" / "contact_sheets"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _tile_title(row: pd.Series) -> str:
    return "\n".join(
        [
            Path(str(row.get("original_path", ""))).name,
            f"{row.get('places365_category', '?')}",
            f"-> {row.get('mapped_background_type', '?')}",
            f"sky={float(row.get('upper_sky_ratio', 0)):.2f} | {row.get('filter_status', '?')}",
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
    if df.empty:
        return None

    rows_ok: list[tuple[pd.Series, Path]] = []
    for _, row in df.iterrows():
        for col in ("output_path", "original_path"):
            raw = str(row.get(col, "")).strip()
            if raw and Path(raw).is_file():
                rows_ok.append((row, Path(raw)))
                break
    if not rows_ok:
        return None

    if len(rows_ok) > max_tiles:
        import numpy as np

        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(rows_ok), size=max_tiles, replace=False))
        rows_ok = [rows_ok[i] for i in idx]

    n = len(rows_ok)
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
        row, path = rows_ok[i]
        try:
            with Image.open(path) as img:
                ax.imshow(img.convert("RGB"))
        except OSError:
            ax.set_title(f"unreadable\n{row.get('places365_category')}", fontsize=7)
            continue
        ax.set_title(_tile_title(row), fontsize=6, loc="left", pad=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select Places365-Standard val_large high-resolution backgrounds "
            "for drone-observation compositing."
        )
    )
    parser.add_argument("--places-root", type=Path, default=DEFAULT_PLACES_ROOT)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--categories-file", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--filelist", type=Path, default=DEFAULT_FILELIST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--no-copy", action="store_true", help="Metadata only; do not copy images")
    parser.add_argument("--make-contact-sheets", action="store_true")
    parser.add_argument("--contact-sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--max-sheet-tiles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    places_root = _resolve(args.places_root)
    images_root = _resolve(args.images_root)
    categories_file = _resolve(args.categories_file)
    filelist = _resolve(args.filelist)
    output_root = _resolve(args.output_root)
    metadata_path = _resolve(args.metadata_out)
    report_path = _resolve(args.report_output)

    if not categories_file.is_file():
        fallback = places_root / "categories_places365.txt"
        if fallback.is_file():
            categories_file = fallback
        else:
            raise FileNotFoundError(f"--categories-file not found: {categories_file}")
    if not filelist.is_file():
        fallback = places_root / "places365_val.txt"
        if fallback.is_file():
            filelist = fallback
        else:
            raise FileNotFoundError(f"--filelist not found: {filelist}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"--images-root not found: {images_root}")

    df, summary = prepare_places365_candidates(
        images_root=images_root,
        categories_file=categories_file,
        filelist=filelist,
        output_root=output_root,
        phash_threshold=args.phash_threshold,
        copy_images=not args.no_copy,
    )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False, columns=METADATA_COLUMNS)
    write_places365_report(report_path, summary)

    print(f"Total validation images found: {summary['total_validation_images']}")
    print(f"Target-category images in filelist: {summary['target_category_images_in_filelist']}")
    print(f"Selected candidate images: {summary['selected_candidate_images']}")
    print(f"  accepted: {summary['accepted_images']}")
    print(f"  review: {summary['review_images']}")
    print(f"  rejected: {summary['rejected_images']}")
    if summary["missing_places_categories"]:
        print(
            "Requested categories not in Places365 taxonomy: "
            + ", ".join(summary["missing_places_categories"])
        )
    print("\nCounts by Places365 category:")
    for name, count in summary["counts_by_places_category"].items():
        print(f"  {name}: {count}")
    print("\nCounts by mapped background category (accept+review):")
    for name, count in summary["counts_by_mapped_category"].items():
        print(f"  {name}: {count}")
    print("\nReject counts by reason:")
    for reason, count in summary["reject_reasons"].items():
        print(f"  {reason}: {count}")
    print(f"\nWrote metadata: {metadata_path}")
    print(f"Wrote report: {report_path}")

    if args.make_contact_sheets:
        sheet_dir = _resolve(args.contact_sheet_dir)
        for cat in CONTACT_SHEET_CATEGORIES:
            if cat == MAPPED_REJECT:
                sub = df[df["filter_status"] == "reject"]
            elif cat == MAPPED_REVIEW:
                sub = df[df["filter_status"] == "review"]
            else:
                sub = df[
                    (df["mapped_background_type"] == cat) & (df["filter_status"] == "accept")
                ]
            if sub.empty:
                continue
            path = sheet_dir / f"places365_highres_{cat}.png"
            saved = _save_contact_sheet(
                sub,
                path,
                max_tiles=args.max_sheet_tiles,
                seed=args.seed,
            )
            if saved:
                print(f"Saved contact sheet: {saved}")


if __name__ == "__main__":
    main()
