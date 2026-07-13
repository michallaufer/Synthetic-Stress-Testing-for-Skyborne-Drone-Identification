#!/usr/bin/env python3
"""Diversity audit for one extracted AirBirds chunk before dataset commitment.

Temporal thinning + perceptual-hash deduplication to estimate how many distinct
backgrounds / hard negatives a fixed-camera time-series chunk provides.

See README.md — AirBirds inspection (§2a-x).
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

from drone_stress.airbirds_inspect import (
    METADATA_COLUMNS,
    audit_airbirds_chunk,
    write_audit_report,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "airbirds_inspection"
DEFAULT_SHEET_DIR = PROJECT_ROOT / "outputs" / "contact_sheets"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _tile_title(row: pd.Series) -> str:
    return "\n".join(
        [
            str(row.get("filename", "")),
            f"birds={int(row.get('num_birds', 0))}",
            f"seq={str(row.get('sequence_key', ''))[:28]}",
            f"phash={str(row.get('phash', ''))[:12]}",
        ]
    )


def _save_contact_sheet(
    df: pd.DataFrame,
    output_path: Path,
    *,
    path_column: str = "image_path",
    max_tiles: int = 24,
    seed: int = 42,
    cols: int = 6,
) -> Path | None:
    if df.empty:
        return None

    rows_ok: list[tuple[pd.Series, Path]] = []
    for _, row in df.iterrows():
        raw = str(row.get(path_column, "")).strip()
        if raw and Path(raw).is_file():
            rows_ok.append((row, Path(raw)))
    if not rows_ok:
        return None

    if len(rows_ok) > max_tiles:
        import numpy as np

        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(rows_ok), size=max_tiles, replace=False))
        rows_ok = [rows_ok[i] for i in idx]

    n = len(rows_ok)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.6, rows_n * 2.8))
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
            ax.set_title(f"unreadable\n{row.get('filename')}", fontsize=7)
            continue
        ax.set_title(_tile_title(row), fontsize=6, loc="left", pad=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit diversity of one extracted AirBirds chunk (temporal + phash)."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of one extracted AirBirds chunk (images + optional YOLO labels)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for metadata CSV and report",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5000,
        help="Maximum images to scan (0 = no cap)",
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=100,
        help="Keep every Nth frame per sequence folder",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=6,
        help="Max Hamming distance for perceptual-hash near-duplicate grouping",
    )
    parser.add_argument(
        "--make-contact-sheets",
        action="store_true",
        help="Write sample contact sheets under outputs/contact_sheets/",
    )
    parser.add_argument("--contact-sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--max-sheet-tiles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = _resolve(args.dataset_root)
    output_root = _resolve(args.output_root)
    sheet_dir = _resolve(args.contact_sheet_dir)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"--dataset-root not found: {dataset_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "airbirds_inspection_metadata.csv"
    report_path = output_root / "airbirds_inspection_report.txt"

    df, summary = audit_airbirds_chunk(
        dataset_root,
        max_images=args.max_images,
        temporal_stride=args.temporal_stride,
        phash_threshold=args.phash_threshold,
    )
    df.to_csv(metadata_path, index=False, columns=METADATA_COLUMNS)
    write_audit_report(report_path, summary)

    print(f"Total images found: {summary['total_images_found']}")
    print(f"Images after temporal thinning: {summary['images_after_temporal_thinning']}")
    print(f"Images after perceptual deduplication: {summary['images_after_phash_dedup']}")
    print(f"Duplicate groups: {summary['duplicate_groups']}")
    print(f"No-bird images: {summary['no_bird_images']}")
    print(f"Bird images: {summary['bird_images']}")
    print(
        "Estimated usable distinct backgrounds: "
        f"{summary['usable_distinct_backgrounds']} "
        f"(no-bird {summary['usable_no_bird_backgrounds']}, "
        f"with-bird {summary['usable_bird_hard_negatives']})"
    )
    print(f"\nWrote metadata: {metadata_path}")
    print(f"Wrote report: {report_path}")

    if args.make_contact_sheets:
        sheet_specs = [
            ("airbirds_raw_sample.png", df),
            (
                "airbirds_temporal_thinned_sample.png",
                df[df["selected_after_temporal_stride"] == True],  # noqa: E712
            ),
            (
                "airbirds_deduped_sample.png",
                df[df["selected_after_phash_dedup"] == True],  # noqa: E712
            ),
            (
                "airbirds_with_birds_sample.png",
                df[df["has_bird_annotation"] == True],  # noqa: E712
            ),
            (
                "airbirds_no_birds_sample.png",
                df[df["has_bird_annotation"] == False],  # noqa: E712
            ),
        ]
        for filename, subset in sheet_specs:
            path = sheet_dir / filename
            saved = _save_contact_sheet(
                subset,
                path,
                max_tiles=args.max_sheet_tiles,
                seed=args.seed,
            )
            if saved:
                print(f"Saved contact sheet: {saved}")


if __name__ == "__main__":
    main()
