#!/usr/bin/env python3
"""CLIP-based suitability filter for Places365 high-resolution background candidates.

Second-pass filter after category-based Places365 selection. CLIP is a ranking /
rejection signal — not ground truth. Inspect contact sheets before compositing.

See README.md — Places365 CLIP filter (§2a-xii).

Requires: pip install -r requirements-clip.txt
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

from drone_stress.places365_backgrounds import MAPPED_REJECT
from drone_stress.places365_clip_filter import (
    CLIP_FILTERED_CATEGORIES,
    DEFAULT_SUITABILITY_THRESHOLD,
    filter_places365_with_clip,
    write_clip_filter_report,
)

DEFAULT_METADATA = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_highres_candidates_metadata.csv"
)
DEFAULT_CANDIDATE_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_highres_candidates"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_clip_filtered"
DEFAULT_METADATA_OUT = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_clip_filtered_metadata.csv"
)
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "places365_clip_filter_report.txt"
DEFAULT_SHEET_DIR = PROJECT_ROOT / "outputs" / "contact_sheets"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _cell_str(row: pd.Series, *columns: str) -> str:
    for col in columns:
        val = row.get(col)
        if val is None or pd.isna(val):
            continue
        text = str(val).strip()
        if text:
            return text
    return ""


def _tile_title(row: pd.Series) -> str:
    path_text = _cell_str(row, "clip_output_path", "output_path", "original_path")
    name = Path(path_text).name if path_text else "?"
    return "\n".join(
        [
            name,
            f"{_cell_str(row, 'mapped_background_type') or '?'} | {_cell_str(row, 'clip_filter_status') or '?'}",
            f"suit={float(pd.to_numeric(row.get('clip_suitability_score'), errors='coerce') or 0):.3f} sky={float(pd.to_numeric(row.get('upper_sky_ratio'), errors='coerce') or 0):.2f}",
            f"bad: {_cell_str(row, 'clip_bad_prompt')[:40]}",
        ]
    )


def _save_contact_sheet(
    df: pd.DataFrame,
    output_path: Path,
    *,
    path_column: str,
    fallback_column: str = "output_path",
    max_tiles: int = 24,
    seed: int = 42,
    cols: int = 6,
) -> Path | None:
    if df.empty:
        return None

    rows_ok: list[tuple[pd.Series, Path]] = []
    for _, row in df.iterrows():
        for col in (path_column, fallback_column, "original_path"):
            raw = _cell_str(row, col)
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
            ax.set_title("unreadable", fontsize=7)
            continue
        ax.set_title(_tile_title(row), fontsize=6, loc="left", pad=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLIP suitability filter for Places365 high-res background candidates."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA_OUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--suitability-threshold",
        type=float,
        default=DEFAULT_SUITABILITY_THRESHOLD,
        help="Reject when clip_suitability_score (good_max - bad_max) is below this",
    )
    parser.add_argument("--top-k-per-category", type=int, default=80)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument(
        "--use-open-clip",
        action="store_true",
        help="Use open_clip_torch instead of Hugging Face transformers",
    )
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--make-contact-sheets", action="store_true")
    parser.add_argument("--contact-sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--max-sheet-tiles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_path = _resolve(args.metadata)
    candidate_root = _resolve(args.candidate_root)
    output_root = _resolve(args.output_root)
    metadata_out = _resolve(args.metadata_out)
    report_path = _resolve(args.report_output)

    if not metadata_path.is_file():
        raise FileNotFoundError(f"--metadata not found: {metadata_path}")
    if not candidate_root.is_dir():
        raise FileNotFoundError(f"--candidate-root not found: {candidate_root}")

    df, summary = filter_places365_with_clip(
        metadata_path,
        candidate_root=candidate_root,
        output_root=output_root,
        suitability_threshold=args.suitability_threshold,
        top_k_per_category=args.top_k_per_category,
        clip_model=args.clip_model,
        clip_device=args.clip_device,
        clip_batch_size=args.clip_batch_size,
        use_open_clip=args.use_open_clip,
        copy_images=not args.no_copy,
    )

    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_out, index=False)
    write_clip_filter_report(report_path, summary)

    print(f"Input rows: {summary['input_rows']}")
    print(f"CLIP scored: {summary['clip_scored']}")
    print(f"CLIP selected: {summary['clip_selected']}")
    print(f"CLIP rejected: {summary['clip_rejected']}")
    print("\nSelected counts by mapped_background_type:")
    for name, count in summary["counts_by_mapped_category"].items():
        print(f"  {name}: {count}")
    print("\nTop CLIP reject reasons:")
    for reason, count in summary["reject_reasons"].items():
        print(f"  {reason}: {count}")
    print(f"\nWrote metadata: {metadata_out}")
    print(f"Wrote report: {report_path}")

    if args.make_contact_sheets:
        sheet_dir = _resolve(args.contact_sheet_dir)
        for cat in list(CLIP_FILTERED_CATEGORIES) + [MAPPED_REJECT]:
            if cat == MAPPED_REJECT:
                sub = df[df["clip_filter_status"] == "reject"]
            else:
                sub = df[
                    (df["mapped_background_type"] == cat)
                    & (df["clip_filter_status"] == "accept")
                ]
            if sub.empty:
                continue
            path = sheet_dir / f"places365_clip_filtered_{cat}.png"
            saved = _save_contact_sheet(
                sub,
                path,
                path_column="clip_output_path",
                fallback_column="output_path",
                max_tiles=args.max_sheet_tiles,
                seed=args.seed,
            )
            if saved:
                print(f"Saved contact sheet: {saved}")


if __name__ == "__main__":
    main()
