#!/usr/bin/env python3
"""Manual final approval for CLIP-filtered Places365 high-resolution backgrounds.

Mode 1 (default): build review manifest + numbered contact sheets.
Mode 2 (--apply-decisions): copy accepted manifest rows into final v1 pool.

See README.md — Places365 manual final approval (§2a-xiii).
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

from drone_stress.places365_clip_filter import CLIP_FILTERED_CATEGORIES
from drone_stress.places365_finalize import (
    _cell_str,
    apply_manifest_decisions,
    create_review_manifest,
    resolve_candidate_image_path,
)

DEFAULT_METADATA = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_clip_filtered_metadata.csv"
)
DEFAULT_CANDIDATE_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_clip_filtered"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_final_review_manifest.csv"
)
DEFAULT_FINAL_ROOT = PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_final_v1"
DEFAULT_FINAL_METADATA = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_final_v1_metadata.csv"
)
DEFAULT_SHEET_DIR = PROJECT_ROOT / "outputs" / "contact_sheets"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _tile_title_review(row: pd.Series, number: int) -> str:
    return "\n".join(
        [
            f"#{number:03d} {_cell_str(row, 'candidate_id')}",
            f"{_cell_str(row, 'mapped_background_type') or '?'}",
            f"sky={float(pd.to_numeric(row.get('upper_sky_ratio'), errors='coerce') or 0):.2f} "
            f"clip={float(pd.to_numeric(row.get('clip_suitability_score'), errors='coerce') or 0):.3f}",
            f"bad: {_cell_str(row, 'clip_bad_prompt')[:36]}",
        ]
    )


def _tile_title_final(row: pd.Series) -> str:
    return "\n".join(
        [
            _cell_str(row, "candidate_id"),
            f"{_cell_str(row, 'final_category') or '?'}",
            f"sky={float(pd.to_numeric(row.get('upper_sky_ratio'), errors='coerce') or 0):.2f}",
            _cell_str(row, "places365_category"),
        ]
    )


def _save_numbered_contact_sheet(
    df: pd.DataFrame,
    output_path: Path,
    *,
    candidate_root: Path | None,
    numbered: bool,
    path_columns: tuple[str, ...] = ("output_path", "final_path"),
    cols: int = 8,
) -> Path | None:
    if df.empty:
        return None

    rows_ok: list[tuple[int, pd.Series, Path]] = []
    for number, (_, row) in enumerate(df.iterrows(), start=1):
        path: Path | None = None
        if candidate_root is not None:
            path = resolve_candidate_image_path(row, candidate_root)
        if path is None:
            for col in path_columns:
                raw = _cell_str(row, col)
                if raw and Path(raw).is_file():
                    path = Path(raw)
                    break
        if path is not None:
            rows_ok.append((number, row, path))
    if not rows_ok:
        return None

    n = len(rows_ok)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.2, rows_n * 2.4))
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
        number, row, path = rows_ok[i]
        try:
            with Image.open(path) as img:
                ax.imshow(img.convert("RGB"))
        except OSError:
            ax.set_title(f"unreadable\n{_cell_str(row, 'candidate_id')}", fontsize=7)
            continue
        title = _tile_title_review(row, number) if numbered else _tile_title_final(row)
        ax.set_title(title, fontsize=5, loc="left", pad=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_review_sheets(
    manifest: pd.DataFrame,
    *,
    candidate_root: Path,
    sheet_dir: Path,
) -> None:
    for cat in CLIP_FILTERED_CATEGORIES:
        sub = manifest[manifest["mapped_background_type"].astype(str) == cat].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("candidate_id")
        path = sheet_dir / f"places365_final_review_{cat}.png"
        saved = _save_numbered_contact_sheet(
            sub,
            path,
            candidate_root=candidate_root,
            numbered=True,
            path_columns=("output_path",),
        )
        if saved:
            print(f"Saved review contact sheet: {saved}")


def _save_final_sheets(
    final_df: pd.DataFrame,
    *,
    sheet_dir: Path,
) -> None:
    for cat in CLIP_FILTERED_CATEGORIES:
        sub = final_df[final_df["final_category"].astype(str) == cat].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("candidate_id")
        path = sheet_dir / f"backgrounds_places365_final_v1_{cat}.png"
        saved = _save_numbered_contact_sheet(
            sub,
            path,
            candidate_root=None,
            numbered=False,
            path_columns=("final_path", "candidate_path", "output_path"),
        )
        if saved:
            print(f"Saved final contact sheet: {saved}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual final approval for CLIP-filtered Places365 backgrounds."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL_ROOT)
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=DEFAULT_FINAL_METADATA,
        help="Final metadata CSV written in --apply-decisions mode",
    )
    parser.add_argument(
        "--apply-decisions",
        action="store_true",
        help="Apply edited manifest decisions and copy accepted backgrounds",
    )
    parser.add_argument("--make-contact-sheets", action="store_true")
    parser.add_argument("--contact-sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    args = parser.parse_args()

    metadata_path = _resolve(args.metadata)
    candidate_root = _resolve(args.candidate_root)
    manifest_path = _resolve(args.manifest)
    final_root = _resolve(args.final_root)
    metadata_out = _resolve(args.metadata_out)
    sheet_dir = _resolve(args.contact_sheet_dir)

    if args.apply_decisions:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                "Run without --apply-decisions first to create it."
            )
        final_df = apply_manifest_decisions(
            manifest_path,
            candidate_root=candidate_root,
            final_root=final_root,
            metadata_out=metadata_out,
        )
        if args.make_contact_sheets:
            _save_final_sheets(final_df, sheet_dir=sheet_dir)
        return

    if not metadata_path.is_file():
        raise FileNotFoundError(f"--metadata not found: {metadata_path}")

    manifest = create_review_manifest(
        metadata_path,
        manifest_path=manifest_path,
        candidate_root=candidate_root,
    )
    if args.make_contact_sheets:
        _save_review_sheets(manifest, candidate_root=candidate_root, sheet_dir=sheet_dir)


if __name__ == "__main__":
    main()
