#!/usr/bin/env python3
"""Visualize extracted RGBA assets and SAM2 QA metadata as contact sheets.

Recursively finds PNG assets under an extraction output directory, optionally
joins asset_metadata.csv, and writes a checkerboard (or light-bg) contact sheet.

See README.md — SAM2 asset QA visualization.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.extract import rgba_on_checkerboard

LABEL_SORT_ORDER = {"accept": 0, "review": 1, "reject": 2, "": 3}


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _norm_path_key(path: Path) -> str:
    try:
        return str(path.resolve()).casefold()
    except OSError:
        return str(path).casefold()


def find_png_files(assets_dir: Path, *, recursive: bool) -> list[Path]:
    assets_dir = assets_dir.resolve()
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Assets directory not found: {assets_dir}")
    pattern = "**/*.png" if recursive else "*.png"
    return sorted(p for p in assets_dir.glob(pattern) if p.is_file())


def load_metadata_index(metadata_path: Path) -> dict[str, pd.Series]:
    """Build lookup keys: resolved output_path, basename, asset_id."""
    df = pd.read_csv(metadata_path)
    index: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        for col in ("output_path", "asset_id"):
            val = row.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            text = str(val).strip()
            if not text:
                continue
            if col == "output_path":
                index[_norm_path_key(Path(text))] = row
                index[Path(text).name.casefold()] = row
            else:
                index[text.casefold()] = row
    return index


def match_metadata_row(png_path: Path, index: dict[str, pd.Series]) -> pd.Series | None:
    if not index:
        return None
    keys = (
        _norm_path_key(png_path),
        png_path.name.casefold(),
        png_path.stem.casefold(),
    )
    for key in keys:
        if key in index:
            return index[key]
    return None


def _cell_value(row: pd.Series | None, column: str, default: str = "") -> str:
    if row is None or column not in row.index:
        return default
    val = row[column]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return str(val).strip()


def _short_reasons(text: str, max_len: int = 42) -> str:
    if not text:
        return "-"
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _tile_title(png_path: Path, row: pd.Series | None, title_column: str | None) -> str:
    name = _cell_value(row, "asset_id", png_path.stem)
    asset_class = _cell_value(row, "asset_class") or _cell_value(row, "class_name")
    method = _cell_value(row, "extraction_method", "unknown")
    label = _cell_value(row, "mask_quality_label", "n/a")
    ratio_raw = row.get("mask_area_ratio_in_crop") if row is not None else None
    if ratio_raw is None or (isinstance(ratio_raw, float) and pd.isna(ratio_raw)):
        ratio_str = "n/a"
    else:
        ratio_str = f"{float(ratio_raw):.3f}"
    reasons = _short_reasons(_cell_value(row, "mask_review_reasons"))

    lines = [
        f"{name} | {asset_class or '?'} | {label}",
        method,
        f"area_ratio={ratio_str}",
        reasons,
    ]
    if title_column and title_column not in ("mask_quality_label",):
        extra = _cell_value(row, title_column)
        if extra:
            lines.append(f"{title_column}={extra}")
    return "\n".join(lines)


def rgba_on_light_background(rgba: np.ndarray, color: tuple[int, int, int] = (245, 245, 245)) -> np.ndarray:
    h, w = rgba.shape[:2]
    bg = np.full((h, w, 3), color, dtype=np.uint8)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    comp = rgb * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return comp.clip(0, 255).astype(np.uint8)


def _sort_key(png_path: Path, row: pd.Series | None, title_column: str | None) -> tuple:
    label = _cell_value(row, "mask_quality_label")
    group_val = _cell_value(row, title_column) if title_column else label
    return (
        LABEL_SORT_ORDER.get(label, 99),
        group_val,
        png_path.name,
    )


def print_summary(
    items: list[tuple[Path, pd.Series | None]],
    *,
    metadata_loaded: bool,
) -> None:
    print(f"\nAssets found: {len(items)}")
    if not metadata_loaded:
        print("Metadata: not loaded (PNG-only visualization)")
        return

    matched = sum(1 for _, row in items if row is not None)
    print(f"Metadata matched: {matched}/{len(items)}")

    def _count_field(field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, row in items:
            if row is None:
                val = "(no metadata)"
            else:
                val = _cell_value(row, field, "(empty)") or "(empty)"
            counts[val] = counts.get(val, 0) + 1
        return dict(sorted(counts.items()))

    print("\nBy mask_quality_label:")
    for label, count in _count_field("mask_quality_label").items():
        print(f"  {label}: {count}")

    print("\nBy extraction_method:")
    for method, count in _count_field("extraction_method").items():
        print(f"  {method}: {count}")

    print("\nBy asset_class:")
    for cls, count in _count_field("asset_class").items():
        print(f"  {cls}: {count}")


def build_contact_sheet(
    items: list[tuple[Path, pd.Series | None]],
    output_path: Path,
    *,
    checkerboard: bool,
    cols: int,
    title_column: str | None,
) -> Path:
    if not items:
        raise ValueError("No PNG assets to visualize")

    n = len(items)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.0))
    axes_flat = _flatten_axes(axes, rows, cols)

    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i >= n:
            ax.set_visible(False)
            continue
        png_path, row = items[i]
        with Image.open(png_path) as img:
            rgba = np.array(img.convert("RGBA"))
        if checkerboard:
            vis = rgba_on_checkerboard(rgba)
        else:
            vis = rgba_on_light_background(rgba)
        ax.imshow(vis)
        ax.set_title(_tile_title(png_path, row, title_column), fontsize=6, loc="left", pad=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _flatten_axes(axes, rows: int, cols: int) -> list:
    if rows == 1 and cols == 1:
        return [axes]
    if rows == 1:
        return list(axes)
    if cols == 1:
        return list(axes)
    return [ax for row in axes for ax in row]


def _sample_items(
    items: list[tuple[Path, pd.Series | None]],
    max_assets: int,
    seed: int,
) -> list[tuple[Path, pd.Series | None]]:
    if len(items) <= max_assets:
        return items
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=max_assets, replace=False)
    return [items[i] for i in sorted(idx)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize extracted RGBA assets with SAM2 QA metadata."
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        required=True,
        help="Directory containing extracted PNG assets",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to asset_metadata.csv (default: <assets-dir>/asset_metadata.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output contact sheet PNG path",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        default=100,
        help="Maximum tiles on the contact sheet (default: 100)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for PNGs recursively under --assets-dir",
    )
    parser.add_argument(
        "--light-background",
        action="store_true",
        help="Use light gray background instead of checkerboard (default: checkerboard)",
    )
    parser.add_argument(
        "--title-column",
        default="mask_quality_label",
        help="Optional metadata column for sort grouping (default: mask_quality_label)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=6,
        help="Contact sheet columns (default: 6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when subsampling with --max-assets (default: 42)",
    )
    args = parser.parse_args()

    assets_dir = _resolve_path(args.assets_dir)
    output_path = _resolve_path(args.output)
    metadata_path = args.metadata
    if metadata_path is None:
        metadata_path = assets_dir / "asset_metadata.csv"
    else:
        metadata_path = _resolve_path(metadata_path)

    png_files = find_png_files(assets_dir, recursive=args.recursive)
    if not png_files:
        raise FileNotFoundError(f"No PNG files found under {assets_dir}")

    metadata_loaded = metadata_path.is_file()
    index: dict[str, pd.Series] = {}
    if metadata_loaded:
        index = load_metadata_index(metadata_path)
        print(f"Loaded metadata: {metadata_path} ({len(index)} lookup keys)")
    else:
        print(f"Metadata not found: {metadata_path} (continuing without QA join)")

    items = [(p, match_metadata_row(p, index)) for p in png_files]
    title_col = args.title_column.strip() if args.title_column else None
    items.sort(key=lambda item: _sort_key(item[0], item[1], title_col))
    items = _sample_items(items, args.max_assets, args.seed)

    checkerboard = not args.light_background
    sheet_path = build_contact_sheet(
        items,
        output_path,
        checkerboard=checkerboard,
        cols=args.cols,
        title_column=title_col,
    )
    print(f"Saved contact sheet: {sheet_path}")
    print_summary(items, metadata_loaded=metadata_loaded)


if __name__ == "__main__":
    main()
