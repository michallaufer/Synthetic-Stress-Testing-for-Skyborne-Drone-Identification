#!/usr/bin/env python3
"""Copy manually selected distractor assets using a keep-list of asset_ids.

Simple strict-approval workflow after semantic filtering — no classifier.
One asset_id per line in the keep-list file; lines starting with # are ignored.

See README.md — COCO distractor manual keep-list (§2a-vi).
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

from drone_stress.extract import rgba_on_checkerboard

DEFAULT_OUTPUT_ROOT = Path(r"C:\datasets\coco2017\assets\coco_distractors_strict_approved")
DEFAULT_SHEET_DIR = PROJECT_ROOT / "outputs" / "contact_sheets"
CLASSES = ("bird", "airplane", "kite")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_keep_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"--keep-list not found: {path}")
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def resolve_asset_png(row: pd.Series) -> Path | None:
    for col in ("approved_asset_path", "source_asset_path", "output_path"):
        val = str(row.get(col, "")).strip()
        if val and Path(val).is_file():
            return Path(val)
    return None


def _tile_title(row: pd.Series) -> str:
    return "\n".join(
        [
            str(row.get("asset_id", "?")),
            str(row.get("class_name", "?")),
            str(row.get("semantic_filter_reason", row.get("strict_selection_method", "")))[:48],
        ]
    )


def _save_contact_sheet(
    df: pd.DataFrame,
    output_path: Path,
    *,
    path_column: str = "strict_output_path",
    max_tiles: int = 48,
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
        path = Path(str(row[path_column]))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy manually keep-listed distractor assets into strict approved folder."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--keep-list", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Strict approved root (default: coco_distractors_strict_approved)",
    )
    parser.add_argument("--copy", action="store_true", help="Copy PNG files to output-root")
    parser.add_argument("--make-contact-sheets", action="store_true")
    parser.add_argument("--contact-sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--max-sheet-tiles", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_path = _resolve(args.metadata)
    keep_list_path = _resolve(args.keep_list)
    output_root = _resolve(args.output_root)

    if not metadata_path.is_file():
        raise FileNotFoundError(f"--metadata not found: {metadata_path}")

    keep_ids = read_keep_list(keep_list_path)
    if not keep_ids:
        raise ValueError(f"No asset_ids in keep-list: {keep_list_path}")

    keep_set = set(keep_ids)
    df = pd.read_csv(metadata_path)
    if "asset_id" not in df.columns:
        raise ValueError(f"Metadata missing asset_id column: {metadata_path}")

    selected = df[df["asset_id"].astype(str).isin(keep_set)].copy()
    if selected.empty:
        raise ValueError(
            f"No metadata rows matched keep-list ({len(keep_set)} ids). "
            "Check asset_id spelling and --metadata path."
        )

    missing_in_metadata = sorted(keep_set - set(selected["asset_id"].astype(str)))
    if missing_in_metadata:
        print(f"Warning: {len(missing_in_metadata)} keep-list id(s) not in metadata:")
        for asset_id in missing_in_metadata[:20]:
            print(f"  {asset_id}")
        if len(missing_in_metadata) > 20:
            print(f"  ... and {len(missing_in_metadata) - 20} more")

    output_root.mkdir(parents=True, exist_ok=True)
    strict_paths: list[str] = []
    copy_ok = 0
    copy_fail = 0

    for _, row in selected.iterrows():
        src = resolve_asset_png(row)
        class_name = str(row.get("class_name", "unknown"))
        dest = output_root / class_name / f"{row['asset_id']}.png"
        if src is None:
            strict_paths.append("")
            copy_fail += 1
            continue
        if args.copy:
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dest)
                strict_paths.append(str(dest.resolve()))
                copy_ok += 1
            except OSError:
                strict_paths.append("")
                copy_fail += 1
        else:
            strict_paths.append(str(dest.resolve()))

    selected["strict_output_path"] = strict_paths
    selected["strict_approved"] = True
    selected["strict_selection_method"] = "manual_keep_list"

    out_csv = output_root / "distractor_metadata_strict_approved.csv"
    selected.to_csv(out_csv, index=False)

    print(f"Keep-list ids: {len(keep_set)}")
    print(f"Matched rows: {len(selected)}")
    if args.copy:
        print(f"Copied: {copy_ok}")
        if copy_fail:
            print(f"Copy failed / missing source: {copy_fail}")
    print("\nCounts by class:")
    for cls in CLASSES:
        count = int((selected["class_name"] == cls).sum())
        if count:
            print(f"  {cls}: {count}")
    other = selected[~selected["class_name"].isin(CLASSES)]
    if not other.empty:
        for cls, count in other["class_name"].value_counts().items():
            print(f"  {cls}: {count}")
    print(f"\nWrote metadata: {out_csv}")
    print(f"Output root: {output_root.resolve()}")

    if args.make_contact_sheets:
        sheet_dir = _resolve(args.contact_sheet_dir)
        for cls in CLASSES:
            cls_df = selected[selected["class_name"] == cls]
            if cls_df.empty:
                continue
            path = sheet_dir / f"coco_distractors_strict_{cls}.png"
            saved = _save_contact_sheet(
                cls_df,
                path,
                max_tiles=args.max_sheet_tiles,
                seed=args.seed,
            )
            if saved:
                print(f"Saved contact sheet: {saved}")


if __name__ == "__main__":
    main()
