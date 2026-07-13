#!/usr/bin/env python3
"""Extract high-precision sky-visible outdoor backgrounds from COCO val2017.

Strict filter for drone compositing — prefers fewer, cleaner backgrounds over recall.
Reuses background_filter sky heuristics and COCO annotation scene checks.

See README.md — COCO sky backgrounds (§2a-ix).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.background_clip import V1_APPROVED_CATEGORIES
from drone_stress.coco_sky_background import (
    OUTPUT_CATEGORIES,
    STRICT_METADATA_COLUMNS,
    classify_coco_sky_background,
    load_coco_annotation_context,
    result_to_strict_metadata_row,
    select_balanced_approved,
    write_coco_sky_report,
)
from drone_stress.scene_filter import load_image_rgb

DEFAULT_IMAGE_ROOT = Path(r"C:\datasets\coco2017\images\val2017")
DEFAULT_ANNOTATION_JSON = Path(r"C:\datasets\coco2017\annotations\instances_val2017.json")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "backgrounds_coco_sky_strict"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "backgrounds_coco_sky_strict_metadata.csv"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "coco_sky_backgrounds_strict_report.txt"
DEFAULT_SHEET_DIR = PROJECT_ROOT / "outputs" / "contact_sheets"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _win_long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\") and len(resolved) >= 240:
        return "\\\\?\\" + resolved
    return resolved


def _ensure_dir(path: Path) -> None:
    if os.name == "nt" and len(str(path.resolve())) >= 248:
        os.makedirs(_win_long_path(path), exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def _copy_image(src: Path, dest: Path) -> None:
    _ensure_dir(dest.parent)
    if os.name == "nt" and (
        len(str(src.resolve())) >= 240 or len(str(dest.resolve())) >= 240
    ):
        import shutil

        shutil.copy2(_win_long_path(src), _win_long_path(dest))
    else:
        import shutil

        shutil.copy2(src, dest)


def _background_id(image_id: int, file_name: str) -> str:
    digest = hashlib.sha256(f"{image_id}:{file_name}".encode()).hexdigest()[:8]
    return f"coco_bg_{image_id:012d}_{digest}"


def _unique_dest(dest_dir: Path, file_name: str, image_id: int) -> Path:
    dest = dest_dir / file_name
    if not dest.exists():
        return dest
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix or ".jpg"
    return dest_dir / f"{stem}_{image_id:012d}{suffix}"


def _tile_title(row: pd.Series) -> str:
    path = str(row.get("output_path") or row.get("source_image", ""))
    name = Path(path).name
    return "\n".join(
        [
            name,
            f"{row.get('background_type', '?')} | {row.get('filter_status', '?')}",
            f"upper_sky={float(row.get('upper_sky_ratio', row.get('sky_score', 0))):.2f}",
            str(row.get("filter_reason", ""))[:48],
        ]
    )


def _save_contact_sheet(
    df: pd.DataFrame,
    output_path: Path,
    *,
    path_column: str,
    fallback_column: str = "source_image",
    max_tiles: int = 24,
    seed: int = 42,
    cols: int = 6,
) -> Path | None:
    if df.empty:
        return None

    def _resolve_path(row: pd.Series) -> Path | None:
        for col in (path_column, fallback_column):
            raw = str(row.get(col, "")).strip()
            if raw and Path(raw).is_file():
                return Path(raw)
        return None

    rows_ok = []
    for _, row in df.iterrows():
        p = _resolve_path(row)
        if p is not None:
            rows_ok.append((row, p))
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
            ax.set_title(f"unreadable\n{row.get('image_id')}", fontsize=7)
            continue
        ax.set_title(_tile_title(row), fontsize=6, loc="left", pad=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract strict COCO val2017 sky-visible backgrounds for compositing."
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--annotation-json", type=Path, default=DEFAULT_ANNOTATION_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--permissive",
        action="store_true",
        help="Use legacy permissive filter (not recommended)",
    )
    parser.add_argument("--copy", action="store_true", help="Copy images into category folders")
    parser.add_argument("--make-contact-sheets", action="store_true")
    parser.add_argument("--contact-sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--max-sheet-tiles", type=int, default=24)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    image_root = _resolve(args.image_root)
    annotation_json = _resolve(args.annotation_json)
    output_root = _resolve(args.output_root)
    metadata_path = _resolve(args.metadata_out)
    report_path = _resolve(args.report_output)
    strict = not args.permissive

    if not image_root.is_dir():
        raise FileNotFoundError(f"--image-root not found: {image_root}")
    if not annotation_json.is_file():
        raise FileNotFoundError(f"--annotation-json not found: {annotation_json}")

    output_root.mkdir(parents=True, exist_ok=True)
    for cat in OUTPUT_CATEGORIES:
        _ensure_dir(output_root / cat)

    images, anns_by_image, cat_id_to_name = load_coco_annotation_context(annotation_json)
    images = sorted(images, key=lambda x: int(x["id"]))

    results = []
    skipped_missing = 0
    for img_rec in images:
        image_id = int(img_rec["id"])
        file_name = str(img_rec["file_name"])
        source_path = image_root / file_name
        if not source_path.is_file():
            skipped_missing += 1
            continue
        try:
            rgb = load_image_rgb(source_path)
        except OSError:
            skipped_missing += 1
            continue

        bg_id = _background_id(image_id, file_name)
        result = classify_coco_sky_background(
            image_id=image_id,
            source_image=source_path,
            rgb=rgb,
            anns=anns_by_image.get(image_id, []),
            cat_id_to_name=cat_id_to_name,
            background_id=bg_id,
            strict=strict,
        )
        results.append(result)

    selected_ids = select_balanced_approved(
        results, max_images=args.max_images, seed=args.seed
    )

    metadata_rows: list[dict] = []
    approved_copied = 0

    for result in results:
        output_path_str = ""
        should_copy = False

        if result.filter_status == "accept" and result.background_id in selected_ids:
            should_copy = args.copy
            dest_dir = output_root / result.background_type
        elif result.filter_status == "review":
            should_copy = args.copy
            dest_dir = output_root / "review"
        else:
            dest_dir = None

        if should_copy and dest_dir is not None:
            dest = _unique_dest(dest_dir, result.source_image.name, result.image_id)
            try:
                _copy_image(result.source_image, dest)
                output_path_str = str(dest.resolve())
                if result.filter_status == "accept":
                    approved_copied += 1
            except OSError as exc:
                print(f"COPY FAILED: {result.source_image} -> {dest}: {exc}", file=sys.stderr)
                output_path_str = ""

        metadata_rows.append(
            result_to_strict_metadata_row(result, output_path=output_path_str)
        )

    df = pd.DataFrame(metadata_rows, columns=STRICT_METADATA_COLUMNS)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False)

    write_coco_sky_report(
        report_path,
        total_scanned=len(images),
        rows=metadata_rows,
        output_root=output_root,
        metadata_csv=metadata_path,
        approved_copied=approved_copied,
        strict=strict,
    )

    reject_reasons = Counter(
        row["filter_reason"]
        for row in metadata_rows
        if row.get("filter_status") == "reject"
    )

    print(f"Total scanned: {len(images)}")
    print(f"Classified: {len(results)}")
    print(f"Skipped missing/unreadable: {skipped_missing}")
    print(f"Mode: {'strict' if strict else 'permissive'}")
    print(f"Approved selected: {len(selected_ids)}")
    if args.copy:
        print(f"Approved copied: {approved_copied}")
        review_copied = int(
            df[
                (df["filter_status"] == "review")
                & df["output_path"].fillna("").astype(str).str.strip().ne("")
            ].shape[0]
        )
        print(f"Review copied: {review_copied}")
    print("\nAccepted by background_type:")
    for cat in V1_APPROVED_CATEGORIES:
        n = int(((df["filter_status"] == "accept") & (df["background_type"] == cat)).sum())
        if n:
            print(f"  {cat}: {n}")
    print(f"\nreview: {int((df['filter_status'] == 'review').sum())}")
    print(f"reject: {int((df['filter_status'] == 'reject').sum())}")
    print("\nTop reject reasons:")
    for reason, count in reject_reasons.most_common(12):
        print(f"  {reason}: {count}")
    print(f"\nWrote metadata: {metadata_path}")
    print(f"Wrote report: {report_path}")

    if args.make_contact_sheets:
        sheet_dir = _resolve(args.contact_sheet_dir)
        prefix = "coco_sky_strict" if strict else "coco_sky_background"
        for cat in list(V1_APPROVED_CATEGORIES) + ["review", "reject"]:
            sub = df[df["background_type"] == cat]
            if cat in V1_APPROVED_CATEGORIES:
                sub = sub[sub["filter_status"] == "accept"]
            if sub.empty:
                continue
            path = sheet_dir / f"{prefix}_{cat}.png"
            saved = _save_contact_sheet(
                sub,
                path,
                path_column="output_path",
                fallback_column="source_image",
                max_tiles=args.max_sheet_tiles,
                seed=args.seed,
            )
            if saved:
                print(f"Saved contact sheet: {saved}")


if __name__ == "__main__":
    main()
