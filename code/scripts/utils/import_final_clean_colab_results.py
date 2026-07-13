#!/usr/bin/env python3
"""Extract final-clean Colab result zips into outputs/ layout."""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Prefer repo-local zip drop folder; fall back to sibling project/results.
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "project" / "results"
    if (PROJECT_ROOT / "project" / "results").is_dir()
    else PROJECT_ROOT / "results"
)

if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))


def _ensure_parent(path: Path) -> None:
    if os.name == "nt":
        from drone_stress.assets import win_long_path

        os.makedirs(win_long_path(path.parent), exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)


def _write_bytes(path: Path, data: bytes) -> None:
    _ensure_parent(path)
    if os.name == "nt":
        from drone_stress.assets import win_long_path

        with open(win_long_path(path), "wb") as dst:
            dst.write(data)
    else:
        path.write_bytes(data)


def _extract_zip(zip_path: Path, dest: Path, strip_prefix: str | None = None) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            out_name = name
            if strip_prefix and out_name.startswith(strip_prefix):
                out_name = out_name[len(strip_prefix) :]
            out_path = dest / out_name
            _write_bytes(out_path, zf.read(name))
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Import final-clean Colab artifacts.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Folder containing downloaded zips (default: project/results)",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    mappings = [
        (
            "yolo11n_drone_colab-9*.zip",
            PROJECT_ROOT / "outputs/training/yolo",
            None,
        ),
        (
            "rtdetr_l_drone_colab-9*.zip",
            PROJECT_ROOT / "outputs/training/rtdetr",
            None,
        ),
        (
            "final_clean_full_curated_v1*.zip",
            PROJECT_ROOT / "outputs/evaluation",
            None,
        ),
    ]

    for pattern, dest_root, strip in mappings:
        matches = sorted(results_dir.glob(pattern))
        if not matches:
            print(f"SKIP (no zip): {pattern}")
            continue
        zip_path = matches[-1]
        n = _extract_zip(zip_path, dest_root, strip_prefix=strip)
        print(f"Extracted {n} files from {zip_path.name} -> {dest_root}")

    from drone_stress.assets import file_accessible

    yolo_best = PROJECT_ROOT / "outputs/training/yolo/yolo11n_drone_colab-9/weights/best.pt"
    rtdetr_best = PROJECT_ROOT / "outputs/training/rtdetr/rtdetr_l_drone_colab-9/weights/best.pt"
    corrected = (
        PROJECT_ROOT
        / "outputs/evaluation/final_clean_full_curated_v1/combined_metrics/summary_by_model_by_eval_subset_CORRECTED.csv"
    )
    print("\nVerification:")
    print(f"  YOLO best.pt:    {file_accessible(yolo_best)}  {yolo_best}")
    print(f"  RT-DETR best.pt: {file_accessible(rtdetr_best)}  {rtdetr_best}")
    print(f"  CORRECTED CSV:   {file_accessible(corrected)}  {corrected}")


if __name__ == "__main__":
    main()
