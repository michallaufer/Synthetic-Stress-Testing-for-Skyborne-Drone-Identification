#!/usr/bin/env python3
"""Create Ultralytics YOLO train/val/test split from a synthetic dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.train_split import create_ultralytics_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create YOLO dataset.yaml + train/val/test split.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data/synthetic/full_curated_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data/training/full_curated_v1_train_split",
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None, help="Smoke split size cap")
    parser.add_argument(
        "--stratify-col",
        type=str,
        default="subset",
        help="Metadata column for per-group splitting (default: subset)",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else PROJECT_ROOT / args.dataset_root
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    metadata = args.metadata
    if metadata and not metadata.is_absolute():
        metadata = PROJECT_ROOT / metadata

    print(
        "WARNING: If dataset-root is full_curated_v1, training on this split and evaluating "
        "on the same benchmark causes train/test leakage. Use only for smoke tests or "
        "generate a separate synthetic train set with a different seed/name."
    )

    report = create_ultralytics_split(
        dataset_root,
        output_root,
        metadata_csv=metadata,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_images=args.max_images,
        stratify_col=args.stratify_col,
    )
    print("\nUltralytics split created")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
