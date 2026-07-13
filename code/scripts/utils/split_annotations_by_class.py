#!/usr/bin/env python3
"""Split an annotation CSV into per-class files plus a combined copy.

Useful after adapters (e.g. coco_to_annotations.py) to prepare class-specific
extraction inputs for scripts/02_extract_assets.py annotate.

Does not modify the input CSV, run SAM2, or generate synthetic images.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "annotations" / "splits"

CLASS_ALIASES: dict[str, str] = {
    "birds": "bird",
    "drones": "drone",
    "uav": "drone",
    "quadcopter": "drone",
    "aeroplane": "airplane",
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def derive_prefix(input_csv: Path) -> str:
    """Derive output filename prefix from the input CSV stem."""
    stem = input_csv.stem
    for suffix in (
        "_distractors_annotations",
        "_annotations_filtered_clip",
        "_annotations_filtered",
        "_annotations",
        "_filtered",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)].rstrip("_")
    return stem


def normalize_class_name(name: object) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    normalized = re.sub(r"\s+", " ", str(name).strip().lower())
    return CLASS_ALIASES.get(normalized, normalized)


def safe_class_slug(class_name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", class_name).strip("_")
    return slug or "unknown"


def split_annotations_by_class(
    df: pd.DataFrame,
    *,
    prefix: str,
    output_dir: Path,
    class_column: str,
) -> list[tuple[Path, int]]:
    if class_column not in df.columns:
        raise ValueError(f"Missing column {class_column!r} in input CSV")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, int]] = []

    all_path = output_dir / f"{prefix}_all.csv"
    df.to_csv(all_path, index=False)
    written.append((all_path, len(df)))

    classes = sorted(c for c in df[class_column].unique() if c)
    for class_name in classes:
        subset = df[df[class_column] == class_name]
        slug = safe_class_slug(class_name)
        class_path = output_dir / f"{prefix}_{slug}.csv"
        subset.to_csv(class_path, index=False)
        written.append((class_path, len(subset)))

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split an annotation CSV by class_name into per-class files "
            "and a combined {prefix}_all.csv (does not modify the input)."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Annotation CSV (must include class column)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for split CSV outputs (default: data/processed/annotations/splits)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix for outputs (default: derived from input stem)",
    )
    parser.add_argument(
        "--class-column",
        default="class_name",
        help="Column with object class (normalized to lowercase)",
    )
    args = parser.parse_args()

    input_csv = _resolve(args.input_csv)
    output_dir = _resolve(args.output_dir)
    prefix = args.prefix or derive_prefix(input_csv)

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if args.class_column not in df.columns:
        raise ValueError(f"Missing column {args.class_column!r} in {input_csv}")

    df = df.copy()
    df[args.class_column] = df[args.class_column].map(normalize_class_name)

    written = split_annotations_by_class(
        df,
        prefix=prefix,
        output_dir=output_dir,
        class_column=args.class_column,
    )

    print(f"Input:  {input_csv} ({len(df)} rows, unchanged)")
    print(f"Prefix: {prefix}")
    print(f"Output: {output_dir.resolve()}")
    print("")
    print("Counts by class:")
    for class_name in sorted(c for c in df[args.class_column].unique() if c):
        count = int((df[args.class_column] == class_name).sum())
        print(f"  {class_name}: {count}")
    print("")
    print("Output files:")
    for path, count in written:
        print(f"  {path.name}: {count}")


if __name__ == "__main__":
    main()
