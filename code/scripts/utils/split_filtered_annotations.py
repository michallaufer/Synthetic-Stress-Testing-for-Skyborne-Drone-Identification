#!/usr/bin/env python3
"""Split a filtered annotation CSV by final disposition and class.

Prepares smaller CSVs for scripts/02_extract_assets.py annotate without running
extraction. See README.md — BirdvsDrone2 Step 2.
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
}

DISPOSITIONS = ("accept", "review", "reject")
EXTRACTION_CLASSES = ("drone", "bird")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def derive_prefix(input_csv: Path) -> str:
    """Derive output filename prefix from the input CSV stem."""
    stem = input_csv.stem
    for suffix in (
        "_annotations_filtered_clip",
        "_annotations_filtered",
        "_filtered_clip",
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


def normalize_class_column(df: pd.DataFrame, class_column: str) -> pd.DataFrame:
    out = df.copy()
    out[class_column] = out[class_column].map(normalize_class_name)
    return out


def split_filtered_annotations(
    df: pd.DataFrame,
    *,
    prefix: str,
    output_dir: Path,
    disposition_column: str = "filter_disposition",
    class_column: str = "class_name",
) -> list[tuple[Path, int]]:
    if disposition_column not in df.columns:
        raise ValueError(f"Missing column {disposition_column!r} in input CSV")
    if class_column not in df.columns:
        raise ValueError(f"Missing column {class_column!r} in input CSV")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, int]] = []

    for disposition in DISPOSITIONS:
        subset = df[df[disposition_column] == disposition]
        path = output_dir / f"{prefix}_{disposition}_all.csv"
        subset.to_csv(path, index=False)
        written.append((path, len(subset)))

        for class_name in EXTRACTION_CLASSES:
            class_subset = subset[subset[class_column] == class_name]
            class_path = output_dir / f"{prefix}_{disposition}_{class_name}.csv"
            class_subset.to_csv(class_path, index=False)
            written.append((class_path, len(class_subset)))

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split a filtered annotation CSV by filter_disposition and class_name. "
            "Writes accept/review/reject CSVs for extraction prep (no extraction run)."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Filtered annotation CSV (must include filter_disposition and class_name)",
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
        help=(
            "Filename prefix for outputs (default: derived from input stem, "
            "e.g. birdvsdrone2 from birdvsdrone2_annotations_filtered_clip.csv)"
        ),
    )
    parser.add_argument(
        "--disposition-column",
        default="filter_disposition",
        help="Column with final disposition: accept, review, reject",
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
    df = normalize_class_column(df, args.class_column)

    unknown_dispositions = sorted(
        set(df[args.disposition_column].dropna().astype(str).unique()) - set(DISPOSITIONS)
    )
    if unknown_dispositions:
        print(
            "Warning: unexpected disposition values (rows kept in all.csv only if matched):",
            ", ".join(unknown_dispositions),
        )

    written = split_filtered_annotations(
        df,
        prefix=prefix,
        output_dir=output_dir,
        disposition_column=args.disposition_column,
        class_column=args.class_column,
    )

    print(f"Input:  {input_csv} ({len(df)} rows)")
    print(f"Prefix: {prefix}")
    print(f"Output: {output_dir.resolve()}")
    print("")
    print("Counts per output file:")
    for path, count in written:
        print(f"  {path.name}: {count}")

    print("")
    print(
        "Next: run 02_extract_assets.py annotate on accept (and optionally review) "
        "class CSVs — not run automatically by this script."
    )


if __name__ == "__main__":
    main()
