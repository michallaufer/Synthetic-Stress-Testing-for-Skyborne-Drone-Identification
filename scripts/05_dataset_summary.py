#!/usr/bin/env python3
"""Summarize a generated synthetic pilot dataset from metadata.csv.

Sanity-check utility only. Does not train or evaluate models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drone_stress.config import PilotConfig

DEFAULT_OUTPUT_TXT = PROJECT_ROOT / "outputs" / "reports" / "pilot_dataset_summary.txt"

AUDIT_COLUMNS = [
    "image_id",
    "subset",
    "background_type",
    "background_filename",
    "foreground_asset_source",
    "target_asset_id",
    "distractor_asset_ids",
    "target_bbox",
    "distractor_bboxes",
    "pasted_object_summary",
]


def _section(title: str, series: pd.Series) -> str:
    lines = [title, series.sort_values(ascending=False).to_string()]
    return "\n".join(lines) + "\n"


def build_summary_text(df: pd.DataFrame, metadata_path: Path) -> str:
    parts: list[str] = [
        "Pilot dataset summary",
        f"metadata: {metadata_path}",
        "",
        f"1. Total images: {len(df)}",
        "",
        "2. Counts by subset:",
        _section("", df["subset"].value_counts(dropna=False)),
        "3. Counts by background_type:",
        _section("", df["background_type"].value_counts(dropna=False)),
        "4. Counts by object_size_px:",
        _section("", df["object_size_px"].value_counts(dropna=False)),
        "5. Counts by gaussian_noise_sigma:",
        _section("", df["gaussian_noise_sigma"].value_counts(dropna=False)),
        "6. Counts by blur_level:",
        _section("", df["blur_level"].value_counts(dropna=False)),
        "7. Counts by target_present:",
        _section("", df["target_present"].value_counts(dropna=False)),
        "8. Counts by distractor_classes:",
        _section("", df["distractor_classes"].value_counts(dropna=False)),
    ]

    if "foreground_asset_source" in df.columns:
        parts.extend(
            [
                "9. Counts by foreground_asset_source:",
                _section("", df["foreground_asset_source"].value_counts(dropna=False)),
            ]
        )

    if "background_filename" in df.columns:
        parts.extend(
            [
                "10. Counts by background_filename:",
                _section("", df["background_filename"].value_counts(dropna=False)),
            ]
        )

    audit_cols = [c for c in AUDIT_COLUMNS if c in df.columns]
    parts.extend(
        [
            "11. Sample audit rows (background_filename, target_bbox, distractor_bboxes, "
            "target_asset_id, distractor_asset_ids):",
            df[audit_cols].head(8).to_string(index=False),
            "",
        ]
    )

    parts.extend(
        [
            "12. First 5 rows (full):",
            df.head(5).to_string(index=False),
            "",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generated pilot dataset metadata.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilot.yaml",
        help="Path to YAML config (default: configs/pilot.yaml)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata.csv (overrides config output path if set)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_TXT,
        help="Where to write the text summary (default: outputs/reports/pilot_dataset_summary.txt)",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = PilotConfig.from_yaml(config_path, project_root=PROJECT_ROOT)

    metadata_path = args.metadata
    if metadata_path is None:
        metadata_path = cfg.metadata_csv
    elif not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Generate the pilot dataset first:\n"
            "  python scripts/03_generate_synthetic.py --config configs/pilot.yaml"
        )

    df = pd.read_csv(metadata_path)
    if df.empty:
        raise ValueError(f"Metadata file has no rows: {metadata_path}")

    summary_text = build_summary_text(df, metadata_path)
    print(summary_text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary_text, encoding="utf-8")
    print(f"Wrote text summary: {args.output}")


if __name__ == "__main__":
    main()
