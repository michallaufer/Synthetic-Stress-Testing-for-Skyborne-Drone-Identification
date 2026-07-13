#!/usr/bin/env python3
"""Summarize a generated synthetic pilot dataset from metadata.csv.

Sanity-check utility only. Does not train or evaluate models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

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

    if "distractor_present" in df.columns:
        parts.extend(
            [
                "9. Counts by distractor_present:",
                _section("", df["distractor_present"].value_counts(dropna=False)),
            ]
        )

    if "distractor_type" in df.columns:
        dist_type = df.loc[df["distractor_type"].astype(str).str.len() > 0, "distractor_type"]
        if not dist_type.empty:
            parts.extend(
                [
                    "10. Counts by distractor_type:",
                    _section("", dist_type.value_counts(dropna=False)),
                ]
            )

    section_num = 11
    if "foreground_asset_source" in df.columns:
        parts.extend(
            [
                f"{section_num}. Counts by foreground_asset_source:",
                _section("", df["foreground_asset_source"].value_counts(dropna=False)),
            ]
        )
        section_num += 1

    if "background_filename" in df.columns:
        parts.extend(
            [
                f"{section_num}. Counts by background_filename (top 30):",
                _section("", df["background_filename"].value_counts(dropna=False).head(30)),
            ]
        )
        section_num += 1

    if "placement_mode" in df.columns:
        parts.extend(
            [
                f"{section_num}. Counts by placement_mode:",
                _section("", df["placement_mode"].value_counts(dropna=False)),
            ]
        )
        section_num += 1
        if "drone_center_y" in df.columns:
            drone_y = pd.to_numeric(df["drone_center_y"], errors="coerce").dropna()
            if not drone_y.empty:
                parts.extend(
                    [
                        f"{section_num}. Drone center_y (px) summary:",
                        f"  min={drone_y.min():.1f} max={drone_y.max():.1f} mean={drone_y.mean():.1f}",
                        "",
                    ]
                )
                section_num += 1
        if "drone_distractor_iou" in df.columns:
            iou = pd.to_numeric(df["drone_distractor_iou"], errors="coerce").dropna()
            if not iou.empty:
                parts.extend(
                    [
                        f"{section_num}. drone_distractor_iou summary (mixed subset):",
                        f"  max={iou.max():.4f} mean={iou.mean():.4f}",
                        "",
                    ]
                )
                section_num += 1

    audit_cols = [c for c in AUDIT_COLUMNS if c in df.columns]
    parts.extend(
        [
            f"{section_num}. Sample audit rows (background_filename, target_bbox, distractor_bboxes, "
            "target_asset_id, distractor_asset_ids):",
            df[audit_cols].head(8).to_string(index=False),
            "",
        ]
    )
    section_num += 1

    parts.extend(
        [
            f"{section_num}. First 5 rows (full):",
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
            "Generate the dataset first:\n"
            f"  python scripts/03_generate_synthetic.py --config configs/{cfg.split}.yaml"
        )

    df = pd.read_csv(metadata_path)
    if df.empty:
        raise ValueError(f"Metadata file has no rows: {metadata_path}")

    summary_text = build_summary_text(df, metadata_path)
    print(summary_text)

    output_path = args.output
    if output_path == DEFAULT_OUTPUT_TXT and cfg.split:
        output_path = PROJECT_ROOT / "outputs" / "reports" / f"{cfg.split}_dataset_summary.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary_text, encoding="utf-8")
    print(f"Wrote text summary: {output_path}")


if __name__ == "__main__":
    main()
