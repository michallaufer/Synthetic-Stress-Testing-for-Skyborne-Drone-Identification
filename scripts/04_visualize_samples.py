#!/usr/bin/env python3
"""Build contact sheets and optional HTML audit report for pilot dataset QA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drone_stress.config import PilotConfig
from drone_stress.qa_visualize import (
    build_contact_sheet,
    build_crops_sheet,
    build_html_audit_report,
    parse_bbox,
)

COLOR_DRONE = (0, 220, 0)
COLOR_DISTRACTOR = (255, 140, 0)


def _legacy_draw_bbox(
    rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int,
    draw_center: bool,
) -> np.ndarray:
    import cv2

    x, y, w, h = bbox
    bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (x, y), (x + w, y + h), color[::-1], thickness)
    if draw_center:
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        cv2.circle(bgr, (cx, cy), max(2, thickness), color[::-1], -1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_legacy_contact_sheet(
    metadata_csv: Path,
    images_dir: Path,
    output_path: Path,
    num_samples: int,
    cols: int,
    seed: int,
    bbox_thickness: int,
    draw_center: bool,
) -> Path:
    df = pd.read_csv(metadata_csv)
    n = min(num_samples, len(df))
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.6))
    if rows == 1 and cols == 1:
        axes_list = [[axes]]
    elif rows == 1:
        axes_list = [list(axes)]
    elif cols == 1:
        axes_list = [[ax] for ax in axes]
    else:
        axes_list = [list(r) for r in axes]

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes_list[r][c]
        ax.axis("off")
        if i >= n:
            continue
        row = sample.iloc[i]
        image_path = images_dir / row["image_id"]
        if not image_path.is_file():
            ax.set_title("missing")
            continue
        bgr = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bbox = parse_bbox(str(row.get("bbox", "[]")))
        if bbox is not None:
            color = COLOR_DRONE if row["target_present"] else COLOR_DISTRACTOR
            rgb = _legacy_draw_bbox(rgb, bbox, color, bbox_thickness, draw_center)
        ax.imshow(rgb)
        ax.set_title(
            f"{row['subset']}\n"
            f"size={row['object_size_px']}px  bg={row['background_type']}",
            fontsize=7,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create pilot dataset contact sheets for visual QA."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilot.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--num-samples", type=int, default=24, help="Images per contact sheet")
    parser.add_argument("--cols", type=int, default=6, help="Grid columns")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument(
        "--qa-mode",
        action="store_true",
        help=(
            "Full QA outputs: labeled boxes, detailed crop tiles, "
            "pilot_contact_sheet_qa.png, pilot_contact_sheet_crops_qa.png, "
            "and outputs/reports/pilot_audit.html"
        ),
    )
    parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Legacy mode: single ambiguous bbox overlay only",
    )
    parser.add_argument("--bbox-thickness", type=int, default=3, help="BBox line thickness (px)")
    parser.add_argument(
        "--no-center-dot",
        action="store_true",
        help="Disable drawing bbox center dots",
    )
    parser.add_argument(
        "--no-write-crops",
        action="store_true",
        help="Skip saving the zoomed crops sheet",
    )
    parser.add_argument("--crop-pad-px", type=int, default=16, help="Padding around bbox for crop")
    parser.add_argument("--crop-size-px", type=int, default=200, help="Crop tile size (px)")
    parser.add_argument(
        "--html-rows",
        type=int,
        default=12,
        help="Rows in HTML audit report (--qa-mode only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default depends on --qa-mode)",
    )
    args = parser.parse_args()

    if args.no_qa and args.qa_mode:
        parser.error("Use only one of --qa-mode or --no-qa")

    cfg = PilotConfig.from_yaml(
        args.config if args.config.is_absolute() else PROJECT_ROOT / args.config,
        project_root=PROJECT_ROOT,
    )

    if not cfg.metadata_csv.is_file():
        raise FileNotFoundError(
            f"Metadata not found: {cfg.metadata_csv}. Run scripts/03_generate_synthetic.py first."
        )

    sheets_dir = PROJECT_ROOT / "outputs" / "contact_sheets"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    draw_center = not args.no_center_dot

    if args.no_qa:
        out = args.output or (sheets_dir / "pilot_contact_sheet.png")
        path = build_legacy_contact_sheet(
            cfg.metadata_csv,
            cfg.images_dir,
            out,
            num_samples=args.num_samples,
            cols=args.cols,
            seed=args.seed,
            bbox_thickness=args.bbox_thickness,
            draw_center=draw_center,
        )
        print(f"Saved contact sheet (legacy): {path}")
        return

    use_full_qa = args.qa_mode
    stem = "pilot_contact_sheet_qa" if use_full_qa else "pilot_contact_sheet"
    out = args.output or (sheets_dir / f"{stem}.png")
    crops_out = sheets_dir / (
        "pilot_contact_sheet_crops_qa.png" if use_full_qa else "pilot_contact_sheet_crops.png"
    )

    path = build_contact_sheet(
        cfg.metadata_csv,
        cfg.images_dir,
        out,
        num_samples=args.num_samples,
        cols=args.cols,
        seed=args.seed,
        bbox_thickness=args.bbox_thickness,
        draw_center=draw_center,
    )
    print(f"Saved contact sheet: {path}")
    print("Legend: green = drone (target), orange = distractor; labels at each box")

    if not args.no_write_crops:
        crops_path = build_crops_sheet(
            cfg.metadata_csv,
            cfg.images_dir,
            crops_out,
            num_samples=args.num_samples,
            cols=args.cols,
            seed=args.seed,
            crop_pad_px=args.crop_pad_px,
            crop_size_px=args.crop_size_px,
            bbox_thickness=args.bbox_thickness,
        )
        print(f"Saved crops sheet: {crops_path}")
        print("Crop tiles: image_id, subset, object_type, class, asset_id, bbox")

    if use_full_qa:
        html_path = build_html_audit_report(
            cfg.metadata_csv,
            cfg.images_dir,
            reports_dir / "pilot_audit.html",
            num_rows=args.html_rows,
            seed=args.seed,
            crop_pad_px=args.crop_pad_px,
            crop_size_px=args.crop_size_px,
        )
        print(f"Saved HTML audit: {html_path}")
    elif not args.qa_mode:
        print("Tip: pass --qa-mode for *_qa.png outputs and outputs/reports/pilot_audit.html")


if __name__ == "__main__":
    main()
