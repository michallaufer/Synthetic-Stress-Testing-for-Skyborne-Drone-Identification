#!/usr/bin/env python3
"""Filter annotation rows by flying-object scene / viewpoint compatibility.

Large bboxes are NOT auto-rejected. Size is a QA signal only; synthetic
generation controls final pasted object size. See README.md - Scene filtering.

Optional CLIP (--use-clip) adds a pre-filtering signal; it does not replace
heuristics or manual visual QA.

Typical pipeline:
  yolo_to_annotations.py -> 06_filter_flying_object_scenes.py -> 02_extract_assets.py annotate
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.scene_filter import (
    ClipSceneScorer,
    ClipScoreResult,
    Disposition,
    FilterThresholds,
    SceneFeatures,
    classify_scene_row,
    compute_scene_features,
    load_image_rgb,
    merge_heuristic_and_clip,
)

DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "flying_object_scene_filter_report.txt"
DEFAULT_CONTACT_SHEET = (
    PROJECT_ROOT / "outputs" / "contact_sheets" / "scene_filter_clip_contact_sheet.png"
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_image_path(row: pd.Series, image_root: Path) -> Path:
    for col in ("image_path", "filename"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            rel = Path(str(row[col]))
            if rel.is_absolute():
                return rel
            return image_root / rel
    raise ValueError(f"Row missing image_path/filename: {row.to_dict()}")


def _clip_columns(clip: ClipScoreResult) -> dict:
    return {
        "clip_used": clip.clip_used,
        "clip_top_prompt": clip.clip_top_prompt,
        "clip_top_score": clip.clip_top_score,
        "clip_second_prompt": clip.clip_second_prompt,
        "clip_second_score": clip.clip_second_score,
        "clip_margin": clip.clip_margin,
        "clip_viewpoint_label": clip.clip_viewpoint_label,
        "clip_filter_hint": clip.clip_filter_hint,
    }


def filter_annotations(
    df: pd.DataFrame,
    image_root: Path,
    thresholds: FilterThresholds,
    filter_purpose: str = "asset_extraction",
    only_disposition: str | None = None,
    clip_scorer: ClipSceneScorer | None = None,
    clip_margin_threshold: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    image_cache: dict[str, object] = {}
    feature_cache: dict[tuple, dict] = {}
    clip_cache: dict[tuple, ClipScoreResult] = {}
    out_rows: list[dict] = []
    stats = {
        "rows_in": len(df),
        "rows_out": 0,
        "filter_purpose": filter_purpose,
        "clip_used": clip_scorer is not None,
        "by_disposition": {d.value: 0 for d in Disposition},
        "by_heuristic_disposition": {d.value: 0 for d in Disposition},
        "by_filter_reason": {},
        "by_viewpoint": {},
        "by_clip_viewpoint": {},
        "by_merge_policy_reason": {},
        "heuristic_accept_preserved_count": 0,
        "heuristic_accept_moved_to_review_count": 0,
        "heuristic_accept_moved_to_reject_count": 0,
        "bbox_max_dims": [],
    }

    for _, row in df.iterrows():
        image_path = _resolve_image_path(row, image_root)
        cache_key = str(image_path.resolve())

        if cache_key not in image_cache:
            try:
                image_cache[cache_key] = load_image_rgb(image_path)
            except OSError:
                image_cache[cache_key] = None

        rgb = image_cache[cache_key]
        if rgb is None:
            continue

        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        bbox = (x, y, w, h)
        feat_key = (cache_key, x, y, w, h)
        if feat_key not in feature_cache:
            feature_cache[feat_key] = compute_scene_features(rgb, bbox, thresholds).__dict__

        features = SceneFeatures(**feature_cache[feat_key])
        class_name = str(row.get("class_name", "object"))
        heuristic = classify_scene_row(features, class_name, thresholds)
        stats["by_heuristic_disposition"][heuristic.disposition.value] += 1

        if clip_scorer is not None:
            clip_key = (cache_key, x, y, w, h)
            if clip_key not in clip_cache:
                clip_cache[clip_key] = clip_scorer.score_rgb(rgb, bbox)
            clip = clip_cache[clip_key]
            merged = merge_heuristic_and_clip(
                heuristic,
                clip,
                features,
                thresholds,
                clip_margin_threshold,
                filter_purpose=filter_purpose,
            )
        else:
            clip = ClipScoreResult.unused()
            merged = merge_heuristic_and_clip(
                heuristic,
                clip,
                features,
                thresholds,
                clip_margin_threshold,
                filter_purpose=filter_purpose,
            )

        final_disp = merged.disposition

        if heuristic.disposition == Disposition.ACCEPT:
            if final_disp == Disposition.ACCEPT:
                stats["heuristic_accept_preserved_count"] += 1
            elif final_disp == Disposition.REVIEW:
                stats["heuristic_accept_moved_to_review_count"] += 1
            elif final_disp == Disposition.REJECT:
                stats["heuristic_accept_moved_to_reject_count"] += 1

        if only_disposition and final_disp.value != only_disposition:
            continue

        out = row.to_dict()
        out.update(
            {
                "filter_purpose": filter_purpose,
                "heuristic_disposition": heuristic.disposition.value,
                "filter_disposition": final_disp.value,
                "filter_reason": heuristic.filter_reason,
                "viewpoint_label": heuristic.viewpoint_label,
                "clip_low_margin": merged.clip_low_margin,
                "merge_policy_reason": merged.merge_policy_reason,
                "sky_ratio_upper": round(features.sky_ratio_upper, 4),
                "sky_ratio_lower": round(features.sky_ratio_lower, 4),
                "bbox_area_ratio": round(features.bbox_area_ratio, 4),
                "bbox_center_y_norm": round(features.bbox_center_y_norm, 4),
                "bbox_max_dim_ratio": round(features.bbox_max_dim_ratio, 4),
                "is_large_bbox": features.is_large_bbox,
            }
        )
        out.update(_clip_columns(clip))
        out_rows.append(out)
        stats["rows_out"] += 1
        stats["by_disposition"][final_disp.value] += 1
        stats["by_filter_reason"][heuristic.filter_reason] = (
            stats["by_filter_reason"].get(heuristic.filter_reason, 0) + 1
        )
        stats["by_viewpoint"][heuristic.viewpoint_label] = (
            stats["by_viewpoint"].get(heuristic.viewpoint_label, 0) + 1
        )
        if clip.clip_used:
            stats["by_clip_viewpoint"][clip.clip_viewpoint_label] = (
                stats["by_clip_viewpoint"].get(clip.clip_viewpoint_label, 0) + 1
            )
        stats["by_merge_policy_reason"][merged.merge_policy_reason] = (
            stats["by_merge_policy_reason"].get(merged.merge_policy_reason, 0) + 1
        )
        stats["bbox_max_dims"].append(max(int(w), int(h)))

    stats["by_filter_purpose"] = {filter_purpose: stats["rows_out"]}

    return pd.DataFrame(out_rows), stats


def _draw_tile_title(row: pd.Series, clip_enabled: bool) -> str:
    line1 = f"{row.get('class_name', '?')} | {row.get('filter_disposition', '?')}"
    line2 = f"heuristic: {row.get('filter_reason', '?')}"
    if clip_enabled and row.get("clip_used"):
        margin = float(row.get("clip_margin", 0.0))
        line3 = f"clip: {row.get('clip_viewpoint_label', '?')} margin={margin:.3f}"
    else:
        line3 = "clip: (off)"
    line4 = f"merge: {row.get('merge_policy_reason', '?')}"
    return "\n".join([line1, line2, line3, line4])


def build_contact_sheet(
    df: pd.DataFrame,
    image_root: Path,
    output_path: Path,
    num_samples: int,
    seed: int,
    clip_enabled: bool,
    cols: int = 4,
) -> Path:
    if df.empty:
        raise ValueError("Cannot build contact sheet from empty dataframe.")

    sample = df.sample(n=min(num_samples, len(df)), random_state=seed).reset_index(drop=True)
    n = len(sample)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 5.0))
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    elif rows == 1:
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
        row = sample.iloc[i]
        image_path = _resolve_image_path(row, image_root)
        try:
            rgb = load_image_rgb(image_path)
        except OSError:
            ax.set_title("missing image", fontsize=7)
            continue
        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        vis = rgb.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ax.imshow(vis)
        ax.set_title(_draw_tile_title(row, clip_enabled), fontsize=8, loc="left", pad=8, wrap=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_report(
    stats: dict,
    report_path: Path,
    input_csv: Path,
    output_csv: Path,
    thresholds: FilterThresholds,
    clip_margin_threshold: float | None = None,
) -> None:
    dims = stats["bbox_max_dims"]
    if dims:
        min_d, med_d, max_d = min(dims), int(statistics.median(dims)), max(dims)
    else:
        min_d = med_d = max_d = 0

    lines = [
        "Flying-object scene filter report",
        f"input_csv: {input_csv.resolve()}",
        f"output_csv: {output_csv.resolve()}",
        f"filter_purpose: {stats.get('filter_purpose', 'asset_extraction')}",
        f"clip_used: {stats.get('clip_used', False)}",
        "",
        "Policy: large bbox alone is NOT auto-rejected (asset_extraction mode).",
        "Reject when heuristic hard-reject or CLIP bad label + weak sky + low object.",
    ]
    if stats.get("clip_used"):
        lines.extend(
            [
                "CLIP is optional pre-filtering only (not ground truth).",
                f"clip_margin_threshold: {clip_margin_threshold}",
            ]
        )
    lines.extend(
        [
            "",
            f"rows_in: {stats['rows_in']}",
            f"rows_out: {stats['rows_out']}",
            "",
            "by_filter_purpose:",
        ]
    )
    for k, v in sorted(stats.get("by_filter_purpose", {}).items()):
        lines.append(f"  {k}: {v}")
    lines.extend(
        [
            "",
            "by_final_disposition:",
        ]
    )
    for k, v in sorted(stats["by_disposition"].items()):
        lines.append(f"  {k}: {v}")
    if stats.get("clip_used"):
        lines.append("")
        lines.append("by_heuristic_disposition:")
        for k, v in sorted(stats["by_heuristic_disposition"].items()):
            lines.append(f"  {k}: {v}")
        lines.extend(
            [
                "",
                f"heuristic_accept_preserved_count: {stats.get('heuristic_accept_preserved_count', 0)}",
                f"heuristic_accept_moved_to_review_count: {stats.get('heuristic_accept_moved_to_review_count', 0)}",
                f"heuristic_accept_moved_to_reject_count: {stats.get('heuristic_accept_moved_to_reject_count', 0)}",
            ]
        )
    lines.append("")
    lines.append("by_heuristic_filter_reason:")
    for k, v in sorted(stats["by_filter_reason"].items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {v}")
    if stats.get("by_clip_viewpoint"):
        lines.append("")
        lines.append("by_clip_viewpoint_label:")
        for k, v in sorted(stats["by_clip_viewpoint"].items(), key=lambda x: -x[1]):
            lines.append(f"  {k}: {v}")
    if stats.get("by_merge_policy_reason"):
        lines.append("")
        lines.append("by_merge_policy_reason:")
        for k, v in sorted(stats["by_merge_policy_reason"].items(), key=lambda x: -x[1]):
            lines.append(f"  {k}: {v}")
    lines.extend(
        [
            "",
            "bbox max(w,h) px - min / median / max (QA signal only):",
            f"  {min_d} / {med_d} / {max_d}",
            "",
            "thresholds:",
            f"  large_bbox_area_ratio: {thresholds.large_bbox_area_ratio}",
            f"  large_bbox_max_dim_ratio: {thresholds.large_bbox_max_dim_ratio}",
            f"  sky_ratio_upper_accept: {thresholds.sky_ratio_upper_accept}",
            f"  sky_ratio_upper_reject: {thresholds.sky_ratio_upper_reject}",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter annotation CSV rows by viewpoint compatibility for flying objects. "
            "Large bboxes go to review unless combined with bad scene clues. "
            "Optional CLIP pre-filtering (--use-clip) does not replace heuristics or QA."
        )
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--filter-purpose",
        choices=("asset_extraction", "real_eval"),
        default="asset_extraction",
        help=(
            "Merge policy: asset_extraction (permissive, default) preserves heuristic "
            "accepts; real_eval is stricter on large/close-up scenes"
        ),
    )
    parser.add_argument(
        "--only-disposition",
        choices=[d.value for d in Disposition],
        default=None,
        help="Optional: write only accept, review, or reject rows",
    )
    parser.add_argument("--large-bbox-area-ratio", type=float, default=0.10)
    parser.add_argument("--large-bbox-max-dim-ratio", type=float, default=0.28)
    parser.add_argument("--sky-ratio-upper-accept", type=float, default=0.28)
    parser.add_argument("--sky-ratio-upper-horizon", type=float, default=0.18)
    parser.add_argument("--sky-ratio-upper-reject", type=float, default=0.12)
    parser.add_argument("--object-low-y-center", type=float, default=0.72)
    parser.add_argument(
        "--use-clip",
        action="store_true",
        help="Enable optional CLIP pre-filtering (requires transformers + torch)",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model id (default: openai/clip-vit-base-patch32)",
    )
    parser.add_argument(
        "--clip-device",
        default="auto",
        help="Device for CLIP: auto, cpu, cuda, or mps",
    )
    parser.add_argument(
        "--clip-margin-threshold",
        type=float,
        default=0.05,
        help="Flag clip_low_margin when top-two CLIP group margin is below this",
    )
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument(
        "--make-contact-sheets",
        action="store_true",
        help="Save PNG contact sheet for visual QA (recommended with --use-clip)",
    )
    parser.add_argument(
        "--contact-sheet-output",
        type=Path,
        default=DEFAULT_CONTACT_SHEET,
    )
    parser.add_argument("--contact-sheet-samples", type=int, default=24)
    parser.add_argument("--contact-sheet-seed", type=int, default=42)
    args = parser.parse_args()

    input_csv = _resolve(args.annotations)
    image_root = _resolve(args.image_root)
    output_csv = _resolve(args.output_csv)
    report_path = _resolve(args.report)

    if not input_csv.is_file():
        raise FileNotFoundError(f"Annotations not found: {input_csv}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    thresholds = FilterThresholds(
        large_bbox_area_ratio=args.large_bbox_area_ratio,
        large_bbox_max_dim_ratio=args.large_bbox_max_dim_ratio,
        sky_ratio_upper_accept=args.sky_ratio_upper_accept,
        sky_ratio_upper_horizon=args.sky_ratio_upper_horizon,
        sky_ratio_upper_reject=args.sky_ratio_upper_reject,
        object_low_y_center=args.object_low_y_center,
    )

    clip_scorer = None
    if args.use_clip:
        print(f"Loading CLIP model: {args.clip_model}")
        clip_scorer = ClipSceneScorer(
            model_name=args.clip_model,
            device=args.clip_device,
            batch_size=args.clip_batch_size,
        )

    df = pd.read_csv(input_csv)
    for col in ("x", "y", "w", "h"):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in {input_csv}")

    out_df, stats = filter_annotations(
        df,
        image_root,
        thresholds,
        filter_purpose=args.filter_purpose,
        only_disposition=args.only_disposition,
        clip_scorer=clip_scorer,
        clip_margin_threshold=args.clip_margin_threshold,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    write_report(
        stats,
        report_path,
        input_csv,
        output_csv,
        thresholds,
        clip_margin_threshold=args.clip_margin_threshold if args.use_clip else None,
    )

    print(f"Wrote {len(out_df)} rows to {output_csv}")
    print(f"Wrote report: {report_path}")
    print(f"Filter purpose: {args.filter_purpose}")
    print(
        "Final disposition counts:",
        ", ".join(f"{k}={v}" for k, v in sorted(stats["by_disposition"].items())),
    )
    if args.use_clip:
        print(
            "Heuristic accept tracking:",
            f"preserved={stats['heuristic_accept_preserved_count']},",
            f"moved_to_review={stats['heuristic_accept_moved_to_review_count']},",
            f"moved_to_reject={stats['heuristic_accept_moved_to_reject_count']}",
        )

    if args.make_contact_sheets:
        sheet_path = _resolve(args.contact_sheet_output)
        build_contact_sheet(
            out_df,
            image_root,
            sheet_path,
            num_samples=args.contact_sheet_samples,
            seed=args.contact_sheet_seed,
            clip_enabled=args.use_clip,
        )
        print(f"Saved contact sheet: {sheet_path}")

    print(
        "\nNote: bbox size is a QA signal only. CLIP (if used) is pre-filtering only, "
        "not ground truth. Synthetic generation controls final object size."
    )


if __name__ == "__main__":
    main()
