"""Semantic usefulness filtering for COCO segmentation distractor assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXTRACTION_METHOD = "coco_segmentation_mask"

DEFAULT_MAX_PER_CLASS = {
    "bird": 100,
    "airplane": 60,
    "kite": 100,
}

REASON_PRIORITY: dict[str, int] = {
    "bird_flying_like_shape": 100,
    "airplane_wide_full_like": 100,
    "kite_ok": 100,
    "airplane_partial_but_wide": 70,
    "bird_flying_shape_technical_review": 65,
    "kite_border_touch_review": 60,
    "bird_ambiguous_shape": 50,
    "airplane_narrow_crop": 45,
    "bird_vertical_portrait_like": 40,
    "kite_small_or_fragment": 35,
    "airplane_likely_nose_tail_fragment": 30,
    "bird_vertical_technical_review": 25,
    "kite_technical_review": 20,
    "airplane_ambiguous": 15,
    "bird_technical_review": 10,
}


@dataclass
class SemanticFilterResult:
    final_distractor_status: str
    semantic_filter_reason: str
    priority: int


def _safe_ratio(numer: float, denom: float, default: float = 1.0) -> float:
    if denom <= 0:
        return default
    return float(numer / denom)


def compute_geometry(row: pd.Series) -> dict:
    crop_w = int(row["crop_x2"]) - int(row["crop_x1"])
    crop_h = int(row["crop_y2"]) - int(row["crop_y1"])
    crop_w = max(crop_w, 1)
    crop_h = max(crop_h, 1)
    bbox_w = max(float(row.get("bbox_w", 1)), 1.0)
    bbox_h = max(float(row.get("bbox_h", 1)), 1.0)
    return {
        "crop_width": crop_w,
        "crop_height": crop_h,
        "crop_aspect_ratio": round(_safe_ratio(crop_w, crop_h), 4),
        "bbox_aspect_ratio": round(_safe_ratio(bbox_w, bbox_h), 4),
    }


def _result(status: str, reason: str) -> SemanticFilterResult:
    return SemanticFilterResult(
        final_distractor_status=status,
        semantic_filter_reason=reason,
        priority=REASON_PRIORITY.get(reason, 0),
    )


def classify_bird(
    *,
    technical_label: str,
    crop_ar: float,
    bbox_ar: float,
    touches: int,
    area_ratio: float,
) -> SemanticFilterResult:
    if technical_label == "reject":
        return _result("reject", "technical_qa_reject")
    if touches >= 3:
        return _result("reject", "mask_touches_many_borders")
    if area_ratio > 0.85 or area_ratio < 0.02:
        return _result("reject", "extreme_mask_area_ratio")

    flying_like = crop_ar >= 1.15 or bbox_ar >= 1.15
    vertical_like = crop_ar < 0.85 and bbox_ar < 0.85

    if technical_label == "accept":
        if flying_like:
            return _result("approved", "bird_flying_like_shape")
        if vertical_like:
            return _result("review", "bird_vertical_portrait_like")
        return _result("review", "bird_ambiguous_shape")

    if flying_like:
        return _result("review", "bird_flying_shape_technical_review")
    if vertical_like:
        return _result("review", "bird_vertical_technical_review")
    return _result("review", "bird_technical_review")


def classify_airplane(
    *,
    technical_label: str,
    crop_ar: float,
    bbox_ar: float,
    touches: int,
    area_ratio: float,
) -> SemanticFilterResult:
    if technical_label == "reject":
        return _result("reject", "technical_qa_reject")

    wide_like = crop_ar >= 1.5 or bbox_ar >= 1.5
    partial_flags = int(touches >= 2) + int(area_ratio > 0.75) + int(crop_ar < 1.0)

    if touches >= 2 and area_ratio > 0.75 and crop_ar < 1.0:
        return _result("reject", "airplane_extreme_partial_crop")
    if partial_flags >= 2:
        return _result("review", "airplane_likely_nose_tail_fragment")

    if wide_like and touches <= 1 and area_ratio <= 0.75:
        if technical_label == "accept":
            return _result("approved", "airplane_wide_full_like")
        return _result("review", "airplane_partial_but_wide")

    if partial_flags >= 1:
        return _result("review", "airplane_likely_nose_tail_fragment")
    if crop_ar < 1.0:
        return _result("review", "airplane_narrow_crop")
    return _result("review", "airplane_ambiguous")


def classify_kite(
    *,
    technical_label: str,
    touches: int,
    area_ratio: float,
) -> SemanticFilterResult:
    if technical_label == "reject" or area_ratio > 0.95 or touches >= 3:
        return _result("reject", "kite_extreme_failure")
    if technical_label == "accept" and 0.03 <= area_ratio <= 0.85:
        if touches >= 2:
            return _result("review", "kite_border_touch_review")
        return _result("approved", "kite_ok")
    if area_ratio < 0.03 or touches >= 2:
        return _result("review", "kite_small_or_fragment")
    return _result("review", "kite_technical_review")


def classify_semantic_usefulness(row: pd.Series) -> SemanticFilterResult:
    technical = str(row.get("mask_quality_label", "")).strip().lower()
    touches = int(row.get("touches_border_count", 0))
    area_ratio = float(row.get("mask_area_ratio_in_crop", 0.0))
    geom = compute_geometry(row)
    crop_ar = geom["crop_aspect_ratio"]
    bbox_ar = geom["bbox_aspect_ratio"]
    class_name = str(row.get("class_name", "")).strip().lower()

    if class_name == "bird":
        return classify_bird(
            technical_label=technical,
            crop_ar=crop_ar,
            bbox_ar=bbox_ar,
            touches=touches,
            area_ratio=area_ratio,
        )
    if class_name == "airplane":
        return classify_airplane(
            technical_label=technical,
            crop_ar=crop_ar,
            bbox_ar=bbox_ar,
            touches=touches,
            area_ratio=area_ratio,
        )
    if class_name == "kite":
        return classify_kite(
            technical_label=technical,
            touches=touches,
            area_ratio=area_ratio,
        )
    return _result("reject", "unknown_class")


def filter_input_rows(
    df: pd.DataFrame,
    *,
    technical_labels: set[str],
) -> pd.DataFrame:
    work = df.copy()
    work["has_alpha"] = work["has_alpha"].astype(str).str.lower().isin({"true", "1", "yes"})
    work["extraction_failed"] = work.get("extraction_failed", False)
    if "extraction_failed" in work.columns:
        work["extraction_failed"] = work["extraction_failed"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    else:
        work["extraction_failed"] = False

    mask = (
        (work["extraction_method"] == EXTRACTION_METHOD)
        & work["has_alpha"]
        & (~work["extraction_failed"])
        & (work["mask_area_px"].astype(float) > 0)
        & (work["mask_quality_label"].astype(str).str.lower().isin(technical_labels))
    )
    return work[mask].copy()


def enrich_with_semantic_labels(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        geom = compute_geometry(row)
        sem = classify_semantic_usefulness(row)
        rows.append(
            {
                **row.to_dict(),
                **geom,
                "final_distractor_status": sem.final_distractor_status,
                "semantic_filter_reason": sem.semantic_filter_reason,
                "_priority": sem.priority,
            }
        )
    return pd.DataFrame(rows)


def select_diverse_approved(
    approved_df: pd.DataFrame,
    *,
    max_per_class: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    if approved_df.empty:
        return approved_df

    rng = np.random.default_rng(seed)
    selected_parts: list[pd.DataFrame] = []

    for class_name, max_n in max_per_class.items():
        subset = approved_df[approved_df["class_name"] == class_name].copy()
        if subset.empty:
            continue
        if len(subset) <= max_n:
            selected_parts.append(subset)
            continue

        subset = subset.sort_values(
            ["_priority", "mask_area_ratio_in_crop", "asset_id"],
            ascending=[False, False, True],
        )
        picked_idx: list = []
        seen_images: set = set()

        for idx, row in subset.iterrows():
            if len(picked_idx) >= max_n:
                break
            img_key = row.get("image_id", row.get("asset_id", idx))
            if img_key not in seen_images:
                picked_idx.append(idx)
                seen_images.add(img_key)

        remaining = subset.loc[~subset.index.isin(picked_idx)]
        if len(picked_idx) < max_n and not remaining.empty:
            extra_n = max_n - len(picked_idx)
            extra_idx = remaining.sample(n=min(extra_n, len(remaining)), random_state=int(rng.integers(0, 2**31)))
            picked_idx.extend(extra_idx.index.tolist())

        selected_parts.append(subset.loc[picked_idx[:max_n]])

    if not selected_parts:
        return approved_df.iloc[0:0]
    return pd.concat(selected_parts, ignore_index=True)


def resolve_source_asset_path(row: pd.Series, assets_root: Path) -> Path | None:
    output_path = str(row.get("output_path", "")).strip()
    if output_path and Path(output_path).is_file():
        return Path(output_path)
    asset_id = str(row.get("asset_id", ""))
    class_name = str(row.get("class_name", ""))
    candidate = assets_root / class_name / f"{asset_id}.png"
    if candidate.is_file():
        return candidate
    candidate = assets_root / f"{asset_id}.png"
    if candidate.is_file():
        return candidate
    return None


def write_filter_report(
    report_path: Path,
    *,
    input_by_class: dict[str, int],
    status_by_class: dict[str, dict[str, int]],
    reason_counts: dict[str, int],
    approved_root: Path,
    review_root: Path,
    approved_csv: Path,
    review_csv: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "COCO distractor semantic filter report",
        "",
        "Input count by class (after technical pre-filter):",
    ]
    for cls in sorted(input_by_class):
        lines.append(f"  {cls}: {input_by_class[cls]}")
    lines.append("\nFinal status by class:")
    for cls in sorted(status_by_class):
        bucket = status_by_class[cls]
        lines.append(f"  {cls}:")
        for status in ("approved", "review", "reject"):
            lines.append(f"    {status}: {bucket.get(status, 0)}")
    lines.append("\nCounts by semantic_filter_reason:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count}")
    lines.extend(
        [
            "",
            "Output paths:",
            f"  approved_root: {approved_root.resolve()}",
            f"  review_root: {review_root.resolve()}",
            f"  approved_metadata: {approved_csv.resolve()}",
            f"  review_metadata: {review_csv.resolve()}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
