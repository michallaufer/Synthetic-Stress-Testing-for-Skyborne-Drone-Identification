"""Heuristic filtering for full-image COCO hard negatives (no cutouts)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drone_stress.background_filter import compute_background_features
from drone_stress.scene_filter import load_image_rgb

DEFAULT_MAX_PER_CATEGORY = {
    "bird": 70,
    "airplane": 70,
    "kite": 80,
}

REASON_PRIORITY: dict[str, int] = {
    "bird_multi_object_sky": 100,
    "bird_distant_outdoor": 95,
    "kite_multi_object": 95,
    "kite_outdoor_scene": 90,
    "airplane_distant_outdoor": 90,
    "airplane_wide_outdoor": 85,
    "sky_boost_approved": 80,
    "kite_small_outdoor": 75,
    "bird_outdoor_candidate": 70,
    "airplane_outdoor_candidate": 70,
    "bird_closeup_review": 40,
    "bird_large_single_review": 40,
    "airplane_closeup_review": 40,
    "airplane_large_ground_review": 40,
    "kite_closeup_review": 40,
    "object_tiny_review": 35,
    "closeup_large_area_review": 35,
    "indoor_no_sky_review": 30,
    "uncertain_review": 25,
    "image_missing": 0,
    "image_unreadable": 0,
    "clearly_unusable": 0,
}


@dataclass
class RowGeometry:
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    image_width: int
    image_height: int
    max_bbox_px: int
    bbox_area_ratio: float
    bbox_center_y_ratio: float
    bbox_aspect_ratio: float
    num_requested_objects: int


@dataclass
class SkyHint:
    sky_ratio_upper: float | None
    outdoor_likely: bool
    indoor_likely: bool


@dataclass
class HardNegativeFilterResult:
    final_hard_negative_status: str
    hard_negative_filter_reason: str
    priority: int
    sky_ratio_upper: float | None


def parse_max_per_category(values: list[str] | None) -> dict[str, int]:
    limits = dict(DEFAULT_MAX_PER_CATEGORY)
    if not values:
        return limits
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected category=count, got {item!r}")
        cat, count = item.split("=", 1)
        limits[cat.strip().lower()] = int(count.strip())
    return limits


def compute_row_geometry(row: pd.Series) -> RowGeometry:
    img_w = max(int(row.get("image_width", 0)), 1)
    img_h = max(int(row.get("image_height", 0)), 1)
    bw = max(int(row.get("largest_distractor_bbox_w", 0)), 0)
    bh = max(int(row.get("largest_distractor_bbox_h", 0)), 0)
    bx = int(row.get("largest_distractor_bbox_x", 0))
    by = int(row.get("largest_distractor_bbox_y", 0))
    img_area = img_w * img_h

    area_val = row.get("largest_distractor_area")
    if pd.notna(area_val) and float(area_val) > 0:
        bbox_area_ratio = float(area_val) / img_area
    else:
        bbox_area_ratio = (bw * bh) / img_area

    cy = by + bh / 2.0
    return RowGeometry(
        bbox_x=bx,
        bbox_y=by,
        bbox_w=bw,
        bbox_h=bh,
        image_width=img_w,
        image_height=img_h,
        max_bbox_px=max(bw, bh),
        bbox_area_ratio=bbox_area_ratio,
        bbox_center_y_ratio=cy / img_h,
        bbox_aspect_ratio=bw / max(bh, 1),
        num_requested_objects=int(row.get("num_requested_objects", 1)),
    )


def resolve_image_path(row: pd.Series, image_root: Path) -> Path | None:
    for key in ("output_path", "original_image_path"):
        raw = str(row.get(key, "")).strip()
        if raw and Path(raw).is_file():
            return Path(raw)
    dominant = str(row.get("dominant_distractor_type", "")).strip()
    for key in ("output_path", "original_image_path"):
        raw = str(row.get(key, "")).strip()
        if raw:
            candidate = image_root / dominant / Path(raw).name
            if candidate.is_file():
                return candidate
    return None


def _optional_sky_hint(image_path: Path | None) -> SkyHint:
    if image_path is None:
        return SkyHint(None, False, False)
    try:
        rgb = load_image_rgb(image_path)
        features = compute_background_features(rgb)
        upper = features.sky_ratio_upper
        full = features.sky_ratio_full
        outdoor = upper >= 0.18 or full >= 0.22
        indoor = upper < 0.10 and full < 0.10
        return SkyHint(round(upper, 4), outdoor, indoor)
    except OSError:
        return SkyHint(None, False, False)


def _result(status: str, reason: str, *, sky: float | None = None) -> HardNegativeFilterResult:
    return HardNegativeFilterResult(
        final_hard_negative_status=status,
        hard_negative_filter_reason=reason,
        priority=REASON_PRIORITY.get(reason, 20),
        sky_ratio_upper=sky,
    )


def _apply_sky_adjustment(
    result: HardNegativeFilterResult,
    sky: SkyHint,
    geom: RowGeometry,
) -> HardNegativeFilterResult:
    if result.final_hard_negative_status == "reject":
        return result
    if sky.indoor_likely and result.final_hard_negative_status == "approved":
        return _result("review", "indoor_no_sky_review", sky=sky.sky_ratio_upper)
    if sky.outdoor_likely and result.final_hard_negative_status == "review":
        if geom.bbox_area_ratio < 0.25 and geom.bbox_center_y_ratio < 0.80:
            return _result("approved", "sky_boost_approved", sky=sky.sky_ratio_upper)
    return result


def classify_bird(geom: RowGeometry, sky: SkyHint) -> HardNegativeFilterResult:
    if sky.indoor_likely and geom.bbox_area_ratio > 0.35 and geom.num_requested_objects < 2:
        return _result("reject", "clearly_unusable", sky=sky.sky_ratio_upper)

    if geom.bbox_area_ratio >= 0.20:
        return _result("review", "bird_closeup_review", sky=sky.sky_ratio_upper)

    if geom.num_requested_objects == 1 and (
        geom.bbox_area_ratio >= 0.12 or geom.max_bbox_px > 280
    ):
        return _result("review", "bird_large_single_review", sky=sky.sky_ratio_upper)

    multi = geom.num_requested_objects >= 2
    distant = 20 <= geom.max_bbox_px <= 250 and geom.bbox_area_ratio < 0.20
    outdoor_ok = sky.outdoor_likely or (
        sky.sky_ratio_upper is not None
        and not sky.indoor_likely
        and sky.sky_ratio_upper >= 0.14
    )

    if multi and (outdoor_ok or sky.sky_ratio_upper is None):
        return _result("approved", "bird_multi_object_sky", sky=sky.sky_ratio_upper)

    if outdoor_ok and distant:
        return _result("approved", "bird_distant_outdoor", sky=sky.sky_ratio_upper)

    if multi or distant:
        return _result("review", "uncertain_review", sky=sky.sky_ratio_upper)

    return _result("review", "uncertain_review", sky=sky.sky_ratio_upper)


def classify_airplane(geom: RowGeometry, sky: SkyHint) -> HardNegativeFilterResult:
    if geom.bbox_area_ratio >= 0.25:
        return _result("review", "airplane_closeup_review", sky=sky.sky_ratio_upper)

    if geom.max_bbox_px > 450 and geom.bbox_center_y_ratio >= 0.72:
        return _result("review", "airplane_large_ground_review", sky=sky.sky_ratio_upper)

    if 30 <= geom.max_bbox_px <= 500 and geom.bbox_area_ratio < 0.25:
        reason = "airplane_wide_outdoor" if geom.bbox_aspect_ratio > 1.2 else "airplane_distant_outdoor"
        if sky.indoor_likely:
            return _result("review", "indoor_no_sky_review", sky=sky.sky_ratio_upper)
        return _result("approved", reason, sky=sky.sky_ratio_upper)

    if geom.max_bbox_px >= 30 and geom.bbox_area_ratio < 0.25:
        return _result("approved", "airplane_outdoor_candidate", sky=sky.sky_ratio_upper)

    return _result("review", "uncertain_review", sky=sky.sky_ratio_upper)


def classify_kite(geom: RowGeometry, sky: SkyHint) -> HardNegativeFilterResult:
    if geom.bbox_area_ratio >= 0.30:
        return _result("review", "kite_closeup_review", sky=sky.sky_ratio_upper)

    multi = geom.num_requested_objects >= 2
    if multi and geom.max_bbox_px >= 20:
        return _result("approved", "kite_multi_object", sky=sky.sky_ratio_upper)

    if geom.max_bbox_px >= 20 and geom.bbox_area_ratio < 0.30:
        if sky.indoor_likely and geom.bbox_area_ratio > 0.15:
            return _result("review", "indoor_no_sky_review", sky=sky.sky_ratio_upper)
        reason = "kite_outdoor_scene" if geom.bbox_area_ratio < 0.15 else "kite_small_outdoor"
        return _result("approved", reason, sky=sky.sky_ratio_upper)

    return _result("review", "uncertain_review", sky=sky.sky_ratio_upper)


def classify_hard_negative(
    row: pd.Series,
    *,
    image_root: Path,
) -> HardNegativeFilterResult:
    """Classify one hard-negative row using metadata + optional sky hint."""
    geom = compute_row_geometry(row)
    class_name = str(row.get("dominant_distractor_type", "")).strip().lower()
    image_path = resolve_image_path(row, image_root)

    if image_path is None:
        return _result("reject", "image_missing")

    sky = _optional_sky_hint(image_path)

    if geom.max_bbox_px < 20:
        return _result("review", "object_tiny_review", sky=sky.sky_ratio_upper)

    if geom.bbox_area_ratio > 0.40:
        return _result("review", "closeup_large_area_review", sky=sky.sky_ratio_upper)

    if sky.indoor_likely and geom.bbox_area_ratio > 0.30:
        return _result("review", "indoor_no_sky_review", sky=sky.sky_ratio_upper)

    if class_name == "bird":
        result = classify_bird(geom, sky)
    elif class_name == "airplane":
        result = classify_airplane(geom, sky)
    elif class_name == "kite":
        result = classify_kite(geom, sky)
    else:
        result = _result("review", "uncertain_review", sky=sky.sky_ratio_upper)

    result = _apply_sky_adjustment(result, sky, geom)

    if (
        result.final_hard_negative_status == "approved"
        and geom.bbox_center_y_ratio >= 0.75
        and geom.num_requested_objects < 2
        and class_name != "kite"
    ):
        return _result("review", "uncertain_review", sky=sky.sky_ratio_upper)

    return result


def enrich_hard_negative_rows(
    df: pd.DataFrame,
    *,
    image_root: Path,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in df.iterrows():
        geom = compute_row_geometry(row)
        filt = classify_hard_negative(row, image_root=image_root)
        out = {
            **row.to_dict(),
            "max_bbox_px": geom.max_bbox_px,
            "bbox_area_ratio": round(geom.bbox_area_ratio, 6),
            "bbox_center_y_ratio": round(geom.bbox_center_y_ratio, 4),
            "bbox_aspect_ratio": round(geom.bbox_aspect_ratio, 4),
            "final_hard_negative_status": filt.final_hard_negative_status,
            "hard_negative_filter_reason": filt.hard_negative_filter_reason,
            "_priority": filt.priority,
        }
        if filt.sky_ratio_upper is not None:
            out["sky_ratio_upper"] = filt.sky_ratio_upper
        rows.append(out)
    return pd.DataFrame(rows)


def select_approved_subset(
    approved_df: pd.DataFrame,
    *,
    max_per_category: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    if approved_df.empty:
        return approved_df

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    cat_col = "dominant_distractor_type"

    for cat, max_n in max_per_category.items():
        subset = approved_df[approved_df[cat_col] == cat].copy()
        if subset.empty:
            continue
        if len(subset) <= max_n:
            parts.append(subset)
            continue

        subset = subset.sort_values(
            ["_priority", "bbox_area_ratio", "bbox_center_y_ratio", "hard_negative_id"],
            ascending=[False, True, True, True],
        )
        head_n = max(max_n // 2 + max_n % 2, 1)
        head = subset.iloc[:head_n]
        tail = subset.iloc[head_n:]
        need = max_n - len(head)
        if need > 0 and not tail.empty:
            extra = tail.sample(n=min(need, len(tail)), random_state=int(rng.integers(0, 2**31)))
            head = pd.concat([head, extra], ignore_index=True)
        parts.append(head.iloc[:max_n])

    if not parts:
        return approved_df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def write_hard_negative_filter_report(
    report_path: Path,
    *,
    input_by_category: dict[str, int],
    status_by_category: dict[str, dict[str, int]],
    reason_counts: dict[str, int],
    strict_root: Path,
    review_root: Path,
    strict_csv: Path,
    review_csv: Path,
    final_selected_total: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "COCO hard-negative filter report",
        "",
        "Input count by category:",
    ]
    for cat in sorted(input_by_category):
        lines.append(f"  {cat}: {input_by_category[cat]}")
    lines.append("\nApproved / review / reject by category:")
    for cat in sorted(status_by_category):
        bucket = status_by_category[cat]
        lines.append(f"  {cat}:")
        for status in ("approved", "review", "reject"):
            lines.append(f"    {status}: {bucket.get(status, 0)}")
    lines.append("\nCounts by hard_negative_filter_reason:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count}")
    lines.extend(
        [
            "",
            f"final_selected_total: {final_selected_total}",
            "",
            "Output paths:",
            f"  strict_root: {strict_root.resolve()}",
            f"  review_root: {review_root.resolve()}",
            f"  strict_metadata: {strict_csv.resolve()}",
            f"  review_metadata: {review_csv.resolve()}",
            f"  {strict_root.resolve() / 'bird'}",
            f"  {strict_root.resolve() / 'airplane'}",
            f"  {strict_root.resolve() / 'kite'}",
            f"  {review_root.resolve() / 'bird'}",
            f"  {review_root.resolve() / 'airplane'}",
            f"  {review_root.resolve() / 'kite'}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
