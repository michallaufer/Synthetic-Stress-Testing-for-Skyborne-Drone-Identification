"""SAM2 multimask scoring and selection for manual folder extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_stress.sam2_extract import (
    _edge_touches_mask,
    _mask_tight_bbox,
    evaluate_mask_qa,
    mask_crop_to_rgba,
)

# Manual extraction thresholds (stricter than generic annotation QA).
BACKGROUND_AREA_RATIO = 0.60
MAX_BBOX_EXPAND_RATIO = 2.5
MIN_OBJECT_AREA_RATIO = 0.02


@dataclass
class MaskSelectionResult:
    mask_crop: np.ndarray
    rgba: np.ndarray
    mask_inverted: bool
    mask_selection_reason: str
    mask_area_ratio: float
    mask_bbox_area_ratio: float
    mask_touches_border_ratio: float
    quality_label: str
    mask_review_reasons: str
    mask_quality_score: float
    needs_manual_review: bool
    mask_area_px: int
    mask_bbox_x: int
    mask_bbox_y: int
    mask_bbox_w: int
    mask_bbox_h: int
    mask_num_touched_borders: int


def _proposal_center_in_mask(mask: np.ndarray, prop_x: int, prop_y: int, prop_w: int, prop_h: int) -> float:
    """Fraction of proposal-center neighborhood covered by mask."""
    h, w = mask.shape[:2]
    cx = int(prop_x + prop_w / 2)
    cy = int(prop_y + prop_h / 2)
    r = max(3, min(prop_w, prop_h) // 6)
    x1 = max(0, cx - r)
    x2 = min(w, cx + r + 1)
    y1 = max(0, cy - r)
    y2 = min(h, cy + r + 1)
    patch = mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())


def _mask_metrics(mask: np.ndarray, prop_x: int, prop_y: int, prop_w: int, prop_h: int) -> dict:
    h, w = mask.shape[:2]
    crop_area = h * w
    area = int(mask.sum())
    ratio = float(area / crop_area) if crop_area else 0.0
    bx, by, bw, bh = _mask_tight_bbox(mask)
    bbox_area_ratio = float((bw * bh) / crop_area) if crop_area else 0.0
    touches = _edge_touches_mask(mask, area)
    num_touched = sum(touches)
    border_ratio = num_touched / 4.0
    center_cov = _proposal_center_in_mask(mask, prop_x, prop_y, prop_w, prop_h)
    prop_area = max(1, prop_w * prop_h)
    mask_bbox_area = max(1, bw * bh)
    expand_ratio = mask_bbox_area / prop_area
    return {
        "area": area,
        "ratio": ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "num_touched": num_touched,
        "border_ratio": border_ratio,
        "center_cov": center_cov,
        "expand_ratio": expand_ratio,
        "bbox_x": bx,
        "bbox_y": by,
        "bbox_w": bw,
        "bbox_h": bh,
    }


def _score_mask_candidate(
    mask: np.ndarray,
    prop_x: int,
    prop_y: int,
    prop_w: int,
    prop_h: int,
    *,
    inverted: bool,
) -> tuple[float, list[str]]:
    m = _mask_metrics(mask, prop_x, prop_y, prop_w, prop_h)
    reasons: list[str] = []
    score = 1.0

    if m["area"] == 0:
        return -1.0, ["empty_mask"]

    if m["ratio"] > BACKGROUND_AREA_RATIO:
        reasons.append("likely_background_blob")
        score -= 0.55
    elif m["ratio"] > 0.45:
        reasons.append("mask_large_review")
        score -= 0.20
    elif m["ratio"] < MIN_OBJECT_AREA_RATIO:
        reasons.append("mask_too_small")
        score -= 0.50

    if m["num_touched"] >= 3:
        reasons.append("mask_touches_many_borders")
        score -= 0.35
    elif m["num_touched"] == 2:
        reasons.append("mask_touches_two_borders")
        score -= 0.12

    if m["expand_ratio"] > MAX_BBOX_EXPAND_RATIO:
        reasons.append("mask_bbox_much_larger_than_proposal")
        score -= 0.25

    if m["center_cov"] < 0.25:
        reasons.append("mask_misses_proposal_center")
        score -= 0.30
    else:
        score += 0.15 * m["center_cov"]

    # Prefer compact masks that are not huge background fills.
    score += 0.10 * (1.0 - abs(m["ratio"] - 0.25))

    if inverted:
        reasons.append("used_inverse_mask")
        score -= 0.05

    return score, reasons


def _inverse_plausible(
    mask: np.ndarray,
    prop_x: int,
    prop_y: int,
    prop_w: int,
    prop_h: int,
) -> bool:
    inv = ~mask.astype(bool)
    m = _mask_metrics(inv, prop_x, prop_y, prop_w, prop_h)
    if m["area"] == 0:
        return False
    if m["ratio"] > BACKGROUND_AREA_RATIO or m["ratio"] < MIN_OBJECT_AREA_RATIO:
        return False
    if m["num_touched"] >= 3:
        return False
    if m["center_cov"] < 0.35:
        return False
    return True


def select_best_mask_in_crop(
    crop_rgb: np.ndarray,
    candidate_masks: list[np.ndarray],
    *,
    prop_x: int,
    prop_y: int,
    prop_w: int,
    prop_h: int,
) -> MaskSelectionResult:
    """
    Score SAM2 candidate masks (and plausible inverses) within a bbox crop.

    Object pixels should be opaque; background transparent in the output RGBA.
    """
    local_prop_x = 0
    local_prop_y = 0
    local_prop_w = crop_rgb.shape[1]
    local_prop_h = crop_rgb.shape[0]

    best_mask: np.ndarray | None = None
    best_score = -999.0
    best_reasons: list[str] = []
    best_inverted = False
    best_label = "review"

    candidates: list[tuple[np.ndarray, bool, str]] = []
    for i, raw in enumerate(candidate_masks):
        m = raw.astype(bool)
        candidates.append((m, False, f"sam2_mask_{i}"))
        if _inverse_plausible(m, local_prop_x, local_prop_y, local_prop_w, local_prop_h):
            candidates.append((~m, True, f"sam2_inverse_{i}"))

    for mask, inverted, tag in candidates:
        score, reasons = _score_mask_candidate(
            mask,
            local_prop_x,
            local_prop_y,
            local_prop_w,
            local_prop_h,
            inverted=inverted,
        )
        if score > best_score:
            best_score = score
            best_mask = mask
            best_inverted = inverted
            best_reasons = [tag, *reasons]

    if best_mask is None:
        empty = np.zeros(crop_rgb.shape[:2], dtype=bool)
        rgba = mask_crop_to_rgba(crop_rgb, empty)
        return MaskSelectionResult(
            mask_crop=empty,
            rgba=rgba,
            mask_inverted=False,
            mask_selection_reason="no_valid_mask",
            mask_area_ratio=0.0,
            mask_bbox_area_ratio=0.0,
            mask_touches_border_ratio=0.0,
            quality_label="reject",
            mask_review_reasons="no_valid_mask",
            mask_quality_score=0.0,
            needs_manual_review=True,
            mask_area_px=0,
            mask_bbox_x=0,
            mask_bbox_y=0,
            mask_bbox_w=0,
            mask_bbox_h=0,
            mask_num_touched_borders=0,
        )

    qa = evaluate_mask_qa(best_mask, sam2_used=True)
    metrics = _mask_metrics(
        best_mask, local_prop_x, local_prop_y, local_prop_w, local_prop_h
    )

    extra_reasons = list(best_reasons)
    if metrics["ratio"] > BACKGROUND_AREA_RATIO:
        qa.mask_quality_label = "reject"
        extra_reasons.append("rejected_background_area")
    elif "likely_background_blob" in extra_reasons:
        qa.mask_quality_label = "reject"
    elif best_score < 0.35:
        qa.mask_quality_label = "review"
        extra_reasons.append("low_mask_score")

    rgba = mask_crop_to_rgba(crop_rgb, best_mask)
    reason = ";".join(extra_reasons)

    return MaskSelectionResult(
        mask_crop=best_mask,
        rgba=rgba,
        mask_inverted=best_inverted,
        mask_selection_reason=reason,
        mask_area_ratio=round(metrics["ratio"], 6),
        mask_bbox_area_ratio=round(metrics["bbox_area_ratio"], 6),
        mask_touches_border_ratio=round(metrics["border_ratio"], 4),
        quality_label=qa.mask_quality_label,
        mask_review_reasons=reason,
        mask_quality_score=qa.mask_quality_score,
        needs_manual_review=qa.needs_manual_review,
        mask_area_px=metrics["area"],
        mask_bbox_x=metrics["bbox_x"],
        mask_bbox_y=metrics["bbox_y"],
        mask_bbox_w=metrics["bbox_w"],
        mask_bbox_h=metrics["bbox_h"],
        mask_num_touched_borders=metrics["num_touched"],
    )
