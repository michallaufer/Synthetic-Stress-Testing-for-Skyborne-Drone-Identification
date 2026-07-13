"""Conservative sky-region placement for synthetic compositing."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from drone_stress.compositor import BBox


@dataclass(frozen=True)
class PlacementConfig:
    """Foreground paste placement policy."""

    mode: str = "legacy"
    x_min_fraction: float = 0.05
    x_max_fraction: float = 0.95
    y_min_fraction: float = 0.03
    y_max_fraction: float = 0.55
    max_attempts: int = 50
    avoid_overlap: bool = True
    max_iou: float = 0.05
    min_object_distance_px: int = 20
    margin_px: int = 8
    sky_region_top_fraction: float = 0.75

    @property
    def uses_upper_sky(self) -> bool:
        return self.mode.lower() == "upper_sky"

    def region_fractions(self) -> dict[str, float]:
        return {
            "placement_region_x_min": self.x_min_fraction,
            "placement_region_x_max": self.x_max_fraction,
            "placement_region_y_min": self.y_min_fraction,
            "placement_region_y_max": self.y_max_fraction,
        }


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union for axis-aligned boxes."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def bbox_center_distance(a: BBox, b: BBox) -> float:
    acx, acy = a.center()
    bcx, bcy = b.center()
    return math.hypot(acx - bcx, acy - bcy)


def paste_top_left_bounds(
    bg_w: int,
    bg_h: int,
    obj_w: int,
    obj_h: int,
    placement: PlacementConfig,
) -> tuple[int, int, int, int]:
    """
    Valid top-left (x, y) range so object center lies in configured fractions
    and the full bbox stays inside the canvas with margin.
    """
    m = placement.margin_px
    cx_min = placement.x_min_fraction * bg_w
    cx_max = placement.x_max_fraction * bg_w
    cy_min = placement.y_min_fraction * bg_h
    cy_max = placement.y_max_fraction * bg_h

    x_lo = max(m, int(math.ceil(cx_min - obj_w / 2.0)))
    x_hi = min(bg_w - obj_w - m, int(math.floor(cx_max - obj_w / 2.0)))
    y_lo = max(m, int(math.ceil(cy_min - obj_h / 2.0)))
    y_hi = min(bg_h - obj_h - m, int(math.floor(cy_max - obj_h / 2.0)))

    if x_lo > x_hi:
        x_lo = m
        x_hi = max(m, bg_w - obj_w - m)
    if y_lo > y_hi:
        y_lo = m
        y_hi = max(m, min(bg_h - obj_h - m, int(bg_h * placement.y_max_fraction) - obj_h))

    return x_lo, x_hi, y_lo, y_hi


def _accepts_placement(
    candidate: BBox,
    avoid_bboxes: list[BBox],
    placement: PlacementConfig,
) -> bool:
    if not avoid_bboxes or not placement.avoid_overlap:
        return True
    for other in avoid_bboxes:
        if bbox_iou(candidate, other) > placement.max_iou:
            return False
        if bbox_center_distance(candidate, other) < placement.min_object_distance_px:
            return False
    return True


def sample_paste_xy(
    bg_w: int,
    bg_h: int,
    obj_w: int,
    obj_h: int,
    rng: random.Random,
    placement: PlacementConfig,
    *,
    avoid_bboxes: list[BBox] | None = None,
) -> tuple[int, int, int]:
    """
    Sample paste top-left (x, y) and return (x, y, attempts_used).

    mode=legacy delegates to sky-top fraction heuristic.
    mode=upper_sky restricts centers to configured upper/mid sky fractions.
    """
    avoid = list(avoid_bboxes or [])
    if placement.uses_upper_sky:
        return _sample_upper_sky(bg_w, bg_h, obj_w, obj_h, rng, placement, avoid)
    return _sample_legacy(bg_w, bg_h, obj_w, obj_h, rng, placement, avoid)


def _sample_legacy(
    bg_w: int,
    bg_h: int,
    obj_w: int,
    obj_h: int,
    rng: random.Random,
    placement: PlacementConfig,
    avoid_bboxes: list[BBox],
) -> tuple[int, int, int]:
    from drone_stress.compositor import random_paste_xy as legacy_random_paste_xy

    attempts = 0
    best_x, best_y = legacy_random_paste_xy(
        bg_w,
        bg_h,
        obj_w,
        obj_h,
        rng,
        placement.sky_region_top_fraction,
        placement.margin_px,
    )
    for attempt in range(1, placement.max_attempts + 1):
        attempts = attempt
        x, y = legacy_random_paste_xy(
            bg_w,
            bg_h,
            obj_w,
            obj_h,
            rng,
            placement.sky_region_top_fraction,
            placement.margin_px,
        )
        candidate = BBox(x=x, y=y, w=obj_w, h=obj_h)
        if _accepts_placement(candidate, avoid_bboxes, placement):
            return x, y, attempts
        best_x, best_y = x, y
    return best_x, best_y, attempts


def _sample_upper_sky(
    bg_w: int,
    bg_h: int,
    obj_w: int,
    obj_h: int,
    rng: random.Random,
    placement: PlacementConfig,
    avoid_bboxes: list[BBox],
) -> tuple[int, int, int]:
    x_lo, x_hi, y_lo, y_hi = paste_top_left_bounds(bg_w, bg_h, obj_w, obj_h, placement)
    if x_hi < x_lo or y_hi < y_lo:
        cx = max(placement.margin_px, (bg_w - obj_w) // 2)
        cy = max(placement.margin_px, (bg_h - obj_h) // 4)
        return cx, cy, 1

    attempts = 0
    fallback_x = max(x_lo, min(x_hi, (x_lo + x_hi) // 2))
    fallback_y = max(y_lo, min(y_hi, (y_lo + y_hi) // 2))

    for attempt in range(1, placement.max_attempts + 1):
        attempts = attempt
        x = rng.randint(x_lo, x_hi)
        y = rng.randint(y_lo, y_hi)
        candidate = BBox(x=x, y=y, w=obj_w, h=obj_h)
        if _accepts_placement(candidate, avoid_bboxes, placement):
            return x, y, attempts
        fallback_x, fallback_y = x, y

    return fallback_x, fallback_y, attempts
