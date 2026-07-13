"""Heuristic background curation: sky-dominated outdoor scene classification."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from drone_stress.scene_filter import sky_mask_ratio

BACKGROUND_CATEGORIES = (
    "clean_sky",
    "cloudy_sky",
    "trees_sky",
    "urban_skyline",
    "horizon",
    "review",
    "reject",
)

ACCEPT_CATEGORIES = frozenset(
    {"clean_sky", "cloudy_sky", "trees_sky", "urban_skyline", "horizon"}
)

MIN_IMAGE_WIDTH = 64
MIN_IMAGE_HEIGHT = 64

METADATA_COLUMNS = [
    "background_id",
    "original_path",
    "output_path",
    "background_type",
    "filter_status",
    "filter_reason",
    "image_width",
    "image_height",
    "sky_ratio_upper",
    "sky_ratio_full",
    "blue_sky_ratio",
    "gray_white_cloud_ratio",
    "lower_green_ratio",
    "lower_dark_structure_ratio",
]


@dataclass
class BackgroundFeatures:
    image_width: int
    image_height: int
    sky_ratio_upper: float
    sky_ratio_full: float
    blue_sky_ratio: float
    gray_white_cloud_ratio: float
    lower_green_ratio: float
    lower_dark_structure_ratio: float
    lower_texture_score: float
    mean_brightness: float


@dataclass
class BackgroundClassification:
    background_type: str
    filter_status: str
    filter_reason: str


def _upper_lower_regions(rgb: np.ndarray, upper_fraction: float = 0.55) -> tuple[np.ndarray, np.ndarray]:
    h = rgb.shape[0]
    split_row = max(1, int(h * upper_fraction))
    return rgb[:split_row, :, :], rgb[split_row:, :, :]


def _blue_sky_ratio(rgb: np.ndarray) -> float:
    if rgb.size == 0:
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    blue = (h >= 90) & (h <= 140) & (s >= 15) & (v >= 70)
    return float(blue.mean())


def _gray_white_cloud_ratio(rgb: np.ndarray) -> float:
    if rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)
    cloud_like = (gray >= 145) & (gray <= 235) & (s <= 70) & (v >= 100)
    return float(cloud_like.mean())


def _green_ratio(rgb: np.ndarray) -> float:
    if rgb.size == 0:
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    green = (h >= 35) & (h <= 90) & (s >= 25) & (v >= 25)
    return float(green.mean())


def _dark_structure_ratio(rgb: np.ndarray) -> float:
    """Dark gray structured regions (buildings, silhouettes) in lower frame."""
    if rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)
    dark = (gray >= 25) & (gray <= 120) & (s <= 80)
    edges = cv2.Canny(gray, 50, 140)
    structured = dark & (edges > 0)
    return float(structured.mean())


def _texture_score(rgb: np.ndarray) -> float:
    if rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_background_features(rgb: np.ndarray, upper_fraction: float = 0.55) -> BackgroundFeatures:
    """Extract simple color/texture features for background curation."""
    h, w = rgb.shape[:2]
    upper, lower = _upper_lower_regions(rgb, upper_fraction)
    sky_upper = sky_mask_ratio(upper)
    sky_lower = sky_mask_ratio(lower) if lower.size else 0.0
    sky_full = sky_mask_ratio(rgb)

    return BackgroundFeatures(
        image_width=int(w),
        image_height=int(h),
        sky_ratio_upper=sky_upper,
        sky_ratio_full=sky_full,
        blue_sky_ratio=_blue_sky_ratio(upper),
        gray_white_cloud_ratio=_gray_white_cloud_ratio(upper),
        lower_green_ratio=_green_ratio(lower) if lower.size else 0.0,
        lower_dark_structure_ratio=_dark_structure_ratio(lower) if lower.size else 0.0,
        lower_texture_score=_texture_score(lower) if lower.size else 0.0,
        mean_brightness=float(rgb.mean()) if rgb.size else 0.0,
    )


def _is_likely_night_outdoor(f: BackgroundFeatures) -> bool:
    """Outdoor night/dusk candidate — should not auto-reject before CLIP merge."""
    return (
        f.mean_brightness < 55.0
        and f.sky_ratio_full >= 0.05
        and (f.lower_dark_structure_ratio > 0.01 or f.lower_texture_score > 80.0)
    )


def classify_background(features: BackgroundFeatures) -> BackgroundClassification:
    """
    Classify a candidate background image into curated categories.

    Judges sky/horizon suitability for compositing, not object presence.
    """
    f = features
    sky_lower_approx = max(0.0, 2.0 * f.sky_ratio_full - f.sky_ratio_upper)
    horizon_structure = f.sky_ratio_upper > sky_lower_approx + 0.06

    # --- Reject: corrupt, too small, indoor, close-up, no sky ---
    if f.image_width < MIN_IMAGE_WIDTH or f.image_height < MIN_IMAGE_HEIGHT:
        return BackgroundClassification("reject", "reject", "image_too_small")
    if _is_likely_night_outdoor(f):
        return BackgroundClassification("review", "review", "night_low_light_candidate")
    if f.sky_ratio_full < 0.06 and f.sky_ratio_upper < 0.08:
        return BackgroundClassification("reject", "reject", "no_sky")
    if f.sky_ratio_upper < 0.08 and f.sky_ratio_full < 0.10:
        return BackgroundClassification("reject", "reject", "no_sky_indoor_or_closeup")
    if f.sky_ratio_upper < 0.10 and f.lower_texture_score < 120:
        return BackgroundClassification("reject", "reject", "indoor_low_sky")
    if f.sky_ratio_upper < 0.12 and not horizon_structure and f.lower_green_ratio < 0.05:
        return BackgroundClassification("reject", "reject", "closeup_or_mostly_ground")

    # --- Cloudy sky ---
    if f.sky_ratio_upper >= 0.28 and f.gray_white_cloud_ratio >= 0.18:
        return BackgroundClassification("cloudy_sky", "accept", "cloudy_upper_sky")

    # --- Trees + sky ---
    if f.sky_ratio_upper >= 0.22 and f.lower_green_ratio >= 0.16:
        return BackgroundClassification("trees_sky", "accept", "sky_with_trees_lower")

    # --- Urban skyline ---
    if (
        f.sky_ratio_upper >= 0.22
        and f.lower_dark_structure_ratio >= 0.025
        and f.lower_green_ratio < 0.14
        and f.lower_texture_score >= 150
    ):
        return BackgroundClassification("urban_skyline", "accept", "sky_with_buildings_lower")

    # --- Clean sky ---
    if (
        f.sky_ratio_upper >= 0.35
        and f.blue_sky_ratio >= 0.12
        and f.lower_green_ratio < 0.12
        and f.lower_dark_structure_ratio < 0.03
        and f.gray_white_cloud_ratio < 0.35
        and f.lower_texture_score < 250
    ):
        return BackgroundClassification("clean_sky", "accept", "clean_blue_sky_low_texture")

    if (
        f.sky_ratio_upper >= 0.38
        and f.gray_white_cloud_ratio < 0.30
        and f.lower_green_ratio < 0.12
        and f.lower_dark_structure_ratio < 0.03
    ):
        return BackgroundClassification("clean_sky", "accept", "clean_upper_sky_low_texture")

    # --- Horizon ---
    if horizon_structure and 0.16 <= f.sky_ratio_upper < 0.38:
        return BackgroundClassification("horizon", "accept", "horizon_structure")

    # --- Uncertain but possibly useful ---
    if f.sky_ratio_upper >= 0.28:
        if f.gray_white_cloud_ratio >= 0.12:
            return BackgroundClassification("cloudy_sky", "review", "uncertain_cloudy_sky")
        if f.lower_green_ratio >= 0.10:
            return BackgroundClassification("trees_sky", "review", "uncertain_trees_sky")
        if f.lower_dark_structure_ratio >= 0.02:
            return BackgroundClassification("urban_skyline", "review", "uncertain_urban_skyline")
        return BackgroundClassification("clean_sky", "review", "uncertain_clean_sky")

    if f.sky_ratio_upper >= 0.14 and horizon_structure:
        return BackgroundClassification("horizon", "review", "uncertain_horizon")

    if f.sky_ratio_upper >= 0.12:
        return BackgroundClassification("review", "review", "uncertain_sky_scene")

    return BackgroundClassification("reject", "reject", "insufficient_sky")


def features_to_metadata_dict(features: BackgroundFeatures) -> dict[str, float | int]:
    return {
        "image_width": features.image_width,
        "image_height": features.image_height,
        "sky_ratio_upper": round(features.sky_ratio_upper, 6),
        "sky_ratio_full": round(features.sky_ratio_full, 6),
        "blue_sky_ratio": round(features.blue_sky_ratio, 6),
        "gray_white_cloud_ratio": round(features.gray_white_cloud_ratio, 6),
        "lower_green_ratio": round(features.lower_green_ratio, 6),
        "lower_dark_structure_ratio": round(features.lower_dark_structure_ratio, 6),
    }
