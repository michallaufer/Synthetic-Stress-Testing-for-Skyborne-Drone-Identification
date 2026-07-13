"""Image compositing, augmentations, and label helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class BBox:
    """Axis-aligned box in pixel coordinates: x, y, width, height (top-left origin)."""

    x: int
    y: int
    w: int
    h: int

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def to_yolo_line(self, class_id: int, img_w: int, img_h: int) -> str:
        cx = (self.x + self.w / 2.0) / img_w
        cy = (self.y + self.h / 2.0) / img_h
        nw = self.w / img_w
        nh = self.h / img_h
        return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def load_rgba(path: str | Path) -> np.ndarray:
    """Load image as RGBA uint8 array."""
    from drone_stress.assets import open_image_file

    with open_image_file(path) as fp:
        img = Image.open(fp).convert("RGBA")
    return np.array(img)


def load_rgb(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    """Load and resize background to RGB uint8 (width, height)."""
    from drone_stress.assets import open_image_file

    with open_image_file(path) as fp:
        img = Image.open(fp).convert("RGB")
    img = img.resize(size, Image.Resampling.LANCZOS)
    return np.array(img)


def resize_asset_rgba(asset: np.ndarray, target_max_px: int) -> np.ndarray:
    h, w = asset.shape[:2]
    scale = target_max_px / max(h, w, 1)
    if scale <= 0:
        raise ValueError("target_max_px must be positive")
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return asset
    resized = cv2.resize(asset, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def random_paste_xy(
    bg_w: int,
    bg_h: int,
    obj_w: int,
    obj_h: int,
    rng: random.Random,
    sky_top_fraction: float,
    margin: int,
) -> tuple[int, int]:
    """
    Sample a paste location with an explicit margin from image edges.

    If the object does not fit inside the sky region with the requested margin,
    fall back to centering it in the best available region (still avoiding edges
    when possible).
    """

    # Horizontal placement with margin.
    x_min = margin
    x_max = bg_w - obj_w - margin
    if x_max >= x_min:
        x = rng.randint(x_min, x_max)
    else:
        # Object is too wide; center within canvas (may violate margin).
        x = max(0, (bg_w - obj_w) // 2)

    # Vertical placement: prefer the top (sky) region.
    sky_h = int(bg_h * sky_top_fraction)
    y_min = margin
    y_max = sky_h - obj_h - margin
    if y_max >= y_min:
        y = rng.randint(y_min, y_max)
    else:
        # Sky region too small; center within full canvas.
        y = max(0, (bg_h - obj_h) // 2)

    return x, y


def asset_has_transparency(asset_rgba: np.ndarray) -> bool:
    """True if alpha channel contains any value < 255."""
    if asset_rgba.ndim != 3 or asset_rgba.shape[2] != 4:
        return False
    alpha = asset_rgba[:, :, 3]
    return bool(np.any(alpha < 255))


def alpha_composite(base_rgb: np.ndarray, asset_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """Paste RGBA asset onto RGB base at (x, y). Returns RGB image."""
    bh, bw = base_rgb.shape[:2]
    ah, aw = asset_rgba.shape[:2]

    base_rgba = np.concatenate(
        [base_rgb, np.full((bh, bw, 1), 255, dtype=np.uint8)],
        axis=2,
    )

    left = max(0, x)
    top = max(0, y)
    right = min(bw, x + aw)
    bottom = min(bh, y + ah)

    if left >= right or top >= bottom:
        return base_rgb.copy()

    fg_left = left - x
    fg_top = top - y
    fg_right = fg_left + (right - left)
    fg_bottom = fg_top + (bottom - top)

    fg_crop = asset_rgba[fg_top:fg_bottom, fg_left:fg_right]
    alpha = fg_crop[:, :, 3:4].astype(np.float32) / 255.0
    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    bg_region = base_rgba[top:bottom, left:right, :3].astype(np.float32)

    blended = (alpha * fg_rgb + (1.0 - alpha) * bg_region).astype(np.uint8)
    base_rgba[top:bottom, left:right, :3] = blended

    return base_rgba[:, :, :3]


def apply_gaussian_noise(image_rgb: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image_rgb
    noise = np.random.normal(0, sigma, image_rgb.shape).astype(np.float32)
    noisy = np.clip(image_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_blur(image_rgb: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 0:
        return image_rgb
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(image_rgb, (k, k), 0)


def size_to_distance_bin(size_px: int, thresholds: dict[str, int]) -> str:
    """Map object size to distance_bin using descending thresholds."""
    ordered = sorted(thresholds.items(), key=lambda kv: kv[1], reverse=True)
    for name, bound in ordered:
        if size_px >= bound:
            return name
    return ordered[-1][0] if ordered else "mid"


def infer_difficulty(
    subset: str,
    size_px: int,
    noise_sigma: int,
    blur_name: str,
) -> str:
    if subset == "hard_negative":
        return "hard_negative"
    score = 0
    if size_px <= 15:
        score += 2
    elif size_px <= 30:
        score += 1
    if noise_sigma >= 20:
        score += 2
    elif noise_sigma >= 10:
        score += 1
    if blur_name != "none":
        score += 1
    if subset == "mixed_challenge":
        score += 1
    if score >= 4:
        return "hard"
    if score >= 2:
        return "medium"
    return "easy"
