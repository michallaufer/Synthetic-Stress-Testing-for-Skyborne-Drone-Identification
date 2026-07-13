"""RGBA alpha-channel measurement and folder audit helpers."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from drone_stress.assets import file_accessible, open_image_file, win_long_path

ALPHA_CLASS_WITH_TRANSPARENCY = "with_transparency"
ALPHA_CLASS_FULLY_OPAQUE = "fully_opaque"
ALPHA_CLASS_FULLY_TRANSPARENT = "fully_transparent"
ALPHA_CLASS_FAILED = "failed"


@dataclass
class AlphaChannelStats:
    has_alpha: bool
    alpha_min: int
    alpha_max: int
    alpha_nonzero_fraction: float
    width: int
    height: int
    alpha_class: str
    error: str = ""

    def summary_line(self) -> str:
        return (
            f"a=[{self.alpha_min},{self.alpha_max}] "
            f"nz={self.alpha_nonzero_fraction:.3f} {self.alpha_class}"
        )


def load_rgba_array(path: Path) -> np.ndarray:
    """Load image as RGBA uint8 (Windows long-path safe)."""
    with open_image_file(path) as fp:
        img = Image.open(fp).convert("RGBA")
    return np.array(img)


def measure_alpha_stats(rgba: np.ndarray) -> AlphaChannelStats:
    """Measure alpha channel stats from an RGBA array."""
    if rgba.ndim != 3 or rgba.shape[2] < 4:
        return AlphaChannelStats(
            has_alpha=False,
            alpha_min=0,
            alpha_max=0,
            alpha_nonzero_fraction=0.0,
            width=int(rgba.shape[1]) if rgba.ndim >= 2 else 0,
            height=int(rgba.shape[0]) if rgba.ndim >= 2 else 0,
            alpha_class=ALPHA_CLASS_FAILED,
            error="not_rgba",
        )

    alpha = rgba[:, :, 3]
    h, w = alpha.shape
    alpha_min = int(alpha.min())
    alpha_max = int(alpha.max())
    nonzero = float(np.count_nonzero(alpha)) / float(alpha.size) if alpha.size else 0.0
    has_alpha = alpha_min < 255 or alpha_max < 255

    if alpha_max == 0:
        alpha_class = ALPHA_CLASS_FULLY_TRANSPARENT
    elif alpha_min == 255 and alpha_max == 255:
        alpha_class = ALPHA_CLASS_FULLY_OPAQUE
    elif alpha_min < 255 and alpha_max > 0:
        alpha_class = ALPHA_CLASS_WITH_TRANSPARENCY
    else:
        alpha_class = ALPHA_CLASS_FULLY_OPAQUE

    return AlphaChannelStats(
        has_alpha=has_alpha and alpha_class == ALPHA_CLASS_WITH_TRANSPARENCY,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_nonzero_fraction=nonzero,
        width=w,
        height=h,
        alpha_class=alpha_class,
    )


def measure_alpha_stats_from_path(path: Path) -> AlphaChannelStats:
    """Load image from disk and measure alpha stats."""
    try:
        if not file_accessible(path):
            return AlphaChannelStats(
                has_alpha=False,
                alpha_min=0,
                alpha_max=0,
                alpha_nonzero_fraction=0.0,
                width=0,
                height=0,
                alpha_class=ALPHA_CLASS_FAILED,
                error="file_not_accessible",
            )
        rgba = load_rgba_array(path)
        return measure_alpha_stats(rgba)
    except Exception as exc:
        return AlphaChannelStats(
            has_alpha=False,
            alpha_min=0,
            alpha_max=0,
            alpha_nonzero_fraction=0.0,
            width=0,
            height=0,
            alpha_class=ALPHA_CLASS_FAILED,
            error=str(exc),
        )


def audit_alpha_paths(paths: list[Path]) -> dict[str, int]:
    """Count alpha classes across a list of image paths."""
    counts = {
        ALPHA_CLASS_WITH_TRANSPARENCY: 0,
        ALPHA_CLASS_FULLY_OPAQUE: 0,
        ALPHA_CLASS_FULLY_TRANSPARENT: 0,
        ALPHA_CLASS_FAILED: 0,
    }
    for path in paths:
        stats = measure_alpha_stats_from_path(path)
        counts[stats.alpha_class] = counts.get(stats.alpha_class, 0) + 1
    return counts


def save_rgba_png(rgba: np.ndarray, dest: Path) -> None:
    """Save RGBA PNG with Windows long-path support."""
    import os

    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    data = buf.getvalue()
    if os.name == "nt" and len(str(dest)) >= 240:
        with open(win_long_path(dest), "wb") as fp:
            fp.write(data)
    else:
        dest.write_bytes(data)
    if not file_accessible(dest):
        raise OSError(f"save reported ok but file missing: {dest}")


def alpha_stats_to_metadata_fields(stats: AlphaChannelStats) -> dict:
    return {
        "has_alpha": stats.has_alpha,
        "alpha_min": stats.alpha_min,
        "alpha_max": stats.alpha_max,
        "alpha_nonzero_fraction": round(stats.alpha_nonzero_fraction, 6),
        "width": stats.width,
        "height": stats.height,
    }
