"""SAM2 box-prompted mask extraction for annotation-driven asset crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Sam2ModelSize = Literal["tiny", "small", "base", "large"]
MaskQualityLabel = Literal["accept", "review", "reject"]

SAM2_HF_IDS: dict[Sam2ModelSize, str] = {
    "tiny": "facebook/sam2-hiera-tiny",
    "small": "facebook/sam2-hiera-small",
    "base": "facebook/sam2-hiera-base-plus",
    "large": "facebook/sam2-hiera-large",
}

SAM2_CONFIG_NAMES: dict[Sam2ModelSize, str] = {
    "tiny": "sam2_hiera_t.yaml",
    "small": "sam2_hiera_s.yaml",
    "base": "sam2_hiera_b+.yaml",
    "large": "sam2_hiera_l.yaml",
}

BORDER_TOUCH_FRACTION = 0.08

# Mask QA thresholds (judge SAM2 output, not source bbox size).
REJECT_MAX_AREA_RATIO = 0.95
REJECT_MIN_AREA_RATIO = 0.01
REVIEW_SMALL_AREA_RATIO = 0.05
REVIEW_LARGE_AREA_RATIO = 0.80
ACCEPT_MIN_AREA_RATIO = 0.05
ACCEPT_MAX_AREA_RATIO = 0.80
ACCEPT_MAX_TOUCHED_BORDERS = 1

REJECT_REASONS = frozenset(
    {"empty_mask", "mask_too_small", "likely_background_blob", "mask_touches_many_borders"}
)


@dataclass
class MaskQAResult:
    mask_area_px: int
    mask_area_ratio_in_crop: float
    mask_bbox_x: int
    mask_bbox_y: int
    mask_bbox_w: int
    mask_bbox_h: int
    mask_bbox_area_ratio_in_crop: float
    mask_touches_top: bool
    mask_touches_bottom: bool
    mask_touches_left: bool
    mask_touches_right: bool
    mask_num_touched_borders: int
    mask_review_reasons: str
    mask_quality_label: MaskQualityLabel
    mask_quality_score: float
    needs_manual_review: bool

    @property
    def review_reasons_list(self) -> list[str]:
        if not self.mask_review_reasons:
            return []
        return [r.strip() for r in self.mask_review_reasons.split(";") if r.strip()]


def ensure_sam2_dependencies() -> None:
    """Verify SAM2 optional deps are installed before extraction starts."""
    try:
        import torch  # noqa: F401
        from sam2.build_sam import build_sam2  # noqa: F401
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SAM2 extraction requires optional deps. Install with: "
            "pip install -r requirements-sam2.txt"
        ) from exc


def resolve_sam2_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def xywh_to_xyxy(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    return x, y, x + w, y + h


def expand_bbox_xywh(
    bbox_xywh: tuple[int, int, int, int],
    expand_ratio: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Expand bbox by ratio of its size, clamped to image bounds."""
    x, y, w, h = bbox_xywh
    pad_x = int(round(w * expand_ratio))
    pad_y = int(round(h * expand_ratio))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_width, x + w + pad_x)
    y2 = min(image_height, y + h + pad_y)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def expanded_prompt_xyxy(
    bbox_xywh: tuple[int, int, int, int],
    expand_ratio: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    ex, ey, ew, eh = expand_bbox_xywh(bbox_xywh, expand_ratio, image_width, image_height)
    return xywh_to_xyxy(ex, ey, ew, eh)


def _mask_tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return tight xywh bbox of True pixels in crop coordinates."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return 0, 0, 0, 0
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    x1 = int(x_indices[0])
    y1 = int(y_indices[0])
    x2 = int(x_indices[-1])
    y2 = int(y_indices[-1])
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def _edge_touches_mask(mask: np.ndarray, area: int) -> tuple[bool, bool, bool, bool]:
    if area <= 0:
        return False, False, False, False
    threshold = BORDER_TOUCH_FRACTION * area
    return (
        bool(mask[0, :].sum() >= threshold),
        bool(mask[-1, :].sum() >= threshold),
        bool(mask[:, 0].sum() >= threshold),
        bool(mask[:, -1].sum() >= threshold),
    )


def _collect_mask_review_reasons(
    *,
    area: int,
    ratio: float,
    num_touched: int,
) -> list[str]:
    reasons: list[str] = []
    if area == 0:
        reasons.append("empty_mask")
    elif ratio < REJECT_MIN_AREA_RATIO:
        reasons.append("mask_too_small")
    elif ratio < REVIEW_SMALL_AREA_RATIO:
        reasons.append("mask_small_review")

    if ratio > REJECT_MAX_AREA_RATIO:
        reasons.append("likely_background_blob")
    elif ratio > REVIEW_LARGE_AREA_RATIO:
        reasons.append("mask_too_large_review")

    if num_touched >= 3:
        reasons.append("mask_touches_many_borders")
    elif num_touched == 2:
        reasons.append("mask_touches_two_borders")

    return reasons


def _assign_mask_quality_label(reasons: list[str], *, sam2_used: bool) -> MaskQualityLabel:
    if any(r in REJECT_REASONS for r in reasons):
        return "reject"
    if reasons:
        return "review"
    if sam2_used:
        return "accept"
    return "review"


def _mask_quality_score(
    ratio: float,
    num_touched: int,
    reasons: list[str],
) -> float:
    if "empty_mask" in reasons or "mask_too_small" in reasons:
        return 0.0
    score = 1.0
    if ratio < REVIEW_SMALL_AREA_RATIO:
        score -= 0.25
    elif ratio > REVIEW_LARGE_AREA_RATIO:
        score -= 0.20
    if "likely_background_blob" in reasons:
        score -= 0.50
    score -= 0.12 * num_touched
    return round(max(0.0, min(1.0, score)), 3)


def evaluate_mask_qa(mask_crop: np.ndarray, *, sam2_used: bool = True) -> MaskQAResult:
    """Evaluate SAM2 segmentation quality on a crop-aligned boolean mask."""
    mask = mask_crop.astype(bool)
    h, w = mask.shape[:2]
    crop_area = h * w
    area = int(mask.sum())
    ratio = float(area / crop_area) if crop_area else 0.0

    bbox_x, bbox_y, bbox_w, bbox_h = _mask_tight_bbox(mask)
    bbox_area_ratio = float((bbox_w * bbox_h) / crop_area) if crop_area else 0.0

    touches_top, touches_bottom, touches_left, touches_right = _edge_touches_mask(mask, area)
    num_touched = sum((touches_top, touches_bottom, touches_left, touches_right))

    reasons = _collect_mask_review_reasons(area=area, ratio=ratio, num_touched=num_touched)

    # Accept only when SAM2 succeeded and all accept criteria hold.
    can_accept = (
        sam2_used
        and area > 0
        and ACCEPT_MIN_AREA_RATIO <= ratio <= ACCEPT_MAX_AREA_RATIO
        and num_touched <= ACCEPT_MAX_TOUCHED_BORDERS
        and not any(r in REJECT_REASONS for r in reasons)
    )
    if can_accept and not reasons:
        label: MaskQualityLabel = "accept"
    else:
        label = _assign_mask_quality_label(reasons, sam2_used=sam2_used)

    score = _mask_quality_score(ratio, num_touched, reasons)
    needs_review = label in ("review", "reject")

    return MaskQAResult(
        mask_area_px=area,
        mask_area_ratio_in_crop=round(ratio, 6),
        mask_bbox_x=bbox_x,
        mask_bbox_y=bbox_y,
        mask_bbox_w=bbox_w,
        mask_bbox_h=bbox_h,
        mask_bbox_area_ratio_in_crop=round(bbox_area_ratio, 6),
        mask_touches_top=touches_top,
        mask_touches_bottom=touches_bottom,
        mask_touches_left=touches_left,
        mask_touches_right=touches_right,
        mask_num_touched_borders=num_touched,
        mask_review_reasons=";".join(reasons),
        mask_quality_label=label,
        mask_quality_score=score,
        needs_manual_review=needs_review,
    )


def fallback_mask_qa(*, reason: str = "sam2_failed_bbox_fallback") -> MaskQAResult:
    """QA record for SAM2 failure → rectangular bbox fallback."""
    return MaskQAResult(
        mask_area_px=0,
        mask_area_ratio_in_crop=0.0,
        mask_bbox_x=0,
        mask_bbox_y=0,
        mask_bbox_w=0,
        mask_bbox_h=0,
        mask_bbox_area_ratio_in_crop=0.0,
        mask_touches_top=False,
        mask_touches_bottom=False,
        mask_touches_left=False,
        mask_touches_right=False,
        mask_num_touched_borders=0,
        mask_review_reasons=reason,
        mask_quality_label="review",
        mask_quality_score=0.0,
        needs_manual_review=True,
    )


def mask_crop_to_rgba(rgb_crop: np.ndarray, mask_crop: np.ndarray) -> np.ndarray:
    """Build RGBA crop where alpha follows the boolean mask."""
    h, w = rgb_crop.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb_crop
    rgba[:, :, 3] = np.where(mask_crop.astype(bool), 255, 0).astype(np.uint8)
    return rgba


class Sam2BoxPredictor:
    """Lazy-loaded SAM2 image predictor with per-image embedding cache."""

    def __init__(
        self,
        model_size: Sam2ModelSize = "tiny",
        checkpoint: Path | None = None,
        device: str = "auto",
    ) -> None:
        self.model_size = model_size
        self.checkpoint = checkpoint
        self.device = resolve_sam2_device(device)
        self._predictor = None
        self._cached_image_key: str | None = None

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor

        ensure_sam2_dependencies()
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if self.checkpoint is not None and self.checkpoint.is_file():
            import sam2

            config_name = SAM2_CONFIG_NAMES[self.model_size]
            config_dir = Path(sam2.__file__).resolve().parent / "configs" / "sam2"
            config_path = config_dir / config_name
            if not config_path.is_file():
                raise FileNotFoundError(f"SAM2 config not found: {config_path}")
            model = build_sam2(str(config_path), str(self.checkpoint), device=self.device)
            self._predictor = SAM2ImagePredictor(model)
        else:
            hf_id = SAM2_HF_IDS[self.model_size]
            self._predictor = SAM2ImagePredictor.from_pretrained(hf_id, device=self.device)

        self._torch = torch
        return self._predictor

    def _set_image(self, rgb: np.ndarray, image_key: str) -> None:
        if self._cached_image_key == image_key:
            return
        predictor = self._load_predictor()
        predictor.set_image(rgb)
        self._cached_image_key = image_key

    def predict_mask(
        self,
        rgb: np.ndarray,
        box_xyxy: tuple[int, int, int, int],
        image_key: str,
    ) -> np.ndarray:
        """
        Run SAM2 with a box prompt on a full RGB image.

        Returns a boolean mask with the same HxW shape as rgb.
        """
        self._set_image(rgb, image_key)
        predictor = self._predictor
        box = np.array(box_xyxy, dtype=np.float32)
        torch = self._torch

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, _, _ = predictor.predict(
                        box=box,
                        multimask_output=False,
                    )
            else:
                masks, _, _ = predictor.predict(
                    box=box,
                    multimask_output=False,
                )

        mask = masks[0]
        if mask.dtype != bool:
            mask = mask > 0
        return mask.astype(bool)

    def predict_masks_multimask(
        self,
        rgb: np.ndarray,
        box_xyxy: tuple[int, int, int, int],
        image_key: str,
    ) -> list[np.ndarray]:
        """Return all SAM2 masks for a box prompt (typically 3 candidates)."""
        self._set_image(rgb, image_key)
        predictor = self._predictor
        box = np.array(box_xyxy, dtype=np.float32)
        torch = self._torch

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, _, _ = predictor.predict(
                        box=box,
                        multimask_output=True,
                    )
            else:
                masks, _, _ = predictor.predict(
                    box=box,
                    multimask_output=True,
                )

        out: list[np.ndarray] = []
        for mask in masks:
            if mask.dtype != bool:
                mask = mask > 0
            out.append(mask.astype(bool))
        return out
