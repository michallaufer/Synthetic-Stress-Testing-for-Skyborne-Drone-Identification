"""Viewpoint-aware scene filtering for flying-object annotation rows.

Large bboxes are not rejected by size alone. Filtering targets viewpoint
compatibility for ground-to-air / horizon compositing, not object scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

# Primary filter reasons (QA / audit)
REASON_LARGE_OBJECT_REVIEW = "large_object_review"
REASON_LARGE_BUT_SKY_CANDIDATE = "large_but_sky_candidate"
REASON_LIKELY_PRODUCT_OR_GROUND_CLOSEUP = "likely_product_or_ground_closeup"
REASON_LIKELY_TOP_DOWN_OR_NON_SKY = "likely_top_down_or_non_sky"
REASON_FLYING_SKY_CANDIDATE = "flying_sky_candidate"
REASON_HORIZON_CANDIDATE = "horizon_candidate"

# Viewpoint labels
VIEW_GROUND_TO_AIR = "ground_to_air"
VIEW_HORIZON_SIDE = "horizon_side_view"
VIEW_TOP_DOWN = "top_down"
VIEW_AIR_TO_AIR = "air_to_air"
VIEW_DRONE_ON_GROUND = "drone_on_ground"
VIEW_BIRD_ON_GROUND = "bird_on_ground"
VIEW_PRODUCT_PHOTO = "product_photo"
VIEW_INDOOR = "indoor"
VIEW_IRRELEVANT = "irrelevant"
VIEW_UNKNOWN = "unknown"


class Disposition(str, Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


FilterPurpose = str  # "asset_extraction" | "real_eval"

HEURISTIC_STRONG_ACCEPT = frozenset(
    {REASON_FLYING_SKY_CANDIDATE, REASON_HORIZON_CANDIDATE}
)
HEURISTIC_HARD_REJECT = frozenset(
    {
        REASON_LIKELY_PRODUCT_OR_GROUND_CLOSEUP,
        REASON_LIKELY_TOP_DOWN_OR_NON_SKY,
    }
)
CLIP_AMBIGUOUS_LABEL = "ambiguous"


@dataclass
class SceneFeatures:
    sky_ratio_upper: float
    sky_ratio_lower: float
    bbox_center_y_norm: float
    bbox_area_ratio: float
    bbox_max_dim_ratio: float
    bbox_bottom_norm: float
    is_large_bbox: bool


@dataclass
class FilterDecision:
    disposition: Disposition
    filter_reason: str
    viewpoint_label: str


@dataclass
class MergeResult:
    disposition: Disposition
    merge_policy_reason: str
    clip_low_margin: bool


@dataclass
class FilterThresholds:
    large_bbox_area_ratio: float = 0.10
    large_bbox_max_dim_ratio: float = 0.28
    sky_ratio_upper_accept: float = 0.28
    sky_ratio_upper_horizon: float = 0.18
    sky_ratio_upper_reject: float = 0.12
    sky_ratio_full_reject: float = 0.06
    object_low_y_center: float = 0.72
    object_ground_bottom: float = 0.90
    horizon_upper_min: float = 0.20
    indoor_upper_max: float = 0.08


def sky_mask_ratio(rgb: np.ndarray) -> float:
    """Heuristic bright / blue low-clutter sky fraction."""
    return _sky_mask_ratio(rgb)


def _sky_mask_ratio(rgb: np.ndarray) -> float:
    """Heuristic bright / blue low-clutter sky fraction."""
    if rgb.size == 0:
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    bright_low_sat = (v >= 120) & (s <= 85)
    blue_sky = (h >= 90) & (h <= 140) & (s >= 15) & (v >= 70)
    mask = bright_low_sat | blue_sky
    return float(mask.mean())


def compute_scene_features(
    rgb: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    thresholds: FilterThresholds,
    upper_fraction: float = 0.55,
) -> SceneFeatures:
    img_h, img_w = rgb.shape[:2]
    x, y, w, h = bbox_xywh
    cx = x + w / 2.0
    cy = y + h / 2.0
    bbox_center_y_norm = cy / max(img_h, 1)
    bbox_bottom_norm = (y + h) / max(img_h, 1)
    bbox_area_ratio = (w * h) / max(img_w * img_h, 1)
    bbox_max_dim_ratio = max(w, h) / max(min(img_w, img_h), 1)

    split_row = max(1, int(img_h * upper_fraction))
    upper = rgb[:split_row, :, :]
    lower = rgb[split_row:, :, :]

    sky_ratio_upper = _sky_mask_ratio(upper)
    sky_ratio_lower = _sky_mask_ratio(lower) if lower.size else 0.0

    is_large_bbox = (
        bbox_area_ratio >= thresholds.large_bbox_area_ratio
        or bbox_max_dim_ratio >= thresholds.large_bbox_max_dim_ratio
    )

    return SceneFeatures(
        sky_ratio_upper=sky_ratio_upper,
        sky_ratio_lower=sky_ratio_lower,
        bbox_center_y_norm=bbox_center_y_norm,
        bbox_area_ratio=bbox_area_ratio,
        bbox_max_dim_ratio=bbox_max_dim_ratio,
        bbox_bottom_norm=bbox_bottom_norm,
        is_large_bbox=is_large_bbox,
    )


def _ground_object_view(class_name: str) -> str:
    if class_name.lower() == "bird":
        return VIEW_BIRD_ON_GROUND
    if class_name.lower() == "drone":
        return VIEW_DRONE_ON_GROUND
    return VIEW_IRRELEVANT


def classify_scene_row(
    features: SceneFeatures,
    class_name: str,
    thresholds: FilterThresholds,
) -> FilterDecision:
    """
    Viewpoint compatibility filter.

    Large bbox alone -> review, not reject.
    Reject large bbox only when combined with bad scene clues.
    """
    f = features
    cls = class_name.lower()

    sky_full_approx = (f.sky_ratio_upper + f.sky_ratio_lower) / 2.0
    object_very_low = f.bbox_center_y_norm >= thresholds.object_low_y_center
    object_on_ground = f.bbox_bottom_norm >= thresholds.object_ground_bottom
    has_horizon_structure = (
        f.sky_ratio_upper >= thresholds.horizon_upper_min
        and f.sky_ratio_upper > f.sky_ratio_lower + 0.05
    )
    flying_position = f.bbox_center_y_norm <= thresholds.object_low_y_center
    good_upper_sky = f.sky_ratio_upper >= thresholds.sky_ratio_upper_accept

    # --- Hard rejects (bad viewpoint / scene type) ---
    if sky_full_approx <= thresholds.sky_ratio_full_reject and f.sky_ratio_upper <= thresholds.indoor_upper_max:
        return FilterDecision(
            Disposition.REJECT,
            REASON_LIKELY_TOP_DOWN_OR_NON_SKY,
            VIEW_INDOOR if f.sky_ratio_upper <= thresholds.indoor_upper_max else VIEW_IRRELEVANT,
        )

    if (
        f.is_large_bbox
        and f.sky_ratio_upper < thresholds.sky_ratio_upper_reject
        and (object_very_low or object_on_ground)
    ):
        return FilterDecision(
            Disposition.REJECT,
            REASON_LIKELY_PRODUCT_OR_GROUND_CLOSEUP,
            VIEW_PRODUCT_PHOTO,
        )

    if (
        f.sky_ratio_upper < thresholds.sky_ratio_upper_reject
        and object_very_low
        and not has_horizon_structure
    ):
        return FilterDecision(
            Disposition.REJECT,
            REASON_LIKELY_TOP_DOWN_OR_NON_SKY,
            VIEW_TOP_DOWN,
        )

    if object_on_ground and f.sky_ratio_upper < thresholds.sky_ratio_upper_horizon:
        return FilterDecision(
            Disposition.REJECT,
            REASON_LIKELY_PRODUCT_OR_GROUND_CLOSEUP,
            _ground_object_view(cls),
        )

    # --- Accept: clear flying / sky-compatible viewpoints ---
    if good_upper_sky and flying_position:
        if f.is_large_bbox:
            return FilterDecision(
                Disposition.REVIEW,
                REASON_LARGE_BUT_SKY_CANDIDATE,
                VIEW_GROUND_TO_AIR,
            )
        return FilterDecision(
            Disposition.ACCEPT,
            REASON_FLYING_SKY_CANDIDATE,
            VIEW_GROUND_TO_AIR,
        )

    if has_horizon_structure and f.sky_ratio_upper >= thresholds.sky_ratio_upper_horizon:
        if f.is_large_bbox:
            return FilterDecision(
                Disposition.REVIEW,
                REASON_LARGE_BUT_SKY_CANDIDATE,
                VIEW_HORIZON_SIDE,
            )
        return FilterDecision(
            Disposition.ACCEPT,
            REASON_HORIZON_CANDIDATE,
            VIEW_HORIZON_SIDE,
        )

    # --- Review: large but not clearly bad (useful extraction source) ---
    if f.is_large_bbox:
        if f.sky_ratio_upper >= thresholds.sky_ratio_upper_horizon:
            return FilterDecision(
                Disposition.REVIEW,
                REASON_LARGE_BUT_SKY_CANDIDATE,
                VIEW_HORIZON_SIDE,
            )
        return FilterDecision(
            Disposition.REVIEW,
            REASON_LARGE_OBJECT_REVIEW,
            VIEW_UNKNOWN,
        )

    # Ambiguous small/medium objects without strong sky cues
    if f.sky_ratio_upper >= thresholds.sky_ratio_upper_horizon:
        return FilterDecision(
            Disposition.REVIEW,
            REASON_HORIZON_CANDIDATE,
            VIEW_HORIZON_SIDE,
        )

    return FilterDecision(
        Disposition.REVIEW,
        REASON_LARGE_OBJECT_REVIEW if f.bbox_area_ratio > 0.05 else REASON_HORIZON_CANDIDATE,
        VIEW_UNKNOWN,
    )


def load_image_rgb(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise OSError(f"Could not read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_expanded_bbox_crop(
    rgb: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    pad_fraction: float = 0.5,
) -> np.ndarray:
    """Crop around bbox with padding for optional CLIP context scoring."""
    img_h, img_w = rgb.shape[:2]
    x, y, w, h = bbox_xywh
    pad_x = int(w * pad_fraction)
    pad_y = int(h * pad_fraction)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(img_w, x + w + pad_x)
    bottom = min(img_h, y + h + pad_y)
    crop = rgb[top:bottom, left:right]
    if crop.size == 0:
        return rgb
    return crop


# ---------------------------------------------------------------------------
# Optional CLIP-assisted viewpoint scoring (not ground truth; pre-filter signal)
# ---------------------------------------------------------------------------

CLIP_PROMPT_GROUPS: dict[str, list[str]] = {
    VIEW_GROUND_TO_AIR: [
        "a drone or bird flying in the sky, photographed from the ground looking upward",
        "a small flying object in the sky viewed from below",
    ],
    VIEW_HORIZON_SIDE: [
        "a drone or bird flying near the horizon, photographed from ground level",
        "a flying object above trees or rooftops in an outdoor scene",
    ],
    VIEW_TOP_DOWN: [
        "a drone photographed from above looking down at the ground",
        "an aerial top-down view of terrain or grass",
    ],
    VIEW_DRONE_ON_GROUND: [
        "a drone sitting on grass or on the ground",
    ],
    VIEW_PRODUCT_PHOTO: [
        "a close-up product photo of a drone",
    ],
    VIEW_INDOOR: [
        "an indoor photo of a drone",
    ],
    VIEW_IRRELEVANT: [
        "a landscape image with no clear flying object",
    ],
    "ambiguous": [
        "an ambiguous blurry image of a distant flying object",
    ],
}

CLIP_ACCEPT_LABELS = frozenset({VIEW_GROUND_TO_AIR, VIEW_HORIZON_SIDE})
CLIP_REJECT_LABELS = frozenset(
    {
        VIEW_TOP_DOWN,
        VIEW_DRONE_ON_GROUND,
        VIEW_PRODUCT_PHOTO,
        VIEW_INDOOR,
        VIEW_IRRELEVANT,
    }
)
CLIP_REVIEW_LABELS = frozenset({"ambiguous"})


@dataclass
class ClipScoreResult:
    clip_used: bool
    clip_top_prompt: str
    clip_top_score: float
    clip_second_prompt: str
    clip_second_score: float
    clip_margin: float
    clip_viewpoint_label: str
    clip_filter_hint: str

    @staticmethod
    def unused() -> ClipScoreResult:
        return ClipScoreResult(
            clip_used=False,
            clip_top_prompt="",
            clip_top_score=0.0,
            clip_second_prompt="",
            clip_second_score=0.0,
            clip_margin=0.0,
            clip_viewpoint_label="",
            clip_filter_hint="",
        )


def clip_label_disposition(label: str) -> Disposition:
    if label in CLIP_ACCEPT_LABELS:
        return Disposition.ACCEPT
    if label in CLIP_REJECT_LABELS:
        return Disposition.REJECT
    return Disposition.REVIEW


def clip_label_filter_hint(label: str) -> str:
    disp = clip_label_disposition(label)
    return disp.value


def resolve_clip_device(device: str) -> str:
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


def _move_batch_to_device(batch: dict, device: str) -> dict:
    import torch

    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _normalize_features(features):
    import torch

    return features / features.norm(dim=-1, keepdim=True)


def _encode_clip_text_features(model, text_inputs: dict):
    """Encode text prompts to unit-normalized projection vectors."""
    import torch

    with torch.no_grad():
        features = model.get_text_features(**text_inputs)
        if not isinstance(features, torch.Tensor):
            text_outputs = model.text_model(**text_inputs)
            pooled = text_outputs.pooler_output
            if pooled is None:
                pooled = text_outputs.last_hidden_state[:, -1, :]
            features = model.text_projection(pooled)
    return _normalize_features(features)


def _encode_clip_image_features(model, pixel_values):
    """Encode images to unit-normalized projection vectors."""
    import torch

    with torch.no_grad():
        features = model.get_image_features(pixel_values=pixel_values)
        if not isinstance(features, torch.Tensor):
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            pooled = vision_outputs.pooler_output
            if pooled is None:
                pooled = vision_outputs.last_hidden_state[:, 0, :]
            features = model.visual_projection(pooled)
    return _normalize_features(features)


class ClipSceneScorer:
    """Batch CLIP scorer for full-frame and optional bbox-context crops."""

    CLIP_INSTALL_HINT = "pip install transformers torch pillow"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        batch_size: int = 16,
    ):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "CLIP filtering requires transformers, torch, and pillow. "
                f"Install with: {ClipSceneScorer.CLIP_INSTALL_HINT}"
            ) from exc

        self._torch = torch
        self.batch_size = batch_size
        self.device = resolve_clip_device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.prompts: list[str] = []
        self.prompt_groups: list[str] = []
        for group, prompts in CLIP_PROMPT_GROUPS.items():
            for prompt in prompts:
                self.prompts.append(prompt)
                self.prompt_groups.append(group)

        text_inputs = self.processor(
            text=self.prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = _move_batch_to_device(
            {k: text_inputs[k] for k in text_inputs if k in ("input_ids", "attention_mask")},
            self.device,
        )

        with torch.no_grad():
            self.text_features = _encode_clip_text_features(self.model, text_inputs)

    def _encode_image_features(self, pil_images: list):
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = _move_batch_to_device(
            {"pixel_values": inputs["pixel_values"]}, self.device
        )["pixel_values"]
        return _encode_clip_image_features(self.model, pixel_values)

    def _score_pil_batch(self, images: list) -> list[np.ndarray]:
        all_probs: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            image_features = self._encode_image_features(batch)
            with self._torch.no_grad():
                logits = image_features @ self.text_features.T
                probs = logits.softmax(dim=-1)
            for row in probs.cpu().numpy():
                all_probs.append(row)
        return all_probs

    def _aggregate_group_scores(self, prompt_probs: np.ndarray) -> dict[str, float]:
        group_scores: dict[str, float] = {}
        for prob, group in zip(prompt_probs, self.prompt_groups):
            group_scores[group] = max(group_scores.get(group, 0.0), float(prob))
        return group_scores

    def score_rgb(
        self,
        full_rgb: np.ndarray,
        bbox_xywh: tuple[int, int, int, int] | None = None,
    ) -> ClipScoreResult:
        from PIL import Image

        full_pil = Image.fromarray(full_rgb)
        crop_rgb = (
            extract_expanded_bbox_crop(full_rgb, bbox_xywh) if bbox_xywh is not None else None
        )
        pil_images = [full_pil]
        if crop_rgb is not None:
            pil_images.append(Image.fromarray(crop_rgb))

        probs_list = self._score_pil_batch(pil_images)
        combined = np.maximum(probs_list[0], probs_list[1]) if len(probs_list) > 1 else probs_list[0]
        group_scores = self._aggregate_group_scores(combined)

        ranked = sorted(group_scores.items(), key=lambda x: -x[1])
        top_group, top_score = ranked[0]
        second_group, second_score = ranked[1] if len(ranked) > 1 else ("", 0.0)

        top_prompt = ""
        top_prompt_score = -1.0
        second_prompt = ""
        for prob, group, prompt in zip(combined, self.prompt_groups, self.prompts):
            if group == top_group and prob > top_prompt_score:
                top_prompt = prompt
                top_prompt_score = float(prob)
            if group == second_group and prob > 0:
                second_prompt = prompt

        return ClipScoreResult(
            clip_used=True,
            clip_top_prompt=top_prompt,
            clip_top_score=round(top_score, 6),
            clip_second_prompt=second_prompt,
            clip_second_score=round(second_score, 6),
            clip_margin=round(top_score - second_score, 6),
            clip_viewpoint_label=top_group,
            clip_filter_hint=clip_label_filter_hint(top_group),
        )


def merge_heuristic_and_clip(
    heuristic: FilterDecision,
    clip: ClipScoreResult,
    features: SceneFeatures,
    thresholds: FilterThresholds,
    clip_margin_threshold: float,
    filter_purpose: FilterPurpose = "asset_extraction",
) -> MergeResult:
    """Combine heuristic and optional CLIP signals under a purpose-specific policy."""
    if not clip.clip_used:
        return MergeResult(
            heuristic.disposition,
            "heuristic_only",
            False,
        )

    clip_low_margin = clip.clip_margin < clip_margin_threshold
    if filter_purpose == "real_eval":
        return _merge_real_eval(
            heuristic, clip, features, thresholds, clip_margin_threshold, clip_low_margin
        )
    return _merge_asset_extraction(
        heuristic, clip, features, thresholds, clip_low_margin
    )


def _weak_sky(features: SceneFeatures, thresholds: FilterThresholds) -> bool:
    return features.sky_ratio_upper < thresholds.sky_ratio_upper_horizon


def _object_low(features: SceneFeatures, thresholds: FilterThresholds) -> bool:
    return features.bbox_center_y_norm >= thresholds.object_low_y_center


def _scene_cues_bad(features: SceneFeatures, thresholds: FilterThresholds) -> bool:
    return _weak_sky(features, thresholds) and _object_low(features, thresholds)


def _clip_is_strong_bad(label: str) -> bool:
    return label in CLIP_REJECT_LABELS and label != VIEW_IRRELEVANT


def _merge_asset_extraction(
    heuristic: FilterDecision,
    clip: ClipScoreResult,
    features: SceneFeatures,
    thresholds: FilterThresholds,
    clip_low_margin: bool,
) -> MergeResult:
    """Permissive merge for asset extraction (large clear objects often useful)."""
    h = heuristic
    label = clip.clip_viewpoint_label
    clip_good = label in CLIP_ACCEPT_LABELS
    clip_bad = label in CLIP_REJECT_LABELS
    clip_ambiguous = label == CLIP_AMBIGUOUS_LABEL
    weak_sky = _weak_sky(features, thresholds)

    if h.filter_reason in HEURISTIC_HARD_REJECT or h.disposition == Disposition.REJECT:
        return MergeResult(Disposition.REJECT, "heuristic_hard_reject", clip_low_margin)

    if clip_bad and _scene_cues_bad(features, thresholds):
        return MergeResult(Disposition.REJECT, "clip_bad_scene_cues_reject", clip_low_margin)

    if h.filter_reason in HEURISTIC_STRONG_ACCEPT:
        if _clip_is_strong_bad(label):
            return MergeResult(
                Disposition.REVIEW,
                "strong_heuristic_clip_bad_review",
                clip_low_margin,
            )
        return MergeResult(
            Disposition.ACCEPT,
            "strong_heuristic_preserved_accept",
            clip_low_margin,
        )

    if h.filter_reason == REASON_LARGE_BUT_SKY_CANDIDATE:
        if clip_good:
            return MergeResult(
                Disposition.ACCEPT,
                "large_sky_clip_good_accept",
                clip_low_margin,
            )
        if clip_ambiguous or clip_low_margin:
            return MergeResult(
                Disposition.REVIEW,
                "large_sky_clip_ambiguous_review",
                clip_low_margin,
            )
        if clip_bad and weak_sky:
            return MergeResult(
                Disposition.REJECT,
                "large_sky_clip_bad_weak_sky_reject",
                clip_low_margin,
            )
        return MergeResult(
            Disposition.REVIEW,
            "large_sky_default_review",
            clip_low_margin,
        )

    if h.filter_reason == REASON_LARGE_OBJECT_REVIEW:
        if clip_good:
            return MergeResult(
                Disposition.ACCEPT,
                "large_object_clip_good_accept",
                clip_low_margin,
            )
        if clip_bad and weak_sky:
            return MergeResult(
                Disposition.REJECT,
                "large_object_clip_bad_weak_sky_reject",
                clip_low_margin,
            )
        return MergeResult(
            Disposition.REVIEW,
            "large_object_default_review",
            clip_low_margin,
        )

    if h.disposition == Disposition.ACCEPT:
        if _clip_is_strong_bad(label):
            return MergeResult(
                Disposition.REVIEW,
                "heuristic_accept_clip_bad_review",
                clip_low_margin,
            )
        return MergeResult(
            Disposition.ACCEPT,
            "heuristic_accept_preserved",
            clip_low_margin,
        )

    if h.disposition == Disposition.REVIEW:
        if clip_good:
            return MergeResult(
                Disposition.REVIEW,
                "heuristic_review_clip_good",
                clip_low_margin,
            )
        if clip_bad and _scene_cues_bad(features, thresholds):
            return MergeResult(
                Disposition.REJECT,
                "heuristic_review_clip_bad_scene_reject",
                clip_low_margin,
            )
        if clip_bad and h.disposition != clip_label_disposition(label):
            return MergeResult(
                Disposition.REVIEW,
                "heuristic_clip_disagree_review",
                clip_low_margin,
            )
        return MergeResult(
            Disposition.REVIEW,
            "heuristic_review_default",
            clip_low_margin,
        )

    return MergeResult(Disposition.REVIEW, "asset_extraction_fallback_review", clip_low_margin)


def _merge_real_eval(
    heuristic: FilterDecision,
    clip: ClipScoreResult,
    features: SceneFeatures,
    thresholds: FilterThresholds,
    clip_margin_threshold: float,
    clip_low_margin: bool,
) -> MergeResult:
    """Stricter merge for real-image evaluation (conservative on large / close-up)."""
    h = heuristic.disposition
    c = clip_label_disposition(clip.clip_viewpoint_label)
    label = clip.clip_viewpoint_label
    clip_good = label in CLIP_ACCEPT_LABELS
    weak_sky = _weak_sky(features, thresholds)

    if heuristic.filter_reason in HEURISTIC_HARD_REJECT or heuristic.disposition == Disposition.REJECT:
        return MergeResult(Disposition.REJECT, "heuristic_hard_reject", clip_low_margin)

    if clip_low_margin:
        disp = Disposition.REVIEW
        reason = "clip_low_margin_review"
    elif h == c:
        disp = h
        reason = f"heuristic_clip_agree_{h.value}"
    else:
        disp = Disposition.REVIEW
        reason = "heuristic_clip_disagree_review"

    if features.is_large_bbox and label in CLIP_REJECT_LABELS and weak_sky:
        return MergeResult(Disposition.REJECT, "large_bbox_clip_bad_scene_reject", clip_low_margin)

    if features.is_large_bbox:
        if clip_good and h == Disposition.ACCEPT and c == Disposition.ACCEPT:
            return MergeResult(Disposition.ACCEPT, "large_bbox_clip_good_keep", clip_low_margin)
        if disp == Disposition.ACCEPT:
            return MergeResult(Disposition.REVIEW, "large_bbox_real_eval_review", clip_low_margin)
        if clip_good and disp == Disposition.REJECT:
            return MergeResult(Disposition.REVIEW, "large_bbox_clip_good_scene_review", clip_low_margin)

    return MergeResult(disp, reason, clip_low_margin)

