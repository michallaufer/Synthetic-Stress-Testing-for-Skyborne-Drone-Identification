"""Optional CLIP-assisted background relabeling (second stage, not ground truth)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drone_stress.background_filter import BackgroundFeatures
from drone_stress.scene_filter import (
    _encode_clip_image_features,
    _encode_clip_text_features,
    _move_batch_to_device,
    resolve_clip_device,
)

# --- V1 approved daytime category names (primary output) ---
V1_CLEAR_UPPER_SKY = "clear_upper_sky"
V1_CLOUDY_SKY = "cloudy_sky"
V1_SKY_NATURAL = "sky_with_natural_landscape"
V1_SKY_BUILT = "sky_with_built_environment"

V1_APPROVED_CATEGORIES = (
    V1_CLEAR_UPPER_SKY,
    V1_CLOUDY_SKY,
    V1_SKY_NATURAL,
    V1_SKY_BUILT,
)

V1_NIGHT_LOW_LIGHT = "night_low_light"
V1_REJECT = "reject_irrelevant"
V1_REVIEW = "review_ambiguous"

# Internal CLIP / heuristic label names (mapped to V1 on output)
CLIP_CATEGORY_CLEAN_SKY = "clean_sky"
CLIP_CATEGORY_CLOUDY_SKY = "cloudy_sky"
CLIP_CATEGORY_SKY_WITH_TREES = "sky_with_trees"
CLIP_CATEGORY_SKY_WITH_BUILDINGS = "sky_with_buildings"
CLIP_CATEGORY_HORIZON = "horizon"
CLIP_CATEGORY_NIGHT_LOW_LIGHT = "night_low_light"
CLIP_CATEGORY_REJECT = "reject_irrelevant"
CLIP_CATEGORY_REVIEW = "review_ambiguous"

CLIP_BACKGROUND_CATEGORIES = (
    CLIP_CATEGORY_CLEAN_SKY,
    CLIP_CATEGORY_CLOUDY_SKY,
    CLIP_CATEGORY_SKY_WITH_TREES,
    CLIP_CATEGORY_SKY_WITH_BUILDINGS,
    CLIP_CATEGORY_HORIZON,
    CLIP_CATEGORY_NIGHT_LOW_LIGHT,
    CLIP_CATEGORY_REJECT,
    CLIP_CATEGORY_REVIEW,
)

DAYTIME_CLIP_LABELS = frozenset(
    {
        CLIP_CATEGORY_CLEAN_SKY,
        CLIP_CATEGORY_CLOUDY_SKY,
        CLIP_CATEGORY_SKY_WITH_TREES,
        CLIP_CATEGORY_SKY_WITH_BUILDINGS,
        CLIP_CATEGORY_HORIZON,
    }
)

STRONG_HEURISTIC_DAYTIME = frozenset(
    {
        CLIP_CATEGORY_CLEAN_SKY,
        CLIP_CATEGORY_CLOUDY_SKY,
        "trees_sky",
        CLIP_CATEGORY_SKY_WITH_TREES,
        "urban_skyline",
        CLIP_CATEGORY_SKY_WITH_BUILDINGS,
        CLIP_CATEGORY_HORIZON,
    }
)

NATURAL_LABELS = frozenset({"trees_sky", CLIP_CATEGORY_SKY_WITH_TREES})
BUILT_LABELS = frozenset({"urban_skyline", CLIP_CATEGORY_SKY_WITH_BUILDINGS})

HEURISTIC_TO_CANONICAL: dict[str, str] = {
    "clean_sky": CLIP_CATEGORY_CLEAN_SKY,
    "cloudy_sky": CLIP_CATEGORY_CLOUDY_SKY,
    "trees_sky": CLIP_CATEGORY_SKY_WITH_TREES,
    "urban_skyline": CLIP_CATEGORY_SKY_WITH_BUILDINGS,
    "horizon": CLIP_CATEGORY_HORIZON,
    "review": CLIP_CATEGORY_REVIEW,
    "reject": CLIP_CATEGORY_REJECT,
    CLIP_CATEGORY_SKY_WITH_TREES: CLIP_CATEGORY_SKY_WITH_TREES,
    CLIP_CATEGORY_SKY_WITH_BUILDINGS: CLIP_CATEGORY_SKY_WITH_BUILDINGS,
}

# Legacy / internal label -> V1 (horizon handled separately)
LABEL_TO_V1: dict[str, str] = {
    "clean_sky": V1_CLEAR_UPPER_SKY,
    CLIP_CATEGORY_CLEAN_SKY: V1_CLEAR_UPPER_SKY,
    "cloudy_sky": V1_CLOUDY_SKY,
    CLIP_CATEGORY_CLOUDY_SKY: V1_CLOUDY_SKY,
    "trees_sky": V1_SKY_NATURAL,
    CLIP_CATEGORY_SKY_WITH_TREES: V1_SKY_NATURAL,
    "urban_skyline": V1_SKY_BUILT,
    CLIP_CATEGORY_SKY_WITH_BUILDINGS: V1_SKY_BUILT,
    V1_NIGHT_LOW_LIGHT: V1_NIGHT_LOW_LIGHT,
    CLIP_CATEGORY_NIGHT_LOW_LIGHT: V1_NIGHT_LOW_LIGHT,
    "reject": V1_REJECT,
    CLIP_CATEGORY_REJECT: V1_REJECT,
    "review": V1_REVIEW,
    CLIP_CATEGORY_REVIEW: V1_REVIEW,
}

SEMANTIC_CATEGORY_NOTES: dict[str, str] = {
    V1_CLEAR_UPPER_SKY: "clear upper sky, not necessarily pure sky-only",
    V1_CLOUDY_SKY: "visible sky dominated by clouds or overcast conditions",
    V1_SKY_NATURAL: "trees, hills, mountains, vegetation, snow slopes",
    V1_SKY_BUILT: "buildings, roads, rooftops, plazas, beach promenade, parking/urban structures",
    V1_NIGHT_LOW_LIGHT: "holdout for future low-light robustness experiment",
    V1_REVIEW: "ambiguous; needs contact-sheet QA",
    V1_REJECT: "unusable for background compositing",
}

# V1 folder names (relative to --v1-output-root, default: data/processed)
V1_APPROVED_ROOT = Path("backgrounds_approved")
V1_HOLDOUT_ROOT = Path("backgrounds_holdout")
V1_REVIEW_ROOT = Path("backgrounds_review")
V1_REJECT_ROOT = Path("backgrounds_reject")

# Default metadata paths when v1 root is data/processed (resolved by script)
V1_GLOBAL_METADATA = Path("data/processed/backgrounds_v1_metadata.csv")
V1_APPROVED_METADATA = Path("data/processed/backgrounds_approved/background_metadata_approved.csv")

CLIP_PROMPT_GROUPS: dict[str, list[str]] = {
    CLIP_CATEGORY_CLEAN_SKY: [
        "a clear blue sky background with little clutter",
        "an open clean sky with no buildings or ground",
    ],
    CLIP_CATEGORY_CLOUDY_SKY: [
        "a cloudy sky background",
        "an overcast sky with clouds",
    ],
    CLIP_CATEGORY_SKY_WITH_TREES: [
        "a sky scene with trees or mountains at the bottom",
        "sky above forest, hills, or vegetation",
    ],
    CLIP_CATEGORY_SKY_WITH_BUILDINGS: [
        "a sky scene with buildings or urban skyline at the bottom",
        "sky above city buildings or rooftops",
    ],
    CLIP_CATEGORY_HORIZON: [
        "a horizon scene with sky above land or sea",
        "a wide outdoor landscape with a visible horizon line",
    ],
    CLIP_CATEGORY_NIGHT_LOW_LIGHT: [
        "a dark nighttime outdoor scene with visible buildings, sky, or horizon",
        "a low light outdoor surveillance scene at night",
        "a dusk or night skyline scene with artificial lights",
    ],
    CLIP_CATEGORY_REJECT: [
        "an indoor image or image with no useful sky",
        "a close-up or irrelevant non-sky background",
        "an almost completely black unusable image with no visible outdoor scene",
    ],
    CLIP_CATEGORY_REVIEW: [
        "an ambiguous outdoor scene with uncertain sky background",
    ],
}

V1_METADATA_COLUMNS = [
    "background_id",
    "original_path",
    "output_path",
    "heuristic_background_type",
    "heuristic_filter_status",
    "clip_used",
    "clip_top_prompt",
    "clip_top_score",
    "clip_second_prompt",
    "clip_second_score",
    "clip_margin",
    "clip_background_type",
    "final_background_type",
    "final_filter_status",
    "final_use_split",
    "merge_reason",
    "semantic_category_notes",
    "copy_success",
    "copy_error",
    "copy_mode",
    "skipped_due_to_copy_error",
    "image_width",
    "image_height",
    "sky_ratio_upper",
    "sky_ratio_full",
]

# Legacy alias for script imports
CLIP_METADATA_COLUMNS = V1_METADATA_COLUMNS


@dataclass
class ClipBackgroundScoreResult:
    clip_used: bool
    clip_top_prompt: str
    clip_top_score: float
    clip_second_prompt: str
    clip_second_score: float
    clip_margin: float
    clip_background_type: str
    clip_filter_hint: str

    @staticmethod
    def unused() -> ClipBackgroundScoreResult:
        return ClipBackgroundScoreResult(
            clip_used=False,
            clip_top_prompt="",
            clip_top_score=0.0,
            clip_second_prompt="",
            clip_second_score=0.0,
            clip_margin=0.0,
            clip_background_type="",
            clip_filter_hint="",
        )

    def to_dict(self) -> dict:
        return {
            "clip_used": self.clip_used,
            "clip_top_prompt": self.clip_top_prompt,
            "clip_top_score": self.clip_top_score,
            "clip_second_prompt": self.clip_second_prompt,
            "clip_second_score": self.clip_second_score,
            "clip_margin": self.clip_margin,
            "clip_background_type": self.clip_background_type,
            "clip_filter_hint": self.clip_filter_hint,
        }


@dataclass
class BackgroundMergeResult:
    final_background_type: str
    final_filter_status: str
    final_use_split: str
    merge_reason: str
    semantic_category_notes: str


def canonical_heuristic_type(heuristic_type: str) -> str:
    return HEURISTIC_TO_CANONICAL.get(heuristic_type, heuristic_type)


def absorb_horizon_to_v1(features: BackgroundFeatures) -> str:
    """Map horizon scenes to nearest approved V1 category."""
    if features.gray_white_cloud_ratio >= 0.14 and features.sky_ratio_upper >= 0.18:
        return V1_CLOUDY_SKY
    if features.lower_green_ratio >= 0.10:
        return V1_SKY_NATURAL
    if features.lower_dark_structure_ratio >= 0.018 or features.lower_texture_score >= 180:
        return V1_SKY_BUILT
    if features.sky_ratio_upper >= 0.28:
        return V1_CLEAR_UPPER_SKY
    return V1_REVIEW


def label_to_v1(label: str, features: BackgroundFeatures) -> str:
    if label == CLIP_CATEGORY_HORIZON:
        return absorb_horizon_to_v1(features)
    return LABEL_TO_V1.get(label, V1_REVIEW)


def clip_compatible_daytime(heuristic_canon: str, clip_canon: str) -> bool:
    if heuristic_canon == clip_canon:
        return True
    if clip_canon == CLIP_CATEGORY_HORIZON and heuristic_canon in STRONG_HEURISTIC_DAYTIME:
        return True
    if heuristic_canon == CLIP_CATEGORY_HORIZON and clip_canon in DAYTIME_CLIP_LABELS:
        return True
    if heuristic_canon in NATURAL_LABELS and clip_canon in NATURAL_LABELS | {CLIP_CATEGORY_HORIZON}:
        return True
    if heuristic_canon in BUILT_LABELS and clip_canon in BUILT_LABELS | {CLIP_CATEGORY_HORIZON}:
        return True
    return False


def is_completely_black_or_corrupt(features: BackgroundFeatures) -> bool:
    if features.image_width == 0 or features.image_height == 0:
        return True
    return features.sky_ratio_full < 0.02 and features.mean_brightness < 12.0


def is_clearly_unusable(features: BackgroundFeatures, heuristic_canon: str) -> bool:
    if is_completely_black_or_corrupt(features):
        return True
    if heuristic_canon == CLIP_CATEGORY_REJECT and features.sky_ratio_upper < 0.08:
        return True
    if features.sky_ratio_full < 0.05 and features.sky_ratio_upper < 0.06:
        return True
    return False


def is_night_outdoor_scene(
    features: BackgroundFeatures,
    clip: ClipBackgroundScoreResult,
    heuristic_reason: str,
) -> bool:
    if is_completely_black_or_corrupt(features):
        return False
    if clip.clip_used and clip.clip_background_type == CLIP_CATEGORY_NIGHT_LOW_LIGHT:
        return True
    if "night_low_light" in heuristic_reason:
        return True
    return (
        features.mean_brightness < 58.0
        and features.sky_ratio_full >= 0.04
        and (features.lower_dark_structure_ratio > 0.008 or features.lower_texture_score > 70)
    )


def v1_output_directory(v1_root: Path, merge: BackgroundMergeResult) -> Path:
    """Return destination folder under --v1-output-root for a merge result."""
    if merge.final_use_split == "approved_daytime":
        return v1_root / V1_APPROVED_ROOT / merge.final_background_type
    if merge.final_use_split == "holdout_night":
        return v1_root / V1_HOLDOUT_ROOT / V1_NIGHT_LOW_LIGHT
    if merge.final_use_split == "review":
        return v1_root / V1_REVIEW_ROOT
    return v1_root / V1_REJECT_ROOT


def v1_metadata_paths(v1_root: Path) -> tuple[Path, Path]:
    """Global and approved-only metadata CSV paths under v1 root."""
    global_path = v1_root / "backgrounds_v1_metadata.csv"
    approved_path = v1_root / V1_APPROVED_ROOT / "background_metadata_approved.csv"
    return global_path, approved_path


def ensure_v1_output_dirs(v1_root: Path) -> None:
    """Create all V1 category / holdout / review / reject folders."""
    for cat in V1_APPROVED_CATEGORIES:
        (v1_root / V1_APPROVED_ROOT / cat).mkdir(parents=True, exist_ok=True)
    (v1_root / V1_HOLDOUT_ROOT / V1_NIGHT_LOW_LIGHT).mkdir(parents=True, exist_ok=True)
    (v1_root / V1_REVIEW_ROOT).mkdir(parents=True, exist_ok=True)
    (v1_root / V1_REJECT_ROOT).mkdir(parents=True, exist_ok=True)


def _approved_result(v1_type: str, reason: str) -> BackgroundMergeResult:
    return BackgroundMergeResult(
        final_background_type=v1_type,
        final_filter_status="accept",
        final_use_split="approved_daytime",
        merge_reason=reason,
        semantic_category_notes=SEMANTIC_CATEGORY_NOTES.get(v1_type, ""),
    )


def _holdout_night(reason: str) -> BackgroundMergeResult:
    return BackgroundMergeResult(
        final_background_type=V1_NIGHT_LOW_LIGHT,
        final_filter_status="holdout",
        final_use_split="holdout_night",
        merge_reason=reason,
        semantic_category_notes=SEMANTIC_CATEGORY_NOTES[V1_NIGHT_LOW_LIGHT],
    )


def _review_result(reason: str) -> BackgroundMergeResult:
    return BackgroundMergeResult(
        final_background_type=V1_REVIEW,
        final_filter_status="review",
        final_use_split="review",
        merge_reason=reason,
        semantic_category_notes=SEMANTIC_CATEGORY_NOTES[V1_REVIEW],
    )


def _reject_result(reason: str) -> BackgroundMergeResult:
    return BackgroundMergeResult(
        final_background_type=V1_REJECT,
        final_filter_status="reject",
        final_use_split="reject",
        merge_reason=reason,
        semantic_category_notes=SEMANTIC_CATEGORY_NOTES[V1_REJECT],
    )


def merge_heuristic_and_clip_background(
    heuristic_type: str,
    heuristic_status: str,
    clip: ClipBackgroundScoreResult,
    features: BackgroundFeatures,
    clip_margin_threshold: float,
    *,
    heuristic_reason: str = "",
) -> BackgroundMergeResult:
    """
    Combine heuristic first-pass labels with optional CLIP relabeling.

    CLIP is a weak signal — strong heuristic daytime categories are preserved
    even when CLIP margin is very low (~0.001).
    """
    del clip_margin_threshold  # retained for API compat; v1 merge does not over-penalize low margin

    h_canon = canonical_heuristic_type(heuristic_type)
    c_canon = clip.clip_background_type if clip.clip_used else ""

    if is_clearly_unusable(features, h_canon):
        return _reject_result("unusable_dark_or_no_scene")

    if is_night_outdoor_scene(features, clip, heuristic_reason):
        return _holdout_night("night_low_light_holdout")

    h_v1 = label_to_v1(h_canon, features) if h_canon != CLIP_CATEGORY_REJECT else None
    c_v1 = (
        label_to_v1(c_canon, features)
        if clip.clip_used and c_canon not in {CLIP_CATEGORY_REJECT, CLIP_CATEGORY_REVIEW, CLIP_CATEGORY_NIGHT_LOW_LIGHT}
        else None
    )

    strong_heuristic = (
        heuristic_status == "accept" and h_canon in STRONG_HEURISTIC_DAYTIME
    )
    useful_sky = features.sky_ratio_upper >= 0.15

    # A: strong heuristic + compatible CLIP -> accept (ignore low margin)
    if strong_heuristic and h_v1 in V1_APPROVED_CATEGORIES:
        if not clip.clip_used or clip_compatible_daytime(h_canon, c_canon):
            if clip.clip_used and c_canon == CLIP_CATEGORY_REJECT and useful_sky:
                return _approved_result(h_v1, "strong_heuristic_over_clip_reject")
            return _approved_result(h_v1, "strong_heuristic_accept")
        if c_v1 in V1_APPROVED_CATEGORIES:
            return _approved_result(c_v1, "strong_heuristic_compatible_clip")

    # Heuristic useful daytime type but review status — still approve if sky is decent
    if h_canon in STRONG_HEURISTIC_DAYTIME and useful_sky and h_v1 in V1_APPROVED_CATEGORIES:
        if not clip.clip_used or clip_compatible_daytime(h_canon, c_canon):
            return _approved_result(h_v1, "heuristic_useful_daytime")
        if clip.clip_used and c_v1 in V1_APPROVED_CATEGORIES:
            return _approved_result(c_v1, "heuristic_useful_compatible_clip")

    # Salvage heuristic reject when sky is clearly present
    if h_canon == CLIP_CATEGORY_REJECT and features.sky_ratio_upper >= 0.22:
        if clip.clip_used and c_canon in DAYTIME_CLIP_LABELS and c_v1 in V1_APPROVED_CATEGORIES:
            return _approved_result(c_v1, "salvage_reject_heuristic_clip_daytime")
        v1 = absorb_horizon_to_v1(features) if features.sky_ratio_upper >= 0.28 else None
        if v1 in V1_APPROVED_CATEGORIES:
            return _approved_result(v1, "salvage_strong_sky_from_reject")

    # CLIP-only salvage for useful sky when heuristic was weak
    if clip.clip_used and c_canon in DAYTIME_CLIP_LABELS and useful_sky and c_v1 in V1_APPROVED_CATEGORIES:
        if h_canon in {CLIP_CATEGORY_REVIEW, CLIP_CATEGORY_REJECT}:
            return _approved_result(c_v1, "clip_daytime_salvage")

    # Heuristic reject + no salvage path
    if h_canon == CLIP_CATEGORY_REJECT and not useful_sky:
        return _reject_result("heuristic_reject_no_useful_sky")

    # Both weak / ambiguous
    if h_canon in {CLIP_CATEGORY_REVIEW, CLIP_CATEGORY_REJECT} and (
        not clip.clip_used or c_canon in {CLIP_CATEGORY_REVIEW, CLIP_CATEGORY_REJECT, ""}
    ):
        if useful_sky:
            return _review_result("borderline_useful_ambiguous")
        return _review_result("ambiguous_weak_signals")

    if clip.clip_used and c_canon == CLIP_CATEGORY_REJECT and not useful_sky:
        return _reject_result("clip_reject_no_useful_sky")

    # Default: weak review only when we cannot confidently approve
    if h_v1 in V1_APPROVED_CATEGORIES and useful_sky:
        return _approved_result(h_v1, "default_heuristic_v1")

    return _review_result("uncertain_merge")


class ClipBackgroundScorer:
    """Batch CLIP scorer for full-frame background images."""

    CLIP_INSTALL_HINT = "pip install -r requirements-clip.txt"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "CLIP background relabeling requires transformers, torch, and pillow. "
                f"Install with: {self.CLIP_INSTALL_HINT}"
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

    def _result_from_probs(self, probs: np.ndarray) -> ClipBackgroundScoreResult:
        group_scores = self._aggregate_group_scores(probs)
        ranked = sorted(group_scores.items(), key=lambda x: -x[1])
        top_group, top_score = ranked[0]
        second_group, second_score = ranked[1] if len(ranked) > 1 else ("", 0.0)

        top_prompt = ""
        top_prompt_score = -1.0
        second_prompt = ""
        for prob, group, prompt in zip(probs, self.prompt_groups, self.prompts):
            if group == top_group and prob > top_prompt_score:
                top_prompt = prompt
                top_prompt_score = float(prob)
            if group == second_group and prob > 0 and not second_prompt:
                second_prompt = prompt

        return ClipBackgroundScoreResult(
            clip_used=True,
            clip_top_prompt=top_prompt,
            clip_top_score=round(top_score, 6),
            clip_second_prompt=second_prompt,
            clip_second_score=round(second_score, 6),
            clip_margin=round(top_score - second_score, 6),
            clip_background_type=top_group,
            clip_filter_hint=top_group,
        )

    def score_rgb_batch(self, rgbs: list[np.ndarray]) -> list[ClipBackgroundScoreResult]:
        from PIL import Image

        if not rgbs:
            return []
        pil_images = [Image.fromarray(rgb) for rgb in rgbs]
        return [self._result_from_probs(p) for p in self._score_pil_batch(pil_images)]
