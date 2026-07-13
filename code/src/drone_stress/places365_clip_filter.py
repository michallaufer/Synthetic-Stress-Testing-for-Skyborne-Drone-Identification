"""CLIP-based suitability filtering for Places365 high-resolution background candidates."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from drone_stress.places365_backgrounds import (
    MAPPED_CLEAR_UPPER_SKY,
    MAPPED_CLOUDY_SKY,
    MAPPED_RUNWAY,
    MAPPED_SKY_BUILT,
    MAPPED_SKY_NATURAL,
)
from drone_stress.scene_filter import (
    _encode_clip_image_features,
    _encode_clip_text_features,
    _move_batch_to_device,
    resolve_clip_device,
)

CLIP_FILTERED_CATEGORIES = (
    MAPPED_CLEAR_UPPER_SKY,
    MAPPED_CLOUDY_SKY,
    MAPPED_SKY_NATURAL,
    MAPPED_SKY_BUILT,
    MAPPED_RUNWAY,
)

GOOD_PROMPTS: tuple[str, ...] = (
    "a realistic outdoor sky background where a small drone could appear",
    "a wide open sky with a horizon",
    "a sky with rooftops or buildings below",
    "a sky with trees or natural landscape below",
    "an airport runway or open outdoor surveillance scene with sky",
)

BAD_PROMPTS: tuple[str, ...] = (
    "an indoor scene",
    "a close-up of a person",
    "a close-up object or product photo",
    "a flower or food close-up",
    "a painting, drawing, artwork, or heavily stylized image",
    "a scene dominated by cars or vehicles",
    "a scene with a large airplane or bird already in the sky",
    "a scene with very little sky",
)

CLIP_METADATA_COLUMNS = [
    "clip_good_prompt",
    "clip_good_score",
    "clip_bad_prompt",
    "clip_bad_score",
    "clip_suitability_score",
    "clip_filter_status",
    "clip_filter_reason",
    "clip_output_path",
]

CLIP_STRING_COLUMNS = (
    "clip_good_prompt",
    "clip_bad_prompt",
    "clip_filter_status",
    "clip_filter_reason",
    "clip_output_path",
)
CLIP_FLOAT_COLUMNS = (
    "clip_good_score",
    "clip_bad_score",
    "clip_suitability_score",
)

DEFAULT_SUITABILITY_THRESHOLD = 0.02


def initialize_clip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure CLIP output columns have dtypes compatible with mixed str/float writes."""
    rows = df.copy()
    for col in CLIP_STRING_COLUMNS:
        if col not in rows.columns:
            rows[col] = pd.Series([pd.NA] * len(rows), dtype="string")
        else:
            rows[col] = rows[col].astype("string")
    for col in CLIP_FLOAT_COLUMNS:
        if col not in rows.columns:
            rows[col] = np.nan
        else:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    if "clip_selected" not in rows.columns:
        rows["clip_selected"] = False
    return rows


@dataclass
class ClipSuitabilityResult:
    clip_good_prompt: str
    clip_good_score: float
    clip_bad_prompt: str
    clip_bad_score: float
    clip_suitability_score: float


class Places365ClipBackgroundScorer:
    """Score Places365 frames with separate good/bad prompt pools."""

    CLIP_INSTALL_HINT = "pip install -r requirements-clip.txt"

    def __init__(
        self,
        *,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        batch_size: int = 16,
        use_open_clip: bool = False,
        open_clip_model: str = "ViT-B-32",
        open_clip_pretrained: str = "openai",
    ):
        self.batch_size = batch_size
        self.device = resolve_clip_device(device)
        self.use_open_clip = use_open_clip

        if use_open_clip:
            try:
                import open_clip
                import torch
            except ImportError as exc:
                raise ImportError(
                    "OpenCLIP backend requires open_clip_torch and torch. "
                    f"Install with: pip install open_clip_torch torch"
                ) from exc
            self._torch = torch
            model, _, preprocess = open_clip.create_model_and_transforms(
                open_clip_model,
                pretrained=open_clip_pretrained,
            )
            self.model = model.to(self.device)
            self.model.eval()
            self.preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(open_clip_model)
            with torch.no_grad():
                good_tokens = self._tokenizer(list(GOOD_PROMPTS)).to(self.device)
                bad_tokens = self._tokenizer(list(BAD_PROMPTS)).to(self.device)
                self.good_features = _normalize_open_clip_text(
                    self.model, good_tokens
                )
                self.bad_features = _normalize_open_clip_text(self.model, bad_tokens)
        else:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor
            except ImportError as exc:
                raise ImportError(
                    f"CLIP filtering requires transformers and torch. {self.CLIP_INSTALL_HINT}"
                ) from exc
            self._torch = torch
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.good_features = self._encode_prompts_hf(GOOD_PROMPTS)
            self.bad_features = self._encode_prompts_hf(BAD_PROMPTS)

    def _encode_prompts_hf(self, prompts: tuple[str, ...]):
        text_inputs = self.processor(
            text=list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = _move_batch_to_device(
            {k: text_inputs[k] for k in text_inputs if k in ("input_ids", "attention_mask")},
            self.device,
        )
        with self._torch.no_grad():
            return _encode_clip_text_features(self.model, text_inputs)

    def _preprocess_pil_batch(self, images: list[Image.Image]):
        if self.use_open_clip:
            tensors = [self.preprocess(img.convert("RGB")) for img in images]
            return self._torch.stack(tensors).to(self.device)
        inputs = self.processor(images=images, return_tensors="pt")
        return _move_batch_to_device(
            {"pixel_values": inputs["pixel_values"]}, self.device
        )["pixel_values"]

    def _encode_image_batch(self, images: list[Image.Image]):
        pixel_values = self._preprocess_pil_batch(images)
        with self._torch.no_grad():
            if self.use_open_clip:
                return _normalize_open_clip_image(self.model, pixel_values)
            return _encode_clip_image_features(self.model, pixel_values)

    def score_pil(self, image: Image.Image) -> ClipSuitabilityResult:
        results = self.score_pil_batch([image])
        return results[0]

    def score_pil_batch(self, images: list[Image.Image]) -> list[ClipSuitabilityResult]:
        if not images:
            return []
        image_features = self._encode_image_batch(images)
        with self._torch.no_grad():
            good_logits = image_features @ self.good_features.T
            bad_logits = image_features @ self.bad_features.T
        good_scores = good_logits.cpu().numpy()
        bad_scores = bad_logits.cpu().numpy()

        out: list[ClipSuitabilityResult] = []
        for row_good, row_bad in zip(good_scores, bad_scores):
            good_idx = int(np.argmax(row_good))
            bad_idx = int(np.argmax(row_bad))
            good_score = float(row_good[good_idx])
            bad_score = float(row_bad[bad_idx])
            out.append(
                ClipSuitabilityResult(
                    clip_good_prompt=GOOD_PROMPTS[good_idx],
                    clip_good_score=round(good_score, 6),
                    clip_bad_prompt=BAD_PROMPTS[bad_idx],
                    clip_bad_score=round(bad_score, 6),
                    clip_suitability_score=round(good_score - bad_score, 6),
                )
            )
        return out


def _normalize_open_clip_text(model, tokens):
    import torch

    with torch.no_grad():
        features = model.encode_text(tokens)
    return features / features.norm(dim=-1, keepdim=True)


def _normalize_open_clip_image(model, pixel_values):
    import torch

    with torch.no_grad():
        features = model.encode_image(pixel_values)
    return features / features.norm(dim=-1, keepdim=True)


def _win_long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\") and len(resolved) >= 240:
        return "\\\\?\\" + resolved
    return resolved


def ensure_dir(path: Path) -> None:
    if os.name == "nt" and len(str(path.resolve())) >= 248:
        os.makedirs(_win_long_path(path), exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def copy_image(src: Path, dest: Path) -> None:
    ensure_dir(dest.parent)
    if os.name == "nt" and (
        len(str(src.resolve())) >= 240 or len(str(dest.resolve())) >= 240
    ):
        shutil.copy2(_win_long_path(src), _win_long_path(dest))
    else:
        shutil.copy2(src, dest)


def unique_dest(dest_dir: Path, filename: str) -> Path:
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".jpg"
    return dest_dir / f"{stem}_clip{suffix}"


def resolve_candidate_image_path(row: pd.Series, candidate_root: Path) -> Path | None:
    for col in ("output_path", "original_path"):
        raw = str(row.get(col, "")).strip()
        if raw:
            path = Path(raw)
            if path.is_file():
                return path

    filename = Path(str(row.get("original_path", ""))).name
    if not filename:
        filename = Path(str(row.get("output_path", ""))).name
    if not filename:
        return None

    mapped = str(row.get("mapped_background_type", "")).strip()
    search_dirs = [mapped, "review"]
    if row.get("filter_status") == "review":
        search_dirs = ["review", mapped]
    for sub in search_dirs:
        if not sub:
            continue
        candidate = candidate_root / sub / filename
        if candidate.is_file():
            return candidate
    return None


def score_candidates_with_clip(
    df: pd.DataFrame,
    *,
    candidate_root: Path,
    scorer: Places365ClipBackgroundScorer,
) -> pd.DataFrame:
    rows = initialize_clip_columns(df)

    eligible_mask = rows["filter_status"].astype(str).isin(["accept", "review"])
    score_indices: list[int] = []
    pil_images: list[Image.Image] = []

    for idx, row in rows[eligible_mask].iterrows():
        image_path = resolve_candidate_image_path(row, candidate_root)
        if image_path is None:
            rows.at[idx, "clip_filter_status"] = "reject"
            rows.at[idx, "clip_filter_reason"] = "image_not_found"
            continue
        try:
            with Image.open(image_path) as img:
                pil_images.append(img.convert("RGB"))
            score_indices.append(idx)
        except OSError:
            rows.at[idx, "clip_filter_status"] = "reject"
            rows.at[idx, "clip_filter_reason"] = "image_unreadable"

    for start in tqdm(
        range(0, len(pil_images), scorer.batch_size),
        desc="CLIP scoring",
        unit="batch",
    ):
        batch_imgs = pil_images[start : start + scorer.batch_size]
        batch_idx = score_indices[start : start + scorer.batch_size]
        results = scorer.score_pil_batch(batch_imgs)
        for idx, result in zip(batch_idx, results):
            rows.at[idx, "clip_good_prompt"] = result.clip_good_prompt
            rows.at[idx, "clip_good_score"] = result.clip_good_score
            rows.at[idx, "clip_bad_prompt"] = result.clip_bad_prompt
            rows.at[idx, "clip_bad_score"] = result.clip_bad_score
            rows.at[idx, "clip_suitability_score"] = result.clip_suitability_score

    skipped = ~eligible_mask
    rows.loc[skipped, "clip_filter_status"] = "reject"
    rows.loc[skipped, "clip_filter_reason"] = "skipped_non_candidate_status"
    return rows


def select_clip_filtered_candidates(
    df: pd.DataFrame,
    *,
    suitability_threshold: float,
    top_k_per_category: int,
) -> pd.DataFrame:
    rows = df.copy()
    rows["clip_selected"] = False

    eligible = rows["filter_status"].astype(str).isin(["accept", "review"])
    scored = eligible & pd.to_numeric(rows["clip_suitability_score"], errors="coerce").notna()

    for idx in rows[scored].index:
        score = float(rows.at[idx, "clip_suitability_score"])
        if score < suitability_threshold:
            rows.at[idx, "clip_filter_status"] = "reject"
            rows.at[idx, "clip_filter_reason"] = "low_clip_suitability"
        else:
            rows.at[idx, "clip_filter_status"] = "candidate"
            rows.at[idx, "clip_filter_reason"] = "clip_passed_threshold"

    candidates = rows[rows["clip_filter_status"] == "candidate"].copy()
    if candidates.empty:
        return rows

    candidates["_upper_sky"] = pd.to_numeric(
        candidates.get("upper_sky_ratio", 0), errors="coerce"
    ).fillna(0.0)
    candidates["_suitability"] = pd.to_numeric(
        candidates["clip_suitability_score"], errors="coerce"
    ).fillna(-999.0)

    selected_idx: set[int] = set()
    for category in CLIP_FILTERED_CATEGORIES:
        pool = candidates[candidates["mapped_background_type"].astype(str) == category]
        if pool.empty:
            continue
        pool = pool.sort_values(
            by=["_suitability", "_upper_sky", "original_path"],
            ascending=[False, False, True],
        )
        for idx in pool.head(top_k_per_category).index:
            selected_idx.add(int(idx))

    for idx in rows.index:
        if rows.at[idx, "clip_filter_status"] != "candidate":
            continue
        if int(idx) in selected_idx:
            rows.at[idx, "clip_filter_status"] = "accept"
            rows.at[idx, "clip_filter_reason"] = "selected_top_k"
            rows.at[idx, "clip_selected"] = True
        else:
            rows.at[idx, "clip_filter_status"] = "reject"
            rows.at[idx, "clip_filter_reason"] = "below_top_k"

    return rows


def copy_clip_selected_images(
    df: pd.DataFrame,
    *,
    candidate_root: Path,
    output_root: Path,
) -> pd.DataFrame:
    rows = df.copy()
    ensure_dir(output_root)
    for category in CLIP_FILTERED_CATEGORIES:
        ensure_dir(output_root / category)

    for idx, row in rows[rows["clip_selected"] == True].iterrows():  # noqa: E712
        src = resolve_candidate_image_path(row, candidate_root)
        if src is None:
            rows.at[idx, "clip_filter_status"] = "reject"
            rows.at[idx, "clip_filter_reason"] = "copy_source_missing"
            rows.at[idx, "clip_selected"] = False
            continue
        mapped = str(row.get("mapped_background_type", "")).strip()
        if mapped not in CLIP_FILTERED_CATEGORIES:
            rows.at[idx, "clip_filter_status"] = "reject"
            rows.at[idx, "clip_filter_reason"] = "unmapped_category"
            rows.at[idx, "clip_selected"] = False
            continue
        dest = unique_dest(output_root / mapped, src.name)
        try:
            copy_image(src, dest)
            rows.at[idx, "clip_output_path"] = str(dest.resolve())
        except OSError:
            rows.at[idx, "clip_filter_status"] = "reject"
            rows.at[idx, "clip_filter_reason"] = "copy_failed"
            rows.at[idx, "clip_selected"] = False
    return rows


def filter_places365_with_clip(
    metadata_csv: Path,
    *,
    candidate_root: Path,
    output_root: Path,
    suitability_threshold: float = DEFAULT_SUITABILITY_THRESHOLD,
    top_k_per_category: int = 80,
    clip_model: str = "openai/clip-vit-base-patch32",
    clip_device: str = "auto",
    clip_batch_size: int = 16,
    use_open_clip: bool = False,
    copy_images: bool = True,
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(metadata_csv)
    if "filter_status" not in df.columns:
        raise ValueError("metadata CSV missing filter_status column")
    df = initialize_clip_columns(df)

    scorer = Places365ClipBackgroundScorer(
        model_name=clip_model,
        device=clip_device,
        batch_size=clip_batch_size,
        use_open_clip=use_open_clip,
    )
    df = score_candidates_with_clip(df, candidate_root=candidate_root, scorer=scorer)
    df = select_clip_filtered_candidates(
        df,
        suitability_threshold=suitability_threshold,
        top_k_per_category=top_k_per_category,
    )
    if copy_images:
        df = copy_clip_selected_images(
            df, candidate_root=candidate_root, output_root=output_root
        )

    summary = {
        "input_rows": len(df),
        "clip_scored": int(
            pd.to_numeric(df["clip_suitability_score"], errors="coerce").notna().sum()
        ),
        "clip_selected": int((df["clip_selected"] == True).sum()),  # noqa: E712
        "clip_rejected": int((df["clip_filter_status"] == "reject").sum()),
        "counts_by_mapped_category": (
            df[df["clip_selected"] == True]  # noqa: E712
            .groupby("mapped_background_type")
            .size()
            .sort_values(ascending=False)
            .to_dict()
        ),
        "reject_reasons": (
            df[df["clip_filter_status"] == "reject"]
            .groupby("clip_filter_reason")
            .size()
            .sort_values(ascending=False)
            .to_dict()
        ),
    }
    return df, summary


def write_clip_filter_report(report_path: Path, summary: dict) -> None:
    lines = [
        "Places365 CLIP background filter report",
        f"input_rows: {summary['input_rows']}",
        f"clip_scored: {summary['clip_scored']}",
        f"clip_selected: {summary['clip_selected']}",
        f"clip_rejected: {summary['clip_rejected']}",
        "",
        "Selected counts by mapped_background_type:",
    ]
    for name, count in summary["counts_by_mapped_category"].items():
        lines.append(f"  {name}: {count}")
    lines.append("\nTop CLIP reject reasons:")
    for reason, count in summary["reject_reasons"].items():
        lines.append(f"  {reason}: {count}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
