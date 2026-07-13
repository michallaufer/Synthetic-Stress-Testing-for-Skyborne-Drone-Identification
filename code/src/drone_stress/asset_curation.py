"""Automated QA curation for SAM2-extracted foreground assets."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from drone_stress.extract import rgba_on_checkerboard

VALID_ASSET_TYPES = frozenset({"drone", "bird", "airplane"})
VALID_QA_LABELS = frozenset({"accept", "review", "reject"})

CURATED_COLUMNS_EXTRA = [
    "resolved_asset_path",
    "qa_auto_label",
    "qa_auto_reasons",
    "qa_final_label",
    "qa_final_notes",
    "alpha_area_ratio",
    "alpha_bbox_w",
    "alpha_bbox_h",
    "alpha_component_count",
    "alpha_largest_component_ratio",
    "alpha_border_ratio",
    "alpha_opaque_rectangle_score",
    "curated_copy_path",
]

REJECT_REASON_TOKENS = frozenset(
    {
        "likely_background_blob",
        "mask_too_large",
        "missing_proposal",
        "failed",
        "no_object",
    }
)


@dataclass
class AlphaStats:
    alpha_area_ratio: float = 0.0
    alpha_bbox_w: int = 0
    alpha_bbox_h: int = 0
    alpha_component_count: int = 0
    alpha_largest_component_ratio: float = 0.0
    alpha_border_ratio: float = 0.0
    alpha_opaque_rectangle_score: float = 0.0
    analysis_error: str = ""


@dataclass
class QAEvaluation:
    qa_auto_label: str
    qa_auto_reasons: list[str] = field(default_factory=list)
    alpha: AlphaStats = field(default_factory=AlphaStats)

    @property
    def reasons_str(self) -> str:
        return ";".join(self.qa_auto_reasons)


def _cell_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in ("true", "1", "yes", "y")


def _cell_float(value, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    s = str(value).strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _cell_str(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none"):
        return ""
    return s


def _reasons_contain(text: str, tokens: set[str]) -> bool:
    lower = text.lower()
    return any(tok in lower for tok in tokens)


def default_short_assets_root(asset_type: str) -> Path:
    return Path(rf"C:\datasets\assets_full\{asset_type}")


def resolve_curated_asset_path(
    row: pd.Series,
    *,
    metadata_path: Path,
    asset_type: str,
    short_assets_root: Path | None = None,
) -> tuple[Path | None, list[str]]:
    """Resolve PNG path from metadata row."""
    metadata_parent = metadata_path.parent.resolve()
    attempted: list[str] = []
    seen: set[str] = set()

    def _try(path: Path) -> Path | None:
        key = str(path)
        if key in seen:
            return None
        seen.add(key)
        attempted.append(key)
        if path.is_file():
            return path.resolve()
        return None

    raw = _cell_str(row.get("output_path", ""))
    asset_id = _cell_str(row.get("asset_id", ""))

    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw))
        candidates.append(metadata_parent / raw)
        candidates.append(metadata_parent / "images" / Path(raw).name)
        candidates.append(metadata_parent / Path(raw).name)

    resolved_existing = _cell_str(row.get("resolved_asset_path", ""))
    if resolved_existing:
        candidates.insert(0, Path(resolved_existing))

    if short_assets_root is not None:
        root = short_assets_root.resolve()
        if asset_id:
            candidates.append(root / f"{asset_id}.png")
        if raw:
            candidates.append(root / Path(raw).name)

    if asset_id:
        candidates.append(metadata_parent / "images" / f"{asset_id}.png")

    for cand in candidates:
        hit = _try(cand)
        if hit is not None:
            return hit, attempted

    return None, attempted


def analyze_alpha_png(path: Path, *, min_component_area: int = 16) -> AlphaStats:
    """Inspect alpha channel for fragmentation, border contact, rectangle-like blobs."""
    try:
        with Image.open(path) as img:
            rgba = np.array(img.convert("RGBA"))
    except OSError as exc:
        return AlphaStats(analysis_error=str(exc))

    h, w = rgba.shape[:2]
    if h == 0 or w == 0:
        return AlphaStats(analysis_error="empty_image")

    alpha = rgba[:, :, 3]
    fg = alpha > 12
    fg_area = int(fg.sum())
    total = h * w
    alpha_area_ratio = float(fg_area / total) if total else 0.0

    if fg_area == 0:
        return AlphaStats(alpha_area_ratio=0.0, alpha_component_count=0)

    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    bbox_w = int(x2 - x1 + 1)
    bbox_h = int(y2 - y1 + 1)

    border_mask = np.zeros_like(fg, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border_alpha_ratio = float((fg & border_mask).sum() / max(fg_area, 1))

    fg_u8 = fg.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    component_areas: list[int] = []
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            component_areas.append(area)

    component_count = len(component_areas)
    largest = max(component_areas) if component_areas else 0
    largest_ratio = float(largest / fg_area) if fg_area else 0.0

    bbox_area = max(1, bbox_w * bbox_h)
    fill_in_bbox = float(fg[y1 : y2 + 1, x1 : x2 + 1].sum() / bbox_area)
    touches_all_borders = (
        fg[0, :].any() and fg[-1, :].any() and fg[:, 0].any() and fg[:, -1].any()
    )
    opaque_rectangle_score = 0.0
    if touches_all_borders and fill_in_bbox > 0.88 and alpha_area_ratio > 0.75:
        opaque_rectangle_score = min(1.0, fill_in_bbox * alpha_area_ratio)

    return AlphaStats(
        alpha_area_ratio=round(alpha_area_ratio, 6),
        alpha_bbox_w=bbox_w,
        alpha_bbox_h=bbox_h,
        alpha_component_count=component_count,
        alpha_largest_component_ratio=round(largest_ratio, 6),
        alpha_border_ratio=round(border_alpha_ratio, 6),
        alpha_opaque_rectangle_score=round(opaque_rectangle_score, 6),
    )


def _global_reject_reasons(row: pd.Series, alpha: AlphaStats, resolved: Path | None) -> list[str]:
    reasons: list[str] = []
    if _cell_bool(row.get("extraction_failed", False)):
        reasons.append("extraction_failed")
    if resolved is None:
        reasons.append("missing_asset_file")
    if _cell_str(row.get("quality_label")).lower() == "reject":
        reasons.append("quality_label_reject")
    if _cell_str(row.get("mask_quality_label")).lower() == "reject":
        reasons.append("mask_quality_label_reject")

    mask_area_ratio = _cell_float(row.get("mask_area_ratio"))
    if mask_area_ratio > 0.70:
        reasons.append("mask_area_ratio_high")
    border_ratio = _cell_float(row.get("mask_touches_border_ratio"))
    if border_ratio > 0.65:
        reasons.append("mask_touches_border_high")
    bbox_area_ratio = _cell_float(row.get("mask_bbox_area_ratio"))
    if bbox_area_ratio > 4.0:
        reasons.append("mask_bbox_area_ratio_high")

    review_text = _cell_str(row.get("mask_review_reasons")) + " " + _cell_str(row.get("mask_selection_reason"))
    if _reasons_contain(review_text, REJECT_REASON_TOKENS):
        reasons.append("mask_review_hard_failure")

    if alpha.analysis_error:
        reasons.append(f"alpha_analysis_error:{alpha.analysis_error}")
    elif alpha.alpha_area_ratio < 0.005:
        reasons.append("alpha_area_tiny")
    elif alpha.alpha_opaque_rectangle_score > 0.65:
        reasons.append("opaque_background_rectangle")
    elif alpha.alpha_border_ratio > 0.72 and alpha.alpha_area_ratio > 0.70:
        reasons.append("alpha_border_background_blob")

    if alpha.alpha_component_count >= 8 and alpha.alpha_largest_component_ratio < 0.45:
        reasons.append("alpha_many_fragments")

    return reasons


def _global_review_reasons(row: pd.Series, alpha: AlphaStats) -> list[str]:
    reasons: list[str] = []
    if _cell_str(row.get("quality_label")).lower() == "review":
        reasons.append("quality_label_review")
    if _cell_str(row.get("mask_quality_label")).lower() == "review":
        reasons.append("mask_quality_label_review")
    if _cell_bool(row.get("mask_inverted", False)):
        reasons.append("mask_inverted")

    mask_area_ratio = _cell_float(row.get("mask_area_ratio"))
    if 0.45 <= mask_area_ratio <= 0.70:
        reasons.append("mask_area_ratio_mid")
    border_ratio = _cell_float(row.get("mask_touches_border_ratio"))
    if 0.35 <= border_ratio <= 0.65:
        reasons.append("mask_touches_border_mid")
    bbox_area_ratio = _cell_float(row.get("mask_bbox_area_ratio"))
    if 2.0 <= bbox_area_ratio <= 4.0:
        reasons.append("mask_bbox_area_ratio_mid")

    proposal = _cell_str(row.get("proposal_method"))
    asset_type = _cell_str(row.get("asset_type"))
    if asset_type in ("bird", "airplane") and proposal == "center_heuristic":
        reasons.append("center_heuristic_proposal")
    if asset_type in ("bird", "airplane"):
        conf = _cell_float(row.get("proposal_confidence"))
        if proposal.startswith("yolo") and conf < 0.25:
            reasons.append("low_yolo_confidence")

    selection = _cell_str(row.get("mask_selection_reason")).lower()
    if "inverse" in selection or "used_inverse_mask" in selection:
        reasons.append("inverse_mask_used")

    if alpha.alpha_component_count >= 4 and alpha.alpha_largest_component_ratio < 0.70:
        reasons.append("alpha_multiple_components")
    if alpha.alpha_border_ratio > 0.45 and alpha.alpha_area_ratio > 0.40:
        reasons.append("alpha_border_contact")

    return reasons


def _class_reject_reasons(row: pd.Series, alpha: AlphaStats, asset_type: str) -> list[str]:
    reasons: list[str] = []
    mask_area_ratio = _cell_float(row.get("mask_area_ratio"))
    border_ratio = _cell_float(row.get("mask_touches_border_ratio"))

    if asset_type == "drone":
        if mask_area_ratio > 0.55 and border_ratio > 0.50:
            reasons.append("drone_large_background_patch")
        if alpha.alpha_area_ratio > 0.65 and alpha.alpha_border_ratio > 0.55:
            reasons.append("drone_background_residue")
        if alpha.alpha_largest_component_ratio < 0.35 and alpha.alpha_component_count > 3:
            reasons.append("drone_broken_shape")
        proposal = _cell_str(row.get("proposal_method"))
        if proposal in ("center_heuristic", "largest_detection_fallback") and mask_area_ratio > 0.50:
            reasons.append("drone_weak_proposal_large_mask")

    elif asset_type == "bird":
        if alpha.alpha_largest_component_ratio < 0.40 and alpha.alpha_component_count >= 5:
            reasons.append("bird_fragmented")
        if mask_area_ratio > 0.62 and border_ratio > 0.55:
            reasons.append("bird_large_sky_chunk")
        if alpha.alpha_area_ratio < 0.02:
            reasons.append("bird_mostly_missing")

    elif asset_type == "airplane":
        if alpha.alpha_area_ratio < 0.015:
            reasons.append("airplane_mostly_missing")
        if mask_area_ratio > 0.75 and border_ratio > 0.70:
            reasons.append("airplane_huge_background_blob")
        if alpha.alpha_opaque_rectangle_score > 0.75:
            reasons.append("airplane_crop_failure")

    return reasons


def _class_review_reasons(row: pd.Series, alpha: AlphaStats, asset_type: str) -> list[str]:
    reasons: list[str] = []
    proposal = _cell_str(row.get("proposal_method"))

    if asset_type == "drone":
        if proposal in ("center_heuristic", "largest_detection_fallback"):
            reasons.append("drone_heuristic_proposal")
        if alpha.alpha_border_ratio > 0.35:
            reasons.append("drone_edge_halo")

    elif asset_type == "bird":
        if _cell_bool(row.get("mask_inverted", False)):
            reasons.append("bird_mask_inverted")
        if alpha.alpha_component_count >= 3 and alpha.alpha_largest_component_ratio < 0.75:
            reasons.append("bird_disconnected_alpha")
        if alpha.alpha_border_ratio > 0.30:
            reasons.append("bird_edge_artifacts")

    elif asset_type == "airplane":
        if _cell_bool(row.get("mask_inverted", False)):
            reasons.append("airplane_mask_inverted")
        if alpha.alpha_border_ratio > 0.55 and alpha.alpha_area_ratio > 0.35:
            reasons.append("airplane_strong_crop")

    return reasons


def evaluate_asset_qa(
    row: pd.Series,
    *,
    asset_type: str,
    resolved: Path | None,
    alpha: AlphaStats | None = None,
) -> QAEvaluation:
    if alpha is None:
        alpha = analyze_alpha_png(resolved) if resolved is not None else AlphaStats()

    reject = _global_reject_reasons(row, alpha, resolved)
    reject.extend(_class_reject_reasons(row, alpha, asset_type))
    reject = list(dict.fromkeys(reject))

    if reject:
        return QAEvaluation(qa_auto_label="reject", qa_auto_reasons=reject, alpha=alpha)

    review = _global_review_reasons(row, alpha)
    review.extend(_class_review_reasons(row, alpha, asset_type))
    review = list(dict.fromkeys(review))

    if review:
        return QAEvaluation(qa_auto_label="review", qa_auto_reasons=review, alpha=alpha)

    return QAEvaluation(qa_auto_label="accept", qa_auto_reasons=["auto_accept"], alpha=alpha)


def _curation_tile_title(row: pd.Series) -> str:
    reasons = _cell_str(row.get("qa_auto_reasons"))
    if len(reasons) > 42:
        reasons = reasons[:39] + "..."
    return "\n".join(
        [
            f"{_cell_str(row.get('asset_id'))} | {_cell_str(row.get('qa_final_label'))}",
            f"auto={_cell_str(row.get('qa_auto_label'))}",
            reasons or "-",
        ]
    )


def build_curation_contact_sheets(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    num_samples: int = 60,
    seed: int = 42,
) -> dict[str, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path | None] = {}
    for label in ("accept", "review", "reject"):
        subset = df[df["qa_final_label"].astype(str).str.lower() == label]
        out = output_dir / f"contact_sheet_{label}.png"
        paths[label] = _save_curation_contact_sheet(subset, out, num_samples=num_samples, seed=seed)
    return paths


def _save_curation_contact_sheet(
    df: pd.DataFrame,
    output_path: Path,
    *,
    num_samples: int,
    seed: int,
    cols: int = 6,
) -> Path | None:
    if df.empty:
        return None

    rows = df.to_dict("records")
    if len(rows) > num_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=num_samples, replace=False)
        rows = [rows[i] for i in sorted(idx)]

    n = len(rows)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 2.6, nrows * 3.0))
    axes_flat = np.array(axes).reshape(-1) if nrows * cols > 1 else np.array([axes])

    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i >= n:
            ax.set_visible(False)
            continue
        row = rows[i]
        path_str = _cell_str(row.get("resolved_asset_path")) or _cell_str(row.get("curated_copy_path"))
        path = Path(path_str) if path_str else None
        if path is None or not path.is_file():
            ax.set_title(f"missing\n{_cell_str(row.get('asset_id'))}", fontsize=7)
            continue
        with Image.open(path) as img:
            rgba = np.array(img.convert("RGBA"))
        ax.imshow(rgba_on_checkerboard(rgba))
        ax.set_title(_curation_tile_title(pd.Series(row)), fontsize=6, loc="left", pad=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _win_long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\") and len(resolved) >= 240:
        return "\\\\?\\" + resolved
    return resolved


def _ensure_dir(path: Path) -> None:
    if os.name == "nt" and len(str(path.resolve())) >= 248:
        os.makedirs(_win_long_path(path), exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def _copy_or_link(src: Path, dest: Path, *, use_symlink: bool) -> None:
    _ensure_dir(dest.parent)
    if dest.exists():
        return
    if use_symlink:
        dest.symlink_to(src)
    elif os.name == "nt" and (
        len(str(src.resolve())) >= 240 or len(str(dest.resolve())) >= 240
    ):
        shutil.copy2(_win_long_path(src), _win_long_path(dest))
    else:
        shutil.copy2(src, dest)


def curate_extracted_assets(
    metadata_path: Path,
    output_dir: Path,
    *,
    asset_type: str,
    overwrite_final: bool = False,
    make_contact_sheets: bool = True,
    short_assets_root: Path | None = None,
    use_symlink: bool = False,
    contact_sheet_samples: int = 60,
) -> dict:
    if asset_type not in VALID_ASSET_TYPES:
        raise ValueError(f"asset_type must be one of {sorted(VALID_ASSET_TYPES)}")

    metadata_path = metadata_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for label in ("accept", "review", "reject"):
        (output_dir / label).mkdir(parents=True, exist_ok=True)

    if short_assets_root is None:
        short_assets_root = default_short_assets_root(asset_type)
        if not short_assets_root.is_dir():
            short_assets_root = None

    df = pd.read_csv(metadata_path, keep_default_na=False)
    curated_rows: list[dict] = []

    missing_files = 0
    extraction_failed = 0
    reason_counts: dict[str, int] = {}

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        resolved, _ = resolve_curated_asset_path(
            row,
            metadata_path=metadata_path,
            asset_type=asset_type,
            short_assets_root=short_assets_root,
        )

        alpha = analyze_alpha_png(resolved) if resolved is not None else AlphaStats()
        qa = evaluate_asset_qa(row, asset_type=asset_type, resolved=resolved, alpha=alpha)

        existing_final = _cell_str(row_dict.get("qa_final_label"))
        if overwrite_final or not existing_final:
            final_label = qa.qa_auto_label
        else:
            final_label = existing_final.lower()
            if final_label not in VALID_QA_LABELS:
                final_label = qa.qa_auto_label

        if resolved is None:
            missing_files += 1
        if _cell_bool(row.get("extraction_failed", False)):
            extraction_failed += 1

        for reason in qa.qa_auto_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        copy_path = ""
        if resolved is not None:
            dest = output_dir / final_label / resolved.name
            try:
                _copy_or_link(resolved, dest, use_symlink=use_symlink)
                copy_path = str(dest.resolve())
            except OSError as exc:
                row_dict["qa_final_notes"] = _cell_str(row_dict.get("qa_final_notes")) + f";copy_failed:{exc}"

        alpha_dict = asdict(alpha)
        row_dict.update(
            {
                "resolved_asset_path": str(resolved) if resolved else "",
                "qa_auto_label": qa.qa_auto_label,
                "qa_auto_reasons": qa.reasons_str,
                "qa_final_label": final_label,
                "curated_copy_path": copy_path,
                **alpha_dict,
            }
        )
        if "qa_final_notes" not in row_dict:
            row_dict["qa_final_notes"] = _cell_str(row.get("qa_final_notes", ""))
        curated_rows.append(row_dict)

    curated_df = pd.DataFrame(curated_rows)
    curated_df.to_csv(output_dir / "asset_metadata_curated.csv", index=False)

    for label in ("accept", "review", "reject"):
        subset = curated_df[curated_df["qa_final_label"].astype(str).str.lower() == label]
        subset.to_csv(output_dir / f"asset_metadata_{label}.csv", index=False)

    sheet_paths: dict[str, Path | None] = {}
    if make_contact_sheets:
        sheet_paths = build_curation_contact_sheets(
            curated_df,
            output_dir,
            num_samples=contact_sheet_samples,
        )

    label_counts = curated_df["qa_final_label"].astype(str).str.lower().value_counts().to_dict()

    return {
        "total": len(curated_df),
        "accepted": int(label_counts.get("accept", 0)),
        "review": int(label_counts.get("review", 0)),
        "rejected": int(label_counts.get("reject", 0)),
        "missing_files": missing_files,
        "extraction_failed": extraction_failed,
        "reason_counts": reason_counts,
        "output_dir": str(output_dir),
        "contact_sheets": {k: str(v) if v else "" for k, v in sheet_paths.items()},
        "curated_csv": str(output_dir / "asset_metadata_curated.csv"),
    }


def print_curation_report(stats: dict) -> None:
    print("\nAsset QA curation report")
    print(f"  total assets: {stats['total']}")
    print(f"  accepted: {stats['accepted']}")
    print(f"  review: {stats['review']}")
    print(f"  rejected: {stats['rejected']}")
    print(f"  missing files: {stats['missing_files']}")
    print(f"  extraction_failed rows: {stats['extraction_failed']}")
    print(f"  output dir: {stats['output_dir']}")
    print(f"  curated csv: {stats['curated_csv']}")
    print("\n  top qa_auto_reasons:")
    for reason, count in sorted(stats["reason_counts"].items(), key=lambda x: -x[1])[:20]:
        print(f"    {reason}: {count}")
    if stats.get("contact_sheets"):
        print("\n  contact sheets:")
        for label, path in stats["contact_sheets"].items():
            if path:
                print(f"    {label}: {path}")
