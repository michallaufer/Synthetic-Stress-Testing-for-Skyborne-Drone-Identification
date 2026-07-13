"""SAM2 extraction from unlabeled manual foreground image folders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from drone_stress.assets import IMAGE_EXTENSIONS, list_images
from drone_stress.bbox_proposal import BboxProposal, YoloProposalService, propose_bbox_for_image
from drone_stress.extract import (
    EXTRACTION_METHOD_SAM2,
    ExtractedAsset,
    Sam2ExtractConfig,
    build_extraction_contact_sheet,
    extraction_save_summary,
    rgba_on_checkerboard,
    safe_save_rgba_image,
)
from drone_stress.manual_sam2_mask import MaskSelectionResult, select_best_mask_in_crop
from drone_stress.sam2_extract import expanded_prompt_xyxy

VALID_ASSET_TYPES = frozenset({"drone", "bird", "airplane"})

MANUAL_METADATA_COLUMNS = [
    "asset_id",
    "asset_type",
    "asset_class",
    "source_image",
    "source_filename",
    "output_path",
    "extraction_method",
    "bbox_proposal_method",
    "bbox_proposal_confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "mask_area",
    "asset_width",
    "asset_height",
    "quality_label",
    "mask_quality_score",
    "mask_review_reasons",
    "sam2_used",
    "sam2_model_size",
    "needs_manual_review",
    "extraction_failed",
    "extraction_error",
    "notes",
    "yolo_available",
    "yolo_model_loaded",
    "yolo_detection_count",
    "proposal_method",
    "proposal_class_name",
    "proposal_confidence",
    "proposal_bbox_x",
    "proposal_bbox_y",
    "proposal_bbox_w",
    "proposal_bbox_h",
    "proposal_notes",
    "mask_area_ratio",
    "mask_bbox_area_ratio",
    "mask_touches_border_ratio",
    "mask_inverted",
    "mask_selection_reason",
    "mask_quality_label",
    "mask_area_px",
    "width",
    "height",
    "source_bbox",
    "debug_image_path",
    "debug_notes",
]

# Windows MAX_PATH safety for debug PNG filenames under long project roots.
_WIN_DEBUG_PATH_BUDGET = 240


@dataclass
class DebugSaveResult:
    success: bool
    path: str
    error: str


def _debug_png_stem(asset_id: str, *, max_stem_len: int = 48) -> str:
    stem = asset_id
    if len(stem) > max_stem_len:
        digest = hashlib.sha256(stem.encode()).hexdigest()[:10]
        stem = f"{stem[:max_stem_len]}_{digest}"
    return stem


def resolve_debug_dir(
    output_dir: Path,
    asset_type: str,
    *,
    short_output_root: Path | None = None,
) -> Path:
    """Prefer short debug root on Windows when asset PNGs use --short-output-root."""
    if short_output_root is not None:
        return short_output_root.resolve() / "debug" / asset_type
    return output_dir.resolve() / "debug"


def resolve_debug_image_path(debug_dir: Path, asset_id: str) -> Path:
    """Build a debug PNG path, shortening the stem if the full path is too long."""
    debug_dir = debug_dir.resolve()
    candidate = debug_dir / f"{_debug_png_stem(asset_id)}_debug.png"
    if len(str(candidate)) <= _WIN_DEBUG_PATH_BUDGET:
        return candidate
    digest = hashlib.sha256(asset_id.encode()).hexdigest()[:16]
    return debug_dir / f"{digest}_debug.png"


def ensure_manual_output_dirs(
    output_dir: Path,
    *,
    metadata_csv: Path,
    contact_sheet: Path | None,
    qa_report: Path,
    save_debug: bool,
    images_dir: Path | None = None,
) -> Path:
    """Create all output folders before extraction starts."""
    output_dir = output_dir.resolve()
    images_root = images_dir or (output_dir / "images")
    images_root.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    qa_report.parent.mkdir(parents=True, exist_ok=True)
    if contact_sheet is not None:
        contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    if save_debug:
        (output_dir / "debug").mkdir(parents=True, exist_ok=True)
    return images_root


def resolve_asset_output_path(
    output_path: str | Path,
    *,
    metadata_parent: Path,
    output_dir: Path,
) -> tuple[Path | None, list[str]]:
    """
    Resolve a metadata output_path to an on-disk PNG.

    Search order: absolute, metadata parent, output_dir, output_dir/images.
    """
    attempted: list[str] = []
    raw = str(output_path).strip()
    if not raw:
        return None, attempted

    candidates = [
        Path(raw),
        metadata_parent / raw,
        output_dir / raw,
        output_dir / "images" / Path(raw).name,
        metadata_parent / "images" / Path(raw).name,
    ]
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        attempted.append(key)
        if cand.is_file():
            return cand.resolve(), attempted
    return None, attempted


def _asset_id_from_path(asset_type: str, source_path: Path) -> str:
    stem = source_path.stem
    prefix = f"{asset_type}_"
    if stem == asset_type or stem.startswith(prefix):
        return stem
    return f"{asset_type}_{stem}"


def _manual_tile_title(record: ExtractedAsset, source_filename: str, extra: dict) -> str:
    label = extra.get("quality_label") or record.mask_quality_label or "n/a"
    method = extra.get("proposal_method", "")
    reason = extra.get("mask_selection_reason", record.mask_review_reasons or "-")
    if len(str(reason)) > 36:
        reason = str(reason)[:33] + "..."
    return "\n".join(
        [
            f"{record.asset_id} | {record.asset_type}",
            f"src: {source_filename}",
            f"{method} | {label}",
            str(reason),
        ]
    )


def record_to_manual_row(
    record: ExtractedAsset,
    *,
    asset_type: str,
    proposal: BboxProposal,
    mask_info: MaskSelectionResult,
    debug_image_path: str = "",
    debug_notes: str = "",
) -> dict:
    bbox = json.loads(record.source_bbox) if record.source_bbox else [0, 0, 0, 0]
    quality = mask_info.quality_label or record.mask_quality_label or ""
    notes_parts = [
        f"proposal={proposal.proposal_method}",
        proposal.notes,
        mask_info.mask_selection_reason,
    ]
    return {
        "asset_id": record.asset_id,
        "asset_type": asset_type,
        "asset_class": record.asset_class,
        "source_image": record.source_image,
        "source_filename": Path(record.source_image).name,
        "output_path": record.output_path,
        "extraction_method": record.extraction_method,
        "bbox_proposal_method": proposal.proposal_method,
        "bbox_proposal_confidence": proposal.confidence,
        "bbox_x": bbox[0],
        "bbox_y": bbox[1],
        "bbox_w": bbox[2],
        "bbox_h": bbox[3],
        "mask_area": mask_info.mask_area_px,
        "asset_width": record.width,
        "asset_height": record.height,
        "quality_label": quality,
        "mask_quality_score": mask_info.mask_quality_score,
        "mask_review_reasons": mask_info.mask_review_reasons,
        "sam2_used": record.sam2_used,
        "sam2_model_size": record.sam2_model_size,
        "needs_manual_review": record.needs_manual_review,
        "extraction_failed": record.extraction_failed,
        "extraction_error": record.extraction_error,
        "notes": "; ".join(p for p in notes_parts if p),
        "yolo_available": proposal.yolo_available,
        "yolo_model_loaded": proposal.yolo_model_loaded,
        "yolo_detection_count": proposal.yolo_detection_count,
        "proposal_method": proposal.proposal_method,
        "proposal_class_name": proposal.proposal_class_name,
        "proposal_confidence": proposal.confidence,
        "proposal_bbox_x": proposal.x,
        "proposal_bbox_y": proposal.y,
        "proposal_bbox_w": proposal.w,
        "proposal_bbox_h": proposal.h,
        "proposal_notes": proposal.notes,
        "mask_area_ratio": mask_info.mask_area_ratio,
        "mask_bbox_area_ratio": mask_info.mask_bbox_area_ratio,
        "mask_touches_border_ratio": mask_info.mask_touches_border_ratio,
        "mask_inverted": mask_info.mask_inverted,
        "mask_selection_reason": mask_info.mask_selection_reason,
        "mask_quality_label": quality,
        "mask_area_px": mask_info.mask_area_px,
        "width": record.width,
        "height": record.height,
        "source_bbox": record.source_bbox,
        "debug_image_path": debug_image_path,
        "debug_notes": debug_notes,
    }


def write_manual_metadata_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=MANUAL_METADATA_COLUMNS)
    df.to_csv(path, index=False)
    return path


def _apply_min_mask_area(record: ExtractedAsset, mask_info: MaskSelectionResult, min_mask_area: int) -> None:
    if min_mask_area <= 0:
        return
    if mask_info.mask_area_px < min_mask_area and mask_info.quality_label != "reject":
        mask_info.quality_label = "reject"
        mask_info.needs_manual_review = True
        record.mask_quality_label = "reject"
        record.needs_manual_review = True
        reason = f"mask_area_below_min({mask_info.mask_area_px}<{min_mask_area})"
        mask_info.mask_selection_reason = f"{mask_info.mask_selection_reason};{reason}"


def _save_debug_panel(
    debug_path: Path,
    *,
    rgb: np.ndarray,
    proposal: BboxProposal,
    mask_full: np.ndarray | None,
    rgba_crop: np.ndarray,
    quality_label: str,
    mask_selection_reason: str,
    proposal_method: str,
) -> DebugSaveResult:
    """Save a 3-panel debug image; never raises (returns error in DebugSaveResult)."""
    debug_path = Path(debug_path).resolve()
    try:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(rgb)
        rect = patches.Rectangle(
            (proposal.x, proposal.y),
            proposal.w,
            proposal.h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        axes[0].add_patch(rect)
        axes[0].set_title(f"proposal: {proposal_method}", fontsize=8)
        axes[0].axis("off")

        if mask_full is not None:
            overlay = rgb.copy().astype(float)
            overlay[mask_full] = overlay[mask_full] * 0.4 + np.array([255, 64, 64]) * 0.6
            axes[1].imshow(overlay.astype(np.uint8))
            axes[1].set_title("selected mask", fontsize=8)
        else:
            axes[1].text(0.5, 0.5, "no mask", ha="center", va="center")
        axes[1].axis("off")

        axes[2].imshow(rgba_on_checkerboard(rgba_crop))
        axes[2].set_title(f"{quality_label}\n{mask_selection_reason[:60]}", fontsize=7)
        axes[2].axis("off")

        fig.tight_layout()
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(debug_path, dpi=110, bbox_inches="tight")
        plt.close(fig)

        if not debug_path.is_file():
            return DebugSaveResult(
                success=False,
                path=str(debug_path),
                error=f"savefig completed but file missing: {debug_path}",
            )
        return DebugSaveResult(success=True, path=str(debug_path), error="")
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return DebugSaveResult(
            success=False,
            path=str(debug_path),
            error=f"{type(exc).__name__}: {exc}",
        )


def _extract_one_manual_image(
    *,
    rgb: np.ndarray,
    image_path: Path,
    asset_id: str,
    asset_type: str,
    dest_path: Path,
    sam2_config: Sam2ExtractConfig,
    predictor,
    source_dataset: str,
    min_yolo_conf: float,
    heuristic_coverage: float,
    min_mask_area: int,
) -> tuple[ExtractedAsset, dict, BboxProposal, MaskSelectionResult]:
    proposal = propose_bbox_for_image(
        rgb,
        asset_type,
        min_conf=min_yolo_conf,
        heuristic_coverage=heuristic_coverage,
    )
    bbox = proposal.as_xywh()
    x, y, w, h = bbox
    img_h, img_w = rgb.shape[:2]

    prompt_xyxy = expanded_prompt_xyxy(bbox, sam2_config.expand_box_ratio, img_w, img_h)
    image_key = str(image_path.resolve())

    record = ExtractedAsset(
        asset_id=asset_id,
        source_image=str(image_path.resolve()),
        asset_type=asset_type if asset_type == "drone" else "distractor",
        asset_class=asset_type,
        source_bbox=json.dumps(list(bbox)),
        output_path="",
        width=w,
        height=h,
        extraction_method=EXTRACTION_METHOD_SAM2,
        needs_sam2_refinement=False,
        source_dataset=source_dataset,
        has_alpha=False,
        sam2_used=True,
        sam2_model_size=sam2_config.model_size,
        sam2_box_prompt_xyxy=json.dumps(list(prompt_xyxy)),
        extraction_failed=True,
    )

    try:
        full_masks = predictor.predict_masks_multimask(rgb, prompt_xyxy, image_key=image_key)
        crop_masks = [m[y : y + h, x : x + w] for m in full_masks]
        crop_rgb = rgb[y : y + h, x : x + w]
        mask_info = select_best_mask_in_crop(
            crop_rgb,
            crop_masks,
            prop_x=0,
            prop_y=0,
            prop_w=w,
            prop_h=h,
        )
        best_full_mask = None
        if full_masks:
            # Reconstruct full-image mask for debug overlay from crop selection.
            best_full_mask = np.zeros((img_h, img_w), dtype=bool)
            best_full_mask[y : y + h, x : x + w] = mask_info.mask_crop

        _apply_min_mask_area(record, mask_info, min_mask_area)

        record.mask_quality_label = mask_info.quality_label
        record.mask_quality_score = mask_info.mask_quality_score
        record.mask_review_reasons = mask_info.mask_review_reasons
        record.mask_area_px = mask_info.mask_area_px
        record.mask_area_ratio_in_crop = mask_info.mask_area_ratio
        record.mask_bbox_x = mask_info.mask_bbox_x
        record.mask_bbox_y = mask_info.mask_bbox_y
        record.mask_bbox_w = mask_info.mask_bbox_w
        record.mask_bbox_h = mask_info.mask_bbox_h
        record.mask_bbox_area_ratio_in_crop = mask_info.mask_bbox_area_ratio
        record.mask_num_touched_borders = mask_info.mask_num_touched_borders
        record.needs_manual_review = mask_info.needs_manual_review
        record.has_alpha = bool(mask_info.mask_area_px > 0)

        dest_path = dest_path.resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        save_result = safe_save_rgba_image(mask_info.rgba, dest_path)
        if save_result.success and dest_path.is_file():
            record.output_path = str(dest_path)
            record.extraction_failed = False
            record.extraction_error = ""
        else:
            record.extraction_failed = True
            record.extraction_error = save_result.error
            record.mask_quality_label = "reject"
            record.needs_manual_review = True

        extra = {
            "quality_label": mask_info.quality_label,
            "proposal_method": proposal.proposal_method,
            "mask_selection_reason": mask_info.mask_selection_reason,
            "_best_full_mask": best_full_mask,
            "_rgba_crop": mask_info.rgba,
        }
        return record, extra, proposal, mask_info
    except Exception as exc:  # noqa: BLE001
        record.extraction_failed = True
        record.extraction_error = str(exc)
        record.mask_quality_label = "reject"
        record.needs_manual_review = True
        record.sam2_used = False
        empty = MaskSelectionResult(
            mask_crop=np.zeros((h, w), dtype=bool),
            rgba=np.zeros((h, w, 4), dtype=np.uint8),
            mask_inverted=False,
            mask_selection_reason=f"sam2_error:{type(exc).__name__}",
            mask_area_ratio=0.0,
            mask_bbox_area_ratio=0.0,
            mask_touches_border_ratio=0.0,
            quality_label="reject",
            mask_review_reasons=str(exc),
            mask_quality_score=0.0,
            needs_manual_review=True,
            mask_area_px=0,
            mask_bbox_x=0,
            mask_bbox_y=0,
            mask_bbox_w=0,
            mask_bbox_h=0,
            mask_num_touched_borders=0,
        )
        return record, {}, proposal, empty


def extract_manual_folder_sam2(
    input_dir: Path,
    output_dir: Path,
    *,
    asset_type: str,
    sam2_config: Sam2ExtractConfig,
    images_dir: Path,
    source_dataset: str = "manual_curated",
    max_assets: int | None = None,
    min_mask_area: int = 0,
    min_yolo_conf: float = 0.15,
    heuristic_coverage: float = 0.72,
    save_debug: bool = False,
    debug_dir: Path | None = None,
    short_output_root: Path | None = None,
) -> tuple[list[ExtractedAsset], list[dict], list[dict], list[str]]:
    if asset_type not in VALID_ASSET_TYPES:
        raise ValueError(f"asset_type must be one of {sorted(VALID_ASSET_TYPES)}, got {asset_type!r}")

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    images_dir = images_dir.resolve()
    images_dir.mkdir(parents=True, exist_ok=True)
    debug_errors: list[str] = []
    if save_debug:
        debug_dir = debug_dir or resolve_debug_dir(
            output_dir, asset_type, short_output_root=short_output_root
        )
        debug_dir = debug_dir.resolve()
        debug_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in list_images(input_dir) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {input_dir}")
    if max_assets is not None:
        image_paths = image_paths[:max_assets]

    from drone_stress.sam2_extract import Sam2BoxPredictor, ensure_sam2_dependencies

    ensure_sam2_dependencies()
    predictor = Sam2BoxPredictor(
        model_size=sam2_config.model_size,  # type: ignore[arg-type]
        checkpoint=sam2_config.checkpoint,
        device=sam2_config.device,
    )

    records: list[ExtractedAsset] = []
    rows: list[dict] = []
    extras: list[dict] = []
    seen_ids: set[str] = set()

    for image_path in tqdm(image_paths, desc=f"SAM2 {asset_type}"):
        with Image.open(image_path) as img:
            rgb = np.array(img.convert("RGB"))

        asset_id = _asset_id_from_path(asset_type, image_path)
        if asset_id in seen_ids:
            asset_id = f"{asset_id}_{len(seen_ids):04d}"
        seen_ids.add(asset_id)

        dest_path = images_dir / f"{asset_id}.png"
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        record, extra, proposal, mask_info = _extract_one_manual_image(
            rgb=rgb,
            image_path=image_path,
            asset_id=asset_id,
            asset_type=asset_type,
            dest_path=dest_path,
            sam2_config=sam2_config,
            predictor=predictor,
            source_dataset=source_dataset,
            min_yolo_conf=min_yolo_conf,
            heuristic_coverage=heuristic_coverage,
            min_mask_area=min_mask_area,
        )

        debug_path_str = ""
        debug_notes = ""
        if save_debug and debug_dir is not None:
            debug_path = resolve_debug_image_path(debug_dir, asset_id)
            debug_result = _save_debug_panel(
                debug_path,
                rgb=rgb,
                proposal=proposal,
                mask_full=extra.get("_best_full_mask"),
                rgba_crop=extra.get("_rgba_crop", mask_info.rgba),
                quality_label=mask_info.quality_label,
                mask_selection_reason=mask_info.mask_selection_reason,
                proposal_method=proposal.proposal_method,
            )
            if debug_result.success:
                debug_path_str = debug_result.path
            else:
                debug_notes = f"debug_save_failed:{debug_result.error}"
                debug_errors.append(f"{asset_id}: {debug_result.error} (attempted {debug_result.path})")

        rows.append(
            record_to_manual_row(
                record,
                asset_type=asset_type,
                proposal=proposal,
                mask_info=mask_info,
                debug_image_path=debug_path_str,
                debug_notes=debug_notes,
            )
        )
        extras.append(
            {
                "asset_id": asset_id,
                "source_filename": image_path.name,
                "proposal_method": proposal.proposal_method,
                "proposal_confidence": proposal.confidence,
                "yolo_detection_count": proposal.yolo_detection_count,
                "quality_label": mask_info.quality_label,
                "debug_notes": debug_notes,
            }
        )
        records.append(record)

    return records, rows, extras, debug_errors


def build_manual_contact_sheet(
    records: list[ExtractedAsset],
    rows: list[dict],
    output_path: Path,
    *,
    metadata_parent: Path,
    output_dir: Path,
    num_samples: int = 48,
    seed: int = 42,
) -> tuple[Path | None, list[str]]:
    """Contact sheet with robust output_path resolution; logs missing paths."""
    if not records:
        return None, []

    row_by_id = {r["asset_id"]: r for r in rows}
    missing_log: list[str] = []
    resolved_records: list[ExtractedAsset] = []

    for record in records:
        row = row_by_id.get(record.asset_id, {})
        resolved, attempted = resolve_asset_output_path(
            record.output_path,
            metadata_parent=metadata_parent,
            output_dir=output_dir,
        )
        if resolved is None:
            missing_log.append(
                f"{record.asset_id}: missing output; tried: {' | '.join(attempted)}"
            )
            continue
        record.output_path = str(resolved)
        resolved_records.append(record)

    if not resolved_records:
        return None, missing_log

    source_names = {r.asset_id: Path(r.source_image).name for r in resolved_records}

    def _title(rec: ExtractedAsset) -> str:
        extra = row_by_id.get(rec.asset_id, {})
        return _manual_tile_title(rec, source_names.get(rec.asset_id, "?"), extra)

    path = build_extraction_contact_sheet(
        resolved_records,
        output_path,
        num_samples=num_samples,
        seed=seed,
        title_fn=_title,
    )
    return path, missing_log


def write_manual_qa_report(
    records: list[ExtractedAsset],
    rows: list[dict],
    report_path: Path,
    *,
    yolo_status: dict,
    missing_paths: list[str],
    per_image_log: list[dict],
    debug_errors: list[str] | None = None,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_stats = extraction_save_summary(records)
    lines = [
        "Manual SAM2 extraction QA report",
        f"total_rows: {len(records)}",
        f"saved_ok: {save_stats['extracted']}",
        f"save_failed: {save_stats['extraction_failed']}",
        "",
        "YOLO:",
        f"  installed: {yolo_status.get('installed')}",
        f"  model_loaded: {yolo_status.get('model_loaded')}",
        f"  load_error: {yolo_status.get('load_error', '')}",
        "",
        "Quality labels:",
    ]
    for label in ("accept", "review", "reject"):
        count = sum(1 for r in rows if r.get("quality_label") == label)
        lines.append(f"  {label}: {count}")

    lines.append("\nProposal methods:")
    methods: dict[str, int] = {}
    for row in rows:
        m = str(row.get("proposal_method", "unknown"))
        methods[m] = methods.get(m, 0) + 1
    for method, count in sorted(methods.items()):
        lines.append(f"  {method}: {count}")

    if missing_paths:
        lines.append("\nMissing output paths (contact sheet):")
        lines.extend(f"  {m}" for m in missing_paths)

    if save_stats["errors_by_type"]:
        lines.append("\nSave errors:")
        for err, count in save_stats["errors_by_type"].items():
            lines.append(f"  [{count}] {err}")

    if debug_errors:
        lines.append("\nDebug image save failures:")
        lines.extend(f"  {err}" for err in debug_errors)

    lines.append("\nPer-image summary:")
    for item in per_image_log:
        debug_note = item.get("debug_notes", "")
        suffix = f" | debug_notes={debug_note}" if debug_note else ""
        lines.append(
            f"  {item.get('asset_id')} | {item.get('source_filename')} | "
            f"proposal={item.get('proposal_method')} conf={item.get('proposal_confidence')} "
            f"yolo_dets={item.get('yolo_detection_count')} | quality={item.get('quality_label')}"
            f"{suffix}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_manual_extraction_pipeline(
    input_dir: Path,
    output_dir: Path,
    *,
    asset_type: str,
    sam2_config: Sam2ExtractConfig,
    metadata_csv: Path | None = None,
    contact_sheet_output: Path | None = None,
    qa_report_output: Path | None = None,
    make_contact_sheet: bool = True,
    contact_sheet_samples: int = 48,
    contact_sheet_seed: int = 42,
    source_dataset: str = "manual_curated",
    max_assets: int | None = None,
    min_mask_area: int = 0,
    min_yolo_conf: float = 0.15,
    heuristic_coverage: float = 0.72,
    save_debug: bool = False,
    images_dir: Path | None = None,
    short_output_root: Path | None = None,
) -> dict:
    output_dir = output_dir.resolve()
    metadata_path = (metadata_csv or (output_dir / "asset_metadata.csv")).resolve()
    sheet_path = (contact_sheet_output or (output_dir / "qa_contact_sheet.png")).resolve()
    report_path = (qa_report_output or (output_dir / "qa_report.txt")).resolve()

    if short_output_root is not None:
        images_root = short_output_root.resolve() / asset_type
        debug_root = resolve_debug_dir(output_dir, asset_type, short_output_root=short_output_root)
    elif images_dir is not None:
        images_root = images_dir.resolve()
        debug_root = output_dir / "debug"
    else:
        images_root = output_dir / "images"
        debug_root = output_dir / "debug"

    save_debug_effective = save_debug or (max_assets is not None and max_assets <= 20)
    ensure_manual_output_dirs(
        output_dir,
        metadata_csv=metadata_path,
        contact_sheet=sheet_path if make_contact_sheet else None,
        qa_report=report_path,
        save_debug=save_debug_effective,
        images_dir=images_root,
    )
    if save_debug_effective:
        debug_root.mkdir(parents=True, exist_ok=True)
        print(f"  debug images: {debug_root.resolve()}")

    yolo_startup = YoloProposalService.startup_status()
    if not yolo_startup.installed:
        print(
            "\n*** WARNING: YOLO unavailable; using center heuristic only. "
            "Extraction quality may be poor. Install: pip install -r requirements-yolo.txt ***\n"
        )
    elif not YoloProposalService.ensure_model():
        print(
            f"\n*** WARNING: YOLO installed but model failed to load: {yolo_startup.load_error}. "
            "Falling back to center heuristic. ***\n"
        )
    else:
        print("YOLO: installed=yes, model_loaded=yes")

    records, rows, per_image_log, debug_errors = extract_manual_folder_sam2(
        input_dir,
        output_dir,
        asset_type=asset_type,
        sam2_config=sam2_config,
        images_dir=images_root,
        source_dataset=source_dataset,
        max_assets=max_assets,
        min_mask_area=min_mask_area,
        min_yolo_conf=min_yolo_conf,
        heuristic_coverage=heuristic_coverage,
        save_debug=save_debug_effective,
        debug_dir=debug_root if save_debug_effective else None,
        short_output_root=short_output_root,
    )

    write_manual_metadata_csv(rows, metadata_path)

    missing_paths: list[str] = []
    if make_contact_sheet:
        _, missing_paths = build_manual_contact_sheet(
            records,
            rows,
            sheet_path,
            metadata_parent=metadata_path.parent,
            output_dir=output_dir,
            num_samples=contact_sheet_samples,
            seed=contact_sheet_seed,
        )

    write_manual_qa_report(
        records,
        rows,
        report_path,
        yolo_status={
            "installed": yolo_startup.installed,
            "model_loaded": YoloProposalService.startup_status().model_loaded,
            "load_error": yolo_startup.load_error,
        },
        missing_paths=missing_paths,
        per_image_log=per_image_log,
        debug_errors=debug_errors,
    )

    save_stats = extraction_save_summary(records)
    quality_counts = {
        label: sum(1 for r in rows if r.get("quality_label") == label)
        for label in ("accept", "review", "reject")
    }

    return {
        "total": len(records),
        "saved_ok": save_stats["extracted"],
        "save_failed": save_stats["extraction_failed"],
        "quality_counts": quality_counts,
        "metadata_csv": str(metadata_path),
        "contact_sheet": str(sheet_path) if make_contact_sheet else "",
        "qa_report": str(report_path),
        "images_dir": str(images_root),
        "yolo_installed": yolo_startup.installed,
        "yolo_model_loaded": YoloProposalService.startup_status().model_loaded,
    }
