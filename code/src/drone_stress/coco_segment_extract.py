"""COCO instance segmentation -> RGBA distractor asset extraction."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from drone_stress.extract import safe_save_rgba_image
from drone_stress.sam2_extract import BORDER_TOUCH_FRACTION, expand_bbox_xywh

EXTRACTION_METHOD = "coco_segmentation_mask"

COCO_CATEGORY_ALIASES: dict[str, str] = {
    "airplane": "airplane",
    "aeroplane": "airplane",
    "bird": "bird",
    "kite": "kite",
}

METADATA_COLUMNS = [
    "asset_id",
    "source_dataset",
    "class_name",
    "original_image_path",
    "output_path",
    "image_id",
    "annotation_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_format",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "extraction_method",
    "has_alpha",
    "mask_area_px",
    "mask_area_ratio_in_crop",
    "touches_border_count",
    "mask_quality_label",
    "needs_manual_review",
    "qa_reasons",
    "extraction_failed",
    "extraction_error",
]

ACCEPT_MIN_RATIO = 0.03
ACCEPT_MAX_RATIO = 0.85
ACCEPT_MAX_TOUCHES = 1
REJECT_MAX_RATIO = 0.95
MIN_MASK_AREA_PX = 4


@dataclass
class ExtractionStats:
    annotations_scanned: int = 0
    exported: int = 0
    extraction_failed: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    by_class: dict[str, int] = field(default_factory=dict)
    qa_by_class: dict[str, dict[str, int]] = field(default_factory=dict)
    qa_totals: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def add_qa(self, class_name: str, label: str) -> None:
        self.qa_totals[label] = self.qa_totals.get(label, 0) + 1
        bucket = self.qa_by_class.setdefault(class_name, {})
        bucket[label] = bucket.get(label, 0) + 1


@dataclass
class SegmentationDecodeResult:
    mask: np.ndarray | None
    error: str


def normalize_class_name(name: str) -> str:
    key = name.strip().lower()
    return COCO_CATEGORY_ALIASES.get(key, key)


def resolve_categories(
    categories: list[dict],
    requested_names: list[str],
) -> dict[int, dict[str, str]]:
    wanted = {normalize_class_name(n) for n in requested_names}
    by_normalized: dict[str, list[dict]] = {}
    for cat in categories:
        norm = normalize_class_name(str(cat.get("name", "")))
        by_normalized.setdefault(norm, []).append(cat)

    missing = wanted - set(by_normalized)
    if missing:
        available = sorted({normalize_class_name(c["name"]) for c in categories})
        raise ValueError(
            f"Unknown categories: {sorted(missing)}. Available: {available}"
        )

    selected: dict[int, dict[str, str]] = {}
    for norm in sorted(wanted):
        cats = by_normalized[norm]
        cat = next((c for c in cats if normalize_class_name(c["name"]) == norm), cats[0])
        selected[int(cat["id"])] = {
            "class_name": norm,
            "supercategory": str(cat.get("supercategory", "")),
        }
    return selected


def _pycocotools_available() -> bool:
    try:
        import pycocotools.mask  # noqa: F401

        return True
    except ImportError:
        return False


def _decode_polygon_mask_cv2(polygons: list, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        if not poly or len(poly) < 6:
            continue
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def decode_coco_segmentation(
    segmentation: object,
    height: int,
    width: int,
) -> SegmentationDecodeResult:
    if segmentation is None:
        return SegmentationDecodeResult(None, "empty_segmentation")
    if isinstance(segmentation, list):
        if not segmentation:
            return SegmentationDecodeResult(None, "empty_segmentation")
        if isinstance(segmentation[0], list):
            if _pycocotools_available():
                from pycocotools import mask as mask_utils

                rles = mask_utils.frPyObjects(segmentation, height, width)
                rle = mask_utils.merge(rles)
                decoded = mask_utils.decode(rle)
                return SegmentationDecodeResult(decoded.astype(bool), "")
            return SegmentationDecodeResult(
                _decode_polygon_mask_cv2(segmentation, height, width), ""
            )
        if all(isinstance(v, (int, float)) for v in segmentation):
            return SegmentationDecodeResult(
                _decode_polygon_mask_cv2([segmentation], height, width), ""
            )
        return SegmentationDecodeResult(None, "unsupported_polygon_format")

    if isinstance(segmentation, dict):
        if not _pycocotools_available():
            return SegmentationDecodeResult(None, "skip_rle_no_pycocotools")
        from pycocotools import mask as mask_utils

        decoded = mask_utils.decode(segmentation)
        return SegmentationDecodeResult(decoded.astype(bool), "")

    return SegmentationDecodeResult(None, "unsupported_segmentation_type")


def _edge_touches_mask(mask: np.ndarray, area: int) -> int:
    if area <= 0:
        return 0
    threshold = BORDER_TOUCH_FRACTION * area
    touches = (
        bool(mask[0, :].sum() >= threshold),
        bool(mask[-1, :].sum() >= threshold),
        bool(mask[:, 0].sum() >= threshold),
        bool(mask[:, -1].sum() >= threshold),
    )
    return sum(touches)


def evaluate_coco_segmentation_qa(mask_crop: np.ndarray) -> tuple[str, list[str], int, float, int]:
    """Return label, reasons, area_px, area_ratio, touches_border_count."""
    mask = mask_crop.astype(bool)
    crop_area = mask.size
    area = int(mask.sum())
    ratio = float(area / crop_area) if crop_area else 0.0
    touches = _edge_touches_mask(mask, area)
    reasons: list[str] = []

    if crop_area == 0:
        return "reject", ["invalid_crop"], area, ratio, touches
    if area == 0:
        return "reject", ["empty_mask"], area, ratio, touches
    if area < MIN_MASK_AREA_PX:
        reasons.append("mask_too_small")
    if ratio > REJECT_MAX_RATIO:
        reasons.append("likely_background_blob")
    elif ratio > ACCEPT_MAX_RATIO:
        reasons.append("mask_too_large_review")
    elif ratio < ACCEPT_MIN_RATIO:
        reasons.append("mask_small_review")

    if touches >= 3:
        reasons.append("mask_touches_many_borders")
    elif touches == 2:
        reasons.append("mask_touches_two_borders")

    reject_reasons = {"empty_mask", "invalid_crop", "likely_background_blob", "mask_touches_many_borders"}
    if any(r in reject_reasons for r in reasons):
        return "reject", reasons, area, ratio, touches

    if (
        ACCEPT_MIN_RATIO <= ratio <= ACCEPT_MAX_RATIO
        and touches <= ACCEPT_MAX_TOUCHES
        and not reasons
    ):
        return "accept", [], area, ratio, touches

    return "review", reasons, area, ratio, touches


def segmentation_to_rgba(
    rgb: np.ndarray,
    full_mask: np.ndarray,
    crop_xyxy: tuple[int, int, int, int],
) -> np.ndarray | None:
    x1, y1, x2, y2 = crop_xyxy
    if x2 <= x1 or y2 <= y1:
        return None
    crop_rgb = rgb[y1:y2, x1:x2]
    crop_mask = full_mask[y1:y2, x1:x2]
    if crop_rgb.size == 0:
        return None
    h, w = crop_rgb.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = crop_rgb
    rgba[:, :, 3] = (crop_mask.astype(np.uint8) * 255)
    return rgba


def _asset_id(class_name: str, image_id: int, ann_id: int, row_index: int) -> str:
    digest = hashlib.sha256(f"{image_id}|{ann_id}|{row_index}".encode()).hexdigest()[:10]
    safe = re.sub(r"[^\w\-.]+", "_", class_name)
    return f"{safe}_{row_index:05d}_{digest}"


def extract_coco_segmentation_assets(
    annotation_json: Path,
    image_root: Path,
    output_dir: Path,
    *,
    category_names: list[str],
    source_dataset: str,
    min_bbox_px: int,
    max_rows: int | None,
    expand_box_ratio: float,
    class_subdirs: bool,
) -> tuple[pd.DataFrame, ExtractionStats]:
    with annotation_json.open(encoding="utf-8") as f:
        coco = json.load(f)

    if not _pycocotools_available():
        print(
            "Note: pycocotools not installed — polygon masks use OpenCV fallback; "
            "RLE segmentations will be skipped. Install with:\n"
            "  pip install pycocotools"
        )

    selected = resolve_categories(coco.get("categories", []), category_names)
    selected_ids = set(selected)
    image_by_id = {int(img["id"]): img for img in coco.get("images", [])}

    stats = ExtractionStats()
    rows: list[dict] = []
    export_index = 0

    for ann in coco.get("annotations", []):
        if max_rows is not None and stats.exported >= max_rows:
            break

        cat_id = int(ann.get("category_id", -1))
        if cat_id not in selected_ids:
            continue

        stats.annotations_scanned += 1
        class_name = selected[cat_id]["class_name"]

        if int(ann.get("iscrowd", 0)) != 0:
            stats.skip("iscrowd")
            continue

        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            stats.skip("invalid_bbox")
            continue

        bx, by, bw, bh = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if bw <= 0 or bh <= 0:
            stats.skip("invalid_bbox")
            continue
        if max(bw, bh) < min_bbox_px:
            stats.skip("bbox_too_small")
            continue

        image_id = int(ann["image_id"])
        img_rec = image_by_id.get(image_id)
        if img_rec is None:
            stats.skip("image_missing")
            continue

        file_name = str(img_rec["file_name"])
        image_path = image_root / file_name
        if not image_path.is_file():
            stats.skip("image_missing")
            continue

        img_w = int(img_rec.get("width", 0))
        img_h = int(img_rec.get("height", 0))
        if img_w <= 0 or img_h <= 0:
            stats.skip("invalid_image_size")
            continue

        segm = ann.get("segmentation")
        decode = decode_coco_segmentation(segm, img_h, img_w)
        if decode.mask is None:
            stats.skip(decode.error or "mask_decode_failed")
            continue

        bbox_xywh = (
            int(round(bx)),
            int(round(by)),
            int(round(bw)),
            int(round(bh)),
        )
        crop_xywh = expand_bbox_xywh(bbox_xywh, expand_box_ratio, img_w, img_h)
        cx, cy, cw, ch = crop_xywh
        crop_x1, crop_y1 = cx, cy
        crop_x2, crop_y2 = cx + cw, cy + ch

        try:
            with Image.open(image_path) as img:
                rgb = np.array(img.convert("RGB"))
        except OSError as exc:
            stats.skip("image_read_error")
            rows.append(
                _failed_row(
                    class_name=class_name,
                    source_dataset=source_dataset,
                    image_path=image_path,
                    image_id=image_id,
                    ann_id=int(ann["id"]),
                    bbox_xywh=bbox_xywh,
                    crop_xyxy=(crop_x1, crop_y1, crop_x2, crop_y2),
                    export_index=export_index,
                    error=str(exc),
                )
            )
            stats.extraction_failed += 1
            export_index += 1
            continue

        rgba = segmentation_to_rgba(rgb, decode.mask, (crop_x1, crop_y1, crop_x2, crop_y2))
        if rgba is None:
            stats.skip("invalid_crop")
            continue

        mask_crop = decode.mask[crop_y1:crop_y2, crop_x1:crop_x2]
        label, reasons, area_px, area_ratio, touches = evaluate_coco_segmentation_qa(mask_crop)

        asset_id = _asset_id(class_name, image_id, int(ann["id"]), export_index)
        if class_subdirs:
            dest_path = output_dir / class_name / f"{asset_id}.png"
        else:
            dest_path = output_dir / f"{asset_id}.png"

        save = safe_save_rgba_image(rgba, dest_path)
        has_alpha = bool(np.any(rgba[:, :, 3] > 0) and np.any(rgba[:, :, 3] < 255))

        row = {
            "asset_id": asset_id,
            "source_dataset": source_dataset,
            "class_name": class_name,
            "original_image_path": str(image_path.resolve()),
            "output_path": str(dest_path.resolve()) if save.success else str(dest_path),
            "image_id": image_id,
            "annotation_id": int(ann["id"]),
            "bbox_x": bbox_xywh[0],
            "bbox_y": bbox_xywh[1],
            "bbox_w": bbox_xywh[2],
            "bbox_h": bbox_xywh[3],
            "bbox_format": "xywh",
            "crop_x1": crop_x1,
            "crop_y1": crop_y1,
            "crop_x2": crop_x2,
            "crop_y2": crop_y2,
            "extraction_method": EXTRACTION_METHOD,
            "has_alpha": has_alpha,
            "mask_area_px": area_px,
            "mask_area_ratio_in_crop": round(area_ratio, 6),
            "touches_border_count": touches,
            "mask_quality_label": label,
            "needs_manual_review": label in ("review", "reject"),
            "qa_reasons": ";".join(reasons),
            "extraction_failed": False,
            "extraction_error": "",
        }

        if not save.success:
            row["extraction_failed"] = True
            row["extraction_error"] = save.error
            row["mask_quality_label"] = "reject"
            row["needs_manual_review"] = True
            row["has_alpha"] = False
            reasons = list(reasons)
            reasons.append("save_failed")
            row["qa_reasons"] = ";".join(reasons)
            stats.extraction_failed += 1
        else:
            stats.exported += 1
            stats.by_class[class_name] = stats.by_class.get(class_name, 0) + 1

        stats.add_qa(class_name, row["mask_quality_label"])
        rows.append(row)
        export_index += 1

    if not rows:
        raise ValueError(
            "No assets extracted. Check categories, min-bbox-px, segmentation, and image paths."
        )

    return pd.DataFrame(rows, columns=METADATA_COLUMNS), stats


def _failed_row(
    *,
    class_name: str,
    source_dataset: str,
    image_path: Path,
    image_id: int,
    ann_id: int,
    bbox_xywh: tuple[int, int, int, int],
    crop_xyxy: tuple[int, int, int, int],
    export_index: int,
    error: str,
) -> dict:
    asset_id = _asset_id(class_name, image_id, ann_id, export_index)
    return {
        "asset_id": asset_id,
        "source_dataset": source_dataset,
        "class_name": class_name,
        "original_image_path": str(image_path.resolve()),
        "output_path": "",
        "image_id": image_id,
        "annotation_id": ann_id,
        "bbox_x": bbox_xywh[0],
        "bbox_y": bbox_xywh[1],
        "bbox_w": bbox_xywh[2],
        "bbox_h": bbox_xywh[3],
        "bbox_format": "xywh",
        "crop_x1": crop_xyxy[0],
        "crop_y1": crop_xyxy[1],
        "crop_x2": crop_xyxy[2],
        "crop_y2": crop_xyxy[3],
        "extraction_method": EXTRACTION_METHOD,
        "has_alpha": False,
        "mask_area_px": 0,
        "mask_area_ratio_in_crop": 0.0,
        "touches_border_count": 0,
        "mask_quality_label": "reject",
        "needs_manual_review": True,
        "qa_reasons": "extraction_failed",
        "extraction_failed": True,
        "extraction_error": error,
    }


def write_extraction_report(
    stats: ExtractionStats,
    report_path: Path,
    *,
    annotation_json: Path,
    image_root: Path,
    output_dir: Path,
    category_names: list[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "COCO segmentation asset extraction report",
        f"annotation_json: {annotation_json.resolve()}",
        f"image_root: {image_root.resolve()}",
        f"output_dir: {output_dir.resolve()}",
        f"categories: {', '.join(category_names)}",
        "",
        f"annotations_scanned: {stats.annotations_scanned}",
        f"exported (save ok): {stats.exported}",
        f"extraction_failed: {stats.extraction_failed}",
        "",
        "exported_by_class:",
    ]
    for cls in sorted(stats.by_class):
        lines.append(f"  {cls}: {stats.by_class[cls]}")
    lines.append("\nskipped_by_reason:")
    for reason, count in sorted(stats.skipped.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count}")
    lines.append("\nQA totals:")
    for label in ("accept", "review", "reject"):
        lines.append(f"  {label}: {stats.qa_totals.get(label, 0)}")
    lines.append("\nQA by class:")
    for cls in sorted(stats.qa_by_class):
        lines.append(f"  {cls}:")
        for label, count in sorted(stats.qa_by_class[cls].items()):
            lines.append(f"    {label}: {count}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qa_report(df: pd.DataFrame, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ok = df[~df["extraction_failed"].astype(bool)]
    ratios = ok["mask_area_ratio_in_crop"].astype(float).tolist()
    if ratios:
        ratio_line = (
            f"mask_area_ratio_in_crop min/median/max: "
            f"{min(ratios):.4f} / {statistics.median(ratios):.4f} / {max(ratios):.4f}"
        )
    else:
        ratio_line = "mask_area_ratio_in_crop: (no successful exports)"

    lines = [
        "COCO segmentation asset QA report",
        f"total_rows: {len(df)}",
        f"extraction_failed: {int(df['extraction_failed'].astype(bool).sum())}",
        ratio_line,
        "",
        "By mask_quality_label:",
    ]
    for label, count in df["mask_quality_label"].value_counts().sort_index().items():
        lines.append(f"  {label}: {count}")
    lines.append("\nBy class_name:")
    for cls, grp in df.groupby("class_name"):
        lines.append(f"  {cls}: {len(grp)}")
        for label, count in grp["mask_quality_label"].value_counts().sort_index().items():
            lines.append(f"    {label}: {count}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
