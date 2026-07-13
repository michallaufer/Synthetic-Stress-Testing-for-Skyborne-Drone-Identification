"""Foreground asset extraction: folder baseline, annotation bbox crops, SAM2 masks.

Extraction stages (in order of quality):
  1. threshold_near_white / rgba_convert_only — toy folder mode only
  2. bbox_crop — dataset annotations -> rectangular RGBA crops (bridge / fallback)
  3. sam2_mask — SAM2 box-prompted segmentation masks (annotate --use-sam2)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from drone_stress.assets import IMAGE_EXTENSIONS, list_images, list_images_direct

EXTRACTION_METHOD_THRESHOLD = "threshold_near_white"
EXTRACTION_METHOD_BBOX_CROP = "bbox_crop"
EXTRACTION_METHOD_BBOX_CROP_FALLBACK = "bbox_crop_fallback"
EXTRACTION_METHOD_SAM2 = "sam2_mask"
EXTRACTION_METHOD_SAM2_FAILED_SAVE = "sam2_mask_failed_save"
EXTRACTION_METHOD_BBOX_CROP_FAILED_SAVE = "bbox_crop_failed_save"

WARN_BANNER = """
================================================================================
TEMPORARY BASELINE EXTRACTION - NOT PRODUCTION QUALITY
================================================================================
Folder mode uses simple near-white / near-uniform background thresholding only.
Annotation mode uses rectangular bbox crops (extraction_method=bbox_crop).
Both need SAM/SAM2 mask refinement (extraction_method=sam2_mask) before training.
Review extracted PNGs before using them in synthetic generation.
================================================================================
"""

ANNOTATION_MODE_BANNER = """
================================================================================
ANNOTATION BBOX CROP EXTRACTION (BRIDGE MODE)
================================================================================
Crops are axis-aligned rectangles from annotation bboxes - not segmentation masks.
needs_sam2_refinement=True on all rows until SAM/SAM2 mask extraction is run.
================================================================================
"""


ANNOTATION_SAM2_BANNER = """
================================================================================
ANNOTATION SAM2 MASK EXTRACTION
================================================================================
SAM2 box prompts produce alpha masks (extraction_method=sam2_mask).
Each asset gets mask_quality_label: accept / review / reject.
Rows with SAM2 failure fall back to bbox_crop_fallback and are flagged for review.
Review SAM2 contact sheets (accept/review/reject) before compositing.
================================================================================
"""


@dataclass
class Sam2ExtractConfig:
    model_size: str = "tiny"
    checkpoint: Path | None = None
    device: str = "auto"
    expand_box_ratio: float = 0.10
    max_rows: int | None = None


@dataclass
class ExtractedAsset:
    asset_id: str
    source_image: str
    asset_type: str
    asset_class: str
    source_bbox: str
    output_path: str
    width: int
    height: int
    extraction_method: str
    needs_sam2_refinement: bool
    source_dataset: str = ""
    has_alpha: bool = False
    sam2_used: bool = False
    sam2_model_size: str = ""
    sam2_box_prompt_xyxy: str = ""
    mask_quality_label: str = ""
    mask_quality_score: float = 0.0
    mask_review_reasons: str = ""
    mask_area_px: int = 0
    mask_area_ratio_in_crop: float = 0.0
    mask_bbox_x: int = 0
    mask_bbox_y: int = 0
    mask_bbox_w: int = 0
    mask_bbox_h: int = 0
    mask_bbox_area_ratio_in_crop: float = 0.0
    mask_touches_top: bool = False
    mask_touches_bottom: bool = False
    mask_touches_left: bool = False
    mask_touches_right: bool = False
    mask_num_touched_borders: int = 0
    needs_manual_review: bool = False
    extraction_failed: bool = False
    extraction_error: str = ""

    @property
    def source_path(self) -> str:
        """Backward-compatible alias for source_image."""
        return self.source_image


def discover_class_images(input_dir: Path) -> list[tuple[str, Path]]:
    """
    Discover (asset_class, image_path) jobs under input_dir.

    - Class subfolders: input_dir/bird/*.jpg -> class bird
    - Flat folder: input_dir/*.jpg -> class 'drone' if folder name contains 'drone',
      else folder name stem
    """
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    jobs: list[tuple[str, Path]] = []
    subdirs = sorted(p for p in input_dir.iterdir() if p.is_dir())

    for subdir in subdirs:
        for path in list_images(subdir):
            jobs.append((subdir.name, path))

    for path in list_images_direct(input_dir):
        if "drone" in input_dir.name.lower():
            asset_class = "drone"
        else:
            asset_class = input_dir.name
        jobs.append((asset_class, path))

    return jobs


def _remove_near_white_background(rgba: np.ndarray, threshold: int) -> np.ndarray:
    """Set alpha=0 where RGB is near white (temporary baseline)."""
    out = rgba.copy()
    rgb = out[:, :, :3].astype(np.int16)
    near_white = (
        (rgb[:, :, 0] >= threshold)
        & (rgb[:, :, 1] >= threshold)
        & (rgb[:, :, 2] >= threshold)
    )
    out[:, :, 3] = np.where(near_white, 0, out[:, :, 3])
    return out


def _remove_near_uniform_background(rgba: np.ndarray, tolerance: int = 18) -> np.ndarray:
    """Set alpha=0 where pixels are close to the mean corner color (temporary baseline)."""
    out = rgba.copy()
    h, w = out.shape[:2]
    corners = np.array(
        [
            out[0, 0, :3],
            out[0, w - 1, :3],
            out[h - 1, 0, :3],
            out[h - 1, w - 1, :3],
        ],
        dtype=np.int16,
    )
    ref = corners.mean(axis=0)
    rgb = out[:, :, :3].astype(np.int16)
    dist = np.abs(rgb - ref).max(axis=2)
    uniform_bg = dist <= tolerance
    out[:, :, 3] = np.where(uniform_bg, 0, out[:, :, 3])
    return out


def extract_rgba_asset(
    source_path: Path,
    remove_background: bool = True,
    white_threshold: int = 240,
    uniform_tolerance: int = 18,
) -> tuple[np.ndarray, str]:
    """
    Load image, convert to RGBA, optionally apply baseline background removal.

    Returns (rgba array, extraction_method label).
    """
    with Image.open(source_path) as img:
        rgba = np.array(img.convert("RGBA"))

    if not remove_background:
        return rgba, "rgba_convert_only"

    rgba = _remove_near_white_background(rgba, white_threshold)
    rgba = _remove_near_uniform_background(rgba, uniform_tolerance)
    return rgba, EXTRACTION_METHOD_THRESHOLD


def _asset_id(asset_class: str, source_path: Path) -> str:
    stem = source_path.stem
    prefix = f"{asset_class}_"
    if stem == asset_class or stem.startswith(prefix):
        return stem
    return f"{asset_class}_{stem}"


def _annotation_asset_id(
    asset_class: str,
    source_path: Path,
    bbox_xywh: tuple[int, int, int, int],
    row_index: int,
) -> str:
    """Compact, deterministic id — full source/bbox details live in the manifest CSV."""
    stem = source_path.stem
    x, y, w, h = bbox_xywh
    digest = hashlib.sha256(f"{stem}|{x}|{y}|{w}|{h}|{row_index}".encode()).hexdigest()[:12]
    safe_class = re.sub(r"[^\w\-.]+", "_", asset_class)
    return f"{safe_class}_{row_index:05d}_{digest}"


def _has_real_alpha(rgba: np.ndarray) -> bool:
    alpha = rgba[:, :, 3]
    return bool(np.any(alpha < 255) and np.any(alpha > 0))


@dataclass
class SaveResult:
    success: bool
    error: str


def safe_save_rgba_image(rgba: np.ndarray, dest_path: Path) -> SaveResult:
    """Save RGBA PNG; ensure parent exists; verify file exists after save."""
    dest_path = Path(dest_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(rgba, mode="RGBA").save(dest_path)
        if not dest_path.is_file():
            return SaveResult(
                success=False,
                error=f"save reported ok but file missing: {dest_path}",
            )
        return SaveResult(success=True, error="")
    except OSError as exc:
        return SaveResult(success=False, error=str(exc))


def resolve_asset_dest_path(
    output_dir: Path,
    *,
    asset_class: str,
    asset_id: str,
    use_class_subdirs: bool,
    short_output_root: Path | None = None,
) -> Path:
    """Resolve PNG destination; optional short root avoids Windows MAX_PATH issues."""
    if use_class_subdirs:
        rel = Path(asset_class) / f"{asset_id}.png"
    else:
        rel = Path(f"{asset_id}.png")
    base = short_output_root.resolve() if short_output_root is not None else output_dir.resolve()
    return base / rel


def _record_save_failure(record: ExtractedAsset, dest_path: Path, error: str) -> ExtractedAsset:
    """Mark a successfully extracted asset as failed when PNG save fails."""
    if record.extraction_method == EXTRACTION_METHOD_SAM2:
        failed_method = EXTRACTION_METHOD_SAM2_FAILED_SAVE
    else:
        failed_method = EXTRACTION_METHOD_BBOX_CROP_FAILED_SAVE
    record.extraction_failed = True
    record.extraction_error = error
    record.output_path = str(dest_path.resolve())
    record.has_alpha = False
    record.extraction_method = failed_method
    record.mask_quality_label = "reject"
    record.needs_manual_review = True
    if not record.mask_review_reasons:
        record.mask_review_reasons = "save_failed"
    elif "save_failed" not in record.mask_review_reasons:
        record.mask_review_reasons = f"{record.mask_review_reasons};save_failed"
    return record


def extraction_save_summary(records: list[ExtractedAsset]) -> dict[str, int | dict[str, int]]:
    """Counts for console / report: extracted ok vs failed saves by error type."""
    extracted = sum(1 for r in records if not r.extraction_failed)
    failed = sum(1 for r in records if r.extraction_failed)
    errors: dict[str, int] = {}
    for r in records:
        if r.extraction_failed and r.extraction_error:
            errors[r.extraction_error] = errors.get(r.extraction_error, 0) + 1
    return {
        "extracted": extracted,
        "extraction_failed": failed,
        "errors_by_type": errors,
    }


def _input_has_class_subfolders(input_dir: Path) -> bool:
    """True if input_dir contains subfolders with at least one image."""
    return any(
        list_images(subdir) for subdir in input_dir.iterdir() if subdir.is_dir()
    )


def resolve_use_class_subdirs(
    asset_type: str,
    input_dir: Path | None = None,
    flat_output: bool = False,
    class_subdirs: bool = False,
) -> bool:
    """
    Decide output layout.

    - --flat-output / class_subdirs=False for drones: flat under output-dir.
    - drone: flat by default; class subfolders if --class-subdirs or input has class folders.
    - distractor: preserve class subfolders by default.
    """
    if flat_output:
        return False
    if class_subdirs:
        return True
    if asset_type == "distractor":
        return True
    if input_dir is not None:
        return _input_has_class_subfolders(input_dir)
    return False


def parse_bbox_columns(
    row: pd.Series,
    bbox_columns: list[str],
    bbox_format: str,
) -> tuple[int, int, int, int]:
    """Parse bbox from a CSV row; return clamp-ready (x, y, w, h) in pixels."""
    if len(bbox_columns) != 4:
        raise ValueError(f"bbox_columns must list exactly 4 names, got {bbox_columns}")
    try:
        vals = [float(row[c]) for c in bbox_columns]
    except KeyError as exc:
        raise KeyError(f"Missing bbox column in annotations CSV: {exc}") from exc

    if bbox_format == "xywh":
        x, y, w, h = vals
    elif bbox_format == "xyxy":
        x1, y1, x2, y2 = vals
        x, y = x1, y1
        w, h = x2 - x1, y2 - y1
    else:
        raise ValueError(f"bbox_format must be 'xywh' or 'xyxy', got {bbox_format!r}")

    return int(round(x)), int(round(y)), int(round(max(1, w))), int(round(max(1, h)))


def clamp_bbox_xywh(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Clamp bbox to image bounds; return None if no visible area remains."""
    x, y, w, h = bbox
    if image_width <= 0 or image_height <= 0:
        return None
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    w = max(1, min(w, image_width - x))
    h = max(1, min(h, image_height - y))
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def crop_bbox_to_rgba(rgb: np.ndarray, bbox_xywh: tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop bbox region and return RGBA with opaque alpha (rectangular mask).

    Used as fallback when SAM2 is disabled or fails.
    """
    x, y, w, h = bbox_xywh
    crop_rgb = rgb[y : y + h, x : x + w]
    if crop_rgb.size == 0:
        raise ValueError(f"Empty crop for bbox {bbox_xywh}")
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = crop_rgb
    rgba[:, :, 3] = 255
    return rgba


def _checkerboard_rgb(height: int, width: int, cell: int = 12) -> np.ndarray:
    yy, xx = np.indices((height, width))
    board = ((xx // cell + yy // cell) % 2).astype(np.uint8)
    light = np.array([240, 240, 240], dtype=np.uint8)
    dark = np.array([192, 192, 192], dtype=np.uint8)
    return np.where(board[..., None], dark, light)


def rgba_on_checkerboard(rgba: np.ndarray, cell: int = 12) -> np.ndarray:
    """Composite RGBA asset over a checkerboard for alpha QA."""
    h, w = rgba.shape[:2]
    bg = _checkerboard_rgb(h, w, cell=cell)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    comp = rgb * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return comp.clip(0, 255).astype(np.uint8)


def _apply_mask_qa_to_asset(asset: ExtractedAsset, qa) -> ExtractedAsset:
    """Copy mask QA fields from MaskQAResult onto an ExtractedAsset."""
    asset.mask_quality_label = qa.mask_quality_label
    asset.mask_quality_score = qa.mask_quality_score
    asset.mask_review_reasons = qa.mask_review_reasons
    asset.mask_area_px = qa.mask_area_px
    asset.mask_area_ratio_in_crop = qa.mask_area_ratio_in_crop
    asset.mask_bbox_x = qa.mask_bbox_x
    asset.mask_bbox_y = qa.mask_bbox_y
    asset.mask_bbox_w = qa.mask_bbox_w
    asset.mask_bbox_h = qa.mask_bbox_h
    asset.mask_bbox_area_ratio_in_crop = qa.mask_bbox_area_ratio_in_crop
    asset.mask_touches_top = qa.mask_touches_top
    asset.mask_touches_bottom = qa.mask_touches_bottom
    asset.mask_touches_left = qa.mask_touches_left
    asset.mask_touches_right = qa.mask_touches_right
    asset.mask_num_touched_borders = qa.mask_num_touched_borders
    asset.needs_manual_review = qa.needs_manual_review
    return asset


def _sam2_tile_title(record: ExtractedAsset) -> str:
    label = record.mask_quality_label or "n/a"
    ratio = record.mask_area_ratio_in_crop
    borders = record.mask_num_touched_borders
    reasons = record.mask_review_reasons or "-"
    if len(reasons) > 48:
        reasons = reasons[:45] + "..."
    return "\n".join(
        [
            f"{record.asset_id} | {record.asset_class} | {label}",
            f"{record.extraction_method} | sam2={record.sam2_used}",
            f"area_ratio={ratio:.3f} | borders={borders}",
            reasons,
        ]
    )


def _asset_tile_title(record: ExtractedAsset) -> str:
    if record.sam2_used or record.extraction_method in (
        EXTRACTION_METHOD_SAM2,
        EXTRACTION_METHOD_BBOX_CROP_FALLBACK,
    ):
        return _sam2_tile_title(record)
    return "\n".join(
        [
            f"{record.asset_class} | {record.extraction_method}",
            f"sam2={record.sam2_used} review={record.needs_manual_review}",
        ]
    )


def build_extraction_contact_sheet(
    records: list[ExtractedAsset],
    output_path: Path,
    num_samples: int = 24,
    seed: int = 42,
    cols: int = 6,
    title_fn=None,
) -> Path | None:
    """Save a contact sheet of extracted assets composited over checkerboard."""
    if not records:
        return None
    if title_fn is None:
        title_fn = _asset_tile_title
    return _save_contact_sheet(
        records,
        output_path,
        num_samples=num_samples,
        seed=seed,
        cols=cols,
        title_fn=title_fn,
    )


def build_sam2_qa_contact_sheets(
    records: list[ExtractedAsset],
    output_dir: Path,
    *,
    num_samples: int = 24,
    seed: int = 42,
    cols: int = 6,
) -> dict[str, Path | None]:
    """
    Write separate SAM2 QA contact sheets grouped by mask_quality_label.

    Returns paths for accept / review / reject sheets (None if empty).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path | None] = {}
    for label, filename in (
        ("accept", "sam2_assets_accept.png"),
        ("review", "sam2_assets_review.png"),
        ("reject", "sam2_assets_reject.png"),
    ):
        subset = [r for r in records if r.mask_quality_label == label]
        out_path = output_dir / filename
        paths[label] = _save_contact_sheet(
            subset,
            out_path,
            num_samples=num_samples,
            seed=seed,
            cols=cols,
            title_fn=_sam2_tile_title,
        )
    return paths


def _save_contact_sheet(
    records: list[ExtractedAsset],
    output_path: Path,
    *,
    num_samples: int,
    seed: int,
    cols: int,
    title_fn,
) -> Path | None:
    if not records:
        return None

    sample = records if len(records) <= num_samples else _sample_records(records, num_samples, seed)
    n = len(sample)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.0))
    axes_flat = _flatten_axes(axes, rows, cols)

    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i >= n:
            ax.set_visible(False)
            continue
        record = sample[i]
        path = Path(record.output_path)
        if not path.is_file():
            ax.set_title(f"missing\n{record.asset_id}", fontsize=7)
            continue
        with Image.open(path) as img:
            rgba = np.array(img.convert("RGBA"))
        vis = rgba_on_checkerboard(rgba)
        ax.imshow(vis)
        ax.set_title(title_fn(record), fontsize=6, loc="left", pad=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _sample_records(records: list[ExtractedAsset], num_samples: int, seed: int) -> list[ExtractedAsset]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=num_samples, replace=False)
    return [records[i] for i in sorted(idx)]


def _flatten_axes(axes, rows: int, cols: int) -> list:
    if rows == 1 and cols == 1:
        return [axes]
    if rows == 1:
        return list(axes)
    if cols == 1:
        return list(axes)
    return [ax for row in axes for ax in row]


def load_annotations_csv(
    annotations_path: Path,
    class_column: str,
    filename_column: str,
    bbox_columns: list[str],
    bbox_format: str,
    source_dataset_column: str | None = None,
) -> pd.DataFrame:
    """Load and validate annotation CSV."""
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    df = pd.read_csv(annotations_path)
    required = {class_column, filename_column, *bbox_columns}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Annotations CSV missing columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    if source_dataset_column and source_dataset_column not in df.columns:
        raise ValueError(
            f"source_dataset_column {source_dataset_column!r} not in CSV columns"
        )
    if bbox_format not in ("xywh", "xyxy"):
        raise ValueError(f"bbox_format must be 'xywh' or 'xyxy', got {bbox_format!r}")
    return df


def extract_assets_from_annotations(
    annotations_path: Path,
    image_root: Path,
    output_dir: Path,
    asset_type: str,
    class_column: str = "class_name",
    filename_column: str = "filename",
    bbox_format: str = "xywh",
    bbox_columns: list[str] | None = None,
    source_dataset: str = "",
    source_dataset_column: str | None = None,
    use_class_subdirs: bool | None = None,
    flat_output: bool = False,
    class_subdirs: bool = False,
    sam2_config: Sam2ExtractConfig | None = None,
    sam2_predictor=None,
    short_output_root: Path | None = None,
) -> list[ExtractedAsset]:
    """
    Extract RGBA crops from annotation bboxes.

    Without SAM2: rectangular bbox crops (extraction_method=bbox_crop).
    With SAM2: box-prompted masks (extraction_method=sam2_mask), bbox fallback on failure.
    """
    if bbox_columns is None:
        bbox_columns = ["x", "y", "w", "h"] if bbox_format == "xywh" else ["x1", "y1", "x2", "y2"]

    image_root = image_root.resolve()
    output_dir = output_dir.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    df = load_annotations_csv(
        annotations_path,
        class_column,
        filename_column,
        bbox_columns,
        bbox_format,
        source_dataset_column,
    )
    if sam2_config and sam2_config.max_rows is not None:
        df = df.head(sam2_config.max_rows)

    if use_class_subdirs is None:
        use_class_subdirs = resolve_use_class_subdirs(
            asset_type,
            flat_output=flat_output,
            class_subdirs=class_subdirs,
        )

    use_sam2 = sam2_config is not None
    if use_sam2 and sam2_predictor is None:
        from drone_stress.sam2_extract import Sam2BoxPredictor, ensure_sam2_dependencies

        ensure_sam2_dependencies()
        sam2_predictor = Sam2BoxPredictor(
            model_size=sam2_config.model_size,  # type: ignore[arg-type]
            checkpoint=sam2_config.checkpoint,
            device=sam2_config.device,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if short_output_root is not None:
        short_output_root = short_output_root.resolve()
        short_output_root.mkdir(parents=True, exist_ok=True)

    records: list[ExtractedAsset] = []
    seen_ids: set[str] = set()

    for row_index, row in df.iterrows():
        filename = str(row[filename_column]).strip()
        asset_class = str(row[class_column]).strip()
        if not filename or not asset_class:
            continue

        image_path = image_root / filename
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found for row {row_index}: {image_path} "
                f"(filename column={filename_column!r})"
            )

        with Image.open(image_path) as img:
            rgb = np.array(img.convert("RGB"))
        img_h, img_w = rgb.shape[:2]

        bbox_raw = parse_bbox_columns(row, bbox_columns, bbox_format)
        bbox = clamp_bbox_xywh(bbox_raw, img_w, img_h)
        if bbox is None:
            continue

        asset_id = _annotation_asset_id(asset_class, image_path, bbox, int(row_index))
        if asset_id in seen_ids:
            asset_id = f"{asset_id}_{row_index:05d}"
        seen_ids.add(asset_id)

        dest_path = resolve_asset_dest_path(
            output_dir,
            asset_class=asset_class,
            asset_id=asset_id,
            use_class_subdirs=use_class_subdirs,
            short_output_root=short_output_root,
        )

        row_dataset = source_dataset
        if source_dataset_column:
            val = row.get(source_dataset_column)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                row_dataset = str(val)

        record, rgba = _extract_annotation_row(
            rgb=rgb,
            bbox=bbox,
            asset_id=asset_id,
            asset_type=asset_type,
            asset_class=asset_class,
            image_path=image_path,
            dest_path=dest_path,
            row_dataset=row_dataset,
            sam2_config=sam2_config,
            sam2_predictor=sam2_predictor,
        )
        save_result = safe_save_rgba_image(rgba, dest_path)
        if save_result.success:
            record.output_path = str(dest_path.resolve())
            records.append(record)
        else:
            records.append(_record_save_failure(record, dest_path, save_result.error))

    if not records:
        raise ValueError(
            f"No assets extracted from {annotations_path}. "
            "Check bbox columns, image paths, and bbox values."
        )

    return records


def _extract_annotation_row(
    *,
    rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    asset_id: str,
    asset_type: str,
    asset_class: str,
    image_path: Path,
    dest_path: Path,
    row_dataset: str,
    sam2_config: Sam2ExtractConfig | None,
    sam2_predictor,
) -> tuple[ExtractedAsset, np.ndarray]:
    """Extract one annotation row; returns (metadata record, RGBA array)."""
    x, y, w, h = bbox
    img_h, img_w = rgb.shape[:2]

    if sam2_config is None:
        rgba = crop_bbox_to_rgba(rgb, bbox)
        asset = ExtractedAsset(
            asset_id=asset_id,
            source_image=str(image_path.resolve()),
            asset_type=asset_type,
            asset_class=asset_class,
            source_bbox=json.dumps(list(bbox)),
            output_path=str(dest_path.resolve()),
            width=w,
            height=h,
            extraction_method=EXTRACTION_METHOD_BBOX_CROP,
            needs_sam2_refinement=True,
            source_dataset=row_dataset,
            has_alpha=True,
            sam2_used=False,
        )
        return asset, rgba

    from drone_stress.sam2_extract import (
        evaluate_mask_qa,
        expanded_prompt_xyxy,
        fallback_mask_qa,
        mask_crop_to_rgba,
    )

    prompt_xyxy = expanded_prompt_xyxy(bbox, sam2_config.expand_box_ratio, img_w, img_h)
    image_key = str(image_path.resolve())

    try:
        full_mask = sam2_predictor.predict_mask(rgb, prompt_xyxy, image_key=image_key)
        mask_crop = full_mask[y : y + h, x : x + w]
        crop_rgb = rgb[y : y + h, x : x + w]
        rgba = mask_crop_to_rgba(crop_rgb, mask_crop)
        qa = evaluate_mask_qa(mask_crop, sam2_used=True)

        asset = ExtractedAsset(
            asset_id=asset_id,
            source_image=str(image_path.resolve()),
            asset_type=asset_type,
            asset_class=asset_class,
            source_bbox=json.dumps(list(bbox)),
            output_path=str(dest_path.resolve()),
            width=w,
            height=h,
            extraction_method=EXTRACTION_METHOD_SAM2,
            needs_sam2_refinement=False,
            source_dataset=row_dataset,
            has_alpha=_has_real_alpha(rgba),
            sam2_used=True,
            sam2_model_size=sam2_config.model_size,
            sam2_box_prompt_xyxy=json.dumps(list(prompt_xyxy)),
        )
        _apply_mask_qa_to_asset(asset, qa)
        return asset, rgba
    except ImportError:
        raise
    except Exception as exc:
        rgba = crop_bbox_to_rgba(rgb, bbox)
        qa = fallback_mask_qa(
            reason=f"sam2_failed_bbox_fallback:{type(exc).__name__}"
        )
        asset = ExtractedAsset(
            asset_id=asset_id,
            source_image=str(image_path.resolve()),
            asset_type=asset_type,
            asset_class=asset_class,
            source_bbox=json.dumps(list(bbox)),
            output_path=str(dest_path.resolve()),
            width=w,
            height=h,
            extraction_method=EXTRACTION_METHOD_BBOX_CROP_FALLBACK,
            needs_sam2_refinement=True,
            source_dataset=row_dataset,
            has_alpha=True,
            sam2_used=False,
            sam2_model_size=sam2_config.model_size,
            sam2_box_prompt_xyxy=json.dumps(list(prompt_xyxy)),
        )
        _apply_mask_qa_to_asset(asset, qa)
        return asset, rgba


def extract_assets(
    input_dir: Path,
    output_dir: Path,
    remove_background: bool = True,
    white_threshold: int = 240,
    uniform_tolerance: int = 18,
    use_class_subdirs: bool | None = None,
    asset_type: str | None = None,
    short_output_root: Path | None = None,
) -> list[ExtractedAsset]:
    """
    Extract PNG assets from input_dir into output_dir (toy folder / threshold mode).

    If use_class_subdirs is True, writes output_dir/<class>/<asset_id>.png.
    If False, writes output_dir/<asset_id>.png.
    """
    jobs = discover_class_images(input_dir)
    if not jobs:
        raise FileNotFoundError(
            f"No images found under {input_dir}. "
            f"Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    if use_class_subdirs is None:
        if asset_type is not None:
            use_class_subdirs = resolve_use_class_subdirs(
                asset_type, input_dir, flat_output=False
            )
        else:
            classes = {cls for cls, _ in jobs}
            use_class_subdirs = len(classes) > 1 or any(
                (input_dir / cls).is_dir() for cls in classes
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    if short_output_root is not None:
        short_output_root = short_output_root.resolve()
        short_output_root.mkdir(parents=True, exist_ok=True)

    records: list[ExtractedAsset] = []
    resolved_type = asset_type or "unknown"

    for asset_class, source_path in jobs:
        asset_id = _asset_id(asset_class, source_path)
        dest_path = resolve_asset_dest_path(
            output_dir,
            asset_class=asset_class,
            asset_id=asset_id,
            use_class_subdirs=use_class_subdirs,
            short_output_root=short_output_root,
        )

        rgba, method = extract_rgba_asset(
            source_path,
            remove_background=remove_background,
            white_threshold=white_threshold,
            uniform_tolerance=uniform_tolerance,
        )

        h, w = rgba.shape[:2]
        record = ExtractedAsset(
            asset_id=asset_id,
            source_image=str(source_path.resolve()),
            asset_type=resolved_type,
            asset_class=asset_class,
            source_bbox="",
            output_path=str(dest_path.resolve()),
            width=int(w),
            height=int(h),
            extraction_method=method,
            needs_sam2_refinement=False,
            source_dataset="",
            has_alpha=_has_real_alpha(rgba),
        )
        save_result = safe_save_rgba_image(rgba, dest_path)
        if save_result.success:
            records.append(record)
        else:
            records.append(_record_save_failure(record, dest_path, save_result.error))

    return records


# ---------------------------------------------------------------------------
# SAM2 refinement helper (batch re-run on existing bbox_crop metadata)
# ---------------------------------------------------------------------------
#
# def refine_assets_with_sam2(
#     records: list[ExtractedAsset],
#     sam2_checkpoint: Path,
#     prompt_from_bbox: bool = True,
# ) -> list[ExtractedAsset]:
#     """Re-run SAM2 on assets that still have needs_sam2_refinement=True."""
#     ...


METADATA_CSV_COLUMNS = [
    "asset_id",
    "source_image",
    "source_dataset",
    "asset_type",
    "asset_class",
    "source_bbox",
    "output_path",
    "width",
    "height",
    "extraction_method",
    "needs_sam2_refinement",
    "has_alpha",
    "sam2_used",
    "sam2_model_size",
    "sam2_box_prompt_xyxy",
    "mask_quality_label",
    "mask_quality_score",
    "mask_review_reasons",
    "mask_area_px",
    "mask_area_ratio_in_crop",
    "mask_bbox_x",
    "mask_bbox_y",
    "mask_bbox_w",
    "mask_bbox_h",
    "mask_bbox_area_ratio_in_crop",
    "mask_touches_top",
    "mask_touches_bottom",
    "mask_touches_left",
    "mask_touches_right",
    "mask_num_touched_borders",
    "needs_manual_review",
    "extraction_failed",
    "extraction_error",
]


def write_extraction_qa_report(records: list[ExtractedAsset], report_path: Path) -> Path:
    """Write a short SAM2 extraction QA summary report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(records)

    def _count_by(field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in records:
            val = getattr(r, field, "") or "(empty)"
            counts[val] = counts.get(val, 0) + 1
        return dict(sorted(counts.items()))

    reason_counts: dict[str, int] = {}
    for r in records:
        for reason in (r.mask_review_reasons or "").split(";"):
            reason = reason.strip()
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    sam2_failures = sum(
        1 for r in records if "sam2_failed_bbox_fallback" in (r.mask_review_reasons or "")
    )
    bbox_fallbacks = sum(
        1 for r in records if r.extraction_method == EXTRACTION_METHOD_BBOX_CROP_FALLBACK
    )
    save_summary = extraction_save_summary(records)

    lines = [
        "SAM2 Extraction QA Report",
        "=========================",
        f"Total assets: {total}",
        f"Extracted (save ok): {save_summary['extracted']}",
        f"Extraction save failed: {save_summary['extraction_failed']}",
        "",
        "By mask_quality_label:",
    ]
    for label, count in _count_by("mask_quality_label").items():
        lines.append(f"  {label}: {count}")
    lines.extend(["", "By extraction_method:"])
    for method, count in _count_by("extraction_method").items():
        lines.append(f"  {method}: {count}")
    lines.extend(["", "By mask_review_reason:"])
    if reason_counts:
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"  {reason}: {count}")
    else:
        lines.append("  (none)")
    lines.extend(
        [
            "",
            f"SAM2 inference failures (bbox fallback): {sam2_failures}",
            f"Bbox fallbacks (extraction_method=bbox_crop_fallback): {bbox_fallbacks}",
        ]
    )
    errors = save_summary["errors_by_type"]
    if errors:
        lines.extend(["", "Save failures by error:"])
        for err, count in sorted(errors.items(), key=lambda x: -x[1]):
            lines.append(f"  [{count}] {err}")
    lines.extend(
        [
            "",
            "Review contact sheets before compositing:",
            "  outputs/contact_sheets/sam2_assets_accept.png",
            "  outputs/contact_sheets/sam2_assets_review.png",
            "  outputs/contact_sheets/sam2_assets_reject.png",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_asset_metadata_csv(records: list[ExtractedAsset], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows)
    for col in METADATA_CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[METADATA_CSV_COLUMNS]
    df.to_csv(csv_path, index=False)


def records_to_sam2_jobs(records: list[ExtractedAsset]) -> list[dict]:
    """
    Prepare job dicts for a future SAM2 refinement pass (placeholder API).

    Each job includes source_image, source_bbox, output_path, and asset_id.
    """
    jobs = []
    for r in records:
        if not r.needs_sam2_refinement:
            continue
        jobs.append(
            {
                "asset_id": r.asset_id,
                "source_image": r.source_image,
                "source_bbox": r.source_bbox,
                "output_path": r.output_path,
                "asset_class": r.asset_class,
                "target_extraction_method": EXTRACTION_METHOD_SAM2,
            }
        )
    return jobs
