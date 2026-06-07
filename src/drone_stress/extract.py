"""Foreground asset extraction: folder baseline, annotation bbox crops, SAM2 (planned).

Extraction stages (in order of quality):
  1. threshold_near_white / rgba_convert_only — toy folder mode only
  2. bbox_crop — dataset annotations -> rectangular RGBA crops (bridge)
  3. sam2_mask — planned: SAM/SAM2 refines bbox crops into proper alpha masks
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from drone_stress.assets import IMAGE_EXTENSIONS, list_images, list_images_direct

EXTRACTION_METHOD_THRESHOLD = "threshold_near_white"
EXTRACTION_METHOD_BBOX_CROP = "bbox_crop"
# Planned: replace bbox rectangles with segmentation masks from SAM 2.
EXTRACTION_METHOD_SAM2 = "sam2_mask"

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
    stem = source_path.stem
    x, y, w, h = bbox_xywh
    base = f"{asset_class}_{stem}_{x}_{y}_{w}_{h}"
    safe = re.sub(r"[^\w\-.]+", "_", base)
    return safe if safe else f"{asset_class}_{row_index:05d}"


def _has_real_alpha(rgba: np.ndarray) -> bool:
    alpha = rgba[:, :, 3]
    return bool(np.any(alpha < 255) and np.any(alpha > 0))


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

    Future SAM2 path will replace this rectangle with a segmentation mask
    (extraction_method=sam2_mask).
    """
    x, y, w, h = bbox_xywh
    crop_rgb = rgb[y : y + h, x : x + w]
    if crop_rgb.size == 0:
        raise ValueError(f"Empty crop for bbox {bbox_xywh}")
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = crop_rgb
    rgba[:, :, 3] = 255
    return rgba


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
) -> list[ExtractedAsset]:
    """
    Extract RGBA crops from annotation bboxes (bridge toward dataset-derived assets).

    Each row: load image, crop bbox, save PNG, record metadata with
    extraction_method=bbox_crop and needs_sam2_refinement=True.

    Planned refinement (not implemented yet):

        refine_assets_with_sam2(records)  # extraction_method -> sam2_mask
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

    if use_class_subdirs is None:
        use_class_subdirs = resolve_use_class_subdirs(
            asset_type,
            flat_output=flat_output,
            class_subdirs=class_subdirs,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
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

        if use_class_subdirs:
            dest_dir = output_dir / asset_class
        else:
            dest_dir = output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{asset_id}.png"

        rgba = crop_bbox_to_rgba(rgb, bbox)
        Image.fromarray(rgba, mode="RGBA").save(dest_path)

        row_dataset = source_dataset
        if source_dataset_column:
            val = row.get(source_dataset_column)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                row_dataset = str(val)

        records.append(
            ExtractedAsset(
                asset_id=asset_id,
                source_image=str(image_path.resolve()),
                asset_type=asset_type,
                asset_class=asset_class,
                source_bbox=json.dumps(list(bbox)),
                output_path=str(dest_path.resolve()),
                width=int(bbox[2]),
                height=int(bbox[3]),
                extraction_method=EXTRACTION_METHOD_BBOX_CROP,
                needs_sam2_refinement=True,
                source_dataset=row_dataset,
                has_alpha=True,
            )
        )

    if not records:
        raise ValueError(
            f"No assets extracted from {annotations_path}. "
            "Check bbox columns, image paths, and bbox values."
        )

    return records


def extract_assets(
    input_dir: Path,
    output_dir: Path,
    remove_background: bool = True,
    white_threshold: int = 240,
    uniform_tolerance: int = 18,
    use_class_subdirs: bool | None = None,
    asset_type: str | None = None,
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
    records: list[ExtractedAsset] = []
    resolved_type = asset_type or "unknown"

    for asset_class, source_path in jobs:
        asset_id = _asset_id(asset_class, source_path)
        if use_class_subdirs:
            dest_dir = output_dir / asset_class
        else:
            dest_dir = output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{asset_id}.png"

        rgba, method = extract_rgba_asset(
            source_path,
            remove_background=remove_background,
            white_threshold=white_threshold,
            uniform_tolerance=uniform_tolerance,
        )
        Image.fromarray(rgba, mode="RGBA").save(dest_path)

        h, w = rgba.shape[:2]
        records.append(
            ExtractedAsset(
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
        )

    return records


# ---------------------------------------------------------------------------
# Future: SAM / SAM2 mask refinement (interface placeholder — not implemented)
# ---------------------------------------------------------------------------
#
# def refine_assets_with_sam2(
#     records: list[ExtractedAsset],
#     sam2_checkpoint: Path,
#     prompt_from_bbox: bool = True,
# ) -> list[ExtractedAsset]:
#     """
#     Refine bbox_crop assets using SAM 2 segmentation masks.
#
#     - Input: RGBA crops or source_image + source_bbox from asset_metadata.csv
#     - Output: RGBA PNGs with true alpha boundaries
#     - Updates extraction_method to EXTRACTION_METHOD_SAM2 ("sam2_mask")
#     - Sets needs_sam2_refinement=False when mask quality passes QC
#     """
#     raise NotImplementedError("SAM/SAM2 mask extraction is not implemented yet.")


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
]


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
