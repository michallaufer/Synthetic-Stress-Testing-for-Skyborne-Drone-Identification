"""Shared helpers for YOLO dataset -> internal annotation CSV adapters."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class ConversionStats:
    images_scanned: int = 0
    label_files_scanned: int = 0
    boxes_read: int = 0
    boxes_exported: int = 0
    boxes_filtered: int = 0
    filter_reasons: dict[str, int] = field(default_factory=dict)
    by_split: dict[str, int] = field(default_factory=dict)
    by_class: dict[str, int] = field(default_factory=dict)
    bbox_max_dims: list[int] = field(default_factory=list)

    def add_filter(self, reason: str) -> None:
        self.boxes_filtered += 1
        self.filter_reasons[reason] = self.filter_reasons.get(reason, 0) + 1


def parse_data_yaml_names(dataset_root: Path) -> dict[int, str]:
    """
    Read class id -> name mapping from data.yaml.

    Supports:
      names: ["bird", "drone"]
      names:
        0: bird
        1: drone
    """
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}. "
            "Place data.yaml in --dataset-root or use seraphim_yolo_to_annotations.py "
            "for single-class export without data.yaml."
        )

    with data_yaml.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    names = raw.get("names")
    if names is None:
        raise ValueError(f"data.yaml at {data_yaml} has no 'names' field")

    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    raise ValueError(
        f"Unsupported 'names' format in {data_yaml}. Expected list or dict."
    )


def resolve_class_filter(
    class_names: dict[int, str],
    class_filter: list[str] | None,
) -> dict[int, str]:
    """Return subset of id->name to export; default is all classes from data.yaml."""
    if not class_filter:
        return class_names

    allowed = {c.strip() for c in class_filter if c.strip()}
    selected = {i: n for i, n in class_names.items() if n in allowed}
    missing = allowed - set(selected.values())
    if missing:
        available = sorted(class_names.values())
        raise ValueError(
            f"--class-filter includes unknown class(es): {sorted(missing)}. "
            f"Available from data.yaml: {available}"
        )
    if not selected:
        raise ValueError("No classes selected after applying --class-filter.")
    return selected


def split_name_candidates(split: str) -> list[str]:
    """Try folder names for a logical split (val <-> valid)."""
    names = [split]
    if split == "val" and "valid" not in names:
        names.append("valid")
    if split == "valid" and "val" not in names:
        names.append("val")
    return names


def resolve_split_dirs(
    dataset_root: Path,
    split: str,
) -> tuple[Path, Path, str] | None:
    """
    Locate (images_dir, labels_dir) for a split.

    Layout B: dataset/<split>/images + dataset/<split>/labels
    Layout A: dataset/images/<split> + dataset/labels/<split>
    """
    for folder_name in split_name_candidates(split):
        layout_b_img = dataset_root / folder_name / "images"
        layout_b_lbl = dataset_root / folder_name / "labels"
        if layout_b_img.is_dir() and layout_b_lbl.is_dir():
            return layout_b_img, layout_b_lbl, folder_name

        layout_a_img = dataset_root / "images" / folder_name
        layout_a_lbl = dataset_root / "labels" / folder_name
        if layout_a_img.is_dir() and layout_a_lbl.is_dir():
            return layout_a_img, layout_a_lbl, folder_name

    return None


def discover_splits(
    dataset_root: Path,
    splits: list[str],
) -> list[tuple[str, Path, Path]]:
    """Return (canonical_split, images_dir, labels_dir) for each requested split."""
    found: list[tuple[str, Path, Path]] = []
    seen: set[tuple[str, str]] = set()

    for split in splits:
        resolved = resolve_split_dirs(dataset_root, split.strip())
        if resolved is None:
            continue
        images_dir, labels_dir, folder_name = resolved
        key = (str(images_dir.resolve()), str(labels_dir.resolve()))
        if key in seen:
            continue
        seen.add(key)
        found.append((folder_name, images_dir, labels_dir))

    return found


def iter_images(images_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths


def label_path_for_image(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    rel = image_path.relative_to(images_dir)
    return labels_dir / rel.parent / f"{rel.stem}.txt"


def yolo_norm_to_xywh(
    xc: float,
    yc: float,
    wn: float,
    hn: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x = (xc - wn / 2.0) * image_width
    y = (yc - hn / 2.0) * image_height
    w = wn * image_width
    h = hn * image_height
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def clip_bbox_xywh(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    x, y, w, h = bbox
    if image_width <= 0 or image_height <= 0:
        return None
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    w = max(0, min(w, image_width - x))
    h = max(0, min(h, image_height - y))
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def validate_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    min_bbox_px: int,
) -> str | None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return "non_positive_size"
    if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
        return "outside_image"
    if max(w, h) < min_bbox_px:
        return "below_min_bbox_px"
    return None


def read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        w, h = img.size
    return int(w), int(h)


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def convert_yolo_dataset(
    dataset_root: Path,
    splits: list[str],
    class_id_to_name: dict[int, str],
    source_dataset: str,
    min_bbox_px: int,
    max_rows: int | None,
) -> tuple[pd.DataFrame, ConversionStats]:
    """Convert YOLO labels to internal annotation rows for selected classes."""
    dataset_root = dataset_root.resolve()
    split_dirs = discover_splits(dataset_root, splits)
    if not split_dirs:
        raise FileNotFoundError(
            f"No YOLO split folders found under {dataset_root} for splits {splits}. "
            "Expected Layout A (images/<split>, labels/<split>) or "
            "Layout B (<split>/images, <split>/labels)."
        )

    stats = ConversionStats()
    rows: list[dict] = []

    for split_name, images_dir, labels_dir in split_dirs:
        for image_path in iter_images(images_dir):
            if max_rows is not None and stats.boxes_exported >= max_rows:
                break

            stats.images_scanned += 1
            label_path = label_path_for_image(image_path, images_dir, labels_dir)
            if not label_path.is_file():
                continue

            stats.label_files_scanned += 1
            try:
                img_w, img_h = read_image_size(image_path)
            except OSError:
                stats.add_filter("image_read_error")
                continue

            rel_image = relative_path(image_path, dataset_root)

            for line in label_path.read_text(encoding="utf-8").splitlines():
                if max_rows is not None and stats.boxes_exported >= max_rows:
                    break

                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 5:
                    stats.boxes_read += 1
                    stats.add_filter("malformed_label_line")
                    continue

                stats.boxes_read += 1
                try:
                    yolo_class_id = int(float(parts[0]))
                    xc, yc, wn, hn = (
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                except ValueError:
                    stats.add_filter("parse_error")
                    continue

                class_name = class_id_to_name.get(yolo_class_id)
                if class_name is None:
                    stats.add_filter("class_id_not_exported")
                    continue

                bbox = yolo_norm_to_xywh(xc, yc, wn, hn, img_w, img_h)
                clipped = clip_bbox_xywh(bbox, img_w, img_h)
                if clipped is None:
                    stats.add_filter("clip_failed")
                    continue

                reject = validate_bbox(clipped, img_w, img_h, min_bbox_px)
                if reject:
                    stats.add_filter(reject)
                    continue

                x, y, w, h = clipped
                rows.append(
                    {
                        "filename": rel_image,
                        "image_path": rel_image,
                        "class_name": class_name,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "source_dataset": source_dataset,
                        "split": split_name,
                        "image_width": img_w,
                        "image_height": img_h,
                        "yolo_class_id": yolo_class_id,
                    }
                )
                stats.boxes_exported += 1
                stats.by_split[split_name] = stats.by_split.get(split_name, 0) + 1
                stats.by_class[class_name] = stats.by_class.get(class_name, 0) + 1
                stats.bbox_max_dims.append(max(w, h))

        if max_rows is not None and stats.boxes_exported >= max_rows:
            break

    if not rows:
        raise ValueError(
            "No annotations exported. Check dataset layout, data.yaml class names, "
            "--class-filter, and --splits."
        )

    return pd.DataFrame(rows), stats


def write_conversion_report(
    stats: ConversionStats,
    report_path: Path,
    dataset_root: Path,
    output_csv: Path,
    title: str,
    class_id_to_name: dict[int, str],
) -> None:
    dims = stats.bbox_max_dims
    if dims:
        min_dim = min(dims)
        max_dim = max(dims)
        median_dim = int(statistics.median(dims))
    else:
        min_dim = max_dim = median_dim = 0

    lines = [
        title,
        f"dataset_root: {dataset_root.resolve()}",
        f"output_csv: {output_csv.resolve()}",
        "",
        "classes_exported:",
    ]
    for cid in sorted(class_id_to_name):
        lines.append(f"  {cid}: {class_id_to_name[cid]}")
    lines.extend(
        [
            "",
            f"images_scanned: {stats.images_scanned}",
            f"label_files_scanned: {stats.label_files_scanned}",
            f"boxes_read: {stats.boxes_read}",
            f"boxes_exported: {stats.boxes_exported}",
            f"boxes_filtered: {stats.boxes_filtered}",
            "",
            "exported_by_split:",
        ]
    )
    for split in sorted(stats.by_split):
        lines.append(f"  {split}: {stats.by_split[split]}")
    lines.append("")
    lines.append("exported_by_class_name:")
    for cls in sorted(stats.by_class):
        lines.append(f"  {cls}: {stats.by_class[cls]}")
    lines.extend(
        [
            "",
            "bbox max(width,height) px - min / median / max:",
            f"  {min_dim} / {median_dim} / {max_dim}",
            "",
            "filter_reasons:",
        ]
    )
    for reason, count in sorted(stats.filter_reasons.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def slugify_dataset_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
