"""AirBirds chunk diversity audit — temporal thinning and perceptual-hash deduplication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from drone_stress.yolo_adapter import IMAGE_EXTENSIONS, yolo_norm_to_xywh

METADATA_COLUMNS = [
    "image_path",
    "filename",
    "width",
    "height",
    "has_bird_annotation",
    "num_birds",
    "min_bbox_size",
    "max_bbox_size",
    "phash",
    "duplicate_group",
    "selected_after_temporal_stride",
    "selected_after_phash_dedup",
    "sequence_key",
]


@dataclass
class ImageRecord:
    image_path: Path
    filename: str
    width: int
    height: int
    has_bird_annotation: bool
    num_birds: int
    min_bbox_size: float | None
    max_bbox_size: float | None
    phash: str
    duplicate_group: int
    selected_after_temporal_stride: bool
    selected_after_phash_dedup: bool
    sequence_key: str

    def to_row(self) -> dict:
        return {
            "image_path": str(self.image_path.resolve()),
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "has_bird_annotation": self.has_bird_annotation,
            "num_birds": self.num_birds,
            "min_bbox_size": self.min_bbox_size,
            "max_bbox_size": self.max_bbox_size,
            "phash": self.phash,
            "duplicate_group": self.duplicate_group,
            "selected_after_temporal_stride": self.selected_after_temporal_stride,
            "selected_after_phash_dedup": self.selected_after_phash_dedup,
            "sequence_key": self.sequence_key,
        }


def natural_sort_key(path: Path) -> list:
    text = path.as_posix()
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def resolve_bird_class_ids(dataset_root: Path) -> set[int]:
    """Resolve YOLO class ids that correspond to birds (default: {0})."""
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.is_file():
        return {0}

    with data_yaml.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    names = raw.get("names")
    if names is None:
        return {0}

    if isinstance(names, list):
        id_to_name = {i: str(name) for i, name in enumerate(names)}
    elif isinstance(names, dict):
        id_to_name = {int(k): str(v) for k, v in names.items()}
    else:
        return {0}

    bird_ids = {
        cid
        for cid, name in id_to_name.items()
        if "bird" in name.lower()
    }
    return bird_ids or {0}


def find_yolo_label(image_path: Path) -> Path | None:
    """Locate a YOLO label file for an image using common AirBirds / YOLO layouts."""
    direct = image_path.with_suffix(".txt")
    if direct.is_file():
        return direct

    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        lower = part.lower()
        if "image" in lower and "label" not in lower:
            replacements: list[str] = []
            if "images" in lower:
                replacements.append(
                    part.replace("images", "labels").replace("Images", "Labels")
                )
            if part.endswith("image"):
                replacements.append(part[:-5] + "labels")
            replacements.append("labels")
            for replacement in replacements:
                if replacement == part:
                    continue
                candidate = Path(*parts[:i], replacement, *parts[i + 1 :]).with_suffix(
                    ".txt"
                )
                if candidate.is_file():
                    return candidate
    return None


def parse_bird_annotations(
    label_path: Path | None,
    *,
    bird_class_ids: set[int],
    img_w: int,
    img_h: int,
) -> tuple[int, float | None, float | None]:
    if label_path is None or not label_path.is_file():
        return 0, None, None

    sizes: list[float] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc, yc, wn, hn = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
        except ValueError:
            continue
        if class_id not in bird_class_ids:
            continue
        _x, _y, w, h = yolo_norm_to_xywh(xc, yc, wn, hn, img_w, img_h)
        sizes.append(float(max(w, h)))

    if not sizes:
        return 0, None, None
    return len(sizes), float(min(sizes)), float(max(sizes))


def compute_phash(image_path: Path) -> str:
    """Compute 64-bit perceptual hash (hex) compatible with Hamming thresholding."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise OSError(f"Cannot read image for phash: {image_path}")

    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:8, :8]
    med = float(np.median(dct_low[1:, 1:]))
    bits = (dct_low > med).astype(np.uint8).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def phash_hamming_distance(hash_a: str, hash_b: str) -> int:
    if not hash_a or not hash_b:
        return 64
    a = int(hash_a, 16)
    b = int(hash_b, 16)
    return (a ^ b).bit_count()


def sequence_key_for_image(image_path: Path, dataset_root: Path) -> str:
    """Group frames by parent folder relative to dataset root (camera / clip folder)."""
    try:
        rel = image_path.parent.relative_to(dataset_root.resolve())
        key = rel.as_posix()
        return key if key else image_path.parent.name
    except ValueError:
        return image_path.parent.name


def iter_image_paths(dataset_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(dataset_root.rglob("*"), key=natural_sort_key):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths


def apply_temporal_thinning(
    records: list[ImageRecord],
    *,
    temporal_stride: int,
) -> list[ImageRecord]:
    by_sequence: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_sequence.setdefault(record.sequence_key, []).append(record)

    selected: list[ImageRecord] = []
    stride = max(1, temporal_stride)
    for _key, group in by_sequence.items():
        group.sort(key=lambda r: natural_sort_key(Path(r.filename)))
        for idx, record in enumerate(group):
            if idx % stride == 0:
                record.selected_after_temporal_stride = True
                selected.append(record)
            else:
                record.selected_after_temporal_stride = False
    return selected


def apply_phash_deduplication(
    thinned: list[ImageRecord],
    *,
    phash_threshold: int,
) -> tuple[list[ImageRecord], int]:
    """Cluster near-duplicates; keep first frame per group in temporal order."""
    representatives: list[tuple[str, int]] = []
    num_groups = 0

    thinned_sorted = sorted(
        thinned,
        key=lambda r: (r.sequence_key, natural_sort_key(Path(r.filename))),
    )

    for record in thinned_sorted:
        matched_group: int | None = None
        for rep_hash, group_id in representatives:
            if phash_hamming_distance(record.phash, rep_hash) <= phash_threshold:
                matched_group = group_id
                break

        if matched_group is None:
            matched_group = num_groups
            representatives.append((record.phash, matched_group))
            num_groups += 1
            record.selected_after_phash_dedup = True
        else:
            record.selected_after_phash_dedup = False

        record.duplicate_group = matched_group

    return thinned_sorted, num_groups


def scan_airbirds_chunk(
    dataset_root: Path,
    *,
    max_images: int,
    bird_class_ids: set[int] | None = None,
) -> list[ImageRecord]:
    dataset_root = dataset_root.resolve()
    bird_ids = bird_class_ids or resolve_bird_class_ids(dataset_root)

    image_paths = iter_image_paths(dataset_root)
    if max_images > 0:
        image_paths = image_paths[:max_images]

    records: list[ImageRecord] = []
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except OSError:
            continue

        label_path = find_yolo_label(image_path)
        num_birds, min_bbox, max_bbox = parse_bird_annotations(
            label_path,
            bird_class_ids=bird_ids,
            img_w=width,
            img_h=height,
        )

        try:
            phash = compute_phash(image_path)
        except OSError:
            phash = ""

        records.append(
            ImageRecord(
                image_path=image_path,
                filename=image_path.name,
                width=int(width),
                height=int(height),
                has_bird_annotation=num_birds > 0,
                num_birds=num_birds,
                min_bbox_size=min_bbox,
                max_bbox_size=max_bbox,
                phash=phash,
                duplicate_group=-1,
                selected_after_temporal_stride=False,
                selected_after_phash_dedup=False,
                sequence_key=sequence_key_for_image(image_path, dataset_root),
            )
        )
    return records


def audit_airbirds_chunk(
    dataset_root: Path,
    *,
    max_images: int = 5000,
    temporal_stride: int = 100,
    phash_threshold: int = 6,
    bird_class_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    records = scan_airbirds_chunk(
        dataset_root,
        max_images=max_images,
        bird_class_ids=bird_class_ids,
    )
    thinned = apply_temporal_thinning(records, temporal_stride=temporal_stride)
    _, num_duplicate_groups = apply_phash_deduplication(
        thinned,
        phash_threshold=phash_threshold,
    )

    df = pd.DataFrame([r.to_row() for r in records], columns=METADATA_COLUMNS)
    deduped = df[df["selected_after_phash_dedup"] == True]  # noqa: E712
    usable = int(len(deduped))
    bird_images = int(df["has_bird_annotation"].sum())
    no_bird_images = int(len(df) - bird_images)

    summary = {
        "total_images_found": len(records),
        "images_after_temporal_thinning": int(df["selected_after_temporal_stride"].sum()),
        "images_after_phash_dedup": usable,
        "duplicate_groups": num_duplicate_groups,
        "no_bird_images": no_bird_images,
        "bird_images": bird_images,
        "usable_distinct_backgrounds": usable,
        "usable_no_bird_backgrounds": int(
            deduped[deduped["has_bird_annotation"] == False].shape[0]  # noqa: E712
        ),
        "usable_bird_hard_negatives": int(
            deduped[deduped["has_bird_annotation"] == True].shape[0]  # noqa: E712
        ),
    }
    return df, summary


def write_audit_report(report_path: Path, summary: dict) -> None:
    lines = [
        "AirBirds diversity audit report",
        f"total_images_found: {summary['total_images_found']}",
        f"images_after_temporal_thinning: {summary['images_after_temporal_thinning']}",
        f"images_after_phash_dedup: {summary['images_after_phash_dedup']}",
        f"duplicate_groups: {summary['duplicate_groups']}",
        f"no_bird_images: {summary['no_bird_images']}",
        f"bird_images: {summary['bird_images']}",
        f"usable_distinct_backgrounds: {summary['usable_distinct_backgrounds']}",
        f"usable_no_bird_backgrounds: {summary['usable_no_bird_backgrounds']}",
        f"usable_bird_hard_negatives: {summary['usable_bird_hard_negatives']}",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
