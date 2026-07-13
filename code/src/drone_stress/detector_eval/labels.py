"""Ground-truth label loading for evaluation."""

from __future__ import annotations

from pathlib import Path

from drone_stress.compositor import BBox


def yolo_line_to_bbox(line: str, img_w: int, img_h: int) -> tuple[int, BBox] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    class_id = int(float(parts[0]))
    cx = float(parts[1]) * img_w
    cy = float(parts[2]) * img_h
    w = float(parts[3]) * img_w
    h = float(parts[4]) * img_h
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    return class_id, BBox(x=x, y=y, w=max(1, int(round(w))), h=max(1, int(round(h))))


def load_gt_drone_boxes(
    labels_dir: Path,
    image_id: str,
    img_w: int,
    img_h: int,
    drone_class_id: int = 0,
) -> list[BBox]:
    """Load drone GT boxes from YOLO label file (class filter optional)."""
    label_path = labels_dir / f"{Path(image_id).stem}.txt"
    if not label_path.is_file():
        return []
    boxes: list[BBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parsed = yolo_line_to_bbox(line, img_w, img_h)
        if parsed is None:
            continue
        class_id, bbox = parsed
        if class_id == drone_class_id:
            boxes.append(bbox)
    return boxes


def bbox_to_xyxy(bbox: BBox) -> tuple[float, float, float, float]:
    return float(bbox.x), float(bbox.y), float(bbox.x + bbox.w), float(bbox.y + bbox.h)
