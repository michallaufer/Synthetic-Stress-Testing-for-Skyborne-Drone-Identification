"""Bounding-box proposals for unlabeled manual foreground folders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AssetType = str  # drone | bird | airplane

COCO_BIRD = 14
COCO_AIRPLANE = 4

COCO_CLASS_NAMES: dict[int, str] = {
    COCO_BIRD: "bird",
    COCO_AIRPLANE: "airplane",
}

YOLO_CLASS_BY_ASSET: dict[str, int] = {
    "bird": COCO_BIRD,
    "airplane": COCO_AIRPLANE,
}


@dataclass
class BboxProposal:
    x: int
    y: int
    w: int
    h: int
    method: str
    confidence: float
    notes: str = ""
    yolo_available: bool = False
    yolo_model_loaded: bool = False
    yolo_detection_count: int = 0
    proposal_class_name: str = ""
    proposal_method: str = ""

    def __post_init__(self) -> None:
        if not self.proposal_method:
            self.proposal_method = self.method

    def as_xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class YoloStartupStatus:
    installed: bool
    model_loaded: bool
    load_error: str = ""


class YoloProposalService:
    """Singleton YOLOv8n loader — one model for the whole extraction run."""

    _model = None
    _load_error: str = ""
    _installed: bool | None = None

    @classmethod
    def startup_status(cls) -> YoloStartupStatus:
        if cls._installed is None:
            try:
                from ultralytics import YOLO  # noqa: F401

                cls._installed = True
            except ImportError:
                cls._installed = False
                cls._load_error = "ultralytics not installed (pip install -r requirements-yolo.txt)"
        model_loaded = cls._model is not None
        return YoloStartupStatus(
            installed=bool(cls._installed),
            model_loaded=model_loaded,
            load_error=cls._load_error,
        )

    @classmethod
    def ensure_model(cls) -> bool:
        status = cls.startup_status()
        if not status.installed:
            return False
        if cls._model is not None:
            return True
        try:
            from ultralytics import YOLO

            cls._model = YOLO("yolov8n.pt")
            return True
        except Exception as exc:  # noqa: BLE001
            cls._load_error = str(exc)
            return False


def _clamp_bbox(
    x: int, y: int, w: int, h: int, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def center_heuristic_bbox(img_w: int, img_h: int, *, coverage: float = 0.72) -> BboxProposal:
    w = max(32, int(round(img_w * coverage)))
    h = max(32, int(round(img_h * coverage)))
    x = max(0, (img_w - w) // 2)
    y = max(0, (img_h - h) // 2)
    x, y, w, h = _clamp_bbox(x, y, w, h, img_w, img_h)
    return BboxProposal(
        x=x,
        y=y,
        w=w,
        h=h,
        method="center_heuristic",
        proposal_method="center_heuristic",
        confidence=0.0,
        notes=f"coverage={coverage:.2f}",
        yolo_available=YoloProposalService.startup_status().installed,
        yolo_model_loaded=YoloProposalService.startup_status().model_loaded,
    )


def _proposal_from_detection(
    boxes: np.ndarray,
    confs: np.ndarray,
    class_ids: np.ndarray,
    idx: int,
    img_w: int,
    img_h: int,
    *,
    method: str,
    yolo_status: YoloStartupStatus,
    detection_count: int,
) -> BboxProposal:
    x1, y1, x2, y2 = boxes[idx]
    x = int(round(x1))
    y = int(round(y1))
    w = max(1, int(round(x2 - x1)))
    h = max(1, int(round(y2 - y1)))
    x, y, w, h = _clamp_bbox(x, y, w, h, img_w, img_h)
    cls_id = int(class_ids[idx])
    cls_name = COCO_CLASS_NAMES.get(cls_id, f"coco_{cls_id}")
    return BboxProposal(
        x=x,
        y=y,
        w=w,
        h=h,
        method=method,
        proposal_method=method,
        confidence=float(confs[idx]),
        notes=f"yolo_class_id={cls_id}",
        yolo_available=yolo_status.installed,
        yolo_model_loaded=yolo_status.model_loaded,
        yolo_detection_count=detection_count,
        proposal_class_name=cls_name,
    )


def _pick_best_yolo_detection(
    boxes: np.ndarray,
    confs: np.ndarray,
    class_ids: np.ndarray,
    img_w: int,
    img_h: int,
    *,
    allowed_classes: set[int] | None = None,
    method: str = "yolo_pretrained",
    yolo_status: YoloStartupStatus | None = None,
    detection_count: int = 0,
) -> BboxProposal | None:
    if boxes is None or len(boxes) == 0:
        return None
    if yolo_status is None:
        yolo_status = YoloProposalService.startup_status()

    best_idx = None
    best_score = -1.0
    for i in range(len(boxes)):
        cls = int(class_ids[i])
        if allowed_classes is not None and cls not in allowed_classes:
            continue
        conf = float(confs[i])
        x1, y1, x2, y2 = boxes[i]
        w = max(1, int(round(x2 - x1)))
        h = max(1, int(round(y2 - y1)))
        area = w * h
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        center_penalty = abs(cx - img_w / 2) / max(img_w, 1) + abs(cy - img_h / 2) / max(
            img_h, 1
        )
        score = conf * np.sqrt(area) * (1.0 - 0.35 * center_penalty)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return None

    return _proposal_from_detection(
        boxes,
        confs,
        class_ids,
        best_idx,
        img_w,
        img_h,
        method=method,
        yolo_status=yolo_status,
        detection_count=detection_count,
    )


def _run_yolo_detections(rgb: np.ndarray, *, min_conf: float) -> tuple | None:
    if not YoloProposalService.ensure_model():
        return None
    model = YoloProposalService._model
    results = model.predict(rgb, verbose=False, conf=min_conf)
    if not results:
        return None
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    return boxes, confs, class_ids, len(boxes)


def propose_bbox_for_image(
    rgb: np.ndarray,
    asset_type: AssetType,
    *,
    min_conf: float = 0.15,
    heuristic_coverage: float = 0.72,
) -> BboxProposal:
    """Propose a bbox for SAM2 box prompting on unlabeled manual images."""
    img_h, img_w = rgb.shape[:2]
    yolo_status = YoloProposalService.startup_status()

    detections = _run_yolo_detections(rgb, min_conf=min_conf)
    detection_count = detections[-1] if detections else 0

    if detections is not None:
        boxes, confs, class_ids, detection_count = detections

        if asset_type == "bird":
            hit = _pick_best_yolo_detection(
                boxes,
                confs,
                class_ids,
                img_w,
                img_h,
                allowed_classes={COCO_BIRD},
                method="yolo_bird",
                yolo_status=yolo_status,
                detection_count=detection_count,
            )
            if hit is not None:
                return hit

        elif asset_type == "airplane":
            hit = _pick_best_yolo_detection(
                boxes,
                confs,
                class_ids,
                img_w,
                img_h,
                allowed_classes={COCO_AIRPLANE},
                method="yolo_airplane",
                yolo_status=yolo_status,
                detection_count=detection_count,
            )
            if hit is not None:
                return hit

        elif asset_type == "drone":
            hit = _pick_best_yolo_detection(
                boxes,
                confs,
                class_ids,
                img_w,
                img_h,
                allowed_classes=None,
                method="largest_detection_fallback",
                yolo_status=yolo_status,
                detection_count=detection_count,
            )
            if hit is not None:
                hit.proposal_method = "largest_detection_fallback"
                hit.method = "largest_detection_fallback"
                hit.notes = f"{hit.notes};no_coco_drone_class"
                return hit

    fallback = center_heuristic_bbox(img_w, img_h, coverage=heuristic_coverage)
    fallback.yolo_detection_count = detection_count
    if asset_type == "drone" and detection_count > 0:
        fallback.notes = f"{fallback.notes};yolo_detections_present_but_not_used"
    elif detection_count == 0 and yolo_status.model_loaded:
        fallback.notes = f"{fallback.notes};yolo_zero_detections"
    elif not yolo_status.installed:
        fallback.notes = f"{fallback.notes};yolo_not_installed"
    elif not yolo_status.model_loaded:
        fallback.notes = f"{fallback.notes};yolo_load_failed:{yolo_status.load_error}"
    return fallback
