"""QA / debug visualization helpers for pilot synthetic datasets."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

COLOR_DRONE_RGB = (0, 220, 0)
COLOR_DISTRACTOR_RGB = (255, 140, 0)
COLOR_DRONE_BGR = (0, 255, 0)
COLOR_DISTRACTOR_BGR = (0, 165, 255)


def parse_json_list(value) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def parse_bbox(value) -> tuple[int, int, int, int] | None:
    coords = parse_json_list(value)
    if len(coords) != 4:
        return None
    x, y, w, h = [int(c) for c in coords]
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def parse_bbox_list(value) -> list[tuple[int, int, int, int]]:
    raw = parse_json_list(value)
    boxes: list[tuple[int, int, int, int]] = []
    for item in raw:
        if isinstance(item, list) and len(item) == 4:
            x, y, w, h = [int(c) for c in item]
            if w > 0 and h > 0:
                boxes.append((x, y, w, h))
    return boxes


def effective_thickness(bbox: tuple[int, int, int, int], base: int) -> int:
    _, _, w, h = bbox
    if max(w, h) <= 20:
        return max(base + 2, 5)
    return base


def extract_crop(
    rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad: int,
    out_size: int,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    x, y, bw, bh = bbox
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(w, x + bw + pad)
    bottom = min(h, y + bh + pad)
    crop = rgb[top:bottom, left:right]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def _draw_bbox_only(
    bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    color_bgr: tuple[int, int, int],
    thickness: int,
    draw_center: bool,
) -> None:
    x, y, w, h = bbox
    t = effective_thickness(bbox, thickness)
    cv2.rectangle(bgr, (x, y), (x + w, y + h), color_bgr, t)
    if draw_center:
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        cv2.circle(bgr, (cx, cy), max(3, t), color_bgr, -1)


def _draw_bbox_labeled(
    bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    color_bgr: tuple[int, int, int],
    label: str,
    thickness: int,
    draw_center: bool,
) -> None:
    _draw_bbox_only(bgr, bbox, color_bgr, thickness, draw_center)
    x, y, w, h = bbox
    img_h, img_w = bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    tiny = max(w, h) <= 24
    scale = 0.35 if tiny else 0.5
    thickness_txt = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness_txt)
    if tiny:
        # Place label below or to the right so it does not cover the object.
        below_y = min(y + h + th + 4, img_h - 2)
        if below_y >= y + h + 2:
            tx, ty = x, below_y
        else:
            tx = min(x + w + 4, max(0, img_w - tw - 2))
            ty = max(th + 2, min(y + th + 2, img_h - 2))
    else:
        tx = x
        ty = max(y - 6, th + 2)
    tx = max(0, min(tx, img_w - tw - 1))
    ty = max(th + 1, min(ty, img_h - 1))
    cv2.putText(bgr, label, (tx, ty), font, scale, color_bgr, thickness_txt, cv2.LINE_AA)


def _crop_region(
    rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return crop and bbox mapped into crop pixel coordinates (before resize)."""
    h, w = rgb.shape[:2]
    x, y, bw, bh = bbox
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(w, x + bw + pad)
    bottom = min(h, y + bh + pad)
    crop = rgb[top:bottom, left:right]
    local = (x - left, y - top, bw, bh)
    return crop, local


def _scale_bbox_to_size(
    bbox: tuple[int, int, int, int],
    src_shape: tuple[int, int],
    out_size: int,
) -> tuple[int, int, int, int]:
    ch, cw = src_shape[:2]
    if cw <= 0 or ch <= 0:
        return (0, 0, 0, 0)
    x, y, bw, bh = bbox
    sx = out_size / float(cw)
    sy = out_size / float(ch)
    return (
        int(round(x * sx)),
        int(round(y * sy)),
        max(1, int(round(bw * sx))),
        max(1, int(round(bh * sy))),
    )


def render_crop_tile(
    rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad: int,
    out_size: int,
    color_bgr: tuple[int, int, int],
    thickness: int,
    draw_center: bool,
) -> np.ndarray:
    """Zoomed crop with bbox/center only (metadata belongs in the matplotlib title)."""
    crop, local_bbox = _crop_region(rgb, bbox, pad)
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    local_scaled = _scale_bbox_to_size(local_bbox, crop.shape, out_size)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    _draw_bbox_only(bgr, local_scaled, color_bgr, thickness, draw_center)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def annotate_image_qa(
    rgb: np.ndarray,
    row: pd.Series,
    bbox_thickness: int = 3,
    draw_center: bool = True,
) -> np.ndarray:
    """Draw labeled drone (green) and distractor (orange) boxes on the image."""
    bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    distractor_classes = parse_json_list(row.get("distractor_classes"))
    distractor_ids = parse_json_list(row.get("distractor_asset_ids"))
    distractor_boxes = parse_bbox_list(row.get("distractor_bboxes"))

    for i, box in enumerate(distractor_boxes):
        cls = distractor_classes[i] if i < len(distractor_classes) else "distractor"
        _draw_bbox_labeled(bgr, box, COLOR_DISTRACTOR_BGR, cls, bbox_thickness, draw_center)

    target_bbox = parse_bbox(row.get("target_bbox"))
    if target_bbox is None and row.get("target_present"):
        target_bbox = parse_bbox(row.get("bbox"))
    if target_bbox is not None:
        _draw_bbox_labeled(bgr, target_bbox, COLOR_DRONE_BGR, "drone", bbox_thickness, draw_center)

    if not row.get("target_present") and target_bbox is None and not distractor_boxes:
        legacy = parse_bbox(row.get("bbox"))
        if legacy is not None:
            cls = distractor_classes[0] if distractor_classes else "distractor"
            _draw_bbox_labeled(bgr, legacy, COLOR_DISTRACTOR_BGR, cls, bbox_thickness, draw_center)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@dataclass
class CropTile:
    image: np.ndarray
    title: str


def _crop_tile_title(
    image_id: str,
    subset: str,
    object_type: str,
    cls: str,
    asset_id: str,
    bbox: tuple[int, int, int, int],
) -> str:
    return (
        f"{image_id} | subset={subset}\n"
        f"object_type={object_type} | class={cls}\n"
        f"asset_id={asset_id}\n"
        f"bbox={list(bbox)}"
    )


def collect_qa_crop_tiles(
    rgb: np.ndarray,
    row: pd.Series,
    pad: int,
    out_size: int,
    bbox_thickness: int = 3,
) -> list[CropTile]:
    """One tile per pasted object (separate drone + distractor for mixed_challenge)."""
    tiles: list[CropTile] = []
    image_id = str(row.get("image_id", "?"))
    subset = str(row.get("subset", "?"))
    distractor_classes = parse_json_list(row.get("distractor_classes"))
    distractor_ids = parse_json_list(row.get("distractor_asset_ids"))
    distractor_boxes = parse_bbox_list(row.get("distractor_bboxes"))

    target_bbox = parse_bbox(row.get("target_bbox"))
    if target_bbox is None and row.get("target_present"):
        target_bbox = parse_bbox(row.get("bbox"))
    target_id = str(row.get("target_asset_id", "") or "")

    if target_bbox is not None:
        crop = render_crop_tile(
            rgb, target_bbox, pad, out_size, COLOR_DRONE_BGR, bbox_thickness, True
        )
        tiles.append(
            CropTile(
                crop,
                _crop_tile_title(image_id, subset, "target", "drone", target_id, target_bbox),
            )
        )

    for i, box in enumerate(distractor_boxes):
        cls = distractor_classes[i] if i < len(distractor_classes) else "distractor"
        aid = str(distractor_ids[i] if i < len(distractor_ids) else "?")
        crop = render_crop_tile(
            rgb, box, pad, out_size, COLOR_DISTRACTOR_BGR, bbox_thickness, True
        )
        tiles.append(
            CropTile(
                crop,
                _crop_tile_title(image_id, subset, "distractor", cls, aid, box),
            )
        )

    if not tiles and not row.get("target_present"):
        legacy = parse_bbox(row.get("bbox"))
        if legacy is not None:
            cls = distractor_classes[0] if distractor_classes else "distractor"
            aid = str(distractor_ids[0] if distractor_ids else "?")
            crop = render_crop_tile(
                rgb, legacy, pad, out_size, COLOR_DISTRACTOR_BGR, bbox_thickness, True
            )
            tiles.append(
                CropTile(
                    crop,
                    _crop_tile_title(image_id, subset, "distractor", cls, aid, legacy),
                )
            )

    return tiles


def qa_contact_title(row: pd.Series) -> str:
    bg_file = row.get("background_filename", "?")
    bg_type = row.get("background_type", "?")
    target_id = row.get("target_asset_id", "") or "-"
    dist_ids = parse_json_list(row.get("distractor_asset_ids"))
    dist_str = ",".join(str(x) for x in dist_ids) if dist_ids else "-"
    return (
        f"{row['subset']} | {row.get('pasted_object_summary', '')}\n"
        f"bg={bg_type} ({bg_file})\n"
        f"drone={target_id} | dist={dist_str}\n"
        f"size={row['object_size_px']}px"
    )


def _axes_grid(n: int, cols: int):
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 4.2))
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    elif rows == 1:
        axes_flat = list(axes)
    elif cols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]
    return fig, axes_flat


def _hide_unused_axes(axes_flat: list, used: int) -> None:
    for i, ax in enumerate(axes_flat):
        if i >= used:
            ax.set_visible(False)
            ax.axis("off")


def build_contact_sheet(
    metadata_csv: Path,
    images_dir: Path,
    output_path: Path,
    num_samples: int,
    cols: int,
    seed: int,
    bbox_thickness: int = 3,
    draw_center: bool = True,
) -> Path:
    df = pd.read_csv(metadata_csv)
    n = min(num_samples, len(df))
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    total_slots = ((n + cols - 1) // cols) * cols
    fig, axes_flat = _axes_grid(total_slots, cols)

    used = 0
    for i in range(n):
        ax = axes_flat[i]
        ax.axis("off")
        row = sample.iloc[i]
        image_path = images_dir / row["image_id"]
        if not image_path.is_file():
            ax.set_title("missing", fontsize=6)
            used += 1
            continue
        bgr = cv2.imread(str(image_path))
        rgb = annotate_image_qa(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), row, bbox_thickness, draw_center)
        ax.imshow(rgb)
        ax.set_title(qa_contact_title(row), fontsize=6)
        used += 1
    _hide_unused_axes(axes_flat, used)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_crops_sheet(
    metadata_csv: Path,
    images_dir: Path,
    output_path: Path,
    num_samples: int,
    cols: int,
    seed: int,
    crop_pad_px: int = 16,
    crop_size_px: int = 200,
    bbox_thickness: int = 3,
) -> Path:
    df = pd.read_csv(metadata_csv)
    n = min(num_samples, len(df))
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)

    all_tiles: list[CropTile] = []
    for _, row in sample.iterrows():
        image_path = images_dir / row["image_id"]
        if not image_path.is_file():
            continue
        bgr = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        all_tiles.extend(
            collect_qa_crop_tiles(rgb, row, crop_pad_px, crop_size_px, bbox_thickness)
        )

    if not all_tiles:
        raise ValueError("No crop tiles generated for crops sheet.")

    crop_cols = min(cols, 4)
    n_tiles = len(all_tiles)
    total_slots = ((n_tiles + crop_cols - 1) // crop_cols) * crop_cols
    fig, axes_flat = _axes_grid(total_slots, crop_cols)

    for i, tile in enumerate(all_tiles):
        ax = axes_flat[i]
        ax.axis("off")
        ax.imshow(tile.image)
        ax.set_title(tile.title, fontsize=5, loc="left", pad=4)
    _hide_unused_axes(axes_flat, n_tiles)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _rgb_to_base64_png(rgb: np.ndarray) -> str:
    pil = Image.fromarray(rgb)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_html_audit_report(
    metadata_csv: Path,
    images_dir: Path,
    output_path: Path,
    num_rows: int,
    seed: int,
    crop_pad_px: int = 20,
    crop_size_px: int = 180,
) -> Path:
    df = pd.read_csv(metadata_csv)
    n = min(num_rows, len(df))
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)

    rows_html: list[str] = []
    for _, row in sample.iterrows():
        image_path = images_dir / row["image_id"]
        if not image_path.is_file():
            continue
        bgr = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        full_rgb = annotate_image_qa(rgb, row, 3, True)
        full_b64 = _rgb_to_base64_png(full_rgb)

        target_html = "<em>none</em>"
        target_bbox = parse_bbox(row.get("target_bbox"))
        if target_bbox is None and row.get("target_present"):
            target_bbox = parse_bbox(row.get("bbox"))
        if target_bbox is not None:
            tcrop = render_crop_tile(
                rgb, target_bbox, crop_pad_px, crop_size_px, COLOR_DRONE_BGR, 3, True
            )
            target_html = (
                f'<img src="data:image/png;base64,{_rgb_to_base64_png(tcrop)}" '
                f'alt="target crop"/><br/>bbox={list(target_bbox)}'
            )

        dist_html = "<em>none</em>"
        dist_boxes = parse_bbox_list(row.get("distractor_bboxes"))
        if dist_boxes:
            parts = []
            for box in dist_boxes:
                dcrop = render_crop_tile(
                    rgb, box, crop_pad_px, crop_size_px, COLOR_DISTRACTOR_BGR, 3, True
                )
                parts.append(
                    f'<img src="data:image/png;base64,{_rgb_to_base64_png(dcrop)}" '
                    f'alt="distractor crop"/><br/>bbox={list(box)}'
                )
            dist_html = "<br/>".join(parts)

        rows_html.append(
            f"""
            <tr>
              <td><code>{row['image_id']}</code><br/>{row['subset']}</td>
              <td><img src="data:image/png;base64,{full_b64}" alt="full"/></td>
              <td>{target_html}</td>
              <td>{dist_html}</td>
              <td><code>{row.get('background_filename','')}</code></td>
              <td><code>{row.get('target_asset_id','')}</code></td>
              <td><code>{row.get('distractor_asset_ids','[]')}</code></td>
              <td><code>{row.get('target_bbox','[]')}</code></td>
              <td><code>{row.get('distractor_bboxes','[]')}</code></td>
            </tr>
            """
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Pilot dataset audit</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; font-size: 12px; }}
    th {{ background: #f0f0f0; }}
    img {{ max-width: 280px; height: auto; }}
    code {{ font-size: 11px; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>Pilot dataset visual audit</h1>
  <p>Green = drone (target), orange = distractor. Generated from metadata sample (n={n}).</p>
  <table>
    <thead>
      <tr>
        <th>image / subset</th>
        <th>full image</th>
        <th>target crop</th>
        <th>distractor crop(s)</th>
        <th>background_filename</th>
        <th>target_asset_id</th>
        <th>distractor_asset_ids</th>
        <th>target_bbox</th>
        <th>distractor_bboxes</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
