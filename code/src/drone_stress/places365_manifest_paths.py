"""Robust image path resolution for Places365 review/export manifests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drone_stress.places365_clip_filter import CLIP_FILTERED_CATEGORIES
from drone_stress.places365_finalize import _cell_str

MANIFEST_PATH_COLUMNS: tuple[str, ...] = (
    "output_path",
    "candidate_path",
    "review_path",
    "original_path",
    "source_path",
    "image_path",
    "filepath",
    "file_path",
    "path",
    "relative_path",
    "rel_path",
    "filename",
    "image_filename",
)


def _filename_from_row(row: pd.Series) -> str:
    for col in MANIFEST_PATH_COLUMNS:
        raw = _cell_str(row, col)
        if raw:
            name = Path(raw).name
            if name:
                return name
    return ""


def resolve_manifest_image_path(
    row: pd.Series,
    candidate_root: Path | None,
) -> Path | None:
    """Resolve manifest image path from absolute columns or candidate-root relative paths."""
    for col in MANIFEST_PATH_COLUMNS:
        raw = _cell_str(row, col)
        if not raw:
            continue
        path = Path(raw)
        if path.is_file():
            return path
        if candidate_root is not None and candidate_root.is_dir():
            rel = candidate_root / raw
            if rel.is_file():
                return rel

    filename = _filename_from_row(row)
    if not filename or candidate_root is None or not candidate_root.is_dir():
        return None

    mapped = _cell_str(
        row,
        "corrected_category",
        "vision_corrected_category",
        "final_category",
        "mapped_background_type",
    )
    for sub in (mapped, *CLIP_FILTERED_CATEGORIES):
        if not sub:
            continue
        candidate = candidate_root / sub / filename
        if candidate.is_file():
            return candidate

    direct = candidate_root / filename
    if direct.is_file():
        return direct
    return None
