"""Discover and load raw background, drone, and distractor assets."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def win_long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\") and len(resolved) >= 240:
        return "\\\\?\\" + resolved
    return resolved


def file_accessible(path: Path) -> bool:
    """True when a file can be read (handles Windows MAX_PATH)."""
    try:
        if path.is_file():
            return path.stat().st_size > 0
        if os.name == "nt":
            long_path = win_long_path(path)
            if os.path.isfile(long_path):
                return os.path.getsize(long_path) > 0
    except OSError:
        return False
    return False


def open_image_file(path: str | Path):
    """Open a binary image file handle (Windows long-path safe)."""
    resolved = Path(path)
    if os.name == "nt" and len(str(resolved.resolve())) >= 240:
        return open(win_long_path(resolved), "rb")
    return open(resolved, "rb")


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [
        p
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return files


def list_images_direct(directory: Path) -> list[Path]:
    """List image files directly under directory (non-recursive)."""
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# Flat-filename aliases (stem token -> configured class name)
DISTRACTOR_STEM_ALIASES: dict[str, str] = {
    "plane": "airplane",
    "aircraft": "airplane",
    "cloud": "cloud_blob",
    "cloudblob": "cloud_blob",
}


def infer_distractor_class_from_stem(stem: str, class_names: list[str]) -> str | None:
    """Infer distractor class from filename stem (flat-layout compatibility)."""
    stem_lower = stem.lower()
    class_set = set(class_names)

    if stem_lower in DISTRACTOR_STEM_ALIASES:
        alias = DISTRACTOR_STEM_ALIASES[stem_lower]
        if alias in class_set:
            return alias

    for name in sorted(class_names, key=len, reverse=True):
        token = name.lower()
        if stem_lower == token or stem_lower.startswith(f"{token}_"):
            return name
    return None


def index_by_subfolder(parent: Path, category_names: list[str]) -> dict[str, list[Path]]:
    """Map category name -> image paths from subfolders or flat parent (backgrounds)."""
    index: dict[str, list[Path]] = {name: [] for name in category_names}

    for name in category_names:
        subdir = parent / name
        if subdir.is_dir():
            index[name] = list_images(subdir)

    flat = list_images_direct(parent)
    flat_in_subdirs = {p for paths in index.values() for p in paths}
    orphan = [p for p in flat if p not in flat_in_subdirs]
    if orphan and not any(index.values()):
        for name in category_names:
            index[name] = orphan.copy()

    return index


# Filename stem hints for flat background files (not in class subfolders).
BACKGROUND_STEM_ALIASES: dict[str, str] = {
    "clear_sky": "clean_sky",
    "skyline": "urban_skyline",
    "sky_001": "clean_sky",
    "sky_002": "clean_sky",
}


def infer_background_type_from_path(
    path: Path,
    parent: Path,
    category_names: list[str],
) -> str:
    """
    Derive background_type from folder layout or controlled filename mapping.

    Returns a configured category name, or 'unknown' if not inferable.
    """
    try:
        rel = path.resolve().relative_to(parent.resolve())
        if len(rel.parts) >= 2:
            folder = rel.parts[0]
            if folder in category_names:
                return folder
    except ValueError:
        pass

    stem = path.stem.lower()
    if stem in BACKGROUND_STEM_ALIASES:
        alias = BACKGROUND_STEM_ALIASES[stem]
        if alias in category_names:
            return alias

    for name in sorted(category_names, key=len, reverse=True):
        token = name.lower()
        if stem == token or stem.startswith(f"{token}_") or token in stem:
            return name

    return "unknown"


def build_labeled_background_pool(
    parent: Path,
    category_names: list[str],
) -> list[tuple[str, Path]]:
    """
    Build (background_type, path) pairs.

    Subfolder files use the folder name as background_type.
    Flat files use controlled filename inference or 'unknown'.
    Never assigns background_type independently of the file.
    """
    if not parent.is_dir():
        return []

    pool: list[tuple[str, Path]] = []
    seen: set[Path] = set()

    for name in category_names:
        subdir = parent / name
        if subdir.is_dir():
            for path in list_images(subdir):
                resolved = path.resolve()
                if resolved not in seen:
                    pool.append((name, path))
                    seen.add(resolved)

    for path in list_images_direct(parent):
        resolved = path.resolve()
        if resolved in seen:
            continue
        bg_type = infer_background_type_from_path(path, parent, category_names)
        pool.append((bg_type, path))
        seen.add(resolved)

    return pool


def build_labeled_background_pool_from_metadata(
    images_dir: Path,
    metadata_csv: Path,
    category_names: list[str],
) -> list[tuple[str, Path]]:
    """
    Build (background_category, path) pairs from sky-approved metadata CSV.

    Uses background_category when valid; otherwise falls back to unknown.
    Only includes rows whose image file exists under images_dir.
    """
    if not metadata_csv.is_file():
        return build_labeled_background_pool(images_dir, category_names)

    df = pd.read_csv(metadata_csv, keep_default_na=False)
    pool: list[tuple[str, Path]] = []
    seen: set[str] = set()
    valid_cats = set(category_names)

    for _, row in df.iterrows():
        file_name = str(row.get("file_name", "")).strip()
        if not file_name:
            continue
        rel = str(row.get("relative_path", "")).strip()
        if rel:
            candidate = images_dir.parent / rel if rel.startswith("images/") else images_dir / file_name
        else:
            candidate = images_dir / file_name
        if not file_accessible(candidate):
            candidate = images_dir / file_name
        if not file_accessible(candidate):
            continue

        category = str(row.get("background_category", "unknown")).strip() or "unknown"
        if category not in valid_cats:
            category = "unknown" if "unknown" in valid_cats else category_names[0]

        key = str(candidate.resolve())
        if key not in seen:
            pool.append((category, candidate.resolve()))
            seen.add(key)

    if pool:
        return pool
    return build_labeled_background_pool(images_dir, category_names)


BackgroundSampling = str  # "uniform_by_file" | "balanced_by_type"


def pick_background(
    labeled_pool: list[tuple[str, Path]],
    rng,
    sampling: BackgroundSampling = "uniform_by_file",
    *,
    background_type: str | None = None,
) -> tuple[str, Path]:
    """
    Sample a background from a labeled pool.

    uniform_by_file (default): each (type, path) pair equally likely — types with
    more files are oversampled relative to types with fewer files.

    balanced_by_type: pick background_type uniformly, then a random file within that type.
    """
    if not labeled_pool:
        raise ValueError("Background pool is empty")

    if background_type is not None:
        matches = [(t, p) for t, p in labeled_pool if t == background_type]
        if matches:
            bg_type, bg_path = matches[rng.randrange(len(matches))]
            return bg_type, bg_path

    if sampling == "balanced_by_type":
        by_type: dict[str, list[Path]] = {}
        for bg_type, path in labeled_pool:
            by_type.setdefault(bg_type, []).append(path)
        available = [t for t, paths in by_type.items() if paths]
        chosen_type = rng.choice(available)
        return chosen_type, rng.choice(by_type[chosen_type])

    bg_type, bg_path = labeled_pool[rng.randrange(len(labeled_pool))]
    return bg_type, bg_path


def index_distractors_by_class(parent: Path, class_names: list[str]) -> dict[str, list[Path]]:
    """
    Map distractor class -> image paths.

    Preferred layout: data/raw/distractors/<class>/*.png

    Hybrid / flat layout: also assigns root-level files by filename stem
    (e.g. bird.jpg, plane.jpg -> airplane, kite_02.png).
    """
    index: dict[str, list[Path]] = {name: [] for name in class_names}

    for name in class_names:
        subdir = parent / name
        if subdir.is_dir():
            for path in list_images(subdir):
                if path not in index[name]:
                    index[name].append(path)

    for path in list_images_direct(parent):
        cls = infer_distractor_class_from_stem(path.stem, class_names)
        if cls is not None and path not in index[cls]:
            index[cls].append(path)

    return index


def require_non_empty(index: dict[str, list[Path]], label: str) -> None:
    if not any(index.values()):
        raise FileNotFoundError(
            f"No {label} images found. Expected subfolders like "
            f"{{category}}/*.jpg under the configured directory, or loose images."
        )


def require_distractor_classes(
    index: dict[str, list[Path]],
    class_names: list[str],
    distractors_dir: Path,
) -> None:
    """Fail clearly if a configured distractor class has no assets."""
    missing = [name for name in class_names if not index.get(name)]
    if missing:
        raise FileNotFoundError(
            f"No distractor assets for class(es): {', '.join(missing)}\n"
            f"Add images under:\n"
            + "\n".join(f"  {distractors_dir / name}/" for name in missing)
            + "\n"
            "Or use flat layout with filenames like bird_001.png, kite_02.png"
        )


def resolve_effective_qa_label(row) -> tuple[str, str]:
    """
    Authoritative QA label for generation filtering.

    Priority:
    1. qa_final_label (human-reviewed, when non-empty)
    2. quality_label
    3. mask_quality_label

    Returns (label, source_column). Label is lowercased.
    """
    for col in ("qa_final_label", "quality_label", "mask_quality_label"):
        value = str(row.get(col, "")).strip().lower()
        if value and value not in ("nan", "none"):
            return value, col
    return "", ""


def load_approved_assets_from_metadata(
    images_dir: Path,
    metadata_csv: Path | None,
    *,
    exclude_quality_labels: frozenset[str] | None = None,
    require_accept_label: bool = False,
) -> list[Path]:
    """
    Return asset PNG paths approved for synthetic generation.

    QA approval logic (explicit):
    - When qa_final_label is non-empty, it is authoritative.
    - Otherwise fall back to quality_label, then mask_quality_label.
    - Rows with extraction_failed=true are always skipped.
    - With require_accept_label=True, only rows whose effective label is
      exactly "accept" are included (recommended for assets_final packs).
    - Otherwise, rows whose effective label is in exclude_quality_labels
      (default: reject) are skipped; review rows are included for legacy pools.

    When metadata is missing, lists all images under images_dir.
    """
    if exclude_quality_labels is None:
        exclude_quality_labels = frozenset({"reject"})

    if metadata_csv is None or not metadata_csv.is_file():
        return list_images(images_dir)

    import pandas as pd

    df = pd.read_csv(metadata_csv, keep_default_na=False)
    paths: list[Path] = []
    seen: set[str] = set()

    for _, row in df.iterrows():
        if str(row.get("extraction_failed", "")).lower() in ("true", "1", "yes"):
            continue

        label, label_source = resolve_effective_qa_label(row)
        if label:
            if require_accept_label:
                if label != "accept":
                    continue
            elif label in exclude_quality_labels:
                continue
        elif require_accept_label:
            continue

        rel = str(row.get("relative_path", "")).strip()
        file_name = str(row.get("file_name", "")).strip()
        out = str(row.get("output_path", "")).strip()

        path: Path | None = None
        if rel:
            candidate = images_dir.parent / rel
            if file_accessible(candidate):
                path = candidate
        if path is None and file_name:
            candidate = images_dir / file_name
            if file_accessible(candidate):
                path = candidate
        if path is None and out:
            candidate = Path(out)
            if file_accessible(candidate):
                path = candidate
        if path is None:
            asset_id = str(row.get("asset_id", "")).strip()
            if asset_id:
                candidate = images_dir / f"{asset_id}.png"
                if file_accessible(candidate):
                    path = candidate
        if path is None:
            continue

        if not file_accessible(path):
            alt = images_dir / path.name
            if file_accessible(alt):
                path = alt
            else:
                continue
        key = str(path.resolve())
        if key not in seen:
            paths.append(path.resolve())
            seen.add(key)
    return sorted(paths)


def index_distractor_pools(
    pools: dict[str, tuple[Path, Path | None]],
    *,
    exclude_quality_labels: frozenset[str] | None = None,
    require_accept_label: bool = False,
) -> dict[str, list[Path]]:
    """Build class -> approved asset paths from per-class dirs + metadata CSVs."""
    index: dict[str, list[Path]] = {}
    for class_name, (asset_dir, metadata_csv) in pools.items():
        index[class_name] = load_approved_assets_from_metadata(
            asset_dir,
            metadata_csv,
            exclude_quality_labels=exclude_quality_labels,
            require_accept_label=require_accept_label,
        )
    return index
