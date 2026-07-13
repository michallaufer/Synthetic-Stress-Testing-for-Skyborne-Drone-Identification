#!/usr/bin/env python3
"""Filter candidate background images into curated sky/horizon scene folders.

Heuristic pre-filter for large background pools (e.g. SkyFinder). Optional
CLIP-assisted relabeling (--use-clip) is a weak second stage, not ground truth.

V1 outputs (with --use-clip):
  data/processed/backgrounds_approved/<category>/
  data/processed/backgrounds_holdout/night_low_light/
  data/processed/backgrounds_review/
  data/processed/backgrounds_reject/

See README.md — Background Filtering Strategy.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.assets import list_images, list_images_direct
from drone_stress.background_clip import (
    V1_APPROVED_CATEGORIES,
    V1_METADATA_COLUMNS,
    V1_NIGHT_LOW_LIGHT,
    ClipBackgroundScorer,
    ensure_v1_output_dirs,
    merge_heuristic_and_clip_background,
    v1_metadata_paths,
    v1_output_directory,
)
from drone_stress.background_filter import (
    BACKGROUND_CATEGORIES,
    METADATA_COLUMNS,
    BackgroundClassification,
    BackgroundFeatures,
    classify_background,
    compute_background_features,
    features_to_metadata_dict,
)
from drone_stress.scene_filter import load_image_rgb

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "backgrounds_curated"
DEFAULT_V1_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_SHEET_DIR = PROJECT_ROOT / "outputs" / "contact_sheets"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "reports" / "background_filter_report.txt"
DEFAULT_V1_REPORT = PROJECT_ROOT / "outputs" / "reports" / "background_filter_v1_report.txt"
METADATA_NAME = "background_metadata.csv"
THUMB_MAX = 320
FILTER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_OUTPUT_FILENAME_LEN = 120


@dataclass
class CopyResult:
    success: bool
    dest_path: str
    error: str
    mode: str


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def discover_images(input_dir: Path, *, recursive: bool) -> list[Path]:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if recursive:
        paths = list_images(input_dir)
    else:
        paths = list_images_direct(input_dir)
    return sorted(p for p in paths if p.suffix.lower() in FILTER_EXTENSIONS)


def _background_id(src: Path, index: int) -> str:
    digest = hashlib.sha256(str(src.resolve()).encode()).hexdigest()[:10]
    return f"bg_{index:06d}_{digest}"


def _sanitize_stem(stem: str, *, max_len: int = 60) -> str:
    safe = re.sub(r'[<>:"/\\|?*\s]+', "_", stem)
    safe = re.sub(r"_+", "_", safe).strip("._")
    if not safe:
        safe = "image"
    return safe[:max_len]


def _safe_output_filename(background_id: str, src: Path) -> str:
    suffix = src.suffix.lower()
    if suffix not in FILTER_EXTENSIONS:
        suffix = ".jpg"
    stem = _sanitize_stem(src.stem)
    name = f"{background_id}_{stem}{suffix}"
    if len(name) > MAX_OUTPUT_FILENAME_LEN:
        digest = hashlib.sha256(str(src.resolve()).encode()).hexdigest()[:8]
        name = f"{background_id}_{digest}{suffix}"
    return name


def _unique_dest_path(dest_dir: Path, filename: str) -> Path:
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    digest = hashlib.sha256(f"{dest_dir}/{filename}".encode()).hexdigest()[:6]
    candidate = dest_dir / f"{stem}_{digest}{suffix}"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = dest_dir / f"{stem}_{digest}_{n:03d}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def _copy_or_link(src: Path, dest: Path, mode: str) -> CopyResult:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "copy":
            shutil.copy2(src, dest)
        elif mode == "symlink":
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src.resolve())
        else:
            raise ValueError(f"Unsupported mode: {mode!r}")
        return CopyResult(
            success=True,
            dest_path=str(dest.resolve()),
            error="",
            mode=mode,
        )
    except OSError as exc:
        print(f"COPY FAILED: {src} -> {dest}: {exc}", file=sys.stderr)
        return CopyResult(success=False, dest_path="", error=str(exc), mode=mode)


def _resize_thumb(rgb: np.ndarray, max_side: int = THUMB_MAX) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return rgb
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _tile_title_heuristic(row: pd.Series) -> str:
    name = Path(str(row["output_path"])).name if row.get("output_path") else "?"
    return "\n".join(
        [
            name,
            str(row.get("background_type", row.get("heuristic_background_type", "?"))),
            f"sky_upper={float(row['sky_ratio_upper']):.2f}",
            str(row.get("filter_reason", ""))[:48],
        ]
    )


def _tile_title_v1(row: pd.Series) -> str:
    name = Path(str(row["output_path"])).name if row.get("output_path") else "?"
    h_type = row.get("heuristic_background_type", row.get("background_type", "?"))
    c_type = row.get("clip_background_type", "?")
    margin = float(row.get("clip_margin", 0.0))
    return "\n".join(
        [
            name,
            f"{row.get('final_background_type', '?')} | {row.get('final_filter_status', '?')}",
            f"{h_type} -> {c_type}",
            f"{row.get('merge_reason', '')} | margin={margin:.3f}",
        ]
    )


def build_category_contact_sheet(
    df: pd.DataFrame,
    output_path: Path,
    *,
    cols: int = 6,
    max_tiles: int = 24,
    seed: int = 42,
    title_fn=None,
    path_column: str = "output_path",
) -> Path | None:
    if df.empty:
        return None
    if title_fn is None:
        title_fn = _tile_title_heuristic

    subset = df if len(df) <= max_tiles else df.sample(n=max_tiles, random_state=seed)
    n = len(subset)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.9))
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    elif rows == 1:
        axes_flat = list(axes)
    elif cols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i >= n:
            ax.set_visible(False)
            continue
        row = subset.iloc[i]
        path = Path(str(row[path_column])) if row.get(path_column) else None
        if path is None or not path.is_file():
            ax.set_title(f"missing\n{row.get('background_id', '?')}", fontsize=7)
            continue
        try:
            rgb = load_image_rgb(path)
        except OSError:
            ax.set_title(f"unreadable\n{row.get('background_id', '?')}", fontsize=7)
            continue
        ax.imshow(_resize_thumb(rgb))
        ax.set_title(title_fn(row), fontsize=6, loc="left", pad=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_filter_report(df: pd.DataFrame, report_path: Path, *, total_scanned: int) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Background Filter Report (heuristic)",
        "====================================",
        f"Total images scanned: {total_scanned}",
        f"Total images processed: {len(df)}",
        "",
        "Copied per category:",
    ]
    for category in BACKGROUND_CATEGORIES:
        count = int((df["background_type"] == category).sum())
        if count:
            lines.append(f"  {category}: {count}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_v1_filter_report(df: pd.DataFrame, report_path: Path, *, total_scanned: int) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    approved = df[df["final_use_split"] == "approved_daytime"]
    lines = [
        "Background Filter V1 Report",
        "===========================",
        f"Total images scanned: {total_scanned}",
        f"Total images processed: {len(df)}",
        "",
        "By heuristic_background_type:",
    ]
    col = "heuristic_background_type" if "heuristic_background_type" in df.columns else "background_type"
    for label, count in df[col].value_counts().sort_index().items():
        lines.append(f"  {label}: {count}")
    lines.append("\nBy clip_background_type:")
    for label, count in df["clip_background_type"].value_counts().sort_index().items():
        if label:
            lines.append(f"  {label}: {count}")
    lines.append("\nBy final_background_type:")
    for label, count in df["final_background_type"].value_counts().sort_index().items():
        lines.append(f"  {label}: {count}")
    lines.append("\nBy final_filter_status:")
    for label, count in df["final_filter_status"].value_counts().sort_index().items():
        lines.append(f"  {label}: {count}")
    lines.append("\nBy final_use_split:")
    for label, count in df["final_use_split"].value_counts().sort_index().items():
        lines.append(f"  {label}: {count}")
    lines.append("\nBy merge_reason:")
    for reason, count in df["merge_reason"].value_counts().sort_index().items():
        lines.append(f"  {reason}: {count}")
    lines.extend(
        [
            "",
            f"Approved daytime backgrounds: {len(approved)}",
            f"Holdout night backgrounds: {int((df['final_use_split'] == 'holdout_night').sum())}",
            f"Review backgrounds: {int((df['final_use_split'] == 'review').sum())}",
            f"Reject backgrounds: {int((df['final_use_split'] == 'reject').sum())}",
            "",
            "Approved daytime by category:",
        ]
    )
    for cat in V1_APPROVED_CATEGORIES:
        count = int((approved["final_background_type"] == cat).sum())
        if count:
            lines.append(f"  {cat}: {count}")

    if "copy_success" in df.columns:
        copy_ok = int(df["copy_success"].astype(bool).sum())
        copy_fail = int((~df["copy_success"].astype(bool)).sum())
        skipped = int(df.get("skipped_due_to_copy_error", pd.Series(dtype=bool)).astype(bool).sum())
        lines.extend(
            [
                "",
                "Copy operations:",
                f"  copy_success: {copy_ok}",
                f"  copy_error: {copy_fail}",
            ]
        )
        if copy_fail and "copy_error" in df.columns:
            lines.append("\nTop copy errors:")
            err_counts = (
                df.loc[~df["copy_success"].astype(bool), "copy_error"]
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .head(10)
            )
            for err, count in err_counts.items():
                lines.append(f"  [{count}] {err}")
        if skipped:
            lines.append(
                f"\nWARNING: {skipped} image(s) skipped due to copy failure "
                "(see copy_error column in metadata)."
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def print_summary(df: pd.DataFrame, *, total_scanned: int, v1_mode: bool) -> None:
    print(f"\nScanned {total_scanned} image(s), processed {len(df)}.")
    if not v1_mode:
        print("\nBy background_type (heuristic):")
        for label in BACKGROUND_CATEGORIES:
            count = int((df["background_type"] == label).sum())
            if count:
                print(f"  {label}: {count}")
        return

    approved = int((df["final_use_split"] == "approved_daytime").sum())
    holdout = int((df["final_use_split"] == "holdout_night").sum())
    review = int((df["final_use_split"] == "review").sum())
    reject = int((df["final_use_split"] == "reject").sum())
    print(f"\napproved_daytime: {approved}")
    print(f"holdout_night: {holdout}")
    print(f"review: {review}")
    print(f"reject: {reject}")
    print("\nApproved by category:")
    approved_df = df[df["final_use_split"] == "approved_daytime"]
    for cat in V1_APPROVED_CATEGORIES:
        count = int((approved_df["final_background_type"] == cat).sum())
        if count:
            print(f"  {cat}: {count}")
    if "copy_success" in df.columns:
        copy_fail = int((~df["copy_success"].astype(bool)).sum())
        if copy_fail:
            print(f"\ncopy_error: {copy_fail} (metadata retained; see copy_error column)")


def _corrupt_features() -> BackgroundFeatures:
    return BackgroundFeatures(
        image_width=0,
        image_height=0,
        sky_ratio_upper=0.0,
        sky_ratio_full=0.0,
        blue_sky_ratio=0.0,
        gray_white_cloud_ratio=0.0,
        lower_green_ratio=0.0,
        lower_dark_structure_ratio=0.0,
        lower_texture_score=0.0,
        mean_brightness=0.0,
    )


def _empty_feature_record() -> dict:
    return features_to_metadata_dict(_corrupt_features())


def _ensure_report_and_sheet_dirs(report_path: Path, sheet_dir: Path | None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if sheet_dir is not None:
        sheet_dir.mkdir(parents=True, exist_ok=True)


def _contact_sheet_ready_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a successfully copied output file (exclude copy_error rows)."""
    if df.empty:
        return df
    if "copy_success" in df.columns:
        mask = df["copy_success"].astype(bool)
        if "output_path" in df.columns:
            mask &= df["output_path"].astype(str).str.len() > 0
        ready = df[mask].copy()
        ready = ready[
            ready["output_path"].astype(str).apply(lambda p: bool(p) and Path(p).is_file())
        ]
        return ready
    if "output_path" not in df.columns:
        return df.iloc[0:0]
    return df[df["output_path"].astype(str).apply(lambda p: bool(p) and Path(p).is_file())]


def _build_heuristic_records(
    image_paths: list[Path],
    *,
    copy_during_heuristic: bool,
    output_dir: Path,
    mode: str,
) -> list[dict]:
    if copy_during_heuristic:
        for category in BACKGROUND_CATEGORIES:
            (output_dir / category).mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for index, src in enumerate(image_paths):
        base = {
            "background_id": _background_id(src, index),
            "original_path": str(src.resolve()),
            "output_path": "",
            "_src_path": src,
        }
        try:
            rgb = load_image_rgb(src)
        except OSError:
            records.append(
                {
                    **base,
                    "background_type": "reject",
                    "filter_status": "reject",
                    "filter_reason": "corrupt_or_unreadable",
                    **_empty_feature_record(),
                    "_rgb": None,
                    "_features": None,
                }
            )
            continue

        features = compute_background_features(rgb)
        result: BackgroundClassification = classify_background(features)
        output_path = ""
        if copy_during_heuristic:
            dest_dir = output_dir / result.background_type
            safe_name = _safe_output_filename(base["background_id"], src)
            dest_path = _unique_dest_path(dest_dir, safe_name)
            copy_result = _copy_or_link(src, dest_path, mode)
            if copy_result.success:
                output_path = copy_result.dest_path

        records.append(
            {
                **base,
                "background_type": result.background_type,
                "filter_status": result.filter_status,
                "filter_reason": result.filter_reason,
                "output_path": output_path,
                **features_to_metadata_dict(features),
                "_rgb": rgb,
                "_features": features,
            }
        )
    return records


def _record_to_v1_row(
    rec: dict,
    merge,
    clip,
    output_path: str,
    *,
    copy_result: CopyResult,
) -> dict:
    feat = {k: v for k, v in rec.items() if k in features_to_metadata_dict(_corrupt_features())}
    row = {
        "background_id": rec["background_id"],
        "original_path": rec["original_path"],
        "output_path": output_path,
        "heuristic_background_type": rec["background_type"],
        "heuristic_filter_status": rec["filter_status"],
        **feat,
    }
    row.update(clip.to_dict())
    row["final_background_type"] = merge.final_background_type
    row["final_filter_status"] = merge.final_filter_status
    row["final_use_split"] = merge.final_use_split
    row["merge_reason"] = merge.merge_reason
    row["semantic_category_notes"] = merge.semantic_category_notes
    row["copy_success"] = copy_result.success
    row["copy_error"] = copy_result.error
    row["copy_mode"] = copy_result.mode
    row["skipped_due_to_copy_error"] = not copy_result.success
    if not copy_result.success:
        row["final_use_split"] = "copy_error"
        row["final_filter_status"] = "review"
        row["output_path"] = ""
        row["skipped_due_to_copy_error"] = True
    return row


def _apply_clip_merge_and_copy_v1(
    records: list[dict],
    *,
    v1_root: Path,
    mode: str,
    clip_scorer: ClipBackgroundScorer,
    clip_margin_threshold: float,
) -> list[dict]:
    ensure_v1_output_dirs(v1_root)

    rgb_indices = [i for i, r in enumerate(records) if r.get("_rgb") is not None]
    clip_results: list = [None] * len(records)
    for start in range(0, len(rgb_indices), clip_scorer.batch_size):
        batch_idx = rgb_indices[start : start + clip_scorer.batch_size]
        rgbs = [records[i]["_rgb"] for i in batch_idx]
        scored = clip_scorer.score_rgb_batch(rgbs)
        for i, clip in zip(batch_idx, scored):
            clip_results[i] = clip

    output_records: list[dict] = []
    for i, rec in enumerate(records):
        features = rec.get("_features")
        clip = clip_results[i]
        if clip is None or features is None:
            from drone_stress.background_clip import ClipBackgroundScoreResult

            clip = ClipBackgroundScoreResult.unused()
            merge = merge_heuristic_and_clip_background(
                rec["background_type"],
                rec["filter_status"],
                clip,
                features if features is not None else _corrupt_features(),
                clip_margin_threshold,
                heuristic_reason=rec.get("filter_reason", ""),
            )
        else:
            merge = merge_heuristic_and_clip_background(
                rec["background_type"],
                rec["filter_status"],
                clip,
                features,
                clip_margin_threshold,
                heuristic_reason=rec.get("filter_reason", ""),
            )

        output_path = ""
        copy_result = CopyResult(success=False, dest_path="", error="source_missing", mode=mode)
        src = rec["_src_path"]
        if src.is_file():
            dest_dir = v1_output_directory(v1_root, merge)
            safe_name = _safe_output_filename(rec["background_id"], src)
            dest_path = _unique_dest_path(dest_dir, safe_name)
            copy_result = _copy_or_link(src, dest_path, mode)
            if copy_result.success:
                output_path = copy_result.dest_path

        output_records.append(
            _record_to_v1_row(rec, merge, clip, output_path, copy_result=copy_result)
        )

    return output_records


def _records_from_existing_metadata(metadata_path: Path) -> tuple[list[dict], int]:
    df = pd.read_csv(metadata_path)
    if "original_path" not in df.columns:
        raise ValueError(f"Metadata missing original_path: {metadata_path}")

    records: list[dict] = []
    for index, row in df.iterrows():
        src = Path(str(row["original_path"]))
        h_type = str(
            row.get("heuristic_background_type", row.get("background_type", "review"))
        )
        h_status = str(
            row.get("heuristic_filter_status", row.get("filter_status", "review"))
        )
        base = {
            "background_id": str(row.get("background_id", _background_id(src, int(index)))),
            "original_path": str(src),
            "background_type": h_type,
            "filter_status": h_status,
            "filter_reason": str(row.get("filter_reason", "relabel_from_metadata")),
            "_src_path": src,
        }
        feat_cols = _empty_feature_record()
        for col in feat_cols:
            if col in row and not pd.isna(row[col]):
                feat_cols[col] = row[col]

        try:
            rgb = load_image_rgb(src)
            features = compute_background_features(rgb)
            feat_cols = features_to_metadata_dict(features)
            records.append({**base, **feat_cols, "_rgb": rgb, "_features": features})
        except OSError:
            records.append(
                {
                    **base,
                    **feat_cols,
                    "background_type": "reject",
                    "filter_status": "reject",
                    "filter_reason": "corrupt_or_unreadable",
                    "_rgb": None,
                    "_features": None,
                }
            )
    return records, len(records)


def filter_backgrounds_heuristic(
    input_dir: Path,
    output_dir: Path,
    *,
    mode: str,
    max_images: int | None,
    recursive: bool,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    image_paths = discover_images(input_dir, recursive=recursive)
    total_scanned = len(image_paths)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {input_dir}")

    if max_images is not None and len(image_paths) > max_images:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(image_paths), size=max_images, replace=False)
        image_paths = [image_paths[i] for i in sorted(idx)]

    output_dir.mkdir(parents=True, exist_ok=True)
    records = _build_heuristic_records(
        image_paths,
        copy_during_heuristic=True,
        output_dir=output_dir,
        mode=mode,
    )
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    return pd.DataFrame(clean), total_scanned


def filter_backgrounds_with_clip(
    input_dir: Path | None,
    *,
    v1_root: Path,
    mode: str,
    max_images: int | None,
    recursive: bool,
    seed: int,
    clip_scorer: ClipBackgroundScorer,
    clip_margin_threshold: float,
    relabel_metadata: Path | None,
) -> tuple[pd.DataFrame, int]:
    if relabel_metadata is not None:
        records, total_scanned = _records_from_existing_metadata(relabel_metadata)
    else:
        if input_dir is None:
            raise ValueError("--input-dir is required unless --relabel-existing-metadata is set")
        image_paths = discover_images(input_dir, recursive=recursive)
        total_scanned = len(image_paths)
        if not image_paths:
            raise FileNotFoundError(f"No images found under {input_dir}")
        if max_images is not None and len(image_paths) > max_images:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(image_paths), size=max_images, replace=False)
            image_paths = [image_paths[i] for i in sorted(idx)]
        records = _build_heuristic_records(
            image_paths,
            copy_during_heuristic=False,
            output_dir=PROJECT_ROOT,
            mode=mode,
        )

    output_records = _apply_clip_merge_and_copy_v1(
        records,
        v1_root=v1_root,
        mode=mode,
        clip_scorer=clip_scorer,
        clip_margin_threshold=clip_margin_threshold,
    )
    return pd.DataFrame(output_records), total_scanned


def _write_v1_metadata(df: pd.DataFrame, v1_root: Path) -> None:
    for col in V1_METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    export = df[V1_METADATA_COLUMNS]
    global_path, approved_path = v1_metadata_paths(v1_root)
    global_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(global_path, index=False)
    print(f"Wrote global metadata: {global_path}")

    approved = export[export["final_use_split"] == "approved_daytime"]
    approved_path.parent.mkdir(parents=True, exist_ok=True)
    approved.to_csv(approved_path, index=False)
    print(f"Wrote approved metadata: {approved_path}")


def _write_v1_contact_sheets(df: pd.DataFrame, sheet_dir: Path, *, max_tiles: int, seed: int) -> None:
    sheet_dir.mkdir(parents=True, exist_ok=True)
    ready = _contact_sheet_ready_df(df)
    approved = ready[ready["final_use_split"] == "approved_daytime"]
    for cat in V1_APPROVED_CATEGORIES:
        cat_df = approved[approved["final_background_type"] == cat]
        if cat_df.empty:
            continue
        path = sheet_dir / f"background_approved_{cat}.png"
        saved = build_category_contact_sheet(
            cat_df, path, max_tiles=max_tiles, seed=seed, title_fn=_tile_title_v1
        )
        if saved:
            print(f"Saved contact sheet: {saved}")

    holdout = ready[ready["final_use_split"] == "holdout_night"]
    if not holdout.empty:
        saved = build_category_contact_sheet(
            holdout,
            sheet_dir / "background_holdout_night_low_light.png",
            max_tiles=max_tiles,
            seed=seed,
            title_fn=_tile_title_v1,
        )
        if saved:
            print(f"Saved contact sheet: {saved}")

    review = ready[ready["final_use_split"] == "review"]
    if not review.empty:
        saved = build_category_contact_sheet(
            review,
            sheet_dir / "background_review_sample.png",
            max_tiles=max_tiles,
            seed=seed,
            title_fn=_tile_title_v1,
        )
        if saved:
            print(f"Saved contact sheet: {saved}")

    reject = ready[ready["final_use_split"] == "reject"]
    if not reject.empty:
        saved = build_category_contact_sheet(
            reject,
            sheet_dir / "background_reject_sample.png",
            max_tiles=max_tiles,
            seed=seed,
            title_fn=_tile_title_v1,
        )
        if saved:
            print(f"Saved contact sheet: {saved}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter candidate backgrounds (heuristic + optional CLIP relabeling)."
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Heuristic-only output dir")
    parser.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--make-contact-sheets", action="store_true")
    parser.add_argument("--contact-sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sheet-tiles", type=int, default=24)
    parser.add_argument("--use-clip", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--clip-margin-threshold", type=float, default=0.04)
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--relabel-existing-metadata", type=Path, default=None)
    parser.add_argument(
        "--v1-output-root",
        type=Path,
        default=None,
        help="V1 output root (default: data/processed). Use a short path on Windows to avoid MAX_PATH.",
    )
    args = parser.parse_args()

    v1_mode = args.use_clip or args.relabel_existing_metadata is not None
    if v1_mode:
        args.use_clip = True

    if not v1_mode and args.input_dir is None:
        parser.error("--input-dir is required for heuristic-only mode")
    if v1_mode and args.input_dir is None and args.relabel_existing_metadata is None:
        parser.error("--input-dir or --relabel-existing-metadata is required with --use-clip")

    input_dir = _resolve_path(args.input_dir) if args.input_dir else None
    relabel_path = (
        _resolve_path(args.relabel_existing_metadata) if args.relabel_existing_metadata else None
    )
    heuristic_output = _resolve_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT
    v1_root = _resolve_path(args.v1_output_root) if args.v1_output_root else DEFAULT_V1_OUTPUT_ROOT

    print(f"Mode:   {args.mode}")
    print(f"CLIP:   {args.use_clip}")
    if v1_mode:
        print(f"V1 output root: {v1_root}")
        print("  backgrounds_{approved,holdout,review,reject}/ under that root")
    else:
        print(f"Output: {heuristic_output}")
    if input_dir:
        print(f"Input:  {input_dir}")
        print(f"Recursive: {args.recursive}")
    if relabel_path:
        print(f"Relabel metadata: {relabel_path}")
    if args.max_images is not None:
        print(f"Max images: {args.max_images} (seed={args.seed})")

    report_path = _resolve_path(args.report_output or (DEFAULT_V1_REPORT if v1_mode else DEFAULT_REPORT))
    sheet_dir = _resolve_path(args.contact_sheet_dir) if args.make_contact_sheets else None
    _ensure_report_and_sheet_dirs(report_path, sheet_dir)
    if v1_mode:
        ensure_v1_output_dirs(v1_root)

    if v1_mode:
        print(f"Loading CLIP model: {args.clip_model}")
        clip_scorer = ClipBackgroundScorer(
            model_name=args.clip_model,
            device=args.clip_device,
            batch_size=args.clip_batch_size,
        )
        df, total_scanned = filter_backgrounds_with_clip(
            input_dir,
            v1_root=v1_root,
            mode=args.mode,
            max_images=args.max_images,
            recursive=args.recursive,
            seed=args.seed,
            clip_scorer=clip_scorer,
            clip_margin_threshold=args.clip_margin_threshold,
            relabel_metadata=relabel_path,
        )
        _write_v1_metadata(df, v1_root)
        write_v1_filter_report(df, report_path, total_scanned=total_scanned)
    else:
        df, total_scanned = filter_backgrounds_heuristic(
            input_dir,
            heuristic_output,
            mode=args.mode,
            max_images=args.max_images,
            recursive=args.recursive,
            seed=args.seed,
        )
        metadata_path = heuristic_output / METADATA_NAME
        for col in METADATA_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df[METADATA_COLUMNS].to_csv(metadata_path, index=False)
        print(f"Wrote metadata: {metadata_path}")
        write_filter_report(df, report_path, total_scanned=total_scanned)

    print_summary(df, total_scanned=total_scanned, v1_mode=v1_mode)
    print(f"Wrote report: {report_path}")

    if args.make_contact_sheets:
        sheet_dir = _resolve_path(args.contact_sheet_dir)
        if v1_mode:
            _write_v1_contact_sheets(
                df, sheet_dir, max_tiles=args.max_sheet_tiles, seed=args.seed
            )
        else:
            for category in BACKGROUND_CATEGORIES:
                cat_df = df[df["background_type"] == category]
                if cat_df.empty:
                    continue
                saved = build_category_contact_sheet(
                    cat_df,
                    sheet_dir / f"background_{category}.png",
                    max_tiles=args.max_sheet_tiles,
                    seed=args.seed,
                )
                if saved:
                    print(f"Saved contact sheet: {saved}")


if __name__ == "__main__":
    main()
