"""Create Ultralytics YOLO-format train/val/test splits from synthetic datasets."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import yaml


def _copy_pair(image_src: Path, label_src: Path, image_dst: Path, label_dst: Path) -> None:
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_src, image_dst)
    if label_src.is_file():
        shutil.copy2(label_src, label_dst)
    else:
        label_dst.write_text("", encoding="utf-8")


def create_ultralytics_split(
    dataset_root: Path,
    output_root: Path,
    *,
    metadata_csv: Path | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_images: int | None = None,
    stratify_col: str = "subset",
) -> dict:
    """
    Copy images/labels into train/val/test folders and write dataset.yaml + split_metadata.csv.

    WARNING: splitting full_curated_v1 and training on it leaks into benchmark evaluation.
    """
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    meta_path = metadata_csv or (dataset_root / "metadata.csv")
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found: {meta_path}")

    df = pd.read_csv(meta_path)
    if max_images is not None and len(df) > max_images:
        df = df.sample(n=max_images, random_state=seed).reset_index(drop=True)

    if stratify_col in df.columns:
        parts: list[pd.DataFrame] = []
        for _, grp in df.groupby(stratify_col, dropna=False):
            parts.append(_random_split_frame(grp, train_ratio, val_ratio, test_ratio, seed))
        split_df = pd.concat(parts, ignore_index=True)
    else:
        split_df = _random_split_frame(df, train_ratio, val_ratio, test_ratio, seed)

    output_root.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        for sub in split_df[split_df["split_name"] == split_name].itertuples():
            image_id = str(sub.image_id)
            stem = Path(image_id).stem
            src_img = images_dir / image_id
            src_lbl = labels_dir / f"{stem}.txt"
            dst_img = output_root / "images" / split_name / image_id
            dst_lbl = output_root / "labels" / split_name / f"{stem}.txt"
            if not src_img.is_file():
                raise FileNotFoundError(f"Missing image: {src_img}")
            _copy_pair(src_img, src_lbl, dst_img, dst_lbl)

    split_csv = output_root / "split_metadata.csv"
    split_df.to_csv(split_csv, index=False)

    dataset_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "drone"},
    }
    yaml_path = output_root / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    counts = split_df["split_name"].value_counts().to_dict()
    return {
        "output_root": str(output_root.resolve()),
        "dataset_yaml": str(yaml_path.resolve()),
        "split_metadata_csv": str(split_csv.resolve()),
        "counts": counts,
        "total": len(split_df),
        "source": str(dataset_root.resolve()),
    }


def _random_split_frame(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    names: list[str] = []
    names.extend(["train"] * n_train)
    names.extend(["val"] * n_val)
    names.extend(["test"] * (n - n_train - n_val))
    out = shuffled.copy()
    out["split_name"] = names
    return out
