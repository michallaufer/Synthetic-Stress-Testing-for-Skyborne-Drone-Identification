# Dataset and weight links

Fill in your Google Drive (or other) URLs before publishing the repo.

## Benchmark dataset

- **Name:** `full_curated_v1` (3600 synthetic images)
- **Local path (after download):** `data/synthetic/full_curated_v1/`
- **Drive link:** _TODO — paste shared Drive folder/zip URL_

Expected contents after extract:

```text
images/          # img_*.png
labels/          # YOLO .txt (empty for distractor-only)
metadata.csv
```

## Fine-tuned checkpoints (not in Git)

| Model | Local path after download | Drive link |
|-------|---------------------------|------------|
| YOLO11n | `outputs/training/yolo/yolo11n_drone_colab-9/weights/best.pt` | _TODO_ |
| RT-DETR-L | `outputs/training/rtdetr/rtdetr_l_drone_colab-9/weights/best.pt` | _TODO_ |

## Colab result zips (optional re-import)

| Zip | Purpose |
|-----|---------|
| `yolo11n_drone_colab-9-*.zip` | Training run + weights |
| `rtdetr_l_drone_colab-9-*.zip` | Training run + weights |
| `final_clean_full_curated_v1-*.zip` | Evaluation metrics / predictions |

Import:

```powershell
python scripts/utils/import_final_clean_colab_results.py --results-dir <folder-with-zips>
```

Published summaries already live under `results/` (no weights).
