# Data

Large datasets are **not** stored in Git. This folder documents layout and ships **tiny samples** only.

## In this repo

| Path | Contents |
|------|----------|
| `sample_images/` | A few synthetic example PNGs from the benchmark |
| `sample_metadata/split_metadata_sample.csv` | First rows of a train/val/test split metadata file |
| `dataset_links.md` | Where to get the full dataset / weights (Drive) |

## Local / Drive (gitignored)

| Path | Contents |
|------|----------|
| `raw/` | Source backgrounds, drones, distractors |
| `processed/` | SAM2/Gemini cutouts, curated backgrounds (`assets_final`, `backgrounds_sky_approved`) |
| `synthetic/full_curated_v1/` | Full 3600-image stress benchmark |
| `training/` | YOLO-format splits for Colab training |

See root `README.md` and `docs/github_vs_drive.md`.
