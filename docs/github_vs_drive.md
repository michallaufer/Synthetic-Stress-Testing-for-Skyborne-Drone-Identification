# GitHub vs Google Drive

Training ran on **Colab** (GPU). Large artifacts live on **Drive**. This repo keeps **code + configs + published results**.

## Put on GitHub

| Item | Path in this repo |
|------|-------------------|
| Code (`code/scripts/`, `code/src/`) | repo under `code/` |
| Colab notebook | `code/notebooks/colab_training_final_clean.ipynb` (+ root `run_gpu.ipynb`) |
| Configs | `configs/` |
| Corrected metrics | `results/combined_metrics/*CORRECTED*.csv` |
| Training logs (no weights) | `results/training/*/args.yaml`, `results.csv` |
| Prediction CSVs | `results/predictions/` |
| Key plots / QA sheets | `visuals/` |
| Experiment write-up | `docs/final_clean_experiment_results.md` |
| Sample images + metadata | `data/sample_images/`, `data/sample_metadata/` |
| How to get the full dataset | `data/dataset_links.md` |

## Keep on Drive only (do **not** commit)

| Item | Why |
|------|-----|
| `*.pt` / `*.pth` weights (`best.pt`, GroundingDINO checkpoint) | Large; regenerable / downloadable |
| Full `data/synthetic/full_curated_v1/images/` | ~3600 PNGs |
| Full `data/processed/` assets & backgrounds | Large source pools |
| Full train/val/test image splits | Large |
| `external/GroundingDINO`, SAM2 clones | Third-party; install via scripts |
| Raw Colab `outputs/weights/` | Same as checkpoints |
| Contact-sheet dumps under `outputs/contact_sheets/` | Optional local QA only |

## Drive folder vs this repo

The Drive tree with `configs/generation/`, `src/data_generation/`, etc. is an **aspirational** layout. This repository’s real layout is documented in the root `README.md` (Repository layout). Map Drive training zips → local `outputs/` via:

```powershell
python scripts/utils/import_final_clean_colab_results.py --results-dir <path-to-zips>
```

Then re-copy published files into `results/` / `visuals/` if needed.
