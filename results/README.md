# Published results (GitHub-safe)

From the **final-clean Colab** experiment. **No model weights** here.

## Authoritative metrics

Use:

`combined_metrics/summary_by_model_by_eval_subset_CORRECTED.csv`

Do **not** report from `summary_by_model_by_eval_subset.csv` (uncorrected YOLO recall denominator).

## Layout

```text
combined_metrics/     # Summary tables
training/             # args.yaml + results.csv only (no best.pt)
predictions/          # Per-subset prediction CSVs for three models
```

Full narrative: `docs/final_clean_experiment_results.md`
