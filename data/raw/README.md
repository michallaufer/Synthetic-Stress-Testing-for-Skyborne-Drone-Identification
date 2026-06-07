# Raw asset folders (pilot)

Place small manual pilot assets here. Large downloads stay out of git (see `.gitignore`).

## Layout

```text
data/raw/backgrounds/
  clean_sky/
  cloudy_sky/
  urban_skyline/
  trees_sky/

data/raw/drones/
  *.png          # RGBA crops preferred

data/raw/distractors/
  bird/           # only bird assets
  airplane/       # only airplane assets
  kite/           # only kite assets
  cloud_blob/     # only cloud-blob assets
```

**Distractor rule:** each class folder must contain only assets for that class. The generator picks a class, then picks a file **only from that folder**, and writes matching `distractor_classes` and `asset_id` in `metadata.csv`.

**Hybrid / flat layout:** you may also place files directly under `distractors/` (e.g. `bird.jpg`, `plane.jpg` → `airplane`, `kite.jpg`). Stems should match the class name or common aliases (`plane` → `airplane`, `cloud` → `cloud_blob`).

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`

Pilot targets (from README): ~10–50 backgrounds, ~5–20 drone assets, ~5–20 distractor assets **per class** as available.
