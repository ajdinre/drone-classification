# Dataset

Keep candidate image links in [`links.csv`](links.csv). Use one row per image. `link` can be a direct image URL or the page where you found it. Use a class folder name from [`model.py`](../model.py) for `label`; leave it blank until the model is confirmed. `source_group` identifies photos from the same photographer, shoot, or collection. Record the license or permission when known, and leave `split` blank until the images are organized.

Downloaded images go in the ignored `data/` directory:

```text
data/
  incoming/                 # images awaiting review
  train/
    no_drone/               # images without a drone
    unknown_drone/          # drones outside the ten target models
    dji_mini_4_pro/         # one folder per target model
    dji_air_3/
    ...
  val/                      # same class folders, when available
  test/                     # same class folders, when available
```

Only put a photo in an exact-model folder when the model is confirmed. Keep duplicates, crops, and photos from the same shoot in one split. The current training script reads `data/train/`; validation and test sets are reserved for later evaluation.
