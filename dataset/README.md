# Dataset

Keep candidate image links in [`links.csv`](links.csv). Use one row per image. `link` can be a direct image URL or the page where you found it. Set `label` to `drone` or `no_drone`, or leave it blank until reviewed. `source_group` identifies photos from the same photographer, shoot, or collection. Record the license or permission when known, and leave `split` blank until the images are organized.

Downloaded images go in the ignored `data/` directory:

```text
data/
  incoming/                 # images awaiting review
  train/
    drone/                  # any visible drone
    no_drone/               # birds, airplanes, skies, and other negatives
  val/                      # same class folders, when available
  test/                     # same class folders, when available
```

Keep duplicates, crops, and photos from the same shoot in one split. The current training script reads `data/train/`; validation and test sets are reserved for later evaluation.
