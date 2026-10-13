# Drone Presence Classification

Analyze a still image to answer one question: **Is a drone visible?**

## Input and output

- **Input:** One JPEG or PNG image.
- **Output:** `contains_drone` (`true` or `false`).
- Birds, airplanes, and empty skies are examples of images without a drone.

Example:

```json
{
  "contains_drone": true
}
```

## PyTorch starter

This is a whole-image classification baseline. It needs labeled images before it can make useful predictions; it does not locate drones within an image yet.

Record candidate image links in [`dataset/links.csv`](dataset/links.csv), then put reviewed training images in `data/train/drone/` or `data/train/no_drone/`. See the [dataset guide](dataset/README.md) for the folder layout. Then run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python predict.py path/to/image.jpg
```

Training saves the checkpoint to `artifacts/model.pt`. Both `data/` and `artifacts/` are ignored by Git.
