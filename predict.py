import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from model import LABELS, build_model, transform


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a drone image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/model.pt"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with Image.open(args.image) as image:
        image_tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        index = model(image_tensor).argmax(dim=1).item()

    label = classes[index]
    print(json.dumps({"contains_drone": label != "no_drone", "model": LABELS[label]}))


if __name__ == "__main__":
    main()
