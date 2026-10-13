import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import LABELS, build_model, transform


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a whole-image drone classifier")
    parser.add_argument("--data", type=Path, default=Path("data/train"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not args.data.is_dir():
        parser.error(f"training images not found: {args.data}")

    dataset = ImageFolder(args.data, transform=transform)
    unexpected = set(dataset.classes) - set(LABELS)
    if unexpected:
        parser.error(f"unknown class folders: {', '.join(sorted(unexpected))}")
    if "no_drone" not in dataset.classes:
        parser.error("add a no_drone folder with training images")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = build_model(len(dataset.classes), pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        print(f"epoch {epoch + 1}/{args.epochs}: loss={total_loss / len(dataset):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.cpu().state_dict(), "classes": dataset.classes},
        args.output,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
