"""
ResNet18 fine-tuning script for brain tumor MRI classification.

Usage (CLI):
    python -m src.models.train \\
        --data_dir data/raw/Training \\
        --val_dir  data/raw/Testing  \\
        --epochs   10               \\
        --batch    32               \\
        --lr       1e-3             \\
        --output   src/checkpoints/resnet18_braintumor.pt

Or run from the Colab notebook (02_model_training.ipynb).
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import CLASS_NAMES, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
from src.models.cnn_model import build_resnet18


def get_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def train(
    data_dir: str,
    val_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_path: str = "src/checkpoints/resnet18_braintumor.pt",
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on: {device}")

    # Datasets — expects ImageFolder structure (one subfolder per class)
    train_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(train=True))
    val_dataset   = datasets.ImageFolder(val_dir,  transform=get_transforms(train=False))

    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)

    model     = build_resnet18(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    history = []

    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # --- Validation phase ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        epoch_loss = running_loss / len(train_loader)
        history.append({"epoch": epoch, "loss": epoch_loss, "val_accuracy": val_acc})
        print(f"Epoch {epoch}/{epochs}  loss={epoch_loss:.4f}  val_acc={val_acc:.4f}")

    # Save weights
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"Model saved → {out}")

    # Save training history
    history_path = out.parent / "training_history.json"
    with open(history_path, "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"Training history saved → {history_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 on brain tumor MRI dataset")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--val_dir",  required=True)
    parser.add_argument("--epochs",   type=int,   default=10)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--output",   default="src/checkpoints/resnet18_braintumor.pt")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        output_path=args.output,
    )
